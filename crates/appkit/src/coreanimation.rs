//! Core Animation's layer tree and basic animations, drawn by the compositor.
//!
//! A layer here has geometry, a background colour with rounded corners,
//! `contents` (an `IOSurfaceRef`, a `CGImageRef` or an `NSImage`) placed by
//! `contentsGravity`, sublayers, `masksToBounds`, opacity and running
//! animations. That is what a program builds a progress bar, a spinner, a badge
//! or a highlight out of, and what a layer-backed view's content arrives as.
//! Borders, shadows, `contentsRect` and transforms are not drawn, and
//! `masksToBounds` clips to the square bounds whatever the corner radius; a
//! program that relies on them gets its layer's background and contents and
//! no more.
//!
//! The state lives in side tables keyed by object address rather than in
//! `crate::layer::Layer`, whose layout a `CAMetalLayer` extends.

use std::collections::HashMap;
use std::sync::Mutex;

use corefoundation::base::CFTypeRef;
use foundation::geometry::{NSPoint, NSRect, NSSize};
use objc::object::Id;
use objc::sel::Sel;

use crate::backing::{Rect, Store};
use crate::define::{class_methods, methods};

/// The `kCAMediaTimingFunction...` names, published as data by `crate::lib`.
pub const LINEAR: &str = "linear";
pub const EASE_IN: &str = "easeIn";
pub const EASE_OUT: &str = "easeOut";
pub const EASE_IN_EASE_OUT: &str = "easeInEaseOut";
pub const DEFAULT: &str = "default";

/// `kCAFillMode...`, likewise.
pub const FILL_FORWARDS: &str = "forwards";
pub const FILL_BACKWARDS: &str = "backwards";
pub const FILL_BOTH: &str = "both";
pub const FILL_REMOVED: &str = "removed";

/// `kCAFilter...`, likewise.
pub const FILTER_NEAREST: &str = "nearest";
pub const FILTER_LINEAR: &str = "linear";
pub const FILTER_TRILINEAR: &str = "trilinear";

/// `kCAGravity...`, likewise.
pub const GRAVITIES: &[(&str, &str, Gravity)] = &[
    ("_kCAGravityCenter", "center", Gravity::Center),
    ("_kCAGravityTop", "top", Gravity::Top),
    ("_kCAGravityBottom", "bottom", Gravity::Bottom),
    ("_kCAGravityLeft", "left", Gravity::Left),
    ("_kCAGravityRight", "right", Gravity::Right),
    ("_kCAGravityTopLeft", "topLeft", Gravity::TopLeft),
    ("_kCAGravityTopRight", "topRight", Gravity::TopRight),
    ("_kCAGravityBottomLeft", "bottomLeft", Gravity::BottomLeft),
    (
        "_kCAGravityBottomRight",
        "bottomRight",
        Gravity::BottomRight,
    ),
    ("_kCAGravityResize", "resize", Gravity::Resize),
    (
        "_kCAGravityResizeAspect",
        "resizeAspect",
        Gravity::ResizeAspect,
    ),
    (
        "_kCAGravityResizeAspectFill",
        "resizeAspectFill",
        Gravity::ResizeAspectFill,
    ),
];

/// Where a layer's contents sit in its bounds. The edges are the ones a
/// person sees: `Top` is the top of the view on screen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Gravity {
    #[default]
    Resize,
    ResizeAspect,
    ResizeAspectFill,
    Center,
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// The control points of each named timing function, as Apple defines them.
fn named_timing(name: &str) -> Option<[f64; 4]> {
    Some(match name {
        LINEAR => [0.0, 0.0, 1.0, 1.0],
        EASE_IN => [0.42, 0.0, 1.0, 1.0],
        EASE_OUT => [0.0, 0.0, 0.58, 1.0],
        EASE_IN_EASE_OUT => [0.42, 0.0, 0.58, 1.0],
        DEFAULT => [0.25, 0.1, 0.25, 1.0],
        _ => return None,
    })
}

//
// Layers
//

struct Node {
    position: NSPoint,
    anchor: NSPoint,
    /// Premultiplied-later straight RGBA, or `None` for no background.
    background: Option<[f64; 4]>,
    /// The `CGColorRef` `-backgroundColor` answers, retained.
    background_object: usize,
    /// Retained.
    sublayers: Vec<usize>,
    superlayer: usize,
    masks_to_bounds: bool,
    hidden: bool,
    opacity: f64,
    /// `-cornerRadius`, in points.
    corner_radius: f64,
    animations: Vec<(String, Running)>,
    /// `-contents`, retained. A surface also holds a use count while it is
    /// here, which is what tells its owner it is on screen.
    contents: usize,
    gravity: Gravity,
    /// `contents` as premultiplied BGRA, for the kinds that are not already:
    /// the object it was made from, its width and height, and its pixels.
    decoded: Option<(usize, usize, usize, Vec<u8>)>,
}

impl Default for Node {
    fn default() -> Node {
        Node {
            position: NSPoint::default(),
            anchor: NSPoint { x: 0.5, y: 0.5 },
            background: None,
            background_object: 0,
            sublayers: Vec::new(),
            superlayer: 0,
            masks_to_bounds: false,
            hidden: false,
            opacity: 1.0,
            corner_radius: 0.0,
            animations: Vec::new(),
            contents: 0,
            gravity: Gravity::Resize,
            decoded: None,
        }
    }
}

/// An animation as `-addAnimation:forKey:` copied it, and when it began.
struct Running {
    /// The object added, retained so `-animationForKey:` can answer it.
    object: usize,
    spec: Spec,
    start: f64,
}

static NODES: Mutex<Option<HashMap<usize, Node>>> = Mutex::new(None);

fn with_node<T>(layer: Id, f: impl FnOnce(&mut Node) -> T) -> T {
    let mut nodes = NODES.lock().unwrap_or_else(|e| e.into_inner());
    f(nodes
        .get_or_insert_with(HashMap::new)
        .entry(layer as usize)
        .or_default())
}

fn read_node<T>(layer: Id, f: impl FnOnce(Option<&Node>) -> T) -> T {
    let nodes = NODES.lock().unwrap_or_else(|e| e.into_inner());
    f(nodes
        .as_ref()
        .and_then(|nodes| nodes.get(&(layer as usize))))
}

/// Drop a layer's tree state as it is deallocated.
pub(crate) fn forget(layer: Id) {
    let node = NODES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|nodes| nodes.remove(&(layer as usize)));
    let Some(node) = node else {
        return;
    };
    for sublayer in &node.sublayers {
        with_node(*sublayer as Id, |child| child.superlayer = 0);
        objc::arc::release(*sublayer as Id);
    }
    for (_, running) in &node.animations {
        objc::arc::release(running.object as Id);
    }
    objc::arc::release(node.background_object as Id);
    let_go_of_contents(node.contents);
}

/// The layer at the top of `layer`'s tree.
fn root_of(layer: Id) -> Id {
    let mut at = layer;
    for _ in 0..256 {
        let up = read_node(at, |node| node.map_or(0, |node| node.superlayer));
        if up == 0 {
            break;
        }
        at = up as Id;
    }
    at
}

/// Have the view showing `layer`'s tree recomposed.
pub(crate) fn changed(layer: Id) {
    // SAFETY: a layer of this crate's.
    let view = unsafe { crate::layer::view_of(root_of(layer)) };
    if !view.is_null() {
        crate::backing::damage_view(view);
    }
}

unsafe extern "C" fn frame(receiver: Id, _sel: Sel) -> NSRect {
    let bounds = crate::layer::bounds_of(receiver);
    read_node(receiver, |node| {
        let (position, anchor) = node
            .map_or((NSPoint::default(), NSPoint { x: 0.5, y: 0.5 }), |node| {
                (node.position, node.anchor)
            });
        NSRect {
            origin: NSPoint {
                x: position.x - anchor.x * bounds.size.width,
                y: position.y - anchor.y * bounds.size.height,
            },
            size: bounds.size,
        }
    })
}

/// `-setFrame:`: the bounds take the size, and the position is placed so the
/// anchor point lands where the frame puts it.
unsafe extern "C" fn set_frame(receiver: Id, _sel: Sel, frame: NSRect) {
    let mut bounds = crate::layer::bounds_of(receiver);
    bounds.size = frame.size;
    crate::layer::set_bounds_of(receiver, bounds);
    with_node(receiver, |node| {
        node.position = NSPoint {
            x: frame.origin.x + node.anchor.x * frame.size.width,
            y: frame.origin.y + node.anchor.y * frame.size.height,
        };
    });
    changed(receiver);
}

unsafe extern "C" fn position(receiver: Id, _sel: Sel) -> NSPoint {
    read_node(receiver, |node| {
        node.map_or(NSPoint::default(), |node| node.position)
    })
}

unsafe extern "C" fn set_position(receiver: Id, _sel: Sel, position: NSPoint) {
    with_node(receiver, |node| node.position = position);
    changed(receiver);
}

unsafe extern "C" fn anchor_point(receiver: Id, _sel: Sel) -> NSPoint {
    read_node(receiver, |node| {
        node.map_or(NSPoint { x: 0.5, y: 0.5 }, |node| node.anchor)
    })
}

unsafe extern "C" fn set_anchor_point(receiver: Id, _sel: Sel, anchor: NSPoint) {
    with_node(receiver, |node| node.anchor = anchor);
    changed(receiver);
}

unsafe extern "C" fn background_color(receiver: Id, _sel: Sel) -> CFTypeRef {
    read_node(receiver, |node| {
        node.map_or(std::ptr::null(), |node| node.background_object as CFTypeRef)
    })
}

/// `-setBackgroundColor:`, a `CGColorRef`.
unsafe extern "C" fn set_background_color(receiver: Id, _sel: Sel, color: CFTypeRef) {
    // SAFETY: null or a `CGColor`.
    let components = unsafe { crate::cgcontext::CGColorGetComponents(color) };
    let background = (!components.is_null()).then(|| {
        // SAFETY: a colour here always has four components.
        let c = unsafe { std::slice::from_raw_parts(components, 4) };
        [c[0], c[1], c[2], c[3]]
    });
    let retained = objc::arc::retain(color as Id) as usize;
    let old = with_node(receiver, |node| {
        node.background = background;
        std::mem::replace(&mut node.background_object, retained)
    });
    objc::arc::release(old as Id);
    changed(receiver);
}

unsafe extern "C" fn sublayers(receiver: Id, _sel: Sel) -> Id {
    let children: Vec<CFTypeRef> = read_node(receiver, |node| {
        node.map_or_else(Vec::new, |node| {
            node.sublayers.iter().map(|l| *l as CFTypeRef).collect()
        })
    });
    if children.is_empty() {
        return std::ptr::null_mut();
    }
    let array = corefoundation::collections::new_array(
        &children,
        corefoundation::callbacks::Policy::from_type_callbacks(),
        false,
    );
    // SAFETY: a fresh array.
    unsafe { corefoundation::api::CFAutorelease(array) as Id }
}

unsafe extern "C" fn superlayer(receiver: Id, _sel: Sel) -> Id {
    read_node(receiver, |node| node.map_or(0, |node| node.superlayer)) as Id
}

/// Take `layer` out of its superlayer's list, releasing the list's reference.
fn detach(layer: Id) {
    let parent = with_node(layer, |node| std::mem::take(&mut node.superlayer));
    if parent == 0 {
        return;
    }
    let removed = with_node(parent as Id, |node| {
        let before = node.sublayers.len();
        node.sublayers.retain(|child| *child != layer as usize);
        before != node.sublayers.len()
    });
    changed(parent as Id);
    if removed {
        objc::arc::release(layer);
    }
}

fn insert(receiver: Id, layer: Id, index: Option<usize>) {
    if layer.is_null() || layer == receiver {
        return;
    }
    objc::arc::retain(layer);
    detach(layer);
    with_node(receiver, |node| {
        let at = index
            .unwrap_or(node.sublayers.len())
            .min(node.sublayers.len());
        node.sublayers.insert(at, layer as usize);
    });
    with_node(layer, |node| node.superlayer = receiver as usize);
    changed(receiver);
}

unsafe extern "C" fn add_sublayer(receiver: Id, _sel: Sel, layer: Id) {
    insert(receiver, layer, None);
}

unsafe extern "C" fn insert_sublayer_at_index(receiver: Id, _sel: Sel, layer: Id, index: u32) {
    insert(receiver, layer, Some(index as usize));
}

unsafe extern "C" fn remove_from_superlayer(receiver: Id, _sel: Sel) {
    detach(receiver);
}

unsafe extern "C" fn masks_to_bounds(receiver: Id, _sel: Sel) -> u8 {
    read_node(receiver, |node| {
        u8::from(node.is_some_and(|node| node.masks_to_bounds))
    })
}

pub(crate) fn set_masks(layer: Id, masks: bool) {
    with_node(layer, |node| node.masks_to_bounds = masks);
    changed(layer);
}

unsafe extern "C" fn set_masks_to_bounds(receiver: Id, _sel: Sel, value: u8) {
    set_masks(receiver, value != 0);
}

unsafe extern "C" fn corner_radius(receiver: Id, _sel: Sel) -> f64 {
    read_node(receiver, |node| node.map_or(0.0, |node| node.corner_radius))
}

unsafe extern "C" fn set_corner_radius(receiver: Id, _sel: Sel, value: f64) {
    with_node(receiver, |node| node.corner_radius = value.max(0.0));
    changed(receiver);
}

unsafe extern "C" fn is_hidden(receiver: Id, _sel: Sel) -> u8 {
    read_node(receiver, |node| {
        u8::from(node.is_some_and(|node| node.hidden))
    })
}

unsafe extern "C" fn set_hidden(receiver: Id, _sel: Sel, value: u8) {
    with_node(receiver, |node| node.hidden = value != 0);
    changed(receiver);
}

unsafe extern "C" fn opacity(receiver: Id, _sel: Sel) -> f32 {
    read_node(receiver, |node| node.map_or(1.0, |node| node.opacity)) as f32
}

unsafe extern "C" fn set_opacity(receiver: Id, _sel: Sel, value: f32) {
    with_node(receiver, |node| {
        node.opacity = f64::from(value).clamp(0.0, 1.0)
    });
    changed(receiver);
}

/// Stop showing `contents`: its use count and the layer's reference go.
fn let_go_of_contents(contents: usize) {
    if contents == 0 {
        return;
    }
    if iosurface::is_surface(contents as CFTypeRef) {
        // SAFETY: a live surface this layer retained.
        unsafe { iosurface::IOSurfaceDecrementUseCount(contents as CFTypeRef) };
    }
    objc::arc::release(contents as Id);
}

unsafe extern "C" fn contents(receiver: Id, _sel: Sel) -> Id {
    read_node(receiver, |node| node.map_or(0, |node| node.contents)) as Id
}

/// `-setContents:`. A surface shown by a layer is in use for as long as it is
/// shown, as the window server's use of one is on a Mac.
unsafe extern "C" fn set_contents(receiver: Id, _sel: Sel, contents: Id) {
    let retained = objc::arc::retain(contents) as usize;
    if iosurface::is_surface(contents as CFTypeRef) {
        // SAFETY: a live surface.
        unsafe { iosurface::IOSurfaceIncrementUseCount(contents as CFTypeRef) };
    }
    let old = with_node(receiver, |node| {
        node.decoded = None;
        std::mem::replace(&mut node.contents, retained)
    });
    let_go_of_contents(old);
    changed(receiver);
}

unsafe extern "C" fn contents_gravity(receiver: Id, _sel: Sel) -> Id {
    let gravity = read_node(receiver, |node| {
        node.map_or(Gravity::Resize, |node| node.gravity)
    });
    let name = GRAVITIES
        .iter()
        .find(|(_, _, g)| *g == gravity)
        .map_or("resize", |(_, name, _)| name);
    corefoundation::base::interned_name(name) as usize as Id
}

/// `-setContentsGravity:`, whose names are Core Animation's; one it does not
/// define leaves the gravity as it was.
unsafe extern "C" fn set_contents_gravity(receiver: Id, _sel: Sel, name: Id) {
    // SAFETY: an `NSString`, or nil.
    let name = unsafe { text(name) };
    if let Some((_, _, gravity)) = GRAVITIES.iter().find(|(_, n, _)| *n == name) {
        set_gravity(receiver, *gravity);
    }
}

pub(crate) fn set_gravity(layer: Id, gravity: Gravity) {
    with_node(layer, |node| node.gravity = gravity);
    changed(layer);
}

//
// Animations
//

/// What an animation object says, copied into a layer when it is added.
#[derive(Clone)]
struct Spec {
    key_path: String,
    from: Option<f64>,
    to: Option<f64>,
    by: Option<f64>,
    /// The value objects, retained, for the getters.
    from_object: usize,
    to_object: usize,
    by_object: usize,
    duration: f64,
    repeat_count: f64,
    repeat_duration: f64,
    autoreverses: bool,
    speed: f64,
    begin_time: f64,
    timing: [f64; 4],
    timing_object: usize,
    removed_on_completion: bool,
    fill_mode: String,
    delegate: usize,
}

impl Default for Spec {
    fn default() -> Spec {
        Spec {
            key_path: String::new(),
            from: None,
            to: None,
            by: None,
            from_object: 0,
            to_object: 0,
            by_object: 0,
            duration: 0.0,
            repeat_count: 0.0,
            repeat_duration: 0.0,
            autoreverses: false,
            speed: 1.0,
            begin_time: 0.0,
            timing: [0.0, 0.0, 1.0, 1.0],
            timing_object: 0,
            removed_on_completion: true,
            fill_mode: FILL_REMOVED.to_string(),
            delegate: 0,
        }
    }
}

impl Spec {
    fn objects(&self) -> [usize; 4] {
        [
            self.from_object,
            self.to_object,
            self.by_object,
            self.timing_object,
        ]
    }

    fn retain_objects(&self) {
        for object in self.objects() {
            objc::arc::retain(object as Id);
        }
    }

    fn release_objects(&self) {
        for object in self.objects() {
            objc::arc::release(object as Id);
        }
    }
}

static SPECS: Mutex<Option<HashMap<usize, Spec>>> = Mutex::new(None);

fn with_spec<T>(animation: Id, f: impl FnOnce(&mut Spec) -> T) -> T {
    let mut specs = SPECS.lock().unwrap_or_else(|e| e.into_inner());
    f(specs
        .get_or_insert_with(HashMap::new)
        .entry(animation as usize)
        .or_default())
}

fn new_object(class: Id) -> Id {
    // SAFETY: the receiver of a class method is a class object.
    let class =
        unsafe { objc::class::from_ptr(class as *const objc::class::Class) }.expect("a class");
    objc::object::create_instance(class, 0)
}

/// `+animation`.
unsafe extern "C" fn animation(class: Id, _sel: Sel) -> Id {
    let object = new_object(class);
    with_spec(object, |_| ());
    objc::arc::autorelease(object)
}

/// `+animationWithKeyPath:`.
unsafe extern "C" fn animation_with_key_path(class: Id, _sel: Sel, key_path: Id) -> Id {
    let object = new_object(class);
    // SAFETY: an `NSString`, or nil.
    let key_path = unsafe { text(key_path) };
    with_spec(object, |spec| spec.key_path = key_path);
    objc::arc::autorelease(object)
}

unsafe extern "C" fn animation_init(receiver: Id, _sel: Sel) -> Id {
    with_spec(receiver, |_| ());
    receiver
}

unsafe extern "C" fn animation_dealloc(receiver: Id, _sel: Sel) {
    let spec = SPECS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|specs| specs.remove(&(receiver as usize)));
    if let Some(spec) = spec {
        spec.release_objects();
    }
    objc::arc::finalize(receiver);
}

/// # Safety
///
/// `string` is null or an `NSString`.
unsafe fn text(string: Id) -> String {
    match string.is_null() {
        true => String::new(),
        false => {
            String::from_utf8_lossy(&unsafe { foundation::string::bytes(string) }).into_owned()
        }
    }
}

fn autoreleased_string(value: &str) -> Id {
    let string = foundation::string::from_str(value);
    // SAFETY: a fresh string.
    unsafe { corefoundation::api::CFAutorelease(string as CFTypeRef) as Id }
}

/// A scalar animation value: an `NSNumber`, read by `-doubleValue`.
fn number(value: Id) -> Option<f64> {
    if value.is_null() {
        return None;
    }
    let number = objc::class::lookup(c"NSNumber")?;
    // SAFETY: an object the runtime allocated.
    let is_number = unsafe { objc::object::class_of(value) }
        .is_some_and(|class| class.chain().any(|c| std::ptr::eq(c, number)));
    if !is_number {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            eprintln!(
                "rosemary: appkit: a CABasicAnimation value that is not an NSNumber is not \
                 interpolated here; the property keeps its model value"
            );
        });
        return None;
    }
    // SAFETY: `-doubleValue` is `double (id, SEL)`.
    Some(unsafe {
        let f: unsafe extern "C" fn(Id, Sel) -> f64 = std::mem::transmute(objc::msg::entry());
        f(value, objc::sel::get("doubleValue"))
    })
}

unsafe extern "C" fn key_path(receiver: Id, _sel: Sel) -> Id {
    autoreleased_string(&with_spec(receiver, |spec| spec.key_path.clone()))
}

unsafe extern "C" fn set_key_path(receiver: Id, _sel: Sel, value: Id) {
    // SAFETY: an `NSString`, or nil.
    let value = unsafe { text(value) };
    with_spec(receiver, |spec| spec.key_path = value);
}

fn set_value(
    receiver: Id,
    value: Id,
    slot: impl FnOnce(&mut Spec) -> (&mut usize, &mut Option<f64>),
) {
    let parsed = number(value);
    let retained = objc::arc::retain(value) as usize;
    let old = with_spec(receiver, |spec| {
        let (object, number) = slot(spec);
        *number = parsed;
        std::mem::replace(object, retained)
    });
    objc::arc::release(old as Id);
}

unsafe extern "C" fn set_from_value(receiver: Id, _sel: Sel, value: Id) {
    set_value(receiver, value, |spec| {
        (&mut spec.from_object, &mut spec.from)
    });
}

unsafe extern "C" fn from_value(receiver: Id, _sel: Sel) -> Id {
    with_spec(receiver, |spec| spec.from_object) as Id
}

unsafe extern "C" fn set_to_value(receiver: Id, _sel: Sel, value: Id) {
    set_value(receiver, value, |spec| (&mut spec.to_object, &mut spec.to));
}

unsafe extern "C" fn to_value(receiver: Id, _sel: Sel) -> Id {
    with_spec(receiver, |spec| spec.to_object) as Id
}

unsafe extern "C" fn set_by_value(receiver: Id, _sel: Sel, value: Id) {
    set_value(receiver, value, |spec| (&mut spec.by_object, &mut spec.by));
}

unsafe extern "C" fn by_value(receiver: Id, _sel: Sel) -> Id {
    with_spec(receiver, |spec| spec.by_object) as Id
}

unsafe extern "C" fn set_duration(receiver: Id, _sel: Sel, value: f64) {
    with_spec(receiver, |spec| spec.duration = value);
}

unsafe extern "C" fn duration(receiver: Id, _sel: Sel) -> f64 {
    with_spec(receiver, |spec| spec.duration)
}

unsafe extern "C" fn set_repeat_count(receiver: Id, _sel: Sel, value: f32) {
    with_spec(receiver, |spec| spec.repeat_count = f64::from(value));
}

unsafe extern "C" fn repeat_count(receiver: Id, _sel: Sel) -> f32 {
    with_spec(receiver, |spec| spec.repeat_count) as f32
}

unsafe extern "C" fn set_repeat_duration(receiver: Id, _sel: Sel, value: f64) {
    with_spec(receiver, |spec| spec.repeat_duration = value);
}

unsafe extern "C" fn repeat_duration(receiver: Id, _sel: Sel) -> f64 {
    with_spec(receiver, |spec| spec.repeat_duration)
}

unsafe extern "C" fn set_autoreverses(receiver: Id, _sel: Sel, value: u8) {
    with_spec(receiver, |spec| spec.autoreverses = value != 0);
}

unsafe extern "C" fn autoreverses(receiver: Id, _sel: Sel) -> u8 {
    with_spec(receiver, |spec| u8::from(spec.autoreverses))
}

unsafe extern "C" fn set_speed(receiver: Id, _sel: Sel, value: f32) {
    with_spec(receiver, |spec| spec.speed = f64::from(value));
}

unsafe extern "C" fn speed(receiver: Id, _sel: Sel) -> f32 {
    with_spec(receiver, |spec| spec.speed) as f32
}

unsafe extern "C" fn set_begin_time(receiver: Id, _sel: Sel, value: f64) {
    with_spec(receiver, |spec| spec.begin_time = value);
}

unsafe extern "C" fn begin_time(receiver: Id, _sel: Sel) -> f64 {
    with_spec(receiver, |spec| spec.begin_time)
}

unsafe extern "C" fn set_removed_on_completion(receiver: Id, _sel: Sel, value: u8) {
    with_spec(receiver, |spec| spec.removed_on_completion = value != 0);
}

unsafe extern "C" fn is_removed_on_completion(receiver: Id, _sel: Sel) -> u8 {
    with_spec(receiver, |spec| u8::from(spec.removed_on_completion))
}

unsafe extern "C" fn set_fill_mode(receiver: Id, _sel: Sel, value: Id) {
    // SAFETY: an `NSString`, or nil.
    let value = unsafe { text(value) };
    with_spec(receiver, |spec| spec.fill_mode = value);
}

unsafe extern "C" fn fill_mode(receiver: Id, _sel: Sel) -> Id {
    autoreleased_string(&with_spec(receiver, |spec| spec.fill_mode.clone()))
}

/// `-setDelegate:`. Kept for the getter; `-animationDidStart:` and
/// `-animationDidStop:finished:` are not sent.
unsafe extern "C" fn set_animation_delegate(receiver: Id, _sel: Sel, value: Id) {
    let retained = objc::arc::retain(value) as usize;
    let old = with_spec(receiver, |spec| {
        std::mem::replace(&mut spec.delegate, retained)
    });
    objc::arc::release(old as Id);
}

unsafe extern "C" fn animation_delegate(receiver: Id, _sel: Sel) -> Id {
    with_spec(receiver, |spec| spec.delegate) as Id
}

unsafe extern "C" fn set_timing_function(receiver: Id, _sel: Sel, function: Id) {
    let points = timing_points(function).unwrap_or([0.0, 0.0, 1.0, 1.0]);
    let retained = objc::arc::retain(function) as usize;
    let old = with_spec(receiver, |spec| {
        spec.timing = points;
        std::mem::replace(&mut spec.timing_object, retained)
    });
    objc::arc::release(old as Id);
}

unsafe extern "C" fn timing_function(receiver: Id, _sel: Sel) -> Id {
    with_spec(receiver, |spec| spec.timing_object) as Id
}

/// `-copyWithZone:`: a new animation saying the same things.
unsafe extern "C" fn animation_copy(receiver: Id, _sel: Sel, _zone: *mut std::ffi::c_void) -> Id {
    // SAFETY: an object the runtime allocated.
    let class = unsafe { objc::object::class_of(receiver) }.expect("an object");
    let copy = objc::object::create_instance(class, 0);
    let spec = with_spec(receiver, |spec| spec.clone());
    spec.retain_objects();
    objc::arc::retain(spec.delegate as Id);
    with_spec(copy, |slot| *slot = spec);
    copy
}

unsafe extern "C" fn animation_copy_plain(receiver: Id, sel: Sel) -> Id {
    // SAFETY: as `-copyWithZone:`.
    unsafe { animation_copy(receiver, sel, std::ptr::null_mut()) }
}

unsafe extern "C" fn add_animation(receiver: Id, _sel: Sel, animation: Id, key: Id) {
    if animation.is_null() {
        return;
    }
    let spec = with_spec(animation, |spec| spec.clone());
    if !is_animatable(&spec.key_path) {
        say_unanimated(&spec.key_path);
    }
    spec.retain_objects();
    // SAFETY: an `NSString`, or nil; a nil key is a key of its own.
    let key = unsafe { text(key) };
    let start = match spec.begin_time {
        0.0 => darwin::mach::media_time(),
        begin => begin,
    };
    let running = Running {
        object: objc::arc::retain(animation) as usize,
        spec,
        start,
    };
    let replaced = with_node(receiver, |node| {
        let old = node
            .animations
            .iter()
            .position(|(k, _)| *k == key)
            .map(|at| node.animations.remove(at).1);
        node.animations.push((key, running));
        old
    });
    if let Some(old) = replaced {
        release_running(&old);
    }
    changed(receiver);
}

fn release_running(running: &Running) {
    running.spec.release_objects();
    objc::arc::release(running.object as Id);
}

unsafe extern "C" fn animation_for_key(receiver: Id, _sel: Sel, key: Id) -> Id {
    // SAFETY: an `NSString`, or nil.
    let key = unsafe { text(key) };
    read_node(receiver, |node| {
        node.and_then(|node| node.animations.iter().find(|(k, _)| *k == key))
            .map_or(0, |(_, running)| running.object)
    }) as Id
}

unsafe extern "C" fn remove_animation_for_key(receiver: Id, _sel: Sel, key: Id) {
    // SAFETY: an `NSString`, or nil.
    let key = unsafe { text(key) };
    let removed = with_node(receiver, |node| {
        node.animations
            .iter()
            .position(|(k, _)| *k == key)
            .map(|at| node.animations.remove(at).1)
    });
    if let Some(running) = removed {
        release_running(&running);
        changed(receiver);
    }
}

unsafe extern "C" fn remove_all_animations(receiver: Id, _sel: Sel) {
    let removed = with_node(receiver, |node| std::mem::take(&mut node.animations));
    for (_, running) in &removed {
        release_running(running);
    }
    changed(receiver);
}

unsafe extern "C" fn animation_keys(receiver: Id, _sel: Sel) -> Id {
    let keys: Vec<String> = read_node(receiver, |node| {
        node.map_or_else(Vec::new, |node| {
            node.animations.iter().map(|(k, _)| k.clone()).collect()
        })
    });
    if keys.is_empty() {
        return std::ptr::null_mut();
    }
    let strings: Vec<CFTypeRef> = keys
        .iter()
        .map(|k| foundation::string::from_str(k) as CFTypeRef)
        .collect();
    let array = corefoundation::collections::new_array(
        &strings,
        corefoundation::callbacks::Policy::from_type_callbacks(),
        false,
    );
    for string in strings {
        objc::arc::release(string as Id);
    }
    // SAFETY: a fresh array.
    unsafe { corefoundation::api::CFAutorelease(array) as Id }
}

/// The key paths this module interpolates.
fn is_animatable(key_path: &str) -> bool {
    matches!(
        key_path,
        "position.x" | "position.y" | "opacity" | "bounds.size.width" | "bounds.size.height"
    )
}

/// Name a key path that is not animated here, once each.
fn say_unanimated(key_path: &str) {
    static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.iter().any(|k| k == key_path) {
        return;
    }
    said.push(key_path.to_string());
    eprintln!(
        "rosemary: appkit: a CABasicAnimation of {key_path:?} is not animated here; the layer \
         shows its model value"
    );
}

//
// Timing functions
//

static TIMINGS: Mutex<Option<HashMap<usize, [f64; 4]>>> = Mutex::new(None);

fn timing_points(function: Id) -> Option<[f64; 4]> {
    TIMINGS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()?
        .get(&(function as usize))
        .copied()
}

fn new_timing(class: Id, points: [f64; 4]) -> Id {
    let object = new_object(class);
    TIMINGS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(object as usize, points);
    objc::arc::autorelease(object)
}

/// `+functionWithName:`, nil for a name Core Animation does not define.
unsafe extern "C" fn function_with_name(class: Id, _sel: Sel, name: Id) -> Id {
    // SAFETY: an `NSString`, or nil.
    match named_timing(&unsafe { text(name) }) {
        Some(points) => new_timing(class, points),
        None => std::ptr::null_mut(),
    }
}

unsafe extern "C" fn function_with_control_points(
    class: Id,
    _sel: Sel,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> Id {
    new_timing(class, [x1, y1, x2, y2].map(f64::from))
}

unsafe extern "C" fn init_with_control_points(
    receiver: Id,
    _sel: Sel,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> Id {
    TIMINGS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(receiver as usize, [x1, y1, x2, y2].map(f64::from));
    receiver
}

unsafe extern "C" fn get_control_point(receiver: Id, _sel: Sel, index: usize, out: *mut f32) {
    if out.is_null() {
        return;
    }
    let [x1, y1, x2, y2] = timing_points(receiver).unwrap_or([0.0, 0.0, 1.0, 1.0]);
    let (x, y) = match index {
        1 => (x1, y1),
        2 => (x2, y2),
        3 => (1.0, 1.0),
        _ => (0.0, 0.0),
    };
    // SAFETY: the caller's `float[2]`.
    unsafe {
        *out = x as f32;
        *out.add(1) = y as f32;
    }
}

unsafe extern "C" fn timing_dealloc(receiver: Id, _sel: Sel) {
    if let Some(timings) = TIMINGS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        timings.remove(&(receiver as usize));
    }
    objc::arc::finalize(receiver);
}

/// The cubic Bezier from (0,0) through the control points to (1,1), evaluated
/// at `x`: the curve's parameter is found by bisection, since x is monotonic.
fn ease(points: [f64; 4], x: f64) -> f64 {
    let [x1, y1, x2, y2] = points;
    let bezier = |t: f64, a: f64, b: f64| {
        let u = 1.0 - t;
        3.0 * u * u * t * a + 3.0 * u * t * t * b + t * t * t
    };
    let x = x.clamp(0.0, 1.0);
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..48 {
        let mid = (low + high) / 2.0;
        if bezier(mid, x1, x2) < x {
            low = mid;
        } else {
            high = mid;
        }
    }
    bezier((low + high) / 2.0, y1, y2)
}

/// Where an animation is at `now`: `Some(fraction)` of the way from its start
/// value to its end value, or `None` when it shows the model value. The
/// second value says whether it has finished.
fn progress(spec: &Spec, start: f64, now: f64) -> (Option<f64>, bool) {
    let fills_forwards = matches!(spec.fill_mode.as_str(), FILL_FORWARDS | FILL_BOTH);
    let fills_backwards = matches!(spec.fill_mode.as_str(), FILL_BACKWARDS | FILL_BOTH);
    let duration = match spec.duration > 0.0 {
        true => spec.duration,
        // Core Animation's default.
        false => 0.25,
    };
    let local = (now - start) * spec.speed;
    if local < 0.0 {
        return (fills_backwards.then(|| ease(spec.timing, 0.0)), false);
    }
    let cycle = duration * if spec.autoreverses { 2.0 } else { 1.0 };
    let total = match (spec.repeat_count > 0.0, spec.repeat_duration > 0.0) {
        (true, _) => spec.repeat_count as f64 * cycle,
        (false, true) => spec.repeat_duration,
        (false, false) => cycle,
    };
    let (phase, finished) = match local >= total {
        true => (total - (total / cycle).floor() * cycle, true),
        false => (local % cycle, false),
    };
    // A finished animation lands on its last frame, which is the end of a
    // cycle rather than the start of the next one.
    let phase = match finished && phase == 0.0 {
        true => cycle,
        false => phase,
    };
    let mut fraction = phase / duration;
    if fraction > 1.0 {
        fraction = 2.0 - fraction;
    }
    let value = ease(spec.timing, fraction);
    match finished {
        true => (fills_forwards.then_some(value), true),
        false => (Some(value), false),
    }
}

/// The value `key_path` shows at `now`, given its model value.
fn presented(animations: &[(String, Running)], key_path: &str, model: f64, now: f64) -> f64 {
    let mut value = model;
    for (_, running) in animations {
        let spec = &running.spec;
        if spec.key_path != key_path {
            continue;
        }
        let (Some(fraction), _) = progress(spec, running.start, now) else {
            continue;
        };
        let from = spec.from.unwrap_or(match (spec.to, spec.by) {
            (Some(to), Some(by)) => to - by,
            _ => model,
        });
        let to = spec.to.or(spec.by.map(|by| from + by)).unwrap_or(model);
        value = from + (to - from) * fraction;
    }
    value
}

/// Drop animations that have finished and asked to be removed; whether any
/// are still running.
fn retire(layer: Id, now: f64) -> bool {
    let (done, running) = with_node(layer, |node| {
        let mut done = Vec::new();
        node.animations.retain(|(_, running)| {
            let (_, finished) = progress(&running.spec, running.start, now);
            let remove = finished && running.spec.removed_on_completion;
            if remove {
                done.push((running.object, running.spec.clone()));
            }
            !remove
        });
        let running = node
            .animations
            .iter()
            .any(|(_, r)| !progress(&r.spec, r.start, now).1);
        (done, running)
    });
    for (object, spec) in done {
        spec.release_objects();
        objc::arc::release(object as Id);
    }
    running
}

//
// Drawing
//

/// A layer's geometry as it is presented at `now`.
struct Presented {
    frame: NSRect,
    bounds_origin: NSPoint,
    background: Option<[f64; 4]>,
    opacity: f64,
    hidden: bool,
    masks_to_bounds: bool,
    corner_radius: f64,
    sublayers: Vec<usize>,
}

fn presentation(layer: Id, now: f64) -> Presented {
    let bounds = crate::layer::bounds_of(layer);
    read_node(layer, |node| {
        let fallback = Node::default();
        let node = node.unwrap_or(&fallback);
        let animations = &node.animations;
        let width = presented(animations, "bounds.size.width", bounds.size.width, now);
        let height = presented(animations, "bounds.size.height", bounds.size.height, now);
        let x = presented(animations, "position.x", node.position.x, now);
        let y = presented(animations, "position.y", node.position.y, now);
        Presented {
            frame: NSRect {
                origin: NSPoint {
                    x: x - node.anchor.x * width,
                    y: y - node.anchor.y * height,
                },
                size: NSSize { width, height },
            },
            bounds_origin: bounds.origin,
            background: node.background,
            opacity: presented(animations, "opacity", node.opacity, now).clamp(0.0, 1.0),
            hidden: node.hidden,
            masks_to_bounds: node.masks_to_bounds,
            corner_radius: node.corner_radius,
            sublayers: node.sublayers.clone(),
        }
    })
}

/// How a view's layer tree lands in its window's pixels.
struct Mapping {
    view_bounds: NSRect,
    rect: Rect,
    scale_x: f64,
    scale_y: f64,
    flipped: bool,
}

impl Mapping {
    /// A rectangle in the root layer's (the view's bounds') coordinates, in
    /// window pixels.
    fn to_pixels(&self, r: NSRect) -> Rect {
        let left = self.rect.x as f64 + (r.origin.x - self.view_bounds.origin.x) * self.scale_x;
        let top = match self.flipped {
            true => self.rect.y as f64 + (r.origin.y - self.view_bounds.origin.y) * self.scale_y,
            false => {
                let max_y = self.view_bounds.origin.y + self.view_bounds.size.height;
                self.rect.y as f64 + (max_y - (r.origin.y + r.size.height)) * self.scale_y
            }
        };
        let right = left + r.size.width * self.scale_x;
        let bottom = top + r.size.height * self.scale_y;
        Rect {
            x: left.round() as i64,
            y: top.round() as i64,
            width: (right.round() - left.round()) as i64,
            height: (bottom.round() - top.round()) as i64,
        }
    }
}

/// Source-over a straight-alpha colour into `store` over `rect`.
fn fill(store: &mut Store, rect: Rect, color: [f64; 4], opacity: f64) {
    let bounds = Rect {
        x: 0,
        y: 0,
        width: i64::from(store.width),
        height: i64::from(store.height),
    };
    let rect = rect.intersection(&bounds);
    let alpha = (color[3] * opacity).clamp(0.0, 1.0);
    if rect.is_empty() || alpha <= 0.0 {
        return;
    }
    let source = [color[2], color[1], color[0]].map(|c| c.clamp(0.0, 1.0) * alpha * 255.0);
    let keep = 1.0 - alpha;
    for row in rect.y..rect.y + rect.height {
        let start = (row as usize * store.width as usize + rect.x as usize) * 4;
        let end = start + rect.width as usize * 4;
        for pixel in store.pixels[start..end].as_chunks_mut::<4>().0 {
            for channel in 0..3 {
                pixel[channel] = (source[channel] + f64::from(pixel[channel]) * keep).round() as u8;
            }
            pixel[3] = (alpha * 255.0 + f64::from(pixel[3]) * keep).round() as u8;
        }
    }
}

/// [`fill`] `shape` with its corners rounded to `radius` pixels, within `clip`.
fn fill_rounded(
    store: &mut Store,
    shape: Rect,
    clip: Rect,
    color: [f64; 4],
    opacity: f64,
    radius: f64,
) {
    let radius = radius
        .min(shape.width as f64 / 2.0)
        .min(shape.height as f64 / 2.0);
    if radius < 0.5 {
        fill(store, shape.intersection(&clip), color, opacity);
        return;
    }
    let bounds = Rect {
        x: 0,
        y: 0,
        width: i64::from(store.width),
        height: i64::from(store.height),
    };
    let area = shape.intersection(&clip).intersection(&bounds);
    if area.is_empty() {
        return;
    }
    let (left, top) = (shape.x as f64, shape.y as f64);
    let (right, bottom) = (left + shape.width as f64, top + shape.height as f64);
    for row in area.y..area.y + area.height {
        let cy = row as f64 + 0.5;
        // How far into a corner's square this row is, if it is in one.
        let dy = (top + radius - cy).max(cy - (bottom - radius)).max(0.0);
        if dy <= 0.0 {
            fill(
                store,
                Rect {
                    x: area.x,
                    y: row,
                    width: area.width,
                    height: 1,
                },
                color,
                opacity,
            );
            continue;
        }
        for column in area.x..area.x + area.width {
            let cx = column as f64 + 0.5;
            let dx = (left + radius - cx).max(cx - (right - radius)).max(0.0);
            // Coverage from the distance to the corner's circle, one pixel of
            // anti-aliasing wide.
            let coverage = (radius - (dx * dx + dy * dy).sqrt() + 0.5).clamp(0.0, 1.0);
            if coverage > 0.0 {
                fill(
                    store,
                    Rect {
                        x: column,
                        y: row,
                        width: 1,
                        height: 1,
                    },
                    color,
                    opacity * coverage,
                );
            }
        }
    }
}

/// A layer's contents as pixels to draw: premultiplied B, G, R, A rows, or
/// R, G, B, A where `swap` says so.
struct Source<'a> {
    width: usize,
    height: usize,
    stride: usize,
    bytes: &'a [u8],
    swap: bool,
}

/// Say once that a surface in `format` is not drawn.
fn say_undrawn_format(format: u32) {
    static SAID: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if !said.contains(&format) {
        said.push(format);
        eprintln!(
            "rosemary: appkit: layer contents in IOSurface pixel format {:?} are not drawn; only \
             'BGRA' and 'RGBA' are",
            String::from_utf8_lossy(&format.to_be_bytes())
        );
    }
}

/// Premultiplied BGRA pixels of a `CGImage` or an `NSImage`, top row first.
fn decode_contents(contents: usize) -> Option<(usize, usize, Vec<u8>)> {
    if let Some(decoded) = crate::graphics::image_bgra(contents as CFTypeRef) {
        return Some(decoded);
    }
    let bitmap = crate::image::bitmap(contents as Id)?;
    let mut pixels = bitmap.pixels;
    for texel in pixels.as_chunks_mut::<4>().0 {
        let a = u16::from(texel[3]);
        let scale = |c: u8| ((u16::from(c) * a + 127) / 255) as u8;
        *texel = [scale(texel[2]), scale(texel[1]), scale(texel[0]), texel[3]];
    }
    Some((bitmap.width as usize, bitmap.height as usize, pixels))
}

/// Run `f` over `layer`'s contents, if it has any this can draw.
fn with_contents<T>(layer: Id, f: impl FnOnce(Source) -> T) -> Option<T> {
    let (contents, cached) = read_node(layer, |node| {
        node.map_or((0, false), |node| {
            let cached = node
                .decoded
                .as_ref()
                .is_some_and(|(object, ..)| *object == node.contents);
            (node.contents, cached)
        })
    });
    if contents == 0 {
        return None;
    }
    if let Some(surface) = iosurface::pixels(contents as CFTypeRef) {
        let swap = match surface.pixel_format {
            iosurface::PIXEL_FORMAT_BGRA => false,
            format if format == u32::from_be_bytes(*b"RGBA") => true,
            format => {
                say_undrawn_format(format);
                return None;
            }
        };
        if surface.bytes_per_element != 4 || surface.width == 0 || surface.height == 0 {
            return None;
        }
        // SAFETY: the surface is retained by the layer, and its allocation
        // covers `bytes_per_row` bytes for each of its rows.
        let bytes = unsafe {
            std::slice::from_raw_parts(surface.data, surface.bytes_per_row * surface.height)
        };
        return Some(f(Source {
            width: surface.width,
            height: surface.height,
            stride: surface.bytes_per_row,
            bytes,
            swap,
        }));
    }
    if !cached {
        let decoded = decode_contents(contents)?;
        with_node(layer, |node| {
            if node.contents == contents {
                node.decoded = Some((contents, decoded.0, decoded.1, decoded.2));
            }
        });
    }
    read_node(layer, |node| {
        let (_, width, height, pixels) = node?.decoded.as_ref()?;
        Some(f(Source {
            width: *width,
            height: *height,
            stride: width * 4,
            bytes: pixels,
            swap: false,
        }))
    })
}

/// Where contents of `width` by `height` window pixels land in `bounds`.
fn place(gravity: Gravity, bounds: Rect, width: f64, height: f64) -> (f64, f64, f64, f64) {
    let (bx, by) = (bounds.x as f64, bounds.y as f64);
    let (bw, bh) = (bounds.width as f64, bounds.height as f64);
    let fit = |scale: f64| {
        let (w, h) = (width * scale, height * scale);
        (bx + (bw - w) / 2.0, by + (bh - h) / 2.0, w, h)
    };
    match gravity {
        Gravity::Resize => return (bx, by, bw, bh),
        Gravity::ResizeAspect => return fit((bw / width).min(bh / height)),
        Gravity::ResizeAspectFill => return fit((bw / width).max(bh / height)),
        _ => {}
    }
    let x = match gravity {
        Gravity::Left | Gravity::TopLeft | Gravity::BottomLeft => bx,
        Gravity::Right | Gravity::TopRight | Gravity::BottomRight => bx + bw - width,
        _ => bx + (bw - width) / 2.0,
    };
    let y = match gravity {
        Gravity::Top | Gravity::TopLeft | Gravity::TopRight => by,
        Gravity::Bottom | Gravity::BottomLeft | Gravity::BottomRight => by + bh - height,
        _ => by + (bh - height) / 2.0,
    };
    (x, y, width, height)
}

/// Draw `layer`'s contents into `store` for a layer at `bounds` in window
/// pixels, `scale` window pixels to the point, within `clip`, source-over.
fn draw_contents(
    store: &mut Store,
    layer: Id,
    bounds: Rect,
    scale: (f64, f64),
    clip: Rect,
    opacity: f64,
) {
    if opacity <= 0.0 || bounds.is_empty() {
        return;
    }
    let gravity = read_node(layer, |node| {
        node.map_or(Gravity::Resize, |node| node.gravity)
    });
    let contents_scale = crate::layer::contents_scale_of(layer);
    with_contents(layer, |source| {
        let (x, y, width, height) = place(
            gravity,
            bounds,
            source.width as f64 / contents_scale * scale.0,
            source.height as f64 / contents_scale * scale.1,
        );
        if width <= 0.0 || height <= 0.0 {
            return;
        }
        let area = Rect {
            x: x.round() as i64,
            y: y.round() as i64,
            width: ((x + width).round() - x.round()) as i64,
            height: ((y + height).round() - y.round()) as i64,
        }
        .intersection(&clip)
        .intersection(&Rect {
            x: 0,
            y: 0,
            width: i64::from(store.width),
            height: i64::from(store.height),
        });
        if area.is_empty() {
            return;
        }
        // A pixel-aligned, unscaled BGRA image needs no nearest-neighbour
        // coordinate calculation. Opaque rows can be copied directly.
        if !source.swap
            && (x - x.round()).abs() < 1e-9
            && (y - y.round()).abs() < 1e-9
            && width == source.width as f64
            && height == source.height as f64
            && opacity >= 1.0
        {
            composite_identity_bgra(store, &source, area, x.round() as i64, y.round() as i64);
            return;
        }
        let columns: Vec<usize> = (area.x..area.x + area.width)
            .map(|column| {
                let u = (column as f64 + 0.5 - x) / width;
                ((u * source.width as f64) as usize).min(source.width - 1)
            })
            .collect();
        let fraction = (opacity.min(1.0) * 256.0).round() as u32;
        for row in area.y..area.y + area.height {
            let v = (row as f64 + 0.5 - y) / height;
            let from = ((v * source.height as f64) as usize).min(source.height - 1);
            let from = &source.bytes[from * source.stride..from * source.stride + source.width * 4];
            let base = (row as usize * store.width as usize + area.x as usize) * 4;
            let to = &mut store.pixels[base..base + area.width as usize * 4];
            for (target, &column) in to.as_chunks_mut::<4>().0.iter_mut().zip(&columns) {
                let t = &from[column * 4..column * 4 + 4];
                let texel = match source.swap {
                    true => [t[2], t[1], t[0], t[3]],
                    false => [t[0], t[1], t[2], t[3]],
                };
                if fraction >= 256 && texel[3] == 255 {
                    *target = texel;
                    continue;
                }
                let alpha = u32::from(texel[3]) * fraction / 256;
                for channel in 0..4 {
                    let s = u32::from(texel[channel]) * fraction / 256;
                    let d = u32::from(target[channel]) * (255 - alpha) / 255;
                    target[channel] = (s + d).min(255) as u8;
                }
            }
        }
    });
}

/// Draw a pixel-aligned BGRA source at native scale, clipping to `area`.
fn composite_identity_bgra(store: &mut Store, source: &Source<'_>, area: Rect, x: i64, y: i64) {
    for row in area.y..area.y + area.height {
        let from = (row - y) as usize * source.stride + (area.x - x) as usize * 4;
        let length = area.width as usize * 4;
        let source_row = &source.bytes[from..from + length];
        let to = (row as usize * store.width as usize + area.x as usize) * 4;
        let target_row = &mut store.pixels[to..to + length];
        if source_row
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| pixel[3] == 255)
        {
            target_row.copy_from_slice(source_row);
        } else {
            for (target, src) in target_row
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(source_row.as_chunks::<4>().0)
            {
                if src[3] == 255 {
                    *target = *src;
                } else {
                    let keep = 255 - u32::from(src[3]);
                    for channel in 0..4 {
                        let blended =
                            u32::from(src[channel]) + u32::from(target[channel]) * keep / 255;
                        target[channel] = blended.min(255) as u8;
                    }
                }
            }
        }
    }
}

/// Which part of a view's layer tree [`render`] draws: the root layer's own
/// background and contents go under the view's drawing, the sublayers over it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pass {
    Background,
    Sublayers,
}

/// Draw `view`'s layer tree into `store`, where the view sits at `rect` and
/// may show within `clip`. Whether anything in it is still animating.
pub(crate) fn render(view: Id, store: &mut Store, rect: Rect, clip: Rect, pass: Pass) -> bool {
    let root = crate::view::layer_of(view);
    if root.is_null() || crate::layer::is_metal(root) {
        return false;
    }
    let view_bounds = crate::view::bounds_of(view);
    if view_bounds.size.width <= 0.0 || view_bounds.size.height <= 0.0 {
        return false;
    }
    let mapping = Mapping {
        view_bounds,
        rect,
        scale_x: rect.width as f64 / view_bounds.size.width,
        scale_y: rect.height as f64 / view_bounds.size.height,
        flipped: crate::view::is_flipped(view),
    };
    let now = darwin::mach::media_time();
    let root_presented = presentation(root, now);
    if root_presented.hidden {
        return false;
    }
    match pass {
        Pass::Background => {
            let whole = mapping.to_pixels(view_bounds);
            if let Some(color) = root_presented.background {
                fill_rounded(
                    store,
                    whole,
                    clip,
                    color,
                    root_presented.opacity,
                    root_presented.corner_radius * mapping.scale_x.min(mapping.scale_y),
                );
            }
            draw_contents(
                store,
                root,
                whole,
                (mapping.scale_x, mapping.scale_y),
                clip,
                root_presented.opacity,
            );
            false
        }
        Pass::Sublayers => {
            let clip = match root_presented.masks_to_bounds {
                true => clip.intersection(&mapping.to_pixels(view_bounds)),
                false => clip,
            };
            let mut animating = retire(root, now);
            // The root layer's space is the view's bounds.
            let origin = NSPoint {
                x: view_bounds.origin.x - root_presented.bounds_origin.x,
                y: view_bounds.origin.y - root_presented.bounds_origin.y,
            };
            for child in root_presented.sublayers {
                animating |= draw_layer(
                    store,
                    &mapping,
                    child as Id,
                    origin,
                    clip,
                    root_presented.opacity,
                    now,
                    0,
                );
            }
            animating
        }
    }
}

/// Draw `layer`, whose superlayer's coordinate space starts at `origin` in the
/// root's space, and its sublayers.
#[allow(clippy::too_many_arguments)]
fn draw_layer(
    store: &mut Store,
    mapping: &Mapping,
    layer: Id,
    origin: NSPoint,
    clip: Rect,
    opacity: f64,
    now: f64,
    depth: usize,
) -> bool {
    if depth > 64 {
        return false;
    }
    let mut animating = retire(layer, now);
    let presented = presentation(layer, now);
    if presented.hidden {
        return animating;
    }
    let frame = NSRect {
        origin: NSPoint {
            x: origin.x + presented.frame.origin.x,
            y: origin.y + presented.frame.origin.y,
        },
        size: presented.frame.size,
    };
    let pixels = mapping.to_pixels(frame);
    let opacity = opacity * presented.opacity;
    if let Some(color) = presented.background {
        let radius = presented.corner_radius * mapping.scale_x.min(mapping.scale_y);
        fill_rounded(store, pixels, clip, color, opacity, radius);
    }
    draw_contents(
        store,
        layer,
        pixels,
        (mapping.scale_x, mapping.scale_y),
        clip,
        opacity,
    );
    let clip = match presented.masks_to_bounds {
        true => clip.intersection(&pixels),
        false => clip,
    };
    let inner = NSPoint {
        x: frame.origin.x - presented.bounds_origin.x,
        y: frame.origin.y - presented.bounds_origin.y,
    };
    for child in presented.sublayers {
        animating |= draw_layer(
            store,
            mapping,
            child as Id,
            inner,
            clip,
            opacity,
            now,
            depth + 1,
        );
    }
    animating
}

/// Whether `layer`'s tree has anything to draw beyond the view's own pixels.
pub(crate) fn has_tree(layer: Id) -> bool {
    read_node(layer, |node| {
        node.is_some_and(|node| {
            node.background.is_some() || node.contents != 0 || !node.sublayers.is_empty()
        })
    })
}

pub(crate) fn install() {
    let layer = objc::class::lookup(c"CALayer").expect("declared before its methods");
    methods!(layer, {
        "frame" => frame, c"{CGRect={CGPoint=dd}{CGSize=dd}}16@0:8";
        "setFrame:" => set_frame, c"v48@0:8{CGRect={CGPoint=dd}{CGSize=dd}}16";
        "position" => position, c"{CGPoint=dd}16@0:8";
        "setPosition:" => set_position, c"v32@0:8{CGPoint=dd}16";
        "anchorPoint" => anchor_point, c"{CGPoint=dd}16@0:8";
        "setAnchorPoint:" => set_anchor_point, c"v32@0:8{CGPoint=dd}16";
        "backgroundColor" => background_color, c"^{CGColor=}16@0:8";
        "setBackgroundColor:" => set_background_color, c"v24@0:8^{CGColor=}16";
        "sublayers" => sublayers, c"@16@0:8";
        "superlayer" => superlayer, c"@16@0:8";
        "addSublayer:" => add_sublayer, c"v24@0:8@16";
        "insertSublayer:atIndex:" => insert_sublayer_at_index, c"v28@0:8@16I24";
        "removeFromSuperlayer" => remove_from_superlayer, c"v16@0:8";
        "masksToBounds" => masks_to_bounds, c"c16@0:8";
        "setMasksToBounds:" => set_masks_to_bounds, c"v20@0:8c16";
        "cornerRadius" => corner_radius, c"d16@0:8";
        "setCornerRadius:" => set_corner_radius, c"v24@0:8d16";
        "isHidden" => is_hidden, c"c16@0:8";
        "setHidden:" => set_hidden, c"v20@0:8c16";
        "opacity" => opacity, c"f16@0:8";
        "contents" => contents, c"@16@0:8";
        "setContents:" => set_contents, c"v24@0:8@16";
        "contentsGravity" => contents_gravity, c"@16@0:8";
        "setContentsGravity:" => set_contents_gravity, c"v24@0:8@16";
        "setOpacity:" => set_opacity, c"v20@0:8f16";
        "addAnimation:forKey:" => add_animation, c"v32@0:8@16@24";
        "animationForKey:" => animation_for_key, c"@24@0:8@16";
        "removeAnimationForKey:" => remove_animation_for_key, c"v24@0:8@16";
        "removeAllAnimations" => remove_all_animations, c"v16@0:8";
        "animationKeys" => animation_keys, c"@16@0:8";
    });

    let base = objc::class::lookup(c"CAAnimation").expect("declared before its methods");
    class_methods!(base, {
        "animation" => animation, c"@16@0:8";
    });
    methods!(base, {
        "init" => animation_init, c"@16@0:8";
        "dealloc" => animation_dealloc, c"v16@0:8";
        "copy" => animation_copy_plain, c"@16@0:8";
        "copyWithZone:" => animation_copy, c"@24@0:8^v16";
        "setDuration:" => set_duration, c"v24@0:8d16";
        "duration" => duration, c"d16@0:8";
        "setRepeatCount:" => set_repeat_count, c"v20@0:8f16";
        "repeatCount" => repeat_count, c"f16@0:8";
        "setRepeatDuration:" => set_repeat_duration, c"v24@0:8d16";
        "repeatDuration" => repeat_duration, c"d16@0:8";
        "setAutoreverses:" => set_autoreverses, c"v20@0:8c16";
        "autoreverses" => autoreverses, c"c16@0:8";
        "setSpeed:" => set_speed, c"v20@0:8f16";
        "speed" => speed, c"f16@0:8";
        "setBeginTime:" => set_begin_time, c"v24@0:8d16";
        "beginTime" => begin_time, c"d16@0:8";
        "setRemovedOnCompletion:" => set_removed_on_completion, c"v20@0:8c16";
        "isRemovedOnCompletion" => is_removed_on_completion, c"c16@0:8";
        "setFillMode:" => set_fill_mode, c"v24@0:8@16";
        "fillMode" => fill_mode, c"@16@0:8";
        "setTimingFunction:" => set_timing_function, c"v24@0:8@16";
        "timingFunction" => timing_function, c"@16@0:8";
        "setDelegate:" => set_animation_delegate, c"v24@0:8@16";
        "delegate" => animation_delegate, c"@16@0:8";
    });

    let property =
        objc::class::lookup(c"CAPropertyAnimation").expect("declared before its methods");
    class_methods!(property, {
        "animationWithKeyPath:" => animation_with_key_path, c"@24@0:8@16";
    });
    methods!(property, {
        "keyPath" => key_path, c"@16@0:8";
        "setKeyPath:" => set_key_path, c"v24@0:8@16";
    });

    let basic = objc::class::lookup(c"CABasicAnimation").expect("declared before its methods");
    methods!(basic, {
        "setFromValue:" => set_from_value, c"v24@0:8@16";
        "fromValue" => from_value, c"@16@0:8";
        "setToValue:" => set_to_value, c"v24@0:8@16";
        "toValue" => to_value, c"@16@0:8";
        "setByValue:" => set_by_value, c"v24@0:8@16";
        "byValue" => by_value, c"@16@0:8";
    });

    let timing =
        objc::class::lookup(c"CAMediaTimingFunction").expect("declared before its methods");
    class_methods!(timing, {
        "functionWithName:" => function_with_name, c"@24@0:8@16";
        "functionWithControlPoints::::" => function_with_control_points, c"@32@0:8f16f20f24f28";
    });
    methods!(timing, {
        "initWithControlPoints::::" => init_with_control_points, c"@32@0:8f16f20f24f28";
        "getControlPointAtIndex:values:" => get_control_point, c"v32@0:8Q16^f24";
        "dealloc" => timing_dealloc, c"v16@0:8";
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(duration: f64, repeat_count: f64) -> Spec {
        Spec {
            key_path: "position.x".into(),
            from: Some(0.0),
            to: Some(100.0),
            duration,
            repeat_count,
            ..Spec::default()
        }
    }

    #[test]
    fn a_linear_animation_moves_evenly_and_repeats() {
        let spec = spec(2.0, f64::INFINITY);
        let running = |start| {
            vec![(
                String::new(),
                Running {
                    object: 0,
                    spec: spec.clone(),
                    start,
                },
            )]
        };
        assert_eq!(
            presented(&running(10.0), "position.x", 5.0, 11.0).round(),
            50.0
        );
        // A second cycle starts over.
        assert_eq!(
            presented(&running(10.0), "position.x", 5.0, 13.0).round(),
            50.0
        );
        // Other key paths keep their model value.
        assert_eq!(presented(&running(10.0), "opacity", 0.5, 11.0), 0.5);
    }

    #[test]
    fn a_finished_animation_returns_to_the_model_value_unless_it_fills_forwards() {
        let mut spec = spec(1.0, 0.0);
        assert_eq!(progress(&spec, 0.0, 2.0), (None, true));
        spec.fill_mode = FILL_FORWARDS.into();
        assert_eq!(progress(&spec, 0.0, 2.0), (Some(1.0), true));
    }

    #[test]
    fn the_ease_curves_keep_their_ends_and_bend_the_middle() {
        let ease_in = named_timing(EASE_IN).unwrap();
        assert!(ease(ease_in, 0.0).abs() < 1e-9);
        assert!((ease(ease_in, 1.0) - 1.0).abs() < 1e-9);
        assert!(ease(ease_in, 0.5) < 0.5);
        let linear = named_timing(LINEAR).unwrap();
        assert!((ease(linear, 0.3) - 0.3).abs() < 1e-6);
    }

    fn send<A, R>(receiver: Id, selector: &str, argument: A) -> R {
        // SAFETY: each test sends a selector whose prototype is `(id, SEL, A)`.
        unsafe {
            let f: unsafe extern "C" fn(Id, Sel, A) -> R = std::mem::transmute(objc::msg::entry());
            f(receiver, objc::sel::get(selector), argument)
        }
    }

    fn rect(x: f64, y: f64, width: f64, height: f64) -> NSRect {
        NSRect {
            origin: NSPoint { x, y },
            size: NSSize { width, height },
        }
    }

    fn red_columns(store: &Store) -> Vec<usize> {
        (0..store.width as usize)
            .filter(|x| store.pixels[x * 4 + 2] == 255)
            .collect()
    }

    #[test]
    fn a_views_sublayer_is_drawn_at_its_animated_position() {
        crate::bootstrap();
        let pool = objc::arc::pool_push();
        let view = objc::object::create_instance(objc::class::lookup(c"NSView").unwrap(), 0);
        let view: Id = send(view, "initWithFrame:", rect(0.0, 0.0, 10.0, 4.0));
        send::<u8, ()>(view, "setWantsLayer:", 1);
        let root: Id = send(view, "layer", ());
        assert!(!root.is_null());
        let class = objc::class::ptr(objc::class::lookup(c"CALayer").unwrap()) as Id;
        let child: Id = send(class, "layer", ());
        send::<NSRect, ()>(child, "setFrame:", rect(0.0, 0.0, 2.0, 4.0));
        // SAFETY: a fresh colour, retained by the layer below.
        let red = unsafe { crate::cgcontext::CGColorCreateGenericRGB(1.0, 0.0, 0.0, 1.0) };
        send::<CFTypeRef, ()>(child, "setBackgroundColor:", red);
        send::<Id, ()>(root, "addSublayer:", child);
        let whole = Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 4,
        };

        let mut store = Store::for_tests(10, 4);
        render(view, &mut store, whole, whole, Pass::Sublayers);
        assert_eq!(red_columns(&store), [0, 1]);

        // Halfway through a slide of `position.x` from 1 to 9: at 5, so the
        // two-point-wide layer covers 4 and 5.
        let animation_class =
            objc::class::ptr(objc::class::lookup(c"CABasicAnimation").unwrap()) as Id;
        let key = foundation::string::from_str("position.x");
        let animation: Id = send(animation_class, "animationWithKeyPath:", key);
        let number = |value: f64| {
            corefoundation::collections::new_number(corefoundation::collections::Number::Float(
                value,
            )) as Id
        };
        let (from, to) = (number(1.0), number(9.0));
        send::<Id, ()>(animation, "setFromValue:", from);
        send::<Id, ()>(animation, "setToValue:", to);
        send::<f64, ()>(animation, "setDuration:", 100.0);
        send::<f64, ()>(
            animation,
            "setBeginTime:",
            darwin::mach::media_time() - 50.0,
        );
        // SAFETY: `-addAnimation:forKey:` is `void (id, SEL, id, id)`.
        unsafe {
            add_animation(
                child,
                objc::sel::get("addAnimation:forKey:"),
                animation,
                key,
            )
        };

        let mut store = Store::for_tests(10, 4);
        let animating = render(view, &mut store, whole, whole, Pass::Sublayers);
        assert!(animating);
        assert_eq!(red_columns(&store), [4, 5]);
        for object in [key, from, to] {
            objc::arc::release(object);
        }
        objc::arc::pool_pop(pool);
    }

    #[test]
    fn identity_bgra_contents_match_per_pixel_blending_and_clip() {
        let mut store = Store::for_tests(5, 4);
        store.pixels.fill(17);
        let mut source_bytes = vec![0u8; 4 * 4 * 4];
        for (i, pixel) in source_bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *pixel = [
                ((i * 7) % 256) as u8,
                40,
                100,
                if i % 5 == 0 { 128 } else { 255 },
            ];
        }
        let source = Source {
            width: 4,
            height: 4,
            stride: 16,
            bytes: &source_bytes,
            swap: false,
        };
        composite_identity_bgra(
            &mut store,
            &source,
            Rect {
                x: 2,
                y: 1,
                width: 3,
                height: 3,
            },
            1,
            0,
        );
        for row in 0..4usize {
            for col in 0..5usize {
                let mut expected = [17u8; 4];
                if (1..4).contains(&row) && (2..5).contains(&col) {
                    let sy = row;
                    let sx = col - 1;
                    let src: [u8; 4] = source_bytes[(sy * 4 + sx) * 4..(sy * 4 + sx + 1) * 4]
                        .try_into()
                        .unwrap();
                    if src[3] == 255 {
                        expected = src;
                    } else {
                        let keep = 255 - u32::from(src[3]);
                        for c in 0..4 {
                            expected[c] = (u32::from(src[c]) + u32::from(expected[c]) * keep / 255)
                                .min(255) as u8;
                        }
                    }
                }
                assert_eq!(
                    &store.pixels[(row * 5 + col) * 4..(row * 5 + col + 1) * 4],
                    &expected,
                    "pixel {col},{row}"
                );
            }
        }
    }

    #[test]
    fn a_fill_blends_over_what_is_there() {
        let mut store = Store::for_tests(2, 1);
        fill(
            &mut store,
            Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            [1.0, 0.0, 0.0, 0.5],
            1.0,
        );
        assert_eq!(&store.pixels[0..4], &[0, 0, 128, 128]);
        assert_eq!(&store.pixels[4..8], &[0, 0, 0, 0]);
    }

    /// A 2x1 BGRA surface: a red pixel then a green one.
    fn surface() -> CFTypeRef {
        let number = |value: i64| {
            corefoundation::collections::new_number(corefoundation::collections::Number::Integer(
                value,
            ))
        };
        let key = |text: &str| corefoundation::base::interned_name(text) as usize as CFTypeRef;
        let options = corefoundation::collections::new_dictionary(
            &[
                (key("IOSurfaceWidth"), number(2)),
                (key("IOSurfaceHeight"), number(1)),
                (key("IOSurfaceBytesPerElement"), number(4)),
                (
                    key("IOSurfacePixelFormat"),
                    number(i64::from(iosurface::PIXEL_FORMAT_BGRA)),
                ),
            ],
            corefoundation::callbacks::Policy::from_key_callbacks(),
            corefoundation::callbacks::Policy::from_type_callbacks(),
            false,
        );
        // SAFETY: a dictionary of numbers, and a surface four bytes a pixel.
        unsafe {
            let surface = iosurface::IOSurfaceCreate(options);
            let base = iosurface::IOSurfaceGetBaseAddress(surface) as *mut u8;
            std::ptr::copy_nonoverlapping([0, 0, 255, 255, 0, 255, 0, 255].as_ptr(), base, 8);
            surface
        }
    }

    unsafe extern "C" fn metal_backing_layer(_receiver: Id, _sel: Sel) -> Id {
        let class = objc::class::ptr(objc::class::lookup(c"CAMetalLayer").unwrap()) as Id;
        send(class, "layer", ())
    }

    /// `-setWantsLayer:` makes the layer with the view's own
    /// `-makeBackingLayer`, as Qt's `QNSView` relies on to get a
    /// `CAMetalLayer`, and the view becomes its delegate.
    #[test]
    fn a_wanted_layer_comes_from_the_views_make_backing_layer() {
        crate::bootstrap();
        let class = objc::class::lookup(c"RosemaryMetalBackedView").unwrap_or_else(|| {
            let class = corefoundation::base::register(
                c"RosemaryMetalBackedView",
                objc::class::lookup(c"NSView"),
                crate::view::INSTANCE_SIZE,
                false,
            );
            methods!(class, {
                "makeBackingLayer" => metal_backing_layer, c"@16@0:8";
            });
            class
        });
        let pool = objc::arc::pool_push();
        let view = objc::object::create_instance(class, 0);
        let view: Id = send(view, "initWithFrame:", rect(0.0, 0.0, 10.0, 4.0));
        assert_eq!(send::<(), u8>(view, "wantsLayer", ()), 0);
        send::<u8, ()>(view, "setWantsLayer:", 1);
        assert_eq!(send::<(), u8>(view, "wantsLayer", ()), 1);
        let layer: Id = send(view, "layer", ());
        assert!(crate::layer::is_metal(layer));
        assert_eq!(crate::layer::delegate_of(layer), view);
        assert_eq!(crate::layer::bounds_of(layer).size.width, 10.0);
        send::<u8, ()>(view, "setWantsLayer:", 0);
        assert!(
            send::<(), Id>(view, "layer", ()).is_null(),
            "AppKit's layer goes"
        );
        objc::arc::pool_pop(pool);
    }

    /// A layer shows a surface at its top-left corner when its gravity says
    /// so, and holds a use count on it for as long as it does.
    #[test]
    fn a_surface_is_drawn_by_gravity_and_in_use_while_shown() {
        crate::bootstrap();
        let pool = objc::arc::pool_push();
        let view = objc::object::create_instance(objc::class::lookup(c"NSView").unwrap(), 0);
        let view: Id = send(view, "initWithFrame:", rect(0.0, 0.0, 4.0, 2.0));
        send::<isize, ()>(view, "setLayerContentsPlacement:", 11);
        assert_eq!(send::<(), isize>(view, "layerContentsPlacement", ()), 11);
        send::<u8, ()>(view, "setWantsLayer:", 1);
        let layer: Id = send(view, "layer", ());
        let surface = surface();
        send::<Id, ()>(layer, "setContents:", surface as Id);
        // SAFETY: a live surface.
        assert_eq!(unsafe { iosurface::IOSurfaceIsInUse(surface) }, 1);

        let whole = Rect {
            x: 0,
            y: 0,
            width: 4,
            height: 2,
        };
        let mut store = Store::for_tests(4, 2);
        render(view, &mut store, whole, whole, Pass::Background);
        // Top-left, unscaled: red, green, then nothing; the second row empty.
        assert_eq!(&store.pixels[0..8], &[0, 0, 255, 255, 0, 255, 0, 255]);
        assert_eq!(&store.pixels[8..32], &[0; 24]);

        send::<Id, ()>(layer, "setContents:", std::ptr::null_mut());
        // SAFETY: a live surface.
        unsafe {
            assert_eq!(iosurface::IOSurfaceIsInUse(surface), 0);
            corefoundation::api::CFRelease(surface);
        }
        objc::arc::pool_pop(pool);
    }

    #[test]
    fn gravity_places_contents_in_the_bounds() {
        let bounds = Rect {
            x: 10,
            y: 20,
            width: 100,
            height: 50,
        };
        assert_eq!(
            place(Gravity::Resize, bounds, 5.0, 5.0),
            (10.0, 20.0, 100.0, 50.0)
        );
        assert_eq!(
            place(Gravity::TopLeft, bounds, 5.0, 5.0),
            (10.0, 20.0, 5.0, 5.0)
        );
        assert_eq!(
            place(Gravity::BottomRight, bounds, 5.0, 5.0),
            (105.0, 65.0, 5.0, 5.0)
        );
        // Fitting a square into a wide rectangle: as tall as it, centred.
        assert_eq!(
            place(Gravity::ResizeAspect, bounds, 10.0, 10.0),
            (35.0, 20.0, 50.0, 50.0)
        );
        assert_eq!(
            place(Gravity::ResizeAspectFill, bounds, 10.0, 10.0),
            (10.0, -5.0, 100.0, 100.0)
        );
    }
}
