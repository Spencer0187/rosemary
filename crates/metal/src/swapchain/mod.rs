mod corners;
mod present;
#[cfg(test)]
mod tests;

use present::*;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, TryLockError};
use std::time::{Duration, Instant};

use ash::vk;
use vk::Handle;

use crate::gpu::{self, Gpu, GpuImage, ImageDescription};

/// How many compositor updates can be queued before one has to finish.
///
/// Independent Core Animation layers are staged before a window presentation.
/// The accumulator submissions must not reuse their command buffers while
/// those small copies are still in the GPU queue.
const FRAMES_IN_FLIGHT: usize = 8;

/// The per-window state presentation goes through.
struct Swapchain {
    gpu: &'static Gpu,
    surface: vk::SurfaceKHR,
    handle: vk::SwapchainKHR,
    format: vk::Format,
    /// The size the swapchain was actually created at, which is the surface's
    /// answer and not necessarily the one that was asked for.
    extent: vk::Extent2D,
    /// The window size that was last asked for, which is what a later frame
    /// is compared against to decide whether the window resized.
    ///
    /// Not [`Self::extent`]: a surface with a `currentExtent` of its own
    /// overrides the request, and comparing against the overridden value
    /// would see a difference on every frame and rebuild the swapchain on
    /// every frame.
    requested: vk::Extent2D,
    /// The event pump never waits for WSI recreation. A window-local
    /// worker consumes the newest requested extent under the presenting lock.
    pending_rebuild: Option<vk::Extent2D>,
    /// vkCreateSwapchainKHR retires oldSwapchain even if creation fails.
    retired: bool,
    images: Vec<vk::Image>,
    /// One presentation semaphore per swapchain image, rather than one per
    /// frame in flight.
    ///
    /// A presentation holds the semaphore it waits on until that image comes
    /// back from `vkAcquireNextImageKHR`, and how many images there are is
    /// the presentation engine's business, not this module's - a swapchain
    /// with four images and two frames in flight would signal a semaphore
    /// the engine was still waiting on. Indexing by the acquired image makes
    /// the acquisition itself the proof that the last presentation of this
    /// image is done with it.
    blitted: Vec<vk::Semaphore>,
    /// Fences attached to presentations through swapchain maintenance1. One
    /// per image lets acquisition identify the fence which may be recycled.
    /// Empty on devices which use the queue-idle fallback.
    presented: Vec<PresentFence>,
    /// Where layers accumulate when the window has more than one. `None`
    /// while a single layer covers the window, which needs no accumulator.
    composite: Option<GpuImage>,
    /// The latest full-window opaque drawable, retained until a later direct
    /// frame replaces it or a second layer needs it as a GPU snapshot.
    direct: Option<DirectLayer>,
    /// The layers which have presented to this window. Its length is what
    /// decides whether the composite is needed.
    layers: Vec<usize>,
    /// For a window whose surface shows its alpha, the pass that makes its
    /// content opaque and rounds its corners. `None` presents opaque.
    corners: Option<corners::Corners>,
    pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame: usize,
    present_mode: vk::PresentModeKHR,
    /// Metal layers accumulate in an offscreen image independently of Vulkan
    /// WSI pacing. The presentation worker publishes the latest window image.
    composite_dirty: bool,
    composite_shape: Option<corners::Shape>,
    composite_dirty_since: Option<Instant>,
    compositor_batch_delay: Duration,
    last_wsi_error: Option<vk::Result>,
    /// A maintenance1 present fence did not signal even after reacquiring
    /// its image. Queue idle is the fallback at retirement for this driver.
    stalled_present_fence: bool,
    #[cfg(test)]
    injected_unsignaled_present_fence: bool,
    #[cfg(test)]
    skipped_stalled_present_fence: bool,
    #[cfg(test)]
    injected_build_vk_error: Option<vk::Result>,
    #[cfg(test)]
    injected_stage_vk_error: Option<vk::Result>,
    #[cfg(test)]
    injected_queue_present_error: Option<vk::Result>,
    #[cfg(test)]
    #[cfg(test)]
    injected_queue_submit_error: Option<vk::Result>,
    /// vkAcquireNextImageKHR succeeded, but its semaphore wait has not yet
    /// been submitted. Failure requires retiring this swapchain, not simply
    /// resetting a fence and reusing the signaled acquisition semaphore.
    acquired_unsubmitted: Option<(vk::Semaphore, vk::Fence)>,
    #[cfg(test)]
    injected_recovery_error: Option<vk::Result>,
    #[cfg(test)]
    injected_build_failure: bool,
    #[cfg(test)]
    injected_unavailable_extent: bool,
    #[cfg(test)]
    rebuild_attempts: usize,
    #[cfg(test)]
    last_create_old_swapchain: vk::SwapchainKHR,
}

/// A backing-store GPU upload can be reused only after every asynchronous
/// WSI read of it has completed. Store uploads own one Arc; pending direct
/// presentation and its GPU copy fence hold additional references.
pub(crate) struct BackingImageLease {
    gpu: &'static Gpu,
    pub image: GpuImage,
}

impl BackingImageLease {
    pub(crate) fn new(gpu: &'static Gpu, image: GpuImage) -> Arc<Self> {
        Arc::new(Self { gpu, image })
    }
}

impl std::ops::Deref for BackingImageLease {
    type Target = GpuImage;
    fn deref(&self) -> &GpuImage {
        &self.image
    }
}

impl Drop for BackingImageLease {
    fn drop(&mut self) {
        self.gpu
            .destroy_image(std::mem::replace(&mut self.image, GpuImage::none()));
    }
}

/// Serialize presentations for one window while leaving swapchain state
/// available to fence retirement during the blocking image acquisition.
/// Dropping the permit releases the window lock before waking the Metal
/// presenter. That ordering lets deferred staging acquire the lock immediately.
struct PresentingGuard<'a> {
    held: Option<std::sync::MutexGuard<'a, ()>>,
    gpu: &'static Gpu,
}

impl Drop for PresentingGuard<'_> {
    fn drop(&mut self) {
        drop(self.held.take());
        self.gpu.presenter.notify_progress();
    }
}

struct WindowSwapchain {
    gpu: &'static Gpu,
    presenting: Mutex<()>,
    state: Mutex<Swapchain>,
    wake: Condvar,
    alive: AtomicBool,
    worker_started: AtomicBool,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    resize_worker_started: AtomicBool,
    resize_worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// 0 = healthy, 1 = surface lost, 2 = device/unknown terminal failure.
    present_failure: std::sync::atomic::AtomicU8,
    #[cfg(test)]
    injected_acquire_error: Mutex<Option<vk::Result>>,
}

struct PreparedPresent {
    slot: usize,
    command_buffer: vk::CommandBuffer,
    acquired: vk::Semaphore,
    done: vk::Fence,
    composite: Option<vk::Image>,
}

#[derive(Clone, Copy)]
struct AcquireRequest {
    handle: vk::SwapchainKHR,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
}

/// Only the image identity and extent cross an acquire. The swapchain
/// retains ownership of the actual Vulkan allocation under `presenting`.
#[derive(Clone, Copy)]
struct PresentSource {
    image: vk::Image,
    width: u32,
    height: u32,
}

impl From<&GpuImage> for PresentSource {
    fn from(image: &GpuImage) -> Self {
        Self {
            image: image.image,
            width: image.width,
            height: image.height,
        }
    }
}

/// The compositor keeps its image and DMA fence alive across a WSI acquire.
/// The Vulkan acquire itself must not hold the swapchain-state mutex.
struct PreparedComposite {
    frame: appkit::view::Frame,
    source: PresentSource,
    prepared: PreparedPresent,
    request: AcquireRequest,
    direct: bool,
}

/// A directly presented drawable stays available for a later layer to join.
/// Its WSI-copy fence protects release if another frame supersedes it.
// Store SDL's opaque window handle as an integer, as the other worker state
// does. Reconstruct the Frame only under the window's presenting mutex.
#[derive(Clone, Copy)]
struct DirectFrame {
    window: usize,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    window_width: u32,
    window_height: u32,
}

impl DirectFrame {
    fn from_frame(frame: &appkit::view::Frame) -> Self {
        Self {
            window: frame.window as usize,
            x: frame.x,
            y: frame.y,
            width: frame.width,
            height: frame.height,
            window_width: frame.window_width,
            window_height: frame.window_height,
        }
    }

    fn frame(self) -> appkit::view::Frame {
        appkit::view::Frame {
            window: self.window as sdl3::Window,
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            window_width: self.window_width,
            window_height: self.window_height,
        }
    }
}

struct DirectLayer {
    frame: DirectFrame,
    source: PresentSource,
    owner: Option<usize>,
    backing: Option<Arc<BackingImageLease>>,
    submitted_slot: Option<usize>,
    queued_at: Instant,
}

impl Drop for DirectLayer {
    fn drop(&mut self) {
        finish_source(self.owner.take());
    }
}

enum CompositorAcquire {
    Ready(Option<(u32, bool)>),
    TimedOut,
    Terminal(vk::Result),
}

/// Wake an idle compositor worker without losing a wake issued immediately
/// before it parks. Condvar still covers the bounded batch-delay wait.
fn wake_compositor_worker(entry: &WindowSwapchain) {
    if let Some(worker) = entry
        .worker
        .lock()
        .expect("layer compositor worker")
        .as_ref()
    {
        worker.thread().unpark();
    }
    entry.wake.notify_all();
}

fn wake_resize_worker(entry: &WindowSwapchain) {
    if let Some(worker) = entry.resize_worker.lock().expect("resize worker").as_ref() {
        worker.thread().unpark();
    }
}

fn ensure_resize_worker(window: usize, entry: &Arc<WindowSwapchain>) {
    if entry.resize_worker_started.swap(true, Ordering::AcqRel) {
        return;
    }
    let owned = Arc::clone(entry);
    let handle = std::thread::Builder::new()
        .name("rosemary-wsi-resize".into())
        .spawn(move || resize_worker(window, owned))
        .expect("window-local WSI resize worker");
    *entry.resize_worker.lock().expect("resize worker") = Some(handle);
}

fn resize_worker(window: usize, entry: Arc<WindowSwapchain>) {
    loop {
        if !entry.alive.load(Ordering::Acquire)
            || entry.gpu.lost.load(Ordering::Acquire)
            || entry.present_failure.load(Ordering::Acquire) != 0
        {
            return;
        }
        if entry
            .state
            .lock()
            .expect("swapchain")
            .pending_rebuild
            .is_none()
        {
            std::thread::park();
            continue;
        }
        let presenting = PresentingGuard {
            held: Some(entry.presenting.lock().expect("window presentation")),
            gpu: entry.gpu,
        };
        #[cfg(test)]
        if TEST_BLOCK_RESIZE_WORKER.load(Ordering::Acquire) == window {
            TEST_RESIZE_WORKER_PAUSED.store(window, Ordering::Release);
            while TEST_BLOCK_RESIZE_WORKER.load(Ordering::Acquire) == window {
                std::thread::sleep(Duration::from_millis(1));
            }
            TEST_RESIZE_WORKER_PAUSED.store(0, Ordering::Release);
        }
        if !entry.alive.load(Ordering::Acquire) || entry.gpu.lost.load(Ordering::Acquire) {
            return;
        }
        let mut state = entry.state.lock().expect("swapchain");
        let Some(extent) = state.pending_rebuild else {
            continue;
        };
        // idle() and swapchain construction can block on GPU progress.
        // Neither operation is ever performed by the AppKit event thread.
        let rebuilt = if state.acquired_unsubmitted.is_some() {
            // A failed submit left an acquired WSI semaphore outstanding.
            // Consuming it and retiring the image can wait for GPU progress;
            // that recovery belongs on this worker, never in AppKit.
            state.recover_unsubmitted_acquire().and_then(|_| {
                if state.requested != extent {
                    state.rebuild(extent)
                } else {
                    Ok(())
                }
            })
        } else {
            state.rebuild(extent)
        };
        let typed_device_loss = state.last_wsi_error.take() == Some(vk::Result::ERROR_DEVICE_LOST);
        match rebuilt {
            Ok(()) => {
                state.pending_rebuild = None;
                let dirty = uses_layer_compositor(state.present_mode) && state.composite_dirty;
                drop(state);
                drop(presenting);
                if dirty {
                    ensure_layer_compositor_worker(window, &entry);
                    wake_compositor_worker(&entry);
                }
                appkit::backing::present_again(window as sdl3::Window);
            }
            Err(reason) if recoverable_composite_rebuild_failure(&reason) => {
                drop(state);
                drop(presenting);
                std::thread::sleep(Duration::from_millis(80));
            }
            Err(reason) => {
                gpu::trace(format!("deferred WSI resize failed: {reason}"));
                let device_lost = typed_device_loss || reason.contains("ERROR_DEVICE_LOST");
                drop(state);
                drop(presenting);
                // No window lock may be held while notifying the device registry.
                if device_lost {
                    record_device_loss(entry.gpu);
                } else {
                    entry.present_failure.store(2, Ordering::Release);
                    wake_compositor_worker(&entry);
                }
                return;
            }
        }
    }
}

pub(crate) fn record_device_loss(gpu: &Gpu) {
    if gpu.lost.swap(true, Ordering::AcqRel) {
        return;
    }
    gpu::trace("Vulkan logical device lost; stopping all associated composited windows");
    let registry = SWAPCHAINS.lock().expect("window swapchains");
    if let Some(entries) = registry.as_ref() {
        for entry in entries
            .values()
            .filter(|entry| std::ptr::eq(entry.gpu, gpu))
        {
            entry.present_failure.store(2, Ordering::Release);
            wake_compositor_worker(entry);
            wake_resize_worker(entry);
        }
    }
}

fn handle_terminal_error(gpu: &'static Gpu, entry: &WindowSwapchain, error: vk::Result) {
    if error == vk::Result::ERROR_DEVICE_LOST {
        record_device_loss(gpu);
    } else {
        entry
            .present_failure
            .store(terminal_present_failure(error), Ordering::Release);
    }
}

// Zero extent and out-of-date surfaces need a later retry, not a terminal
// failure. A minimized window may remain in this state indefinitely.
fn recoverable_composite_rebuild_failure(reason: &str) -> bool {
    reason.contains("the window has no area to present into")
        || reason.contains("ERROR_OUT_OF_DATE_KHR")
        || reason.contains("SUBOPTIMAL_KHR")
        || reason.contains("ERROR_INITIALIZATION_FAILED")
}

// Converted SDL windows cannot use SDL_GetWindowSurface as fallback.
// Terminal allocation/device failures stop the window, while transient
// staging failures leave the Metal drawable queued for a later retry.
fn terminal_composite_staging_failure(reason: &str) -> bool {
    reason.contains("ERROR_DEVICE_LOST")
        || reason.contains("ERROR_OUT_OF_HOST_MEMORY")
        || reason.contains("ERROR_OUT_OF_DEVICE_MEMORY")
}

/// CPU backing-store damage is needed only once the *first* frame of a new
/// Metal layer has actually entered the Vulkan window accumulator.
fn composite_first_stage_needs_store(
    joined: bool,
    staged: &Result<Option<bool>, String>,
    layer: usize,
    window: usize,
) -> bool {
    joined && layer != window && matches!(staged, Ok(Some(true)))
}

fn terminal_present_failure(error: vk::Result) -> u8 {
    if error == vk::Result::ERROR_SURFACE_LOST_KHR {
        1
    } else {
        2
    }
}

/// One frame's worth of recording and synchronization.
struct Frame {
    command_buffer: vk::CommandBuffer,
    /// Signalled when the presentation engine is done with the acquired
    /// image, waited on before anything is written into it. Per frame rather
    /// than per image, because which image an acquisition returns is not
    /// known until it has returned.
    acquired: vk::Semaphore,
    /// Signals when WSI finishes signaling the acquisition semaphore.
    /// Needed to safely retire an unconsumed semaphore if recovery fails.
    acquire_done: vk::Fence,
    acquire_pending: bool,
    /// Signalled with the submission, so that this frame's command buffer and
    /// its acquisition semaphore are not reused while the device still has
    /// them.
    done: vk::Fence,
    submitted: bool,
    /// Retained drawable whose image this frame's command buffer reads.
    source: Option<usize>,
    /// Store upload source kept immutable until the associated GPU copy fence.
    backing: Option<Arc<BackingImageLease>>,
}

struct PresentFence {
    fence: vk::Fence,
    /// Whether this fence was attached to a presentation and has not yet been
    /// waited for. A fresh fence is unsignalled but has no operation to wait
    /// for, so the distinction is required during initial teardown.
    pending: bool,
}

static SWAPCHAINS: Mutex<Option<HashMap<usize, Arc<WindowSwapchain>>>> = Mutex::new(None);

/// A Vulkan initializer never owns the global map mutex while creating
/// swapchain resources. Each window can initialize independently.
static INITIALIZING: Mutex<Option<HashMap<usize, Arc<AtomicBool>>>> = Mutex::new(None);

/// An initializer removes only its own token. SDL may recycle a window
/// address while a cancelled initializer is still unwinding.
struct InitializerRegistration {
    window: usize,
    token: Arc<AtomicBool>,
}

impl Drop for InitializerRegistration {
    fn drop(&mut self) {
        let mut pending = INITIALIZING
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(all) = pending.as_mut() {
            if all
                .get(&self.window)
                .is_some_and(|current| Arc::ptr_eq(current, &self.token))
            {
                all.remove(&self.window);
            }
        }
    }
}

struct InitRetry {
    after: Instant,
    delay: Duration,
    unknown_failures: u32,
}
static INIT_RETRIES: Mutex<Option<HashMap<usize, InitRetry>>> = Mutex::new(None);

fn retry_ready(window: usize) -> bool {
    let retry = INIT_RETRIES.lock().expect("WSI initialization retry");
    retry
        .as_ref()
        .and_then(|retries| retries.get(&window))
        .is_none_or(|retry| Instant::now() >= retry.after)
}

// Minimized windows can be unavailable indefinitely. Cap backoff, not retries
// for identified transient failures. Unclassified SDL failures have a budget.
const MAX_UNCLASSIFIED_SURFACE_FAILURES: u32 = 8;

fn schedule_retry(window: usize, unclassified_failure: bool) -> bool {
    let mut retries = INIT_RETRIES.lock().expect("WSI initialization retry");
    let retry = retries.get_or_insert_with(HashMap::new);
    let unknown_failures = retry
        .get(&window)
        .map_or(0, |previous| previous.unknown_failures)
        .saturating_add(u32::from(unclassified_failure));
    if unclassified_failure && unknown_failures >= MAX_UNCLASSIFIED_SURFACE_FAILURES {
        return false;
    }
    let delay = retry
        .get(&window)
        .map_or(Duration::from_millis(100), |previous| {
            (previous.delay * 2).min(Duration::from_secs(2))
        });
    retry.insert(
        window,
        InitRetry {
            after: Instant::now() + delay,
            delay,
            unknown_failures,
        },
    );
    true
}

fn reset_retry(window: usize) {
    if let Some(retries) = INIT_RETRIES
        .lock()
        .expect("WSI initialization retry")
        .as_mut()
    {
        retries.remove(&window);
    }
}

fn permanent_sdl_surface_failure(failure: &sdl3::video::vulkan::SurfaceFailure) -> bool {
    let text = failure.reason.to_ascii_lowercase();
    !failure.transient
        && (text.contains("not supported")
            || text.contains("unsupported")
            || text.contains("not available")
            || text.contains("incompatible")
            || text.contains("failed to load")
            || text.contains("missing")
            || text.contains("abandoned"))
}

fn permanent_init_failure(reason: &str) -> bool {
    reason.contains("belongs to OpenGL")
        || reason.contains("cannot present to this window")
        || reason.contains("VK_KHR_surface is gone")
        || reason.contains("VK_KHR_swapchain is gone")
}

#[cfg(test)]
static TEST_SLOW_INITIALIZER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_SLOW_INITIALIZER_STARTED: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_BLOCK_RESIZE_WORKER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_RESIZE_WORKER_PAUSED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_WINDOW_RETIRED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_ACTIVE_RETIREMENTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn wait_for_test_retirements() {
    let deadline = Instant::now() + Duration::from_secs(8);
    while TEST_ACTIVE_RETIREMENTS.load(Ordering::Acquire) != 0 {
        assert!(
            Instant::now() < deadline,
            "Vulkan retirement failed to finish"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    sdl3::video::vulkan::service_surface_requests();
}
#[cfg(test)]
static TEST_COMPOSITOR_IDLE_PARKED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_FAIL_FIRST_CREATE_WINDOW: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_FAIL_DEVICE_CREATE_WINDOW: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_SURFACE_GUARD_RELEASES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn release_created_surface(instance: u64, surface: u64) {
    #[cfg(test)]
    if surface == u64::MAX {
        TEST_SURFACE_GUARD_RELEASES.fetch_add(1, Ordering::AcqRel);
        return;
    }
    sdl3::video::vulkan::destroy_surface(instance, surface);
}

/// Whether a swapchain could not be made for this window, so that a run does
/// not retry - and re-report - a surface the host will never give.
static REFUSED: Mutex<Option<HashMap<usize, ()>>> = Mutex::new(None);

struct SourceOwner(Option<usize>);

impl SourceOwner {
    fn take(&mut self) -> Option<usize> {
        self.0.take()
    }
}

impl Drop for SourceOwner {
    fn drop(&mut self) {
        if let Some(source) = self.0.take() {
            // A pre-WSI failure may still fall back to CPU readback, so its
            // original Metal drawable submission owns the drawable slot.
            objc::arc::release(source as objc::object::Id);
        }
    }
}

/// Drop everything held for an SDL window that is about to be destroyed.
///
/// `appkit` remakes a window when what it is changes - a hidden dialog shown
/// again as a menu - and the map here is keyed by the window's address, which
/// SDL reuses. An entry left behind would be presented to a surface that no
/// longer exists.
pub(crate) fn forget_window(window: sdl3::Window) {
    let key = window as usize;
    reset_retry(key);
    if let Some(initializing) = INITIALIZING
        .lock()
        .expect("initializers")
        .as_mut()
        .and_then(|all| all.remove(&key))
    {
        initializing.store(true, Ordering::Release);
    }
    let entry = SWAPCHAINS
        .lock()
        .expect("swapchains")
        .as_ref()
        .and_then(|all| all.get(&key))
        .cloned();
    if let Some(entry) = &entry {
        // A resize may be blocked in vkQueueWaitIdle with the window lock.
        // Cancellation cannot wait for that lock on the AppKit thread.
        entry.alive.store(false, Ordering::Release);
        entry.wake.notify_all();
        wake_compositor_worker(entry);
        wake_resize_worker(entry);
    }
    let removed = SWAPCHAINS
        .lock()
        .expect("swapchains")
        .as_mut()
        .and_then(|all| all.remove(&key));
    drop(entry);
    if let Some(entry) = removed {
        // SDL_DestroyWindow can run as soon as we return. A token prevents
        // SDL from disposing of the native VkSurface until this swapchain
        // and every window-local worker have actually retired.
        let retired = Arc::new(AtomicBool::new(false));
        sdl3::video::vulkan::hold_window_for_teardown(window, Arc::clone(&retired));
        #[cfg(test)]
        TEST_ACTIVE_RETIREMENTS.fetch_add(1, Ordering::AcqRel);
        std::thread::Builder::new()
            .name("rosemary-wsi-retire".into())
            .spawn(move || {
                let worker = entry.worker.lock().expect("layer compositor worker").take();
                if let Some(worker) = worker {
                    worker.thread().unpark();
                    let _ = worker.join();
                }
                let resize = entry.resize_worker.lock().expect("resize worker").take();
                if let Some(resize) = resize {
                    resize.thread().unpark();
                    let _ = resize.join();
                }
                // A concurrent presentation may retain another Arc. Wait
                // without holding the AppKit or SDL video thread.
                while Arc::strong_count(&entry) != 1 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                drop(entry);
                #[cfg(test)]
                TEST_WINDOW_RETIRED.store(key, Ordering::Release);
                retired.store(true, Ordering::Release);
                #[cfg(test)]
                TEST_ACTIVE_RETIREMENTS.fetch_sub(1, Ordering::AcqRel);
            })
            .expect("asynchronous Vulkan window retirement");
    }
    if let Some(refused) = REFUSED.lock().expect("refused windows").as_mut() {
        refused.remove(&key);
    }
    sdl3::video::forget_cached_window_refresh_rate(window);
}

/// Blit `source` into `window`'s swapchain image at `frame` and present it.
///
/// `Ok(false)` is "this window cannot be presented to with a swapchain",
/// which is not an error: the caller falls back to reading the frame back and
/// blitting it through SDL's window surface, and every host without
/// `VK_KHR_surface` takes that path.
///
/// `holes` are rectangles of other layers, which keep what those layers last
/// presented there: the window's store is presented around its views' layers.
#[cfg(test)]
pub(crate) fn present(
    gpu: &'static Gpu,
    layer: usize,
    frame: &appkit::view::Frame,
    source: &GpuImage,
    holes: &[(usize, Region)],
) -> Result<bool, String> {
    let limit = Instant::now() + Duration::from_secs(3);
    loop {
        match present_impl(gpu, layer, frame, source, holes, None, None)? {
            Some(shown) => return Ok(shown),
            None if Instant::now() < limit => {
                // These tests do not run an AppKit event pump.
                sdl3::video::vulkan::service_surface_requests();
                std::thread::sleep(Duration::from_millis(2));
            }
            None => return Err("timed out initializing test swapchain".into()),
        }
    }
}

/// Fast preflight before uploading a backing store. The post-upload
/// `present_nonblocking` check handles races with newly acquired WSI locks.
pub(crate) fn window_rendering_disabled(window: sdl3::Window) -> bool {
    let Ok(guard) = SWAPCHAINS.try_lock() else {
        return false;
    };
    guard
        .as_ref()
        .and_then(|all| all.get(&(window as usize)))
        .is_some_and(|entry| {
            entry.gpu.lost.load(Ordering::Acquire)
                || entry.present_failure.load(Ordering::Acquire) != 0
        })
}

/// A permanent failure can still use CPU fallback only if SDL never
/// converted the window to Vulkan.
pub(crate) fn window_refused(window: sdl3::Window) -> Option<bool> {
    let refused = REFUSED.lock().expect("refused windows");
    refused
        .as_ref()
        .filter(|m| m.contains_key(&(window as usize)))
        .map(|_| sdl3::video::vulkan::owns_window(window))
}

/// Do not upload the backing store on every pump while initialization is
/// still running or backing off. The first frame still requests a surface.
pub(crate) fn window_initialization_pending(window: sdl3::Window) -> bool {
    let key = window as usize;
    let entry_exists = match SWAPCHAINS.try_lock() {
        Ok(registry) => registry.as_ref().is_some_and(|m| m.contains_key(&key)),
        Err(_) => return true,
    };
    if entry_exists {
        return false;
    }
    !retry_ready(key)
        || INITIALIZING
            .lock()
            .expect("initializers")
            .as_ref()
            .is_some_and(|m| m.contains_key(&key))
        || sdl3::video::vulkan::surface_request_pending(window)
}

/// A live swapchain can temporarily defer staging while its presentation
/// lock or a GPU-copy fence is occupied. Unlike missing WSI metadata, these
/// deferred drawables must be retried before their layer's ring fills.
pub(crate) fn window_has_active_swapchain(window: sdl3::Window) -> bool {
    let registry = match SWAPCHAINS.try_lock() {
        Ok(registry) => registry,
        // Do not turn transient contention with teardown into a long delay.
        Err(_) => return true,
    };
    registry
        .as_ref()
        .and_then(|all| all.get(&(window as usize)))
        .is_some_and(|entry| {
            entry.alive.load(Ordering::Acquire)
                && !entry.gpu.lost.load(Ordering::Acquire)
                && entry.present_failure.load(Ordering::Acquire) == 0
        })
}

pub(crate) fn window_presenting_busy(window: sdl3::Window) -> bool {
    let guard = match SWAPCHAINS.try_lock() {
        Ok(guard) => guard,
        Err(_) => return true,
    };
    let entry = guard.as_ref().and_then(|all| all.get(&(window as usize)));
    entry.is_some_and(|entry| entry.presenting.try_lock().is_err())
}

/// Exercise synthetic unowned sources in Vulkan regression tests. Production
/// backing stores use retained leases to preserve their GPU-copy lifetime.
#[cfg(test)]
pub(crate) fn present_nonblocking(
    gpu: &'static Gpu,
    layer: usize,
    frame: &appkit::view::Frame,
    source: &GpuImage,
    holes: &[(usize, Region)],
) -> Result<Option<bool>, String> {
    present_impl(gpu, layer, frame, source, holes, None, None)
}

/// Present a command buffer's drawable, consuming its retained `owner` on
/// every return path. A successful copy holds it until the frame fence wakes.
pub(crate) fn present_owned(
    gpu: &'static Gpu,
    layer: usize,
    frame: &appkit::view::Frame,
    source: &GpuImage,
    holes: &[(usize, Region)],
    owner: usize,
) -> Result<Option<bool>, String> {
    present_impl(gpu, layer, frame, source, holes, Some(owner), None)
}

/// Retain one immutable backing-store GPU image through its direct WSI copy.
/// The caller's pool never writes an image with outstanding Arc references.
pub(crate) fn present_backing_nonblocking(
    gpu: &'static Gpu,
    layer: usize,
    frame: &appkit::view::Frame,
    source: Arc<BackingImageLease>,
    holes: &[(usize, Region)],
) -> Result<Option<bool>, String> {
    let lease = Arc::clone(&source);
    present_impl(gpu, layer, frame, &source.image, holes, None, Some(lease))
}

fn present_impl(
    gpu: &'static Gpu,
    layer: usize,
    frame: &appkit::view::Frame,
    source: &GpuImage,
    holes: &[(usize, Region)],
    owner: Option<usize>,
    backing: Option<Arc<BackingImageLease>>,
) -> Result<Option<bool>, String> {
    let mut owner = SourceOwner(owner);
    if gpu.lost.load(Ordering::Acquire) {
        finish_source(owner.take());
        return Ok(Some(true));
    }
    if gpu.swapchain_api().is_none() || crate::vulkan::surface_api().is_none() {
        return Ok(Some(false));
    }
    let window = frame.window as usize;
    // Missing main-thread metadata is transient, not a permanently refused
    // surface. Defer the drawable or backing store until AppKit publishes it.
    if sdl3::video::presentation_window(frame.window).is_none() {
        return Ok(None);
    }
    if REFUSED
        .lock()
        .expect("refused windows")
        .as_ref()
        .is_some_and(|refused| refused.contains_key(&window))
    {
        // A failed SDL Vulkan conversion may only fall back while SDL has
        // not converted the window to Vulkan. Otherwise SDL_GetWindowSurface
        // is invalid even when there is no longer a live VkSurface.
        let converted = sdl3::video::vulkan::owns_window(frame.window);
        if converted {
            finish_source(owner.take());
        }
        return Ok(Some(converted));
    }
    let window_swapchain = {
        // This mutex covers only map operations. Never run Vulkan or SDL
        // initialization while holding it. In particular AppKit may not wait
        // for another window's VkSurface or swapchain creation.
        let guard = match SWAPCHAINS.try_lock() {
            Ok(guard) => guard,
            Err(_) => return Ok(None),
        };
        guard.as_ref().and_then(|all| all.get(&window)).cloned()
    };
    let Some(window_swapchain) = window_swapchain else {
        if !retry_ready(window) {
            return Ok(None);
        }
        let instance = crate::vulkan::instance_handle().ok_or("Vulkan instance unavailable")?;
        // The OpenGL flag was captured on SDL's owner thread. Creating a
        // Vulkan surface would irreversibly replace OpenGL on Wayland.
        if sdl3::video::presentation_window(frame.window)
            .is_some_and(|w| w.flags & sdl3::video::flags::OPENGL != 0)
        {
            REFUSED
                .lock()
                .expect("refused windows")
                .get_or_insert_with(HashMap::new)
                .insert(window, ());
            return Ok(Some(false));
        }
        let (surface, init_token) = match sdl3::video::vulkan::acquire_prepared_surface(
            frame.window,
            instance,
            crate::vulkan::library(),
        ) {
            sdl3::video::vulkan::PreparedSurface::Pending => return Ok(None),
            sdl3::video::vulkan::PreparedSurface::Retired => return Ok(None),
            sdl3::video::vulkan::PreparedSurface::Abandoned => {
                // A terminated initializer cannot be retried. SDL converted
                // windows cannot legally use its CPU surface fallback.
                let converted = sdl3::video::vulkan::owns_window(frame.window);
                REFUSED
                    .lock()
                    .expect("refused windows")
                    .get_or_insert_with(HashMap::new)
                    .insert(window, ());
                if converted {
                    finish_source(owner.take());
                }
                return Ok(Some(converted));
            }
            sdl3::video::vulkan::PreparedSurface::Failed(failure) => {
                let converted = sdl3::video::vulkan::owns_window(frame.window);
                if !converted
                    && !failure.transient
                    && (permanent_sdl_surface_failure(&failure) || !schedule_retry(window, true))
                {
                    gpu::trace(format!(
                        "SDL Vulkan window permanently refused; using CPU fallback: {}",
                        failure.reason
                    ));
                    sdl3::video::vulkan::refuse_failed_surface(frame.window, instance);
                    REFUSED
                        .lock()
                        .expect("refused windows")
                        .get_or_insert_with(HashMap::new)
                        .insert(window, ());
                    return Ok(Some(false));
                }
                if converted || failure.transient {
                    schedule_retry(window, false);
                }
                let _ = sdl3::video::vulkan::retry_surface(
                    frame.window,
                    instance,
                    crate::vulkan::library(),
                );
                return Ok(None);
            }
            sdl3::video::vulkan::PreparedSurface::Ready(surface, token) => (surface, token),
        };
        // The real VkSurface is now owned by this initializer. Do not call
        // SDL while building its Vulkan swapchain or holding the shared map.
        let cancelled = Arc::new(AtomicBool::new(false));
        let accepted = {
            let mut pending = INITIALIZING.lock().expect("initializers");
            let all = pending.get_or_insert_with(HashMap::new);
            if all.contains_key(&window) {
                false
            } else {
                all.insert(window, Arc::clone(&cancelled));
                true
            }
        };
        if !accepted {
            // A concurrent initializer already owns this window. A prepared
            // handle obtained during the race still needs to be retired.
            sdl3::video::vulkan::destroy_surface(instance, surface);
            return Ok(None);
        }
        let extent = vk::Extent2D {
            width: frame.window_width.max(1),
            height: frame.window_height.max(1),
        };
        let cancellation = Arc::clone(&cancelled);
        let spawned = std::thread::Builder::new()
            .name("metal:swapchain-init".into())
            .spawn(move || {
                let cancelled = cancellation;
                let registration = InitializerRegistration {
                    window,
                    token: Arc::clone(&cancelled),
                };
                let _init_token = init_token;
                #[cfg(test)]
                if TEST_SLOW_INITIALIZER.load(Ordering::Acquire) == window {
                    TEST_SLOW_INITIALIZER_STARTED.store(true, Ordering::Release);
                    std::thread::sleep(Duration::from_millis(600));
                }
                if cancelled.load(Ordering::Acquire) {
                    sdl3::video::vulkan::destroy_surface(instance, surface);
                    return;
                }
                if gpu.lost.load(Ordering::Acquire) {
                    // SDL already converted this window to Vulkan. Mark its
                    // taken surface terminal, so the backing-store preflight
                    // will no longer mistake it for an active initializer.
                    sdl3::video::vulkan::abandon_surface(window as sdl3::Window, instance);
                    sdl3::video::vulkan::destroy_surface(instance, surface);
                    return;
                }
                let result =
                    Swapchain::create(gpu, window as sdl3::Window, extent, surface, instance);
                let pending = INITIALIZING.lock().expect("initializers");
                // A destroyed window may have been recreated at the same
                // pointer. Only the exact initializer may publish its result.
                let current = pending
                    .as_ref()
                    .and_then(|all| all.get(&window))
                    .is_some_and(|existing| Arc::ptr_eq(existing, &cancelled));
                if !current || cancelled.load(Ordering::Acquire) {
                    return;
                }
                let device_lost_on_init = result
                    .as_ref()
                    .err()
                    .is_some_and(|reason| reason.contains("ERROR_DEVICE_LOST"));
                let mut registry = SWAPCHAINS.lock().expect("window swapchains");
                match result {
                    Ok(mut swapchain) => {
                        reset_retry(window);
                        gpu::milestone("window swapchain");
                        if swapchain.corners.is_none() {
                            appkit::decor::presented_opaque(window as sdl3::Window);
                        }
                        let rate = sdl3::video::cached_window_refresh_rate(window as sdl3::Window)
                            .unwrap_or(60.0)
                            .max(30.0);
                        swapchain.compositor_batch_delay =
                            compositor_batch_delay(swapchain.present_mode, rate);
                        registry.get_or_insert_with(HashMap::new).insert(
                            window,
                            Arc::new(WindowSwapchain {
                                gpu,
                                presenting: Mutex::new(()),
                                state: Mutex::new(swapchain),
                                wake: Condvar::new(),
                                alive: AtomicBool::new(true),
                                worker_started: AtomicBool::new(false),
                                worker: Mutex::new(None),
                                resize_worker_started: AtomicBool::new(false),
                                resize_worker: Mutex::new(None),
                                present_failure: std::sync::atomic::AtomicU8::new(0),
                                #[cfg(test)]
                                injected_acquire_error: Mutex::new(None),
                            }),
                        );
                    }
                    Err(reason) => {
                        gpu::trace(format!("window swapchain initialization failed: {reason}"));
                        if device_lost_on_init {
                            // Propagate after releasing the global registry:
                            // record_device_loss also needs that mutex.
                        } else if permanent_init_failure(&reason) {
                            REFUSED
                                .lock()
                                .expect("refused windows")
                                .get_or_insert_with(HashMap::new)
                                .insert(window, ());
                        } else {
                            schedule_retry(window, false);
                            // SDL is still responsible for its window's
                            // surface on the video thread, even after many
                            // temporary initialization failures.
                            let _ = sdl3::video::vulkan::retry_surface(
                                window as sdl3::Window,
                                instance,
                                crate::vulkan::library(),
                            );
                        }
                    }
                }
                drop(registry);
                if device_lost_on_init {
                    // The failed Swapchain::create already released its
                    // VkSurface. Mark SDL's transferred request terminal
                    // before dropping our initializer generation and
                    // reporting the lost device. The pending mutex still
                    // protects this exact generation against window reuse.
                    sdl3::video::vulkan::abandon_surface(window as sdl3::Window, instance);
                }
                drop(pending);
                // Clear before notifying the device-loss path, not merely
                // when the initializer's closure eventually returns.
                drop(registration);
                if device_lost_on_init {
                    record_device_loss(gpu);
                }
            });
        if let Err(error) = spawned {
            cancelled.store(true, Ordering::Release);
            let mut pending = INITIALIZING.lock().expect("initializers");
            if let Some(all) = pending.as_mut() {
                if all
                    .get(&window)
                    .is_some_and(|job| Arc::ptr_eq(job, &cancelled))
                {
                    all.remove(&window);
                }
            }
            drop(pending);
            // The closure never executed, so no guard took this VkSurface.
            sdl3::video::vulkan::destroy_surface(instance, surface);
            sdl3::video::vulkan::abandon_surface(frame.window, instance);
            return Err(format!("cannot start swapchain initializer: {error}"));
        }
        return Ok(None);
    };
    // Presentations must reserve frame slots in order, but waiting for an
    // image must not hold the state lock: nextDrawable uses that time to
    // retire completed drawables and release their ring slots.
    let presenting = match window_swapchain.presenting.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::WouldBlock) => return Ok(None),
        Err(TryLockError::Poisoned(error)) => error.into_inner(),
    };
    let presenting = PresentingGuard {
        held: Some(presenting),
        gpu: window_swapchain.gpu,
    };
    // The SDL window can be torn down after we cloned the entry, before
    // acquiring its per-window lock. Do not touch its retired surface.
    if !window_swapchain.alive.load(Ordering::Acquire) {
        drop(presenting);
        // The caller already transferred this drawable to Submitted. A
        // plain ARC release cannot return its slot to the drawable ring.
        finish_source(owner.take());
        return Ok(Some(true));
    }
    match window_swapchain.present_failure.load(Ordering::Acquire) {
        1 => {
            // SDL_GetWindowSurface is incompatible with a window converted to
            // Vulkan. A lost surface stays unavailable until window teardown;
            // do not incorrectly switch that same window to SDL readback.
            drop(presenting);
            finish_source(owner.take());
            return Ok(Some(true));
        }
        2 => {
            drop(presenting);
            finish_source(owner.take());
            return Ok(Some(true));
        }
        _ => {}
    }
    let mut swapchain = window_swapchain.state.lock().expect("window swapchain");
    let extent = vk::Extent2D {
        width: frame.window_width.max(1),
        height: frame.window_height.max(1),
    };
    // Defer all potentially unbounded VkQueueWaitIdle/vkWaitForFences calls
    // to the window-local worker, including a previously retired swapchain.
    // No new layer is registered before recreation completes.
    if swapchain.pending_rebuild.is_some() || swapchain.retired || swapchain.requested != extent {
        swapchain.pending_rebuild = Some(extent);
        ensure_resize_worker(window, &window_swapchain);
        wake_resize_worker(&window_swapchain);
        return Ok(None);
    }
    let joined = !swapchain.layers.contains(&layer);
    if joined {
        swapchain.layers.push(layer);
    }
    // Only a layer that has presented here has anything of its own to keep:
    // over any other hole the source is still what the window shows.
    let holes: Vec<Region> = holes
        .iter()
        .filter(|(owner, _)| *owner != layer && swapchain.layers.contains(owner))
        .map(|(_, region)| *region)
        .collect();
    let regions = match holes.is_empty() {
        true => None,
        false => Some(subtract(
            Region {
                x: frame.x,
                y: frame.y,
                width: frame.width as i32,
                height: frame.height as i32,
            },
            &holes,
        )),
    };
    // Core Animation combines independent layers into one window image.
    // Host WSI presentation mode only determines how that completed image
    // reaches the display; never enqueue one host present per Metal layer.
    if uses_layer_compositor(swapchain.present_mode) {
        // The main event pump refreshes this snapshot when the display or its
        // mode changes. Never call main-thread-only SDL APIs from Metal.
        let rate = sdl3::video::cached_window_refresh_rate(frame.window)
            .unwrap_or(60.0)
            .max(30.0);
        swapchain.compositor_batch_delay = compositor_batch_delay(swapchain.present_mode, rate);
        #[cfg(not(test))]
        ensure_layer_compositor_worker(window, &window_swapchain);
        swapchain.composite_shape = Some(corners::Shape::of(
            frame.window,
            swapchain.extent.width,
            swapchain.extent.height,
        ));
        // A single opaque Metal drawable can be copied straight into the WSI
        // image by the same window worker, without an offscreen full-frame copy.
        // The retained drawable also supplies the initial snapshot if a second
        // layer joins later. Other cases must stage into the GPU accumulator.
        let direct = fast_direct_layer(
            swapchain.present_mode,
            swapchain.layers.len(),
            frame,
            regions.as_deref(),
            swapchain.corners.is_some(),
            owner.0.is_some() || backing.is_some(),
        );
        let converting = !direct && swapchain.direct.is_some();
        let converted = if converting {
            swapchain.stage_direct_snapshot(true)
        } else {
            Ok(None)
        };
        // Keep the submitted drawable's additional retained reference until
        // staging actually transfers it to a GPU fence. Error paths must not
        // release the ring slot before the dispatcher decides to retry.
        let converted_staged = matches!(converted, Ok(Some(true)));
        let mut backing = backing;
        let staged = match converted {
            Ok(_) if direct => {
                swapchain.queue_direct(
                    frame,
                    PresentSource::from(source),
                    &mut owner,
                    backing.take(),
                );
                Ok(Some(true))
            }
            // A busy GPU staging slot cannot yet preserve the previous
            // full-window drawable. Defer the incoming layer rather than
            // presenting an incomplete window image over that drawable.
            Ok(None) if converting => Ok(None),
            Ok(None) | Ok(Some(true)) => {
                let result =
                    swapchain.stage_composite(frame, source, regions.as_deref(), &mut owner, true);
                if matches!(result, Ok(Some(true))) {
                    // A backing-store source must also survive its staging
                    // copy when this window already has multiple layers.
                    let slot = swapchain.frame.wrapping_sub(1) % swapchain.frames.len();
                    debug_assert!(swapchain.frames[slot].backing.is_none());
                    swapchain.frames[slot].backing = backing.take();
                }
                result
            }
            Ok(Some(false)) => unreachable!(),
            Err(error) => Err(error),
        };
        let typed_device_loss =
            swapchain.last_wsi_error.take() == Some(vk::Result::ERROR_DEVICE_LOST);
        let refresh_store = composite_first_stage_needs_store(joined, &staged, layer, window);
        if !matches!(staged, Ok(Some(true))) && joined {
            // A saved background snapshot does not mean the new layer's own
            // frame was staged: keep the layer unknown until that succeeds.
            swapchain.layers.retain(|&known| known != layer);
        }
        if matches!(staged, Ok(Some(true))) || converted_staged {
            wake_compositor_worker(&window_swapchain);
        }
        drop(swapchain);
        drop(presenting);
        if refresh_store {
            // A deferred or failed first frame has not changed the Vulkan
            // accumulator; it must not dirty the AppKit CPU backing store.
            appkit::backing::present_again(frame.window);
        }
        return match staged {
            Ok(staged) => Ok(staged),
            Err(reason) if sdl3::video::vulkan::owns_window(frame.window) => {
                gpu::trace(format!("converted layer staging failed: {reason}"));
                if typed_device_loss || terminal_composite_staging_failure(&reason) {
                    if typed_device_loss || reason.contains("ERROR_DEVICE_LOST") {
                        record_device_loss(gpu);
                    } else {
                        window_swapchain.present_failure.store(2, Ordering::Release);
                        wake_compositor_worker(&window_swapchain);
                    }
                    finish_source(owner.take());
                    Ok(Some(true))
                } else {
                    // The presentation queue still owns its original ARC
                    // reference and can retry when the frame slot recovers.
                    Ok(None)
                }
            }
            Err(reason) => Err(reason),
        };
    }
    Err(format!(
        "unsupported Vulkan window presentation mode {:?}",
        swapchain.present_mode
    ))
}

/// Some Gamescope WSI versions omit a present-fence signal when
/// vkQueuePresentKHR fails. Only trust the fence after a successful result
/// in that opt-in WSI path. Other drivers still enqueue ordinary OUT_OF_DATE
/// and SURFACE_LOST operations; device loss and OOM cannot be assumed queued.
/// Mirrors DXVK Presenter::presentImage's fence-result classification.
fn present_fence_will_signal(result: &Result<bool, vk::Result>, gamescope_wsi: bool) -> bool {
    if gamescope_wsi {
        return result.is_ok();
    }
    !matches!(
        result,
        Err(vk::Result::ERROR_DEVICE_LOST
            | vk::Result::ERROR_OUT_OF_HOST_MEMORY
            | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
    )
}

/// Core Animation composites Metal layers into a window independently of
/// when the host display scans out the latest completed composition.
/// All supported WSI modes share that compositing path.
fn uses_layer_compositor(mode: vk::PresentModeKHR) -> bool {
    matches!(
        mode,
        vk::PresentModeKHR::FIFO
            | vk::PresentModeKHR::FIFO_RELAXED
            | vk::PresentModeKHR::MAILBOX
            | vk::PresentModeKHR::IMMEDIATE
    )
}

/// Bound source-copy work independently of WSI's own image count. A guest
/// drawing multiple layers per frame should not wait for each WSI image.
fn frame_slots(_mode: vk::PresentModeKHR) -> usize {
    FRAMES_IN_FLIGHT
}

/// In immediate mode the compositor publishes promptly, while FIFO's longer
/// window batch gathers a full refresh's independent layer changes. Neither
/// timer changes the guest's command-buffer ordering or source completion.
fn compositor_batch_delay(mode: vk::PresentModeKHR, rate: f32) -> Duration {
    let rate = rate.max(30.0);
    if matches!(
        mode,
        vk::PresentModeKHR::IMMEDIATE | vk::PresentModeKHR::MAILBOX
    ) {
        Duration::from_secs_f32((0.35 / rate).min(0.002))
    } else {
        Duration::from_secs_f32((0.65 / rate).min(0.004))
    }
}

/// Only a full-window opaque drawable with a retained lifetime may bypass
/// offscreen composition. FIFO retains its previous batching behavior.
fn fast_direct_layer(
    mode: vk::PresentModeKHR,
    layers: usize,
    frame: &appkit::view::Frame,
    regions: Option<&[Region]>,
    corners: bool,
    owned: bool,
) -> bool {
    matches!(
        mode,
        vk::PresentModeKHR::IMMEDIATE | vk::PresentModeKHR::MAILBOX
    ) && layers == 1
        && frame.covers_window()
        && regions.is_none()
        && !corners
        && owned
}

/// Direct opaque presents have nothing to coalesce, so they do not wait for
/// the multi-layer Core Animation batching heuristic.
fn pending_batch_delay(swapchain: &Swapchain) -> Duration {
    if swapchain.direct.is_some() {
        Duration::ZERO
    } else {
        swapchain.compositor_batch_delay
    }
}

// A static window must not permanently reserve one of a CAMetalLayer's three
// drawable slots. Only idle windows snapshot once; active direct frames never
// pay the extra full-frame GPU transfer.
const DIRECT_IDLE_SNAPSHOT_DELAY: Duration = Duration::from_millis(100);

/// One presentation worker per window prevents an occluded window from
/// stalling other windows and decouples Metal drawables from WSI acquisitions.
fn ensure_layer_compositor_worker(window: usize, entry: &Arc<WindowSwapchain>) {
    if entry.worker_started.swap(true, Ordering::AcqRel) {
        return;
    }
    let worker_entry = Arc::clone(entry);
    let handle = std::thread::Builder::new()
        .name("rosemary-layer-present".into())
        .spawn(move || layer_compositor_worker(window, worker_entry))
        .expect("layer presenter thread");
    // Started under the window presentation mutex, so forget_window cannot
    // run between launching the worker and recording its join handle.
    *entry.worker.lock().expect("layer compositor worker") = Some(handle);
}

fn layer_compositor_worker(window: usize, entry: Arc<WindowSwapchain>) {
    let mut prepare_failures = 0u32;
    let mut acquire_timeouts = 0u32;
    loop {
        // Wait without polling when clean. The writer sets composite_dirty under
        // the same mutex before signaling us, so notifications are not lost.
        let mut state = entry.state.lock().expect("window swapchain");
        while !state.composite_dirty
            && entry.alive.load(Ordering::Acquire)
            && !entry.gpu.lost.load(Ordering::Acquire)
            && entry.present_failure.load(Ordering::Acquire) == 0
        {
            // The idle worker must not retain a window or device after
            // terminal loss. Unlike Condvar notification, unpark's token is
            // retained when a loss races the transition to sleep.
            let snapshot_wait = state.direct.as_ref().and_then(|direct| {
                direct.submitted_slot.and_then(|_| {
                    DIRECT_IDLE_SNAPSHOT_DELAY.checked_sub(direct.queued_at.elapsed())
                })
            });
            if state.direct.as_ref().is_some_and(|direct| {
                direct.submitted_slot.is_some()
                    && direct.queued_at.elapsed() >= DIRECT_IDLE_SNAPSHOT_DELAY
            }) {
                break;
            }
            #[cfg(test)]
            TEST_COMPOSITOR_IDLE_PARKED.store(window, Ordering::Release);
            drop(state);
            if let Some(wait) = snapshot_wait {
                std::thread::park_timeout(wait);
            } else {
                std::thread::park();
            }
            #[cfg(test)]
            TEST_COMPOSITOR_IDLE_PARKED.store(0, Ordering::Release);
            state = entry.state.lock().expect("window swapchain");
        }
        if !entry.alive.load(Ordering::Acquire)
            || entry.gpu.lost.load(Ordering::Acquire)
            || entry.present_failure.load(Ordering::Acquire) != 0
        {
            return;
        }
        if !state.composite_dirty {
            drop(state);
            let _presenting = PresentingGuard {
                held: Some(entry.presenting.lock().expect("window presentation")),
                gpu: entry.gpu,
            };
            let mut state = entry.state.lock().expect("window swapchain");
            match state.snapshot_idle_direct() {
                Ok(_) => {}
                Err(error) => {
                    let device_lost = state.last_wsi_error.take()
                        == Some(vk::Result::ERROR_DEVICE_LOST)
                        || error.contains("ERROR_DEVICE_LOST");
                    drop(state);
                    drop(_presenting);
                    if device_lost {
                        record_device_loss(entry.gpu);
                        return;
                    }
                    gpu::trace(format!("idle direct snapshot failed: {error}"));
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            continue;
        }
        if let Some(remaining) = state
            .composite_dirty_since
            .and_then(|since| pending_batch_delay(&state).checked_sub(since.elapsed()))
        {
            let _wait = entry
                .wake
                .wait_timeout(state, remaining)
                .expect("compositor batch wait");
            continue;
        }
        drop(state);

        let _presenting = PresentingGuard {
            held: Some(entry.presenting.lock().expect("window presentation")),
            gpu: entry.gpu,
        };
        if !entry.alive.load(Ordering::Acquire) || entry.gpu.lost.load(Ordering::Acquire) {
            return;
        }
        let mut state = entry.state.lock().expect("window swapchain");
        // A resize may have reset the accumulator while we acquired the
        // presentation lock. Do not try to submit a retired image.
        if !uses_layer_compositor(state.present_mode) || !state.composite_dirty {
            continue;
        }
        let prepared = state.prepare_composite(window as sdl3::Window);
        let typed_device_loss = state.last_wsi_error.take() == Some(vk::Result::ERROR_DEVICE_LOST);
        let gpu = state.gpu;
        drop(state);
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                if typed_device_loss || error.contains("ERROR_DEVICE_LOST") {
                    drop(_presenting);
                    record_device_loss(gpu);
                    return;
                }
                if recoverable_composite_rebuild_failure(&error) {
                    prepare_failures = 0;
                    drop(_presenting);
                    std::thread::sleep(Duration::from_millis(80));
                    continue;
                }
                prepare_failures += 1;
                if prepare_failures >= 3 {
                    gpu::trace(format!(
                        "presentation disabled after persistent preparation failures: {error}"
                    ));
                    entry.present_failure.store(2, Ordering::Release);
                    return;
                }
                drop(_presenting);
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        #[cfg(test)]
        let injected = entry
            .injected_acquire_error
            .lock()
            .expect("WSI injection")
            .take();
        #[cfg(not(test))]
        let injected: Option<vk::Result> = None;
        let acquired = match injected {
            Some(error) => CompositorAcquire::Terminal(error),
            None => crate::stats::timed(crate::stats::Counter::PresentAcquireMicros, || {
                acquire_compositor_image(gpu, prepared.request, acquire_timeouts == 0)
            }),
        };
        if matches!(acquired, CompositorAcquire::TimedOut) {
            // Retain the dirty accumulator. Only this window waits; another
            // window's presenter is completely independent. Wait with the
            // presentation lock released: a minimized or occluded window can
            // time out indefinitely, and every producer needs that lock.
            acquire_timeouts = acquire_timeouts.saturating_add(1);
            drop(_presenting);
            std::thread::sleep(compositor_acquire_backoff(acquire_timeouts));
            continue;
        }
        acquire_timeouts = 0;
        let ready = match acquired {
            CompositorAcquire::Ready(frame) => frame,
            CompositorAcquire::TimedOut => unreachable!(),
            CompositorAcquire::Terminal(error) => {
                gpu::trace(format!(
                    "presentation disabled after terminal WSI acquire error: {error:?}"
                ));
                drop(_presenting);
                handle_terminal_error(gpu, &entry, error);
                return;
            }
        };
        let mut state = entry.state.lock().expect("window swapchain");
        let old_swapchain = state.handle;
        match state.complete_composite(prepared, Ok(ready)) {
            Ok(None) => {
                prepare_failures = 0;
                if state.handle != old_swapchain {
                    // Recovery lost the accumulator. Request a new store
                    // upload outside the WSI and AppKit store mutexes.
                    drop(state);
                    drop(_presenting);
                    appkit::backing::present_again(window as sdl3::Window);
                    continue;
                }
            }
            Ok(Some(error)) => {
                gpu::trace(format!(
                    "presentation disabled after terminal queue presentation error: {error:?}"
                ));
                drop(state);
                drop(_presenting);
                handle_terminal_error(gpu, &entry, error);
                return;
            }
            Err(error) => {
                let typed_device_loss =
                    state.last_wsi_error.take() == Some(vk::Result::ERROR_DEVICE_LOST);
                if typed_device_loss || error.contains("ERROR_DEVICE_LOST") {
                    drop(state);
                    drop(_presenting);
                    record_device_loss(gpu);
                    return;
                }
                if recoverable_composite_rebuild_failure(&error) {
                    prepare_failures = 0;
                    drop(state);
                    drop(_presenting);
                    std::thread::sleep(Duration::from_millis(80));
                    continue;
                }
                prepare_failures += 1;
                if prepare_failures >= 3 {
                    gpu::trace(format!(
                        "presentation disabled after persistent presentation failures: {error}"
                    ));
                    entry.present_failure.store(2, Ordering::Release);
                    return;
                }
                gpu::trace(format!("window composite presentation failed: {error}"));
                drop(state);
                drop(_presenting);
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

impl Swapchain {
    /// Move an obsolete direct drawable to its WSI submit fence so that a
    /// source cannot be released while the GPU is still copying from it.
    fn retire_direct(&mut self) {
        let Some(mut direct) = self.direct.take() else {
            return;
        };
        if let Some(slot) = direct.submitted_slot {
            if self.frames[slot].submitted {
                debug_assert!(self.frames[slot].source.is_none());
                debug_assert!(self.frames[slot].backing.is_none());
                self.frames[slot].source = direct.owner.take();
                self.frames[slot].backing = direct.backing.take();
            }
        }
        // An unsubmitted drawable or a completed WSI copy is safe to release.
    }

    fn queue_direct(
        &mut self,
        frame: &appkit::view::Frame,
        source: PresentSource,
        owner: &mut SourceOwner,
        backing: Option<Arc<BackingImageLease>>,
    ) {
        self.retire_direct();
        self.direct = Some(DirectLayer {
            frame: DirectFrame::from_frame(frame),
            source,
            owner: owner.take(),
            backing,
            submitted_slot: None,
            queued_at: Instant::now(),
        });
        self.composite_dirty = true;
        self.composite_dirty_since = Some(Instant::now());
    }

    /// Materialize the last directly displayed layer only when another layer
    /// actually joins. This preserves a static underlying layer while avoiding
    /// a full-frame intermediate copy in the steady-state single-layer case.
    fn stage_direct_snapshot(&mut self, nonblocking: bool) -> Result<Option<bool>, String> {
        let Some(mut direct) = self.direct.take() else {
            return Ok(None);
        };
        let mut owner = SourceOwner(direct.owner.take());
        let result = self.stage_composite_source(
            &direct.frame.frame(),
            direct.source,
            None,
            &mut owner,
            nonblocking,
        );
        if matches!(result, Ok(Some(true))) {
            let slot = self.frame.wrapping_sub(1) % self.frames.len();
            debug_assert!(self.frames[slot].backing.is_none());
            self.frames[slot].backing = direct.backing.take();
        } else {
            direct.owner = owner.take();
            self.direct = Some(direct);
        }
        result
    }

    /// After an idle window is already on screen, snapshot it once and return
    /// its drawable to the guest. No extra WSI present is needed for this copy.
    /// Called only under the window presentation mutex, on the worker thread.
    fn snapshot_idle_direct(&mut self) -> Result<bool, String> {
        if self.composite_dirty
            || !self.direct.as_ref().is_some_and(|direct| {
                direct.submitted_slot.is_some()
                    && direct.queued_at.elapsed() >= DIRECT_IDLE_SNAPSHOT_DELAY
            })
        {
            return Ok(false);
        }
        let slot = self.frame % self.frames.len();
        if !matches!(self.stage_direct_snapshot(false)?, Some(true)) {
            return Ok(false);
        }
        // This is an idle-only transfer: waiting on the compositor worker
        // returns the retained drawable immediately without delaying AppKit.
        self.gpu.wait(self.frames[slot].done)?;
        self.frames[slot].submitted = false;
        finish_source(self.frames[slot].source.take());
        self.frames[slot].backing.take();
        self.composite_dirty = false;
        self.composite_dirty_since = None;
        Ok(true)
    }

    /// Copy a guest layer into the window accumulator without making the
    /// compositor display an intermediate image. The regular Metal drawable
    /// ownership/fence lifetime still applies to this submission.
    fn stage_composite(
        &mut self,
        frame: &appkit::view::Frame,
        source: &GpuImage,
        regions: Option<&[Region]>,
        owner: &mut SourceOwner,
        nonblocking: bool,
    ) -> Result<Option<bool>, String> {
        #[cfg(test)]
        if let Some(error) = self.injected_stage_vk_error.take() {
            self.last_wsi_error = Some(error);
            return Err(format!("injected composite staging failure: {error:?}"));
        }
        self.stage_composite_source(
            frame,
            PresentSource::from(source),
            regions,
            owner,
            nonblocking,
        )
    }

    fn stage_composite_source(
        &mut self,
        frame: &appkit::view::Frame,
        source: PresentSource,
        regions: Option<&[Region]>,
        owner: &mut SourceOwner,
        nonblocking: bool,
    ) -> Result<Option<bool>, String> {
        let slot = self.frame % self.frames.len().max(1);
        let this = &self.frames[slot];
        let (command_buffer, done) = (this.command_buffer, this.done);
        // vkQueueSubmit may fail after resetting the fence, without queueing
        // any work to signal it. Only wait for slots actually submitted.
        // AppKit must not wait for a GPU frame held by another operation.
        if this.submitted {
            if nonblocking {
                // SAFETY: the fence belongs to a submitted frame on this GPU.
                if !self.gpu.fence_signaled(done)? {
                    return Ok(None);
                }
            } else {
                self.gpu.wait(done)?;
            }
        }
        self.frames[slot].submitted = false;
        finish_source(self.frames[slot].source.take());
        self.frames[slot].backing.take();
        let composite = self.composite()?;

        let device = &self.gpu.device;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: this command buffer's fence has completed, and all images
        // remain in GENERAL for copies between guest layers.
        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("vkResetCommandBuffer failed: {e:?}"))?;
            device
                .begin_command_buffer(command_buffer, &begin)
                .map_err(|e| format!("vkBeginCommandBuffer failed: {e:?}"))?;
            prepare_source(device, command_buffer, source.image);
            // A previous window presentation may still be reading the
            // accumulator (transfer or corners compute), even when its layer
            // writes have finished. Order those reads before a new write.
            let image_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(
                    vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_READ,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(composite)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier],
            );
            match regions {
                Some(regions) => {
                    for region in regions {
                        let part = appkit::view::Frame {
                            x: region.x,
                            y: region.y,
                            width: region.width.max(0) as u32,
                            height: region.height.max(0) as u32,
                            ..*frame
                        };
                        if let Some(into) = clamp(&part, self.extent) {
                            blit(
                                device,
                                command_buffer,
                                source.image,
                                source_of(into, frame, (source.width, source.height)),
                                composite,
                                vk::ImageLayout::GENERAL,
                                into,
                            );
                        }
                    }
                }
                None => {
                    if let Some(into) = clamp(frame, self.extent) {
                        blit(
                            device,
                            command_buffer,
                            source.image,
                            source_of(into, frame, (source.width, source.height)),
                            composite,
                            vk::ImageLayout::GENERAL,
                            into,
                        );
                    }
                }
            }
            device
                .end_command_buffer(command_buffer)
                .map_err(|e| format!("vkEndCommandBuffer failed: {e:?}"))?;
        }
        // The Vulkan queue orders this composite update after the guest's
        // Metal command buffer and before a later WSI copy. No WSI semaphore
        // is required: the composite is an ordinary device-owned image.
        self.gpu
            .submit_synchronized(command_buffer, done, &[], &[])?;
        self.frames[slot].submitted = true;
        self.frames[slot].source = owner.take();
        self.frame = self.frame.wrapping_add(1);
        if !self.composite_dirty {
            self.composite_dirty_since = Some(Instant::now());
        }
        self.composite_dirty = true;
        Ok(Some(true))
    }

    /// Present the whole layer accumulator once instead of placing one
    /// WSI entry in the display queue for each Metal layer update.
    fn prepare_composite(&mut self, window: sdl3::Window) -> Result<PreparedComposite, String> {
        if self.retired {
            self.rebuild(self.requested)?;
        }
        let direct = self.direct.is_some();
        let (frame, source) = if let Some(single) = self.direct.as_ref() {
            (single.frame.frame(), single.source)
        } else {
            (
                appkit::view::Frame {
                    window,
                    x: 0,
                    y: 0,
                    width: self.extent.width,
                    height: self.extent.height,
                    window_width: self.requested.width,
                    window_height: self.requested.height,
                },
                PresentSource::from(self.composite.as_ref().expect("window layer accumulator")),
            )
        };
        let (prepared, request) = self
            .prepare_present(&frame, if direct { None } else { Some(&[]) }, false)?
            .ok_or("layer compositor failed to reserve present frame")?;
        Ok(PreparedComposite {
            frame,
            source,
            prepared,
            request,
            direct,
        })
    }

    fn complete_composite(
        &mut self,
        ready: PreparedComposite,
        acquired: Result<Option<(u32, bool)>, vk::Result>,
    ) -> Result<Option<vk::Result>, String> {
        let before = self.handle;
        let frame_before = self.frame;
        let direct_slot = ready.prepared.slot;
        let result = self.finish_present(
            &ready.frame,
            ready.source,
            if ready.direct { None } else { Some(&[]) },
            ready.prepared,
            acquired,
        );
        if ready.direct && self.frame != frame_before {
            if let Some(single) = self.direct.as_mut() {
                single.submitted_slot = Some(direct_slot);
            }
        }
        if self.acquired_unsubmitted.is_some() {
            // A failed WSI copy must consume the acquisition semaphore and
            // retire the acquired image before another compositor acquisition.
            if let Err(error) = &result {
                gpu::trace(format!(
                    "window presentation submission dropped after acquire: {error}"
                ));
            }
            if let Err(error) = self.recover_unsubmitted_acquire() {
                // Never retry with unresolved WSI synchronization.
                gpu::trace(format!("window acquire recovery failed: {error}"));
                return Ok(Some(vk::Result::ERROR_UNKNOWN));
            }
            return Ok(None);
        }
        let failed = result?;
        // An out-of-date rebuild retained the previous composite snapshot.
        // Clear dirty only when that snapshot reached the new swapchain.
        if failed.is_none() && self.handle == before {
            self.composite_dirty = false;
            self.composite_dirty_since = None;
        }
        Ok(failed)
    }

    /// Deterministic WSI integration-test entry point. The production worker
    /// has a bounded acquire and does not hold the swapchain-state lock.
    #[cfg(test)]
    fn flush_composite(&mut self, window: sdl3::Window) -> Result<(), String> {
        if !self.composite_dirty {
            return Ok(());
        }
        let ready = self.prepare_composite(window)?;
        let acquired = acquire_image(self.gpu, ready.request, false);
        match self.complete_composite(ready, acquired)? {
            None => Ok(()),
            Some(error) => Err(format!("terminal queue presentation error: {error:?}")),
        }
    }
}

/// Hand back every drawable whose copy into a swapchain image has finished.
///
/// A frame slot otherwise keeps its source until the slot comes round again,
/// several presentations later. Core Animation has a drawable back once it is
/// on screen, and a guest that holds one drawable of three - Unity keeps the
/// one it fetched on resize - runs its ring dry waiting for a presentation
/// that needs a drawable to happen. Must not be called with the drawable
/// cache locked: finishing a source takes that lock.
pub(crate) fn retire_copied() {
    retire_copied_in(&every_swapchain());
}

fn every_swapchain() -> Vec<Arc<WindowSwapchain>> {
    SWAPCHAINS
        .lock()
        .expect("window swapchains")
        .as_ref()
        .map(|all| all.values().cloned().collect::<Vec<_>>())
        .unwrap_or_default()
}

/// Whether any of `swapchains` had a drawable to hand back.
fn retire_copied_in(swapchains: &[Arc<WindowSwapchain>]) -> bool {
    let mut any = false;
    for swapchain in swapchains {
        // Retiring copied drawables is opportunistic. A presentation may be
        // waiting for a swapchain image under this window's lock; do not make
        // drawable acquisition wait behind it merely to check a fence.
        let retired = match swapchain.state.try_lock() {
            Ok(mut state) => state.retire_copied(),
            Err(TryLockError::WouldBlock) => false,
            Err(TryLockError::Poisoned(error)) => error.into_inner().retire_copied(),
        };
        if retired {
            // Source ownership has already returned to its drawable cache.
            // Wake consumers only after releasing the swapchain state mutex.
            swapchain.gpu.presenter.notify_progress();
            any = true;
        }
    }
    any
}

/// [`retire_copied`], waiting up to `timeout` for a copy still running when
/// none has finished yet, because nothing signals the drawable cache when one
/// does. Whether there was a copy to retire or wait for, so that the caller
/// knows to look at its ring again before waiting on anything else. Must not
/// be called with the drawable cache locked.
pub(crate) fn wait_copied(timeout: std::time::Duration) -> bool {
    let entries = every_swapchain();
    // Every window first: the ring being waited on may be another window's
    // than the first one with a copy still running.
    if retire_copied_in(&entries) {
        return true;
    }
    // Hold a single window's state while using its fences: resize and teardown
    // can destroy them, but cannot do so while this guard is held. Never hold
    // the global swapchain-map lock across a Vulkan wait.
    for entry in &entries {
        let state = match entry.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => continue,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
        };
        let fences = state.copying();
        if fences.is_empty() {
            continue;
        }
        // SAFETY: fences of this device's, each attached to a submission,
        // and kept alive by the state guard.
        let waited = unsafe {
            state
                .gpu
                .device
                .wait_for_fences(&fences, false, timeout.as_nanos() as u64)
        };
        drop(state);
        retire_copied_in(&entries);
        // A wait that failed outright did not wait, and the caller must not
        // take it as having done so or it spins on a lost device.
        return matches!(waited, Ok(()) | Err(vk::Result::TIMEOUT));
    }
    false
}

fn finish_source(source: Option<usize>) {
    if let Some(source) = source {
        crate::present::presentation_complete(source);
    }
}

/// A rectangle of window pixels, top-left origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl From<appkit::backing::Rect> for Region {
    fn from(rect: appkit::backing::Rect) -> Region {
        Region {
            x: rect.x.clamp(0, i64::from(i32::MAX)) as i32,
            y: rect.y.clamp(0, i64::from(i32::MAX)) as i32,
            width: rect.width.clamp(0, i64::from(i32::MAX)) as i32,
            height: rect.height.clamp(0, i64::from(i32::MAX)) as i32,
        }
    }
}

/// `whole` less every rectangle in `holes`, as rectangles that do not overlap.
pub(crate) fn subtract(whole: Region, holes: &[Region]) -> Vec<Region> {
    let mut pieces = vec![whole];
    for hole in holes {
        let mut next = Vec::with_capacity(pieces.len() + 4);
        for piece in pieces {
            let left = piece.x.max(hole.x);
            let top = piece.y.max(hole.y);
            let right = (piece.x + piece.width).min(hole.x + hole.width);
            let bottom = (piece.y + piece.height).min(hole.y + hole.height);
            if left >= right || top >= bottom {
                next.push(piece);
                continue;
            }
            let (piece_right, piece_bottom) = (piece.x + piece.width, piece.y + piece.height);
            let bands = [
                (piece.x, piece.y, piece.width, top - piece.y),
                (piece.x, bottom, piece.width, piece_bottom - bottom),
                (piece.x, top, left - piece.x, bottom - top),
                (right, top, piece_right - right, bottom - top),
            ];
            next.extend(
                bands
                    .into_iter()
                    .filter(|&(_, _, width, height)| width > 0 && height > 0)
                    .map(|(x, y, width, height)| Region {
                        x,
                        y,
                        width,
                        height,
                    }),
            );
        }
        pieces = next;
    }
    pieces
}

impl Swapchain {
    fn create(
        gpu: &'static Gpu,
        window: sdl3::Window,
        extent: vk::Extent2D,
        surface: u64,
        instance: u64,
    ) -> Result<Self, String> {
        // Establish VkSurface ownership before *any* validation may fail.
        // The window metadata can be invalidated by SDL resizing the window
        // after its surface has been prepared on the video thread.
        let surface = gpu::Guard::new(vk::SurfaceKHR::from_raw(surface), |surface| {
            release_created_surface(instance, surface.as_raw());
        });
        #[cfg(test)]
        if TEST_FAIL_FIRST_CREATE_WINDOW
            .compare_exchange(window as usize, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Err("the window has no area to present into".into());
        }
        #[cfg(test)]
        if TEST_FAIL_DEVICE_CREATE_WINDOW
            .compare_exchange(window as usize, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Err("vkCreateSwapchainKHR failed with ERROR_DEVICE_LOST".into());
        }
        // Never an OpenGL window. SDL will convert one in place, but it
        // converts the flags and the loaded libraries and not the backend's
        // own state: on Wayland the `EGLSurface` SDL created with the window
        // stays alive on the `wl_surface`, and when the Vulkan driver claims
        // the same surface the compositor answers
        // `wp_linux_drm_syncobj_manager_v1: surface already exists` and
        // closes the connection.
        //
        // A window with no graphics API - which is how this runtime makes
        // them, exactly so that this is possible - has no `EGLSurface` and
        // is converted safely, so only a guest that has actually asked for
        // an `NSOpenGLContext` ends up here.
        let snapshot = sdl3::video::presentation_window(window)
            .ok_or("window video properties have not been published")?;
        if snapshot.flags & sdl3::video::flags::OPENGL != 0 {
            return Err("the window belongs to OpenGL, which cannot share it".into());
        }
        // The transferred surface is already guarded above.
        if !gpu.can_present(*surface) {
            return Err("the Vulkan queue Metal submits to cannot present to this window".into());
        }
        // Serialize Vulkan WSI-object setup, not unrelated shader compilation.
        let wsi = gpu.wsi.lock().expect("Vulkan WSI");
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gpu.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: a well-formed create info on this live device.
        let pool = unsafe { gpu.device.create_command_pool(&pool_info, None) }
            .map_err(|error| format!("vkCreateCommandPool failed with {error:?}"))?;
        // The command buffers are the pool's, so they go with it.
        let pool = gpu::Guard::new(pool, |pool| {
            // SAFETY: a pool of this device's, nothing allocated from it
            // having been submitted.
            unsafe { gpu.device.destroy_command_pool(pool, None) };
        });
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        // SAFETY: the pool was just created by this device.
        let command_buffers = unsafe { gpu.device.allocate_command_buffers(&allocate) }
            .map_err(|error| format!("vkAllocateCommandBuffers failed with {error:?}"))?;
        let mut frames = gpu::GuardedVec::new(|frame: &Frame| {
            // SAFETY: objects of this device's, never submitted.
            unsafe {
                gpu.device.destroy_semaphore(frame.acquired, None);
                gpu.device.destroy_fence(frame.acquire_done, None);
                gpu.device.destroy_fence(frame.done, None);
            }
        });
        for command_buffer in command_buffers {
            // A fence created signalled, because the first use of every frame
            // waits on it before it has ever been submitted.
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            // SAFETY: a well-formed create info on this live device.
            let acquired = unsafe {
                gpu.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(|error| format!("vkCreateSemaphore failed with {error:?}"))?;
            let acquired = gpu::Guard::new(acquired, |acquired| {
                // SAFETY: a semaphore of this device's, never submitted.
                unsafe { gpu.device.destroy_semaphore(acquired, None) };
            });
            // SAFETY: a well-formed create info on this live device.
            let done = unsafe { gpu.device.create_fence(&fence_info, None) }
                .map_err(|error| format!("vkCreateFence failed with {error:?}"))?;
            let done = gpu::Guard::new(done, |fence| {
                // A failed acquisition-fence creation must not leak this fence.
                unsafe { gpu.device.destroy_fence(fence, None) };
            });
            // A separate acquire fence proves the WSI signal finished
            // even if neither the graphics nor recovery queue submit works.
            let acquire_done = unsafe {
                gpu.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .map_err(|error| format!("vkCreateFence failed with {error:?}"))?;
            frames.push(Frame {
                command_buffer,
                acquired: acquired.take(),
                acquire_done,
                acquire_pending: false,
                done: done.take(),
                submitted: false,
                source: None,
                backing: None,
            });
        }
        drop(wsi);
        // Only a window made with room for rounded corners has a surface
        // that shows alpha; see `appkit::window`.
        let transparent = snapshot.flags & sdl3::video::flags::TRANSPARENT != 0;
        let corners = match transparent {
            true => corners::Corners::new(gpu)
                .inspect_err(|reason| {
                    gpu::trace(format!(
                        "window corners unavailable, presenting opaque: {reason}"
                    ))
                })
                .ok(),
            false => None,
        };
        let mut swapchain = Self {
            gpu,
            surface: surface.take(),
            handle: vk::SwapchainKHR::null(),
            format: vk::Format::UNDEFINED,
            extent,
            requested: extent,
            pending_rebuild: None,
            retired: false,
            images: Vec::new(),
            blitted: Vec::new(),
            presented: Vec::new(),
            composite: None,
            direct: None,
            layers: Vec::new(),
            corners,
            pool: pool.take(),
            frames: frames.take(),
            frame: 0,
            present_mode: vk::PresentModeKHR::FIFO,
            composite_dirty: false,
            composite_shape: None,
            composite_dirty_since: None,
            compositor_batch_delay: Duration::from_millis(4),
            last_wsi_error: None,
            stalled_present_fence: false,
            #[cfg(test)]
            injected_unsignaled_present_fence: false,
            #[cfg(test)]
            skipped_stalled_present_fence: false,
            #[cfg(test)]
            injected_build_vk_error: None,
            #[cfg(test)]
            injected_stage_vk_error: None,
            #[cfg(test)]
            injected_queue_present_error: None,
            #[cfg(test)]
            #[cfg(test)]
            injected_queue_submit_error: None,
            acquired_unsubmitted: None,
            #[cfg(test)]
            injected_recovery_error: None,
            #[cfg(test)]
            injected_build_failure: false,
            #[cfg(test)]
            injected_unavailable_extent: false,
            #[cfg(test)]
            rebuild_attempts: 0,
            #[cfg(test)]
            last_create_old_swapchain: vk::SwapchainKHR::null(),
        };
        // From here the `Drop` above owns everything, including whatever
        // `build` leaves behind when it fails.
        swapchain.build(extent)?;
        Ok(swapchain)
    }

    /// Create the `VkSwapchainKHR` itself, replacing whatever is there.
    ///
    /// Every replacement is built whole before any of it is installed: a
    /// failure part-way leaves the swapchain exactly as it was, rather than a
    /// `self.handle` naming a swapchain this call has already destroyed.
    fn build(&mut self, requested: vk::Extent2D) -> Result<(), String> {
        self.last_wsi_error = None;
        #[cfg(test)]
        {
            self.rebuild_attempts += 1;
            if self.injected_unavailable_extent {
                return Err("the window has no area to present into".into());
            }
        }
        let surface_api = crate::vulkan::surface_api().ok_or("VK_KHR_surface is gone")?;
        let api = self.gpu.swapchain_api().ok_or("VK_KHR_swapchain is gone")?;
        // SAFETY: a physical device and a surface of this instance's.
        let capabilities = unsafe {
            surface_api.get_physical_device_surface_capabilities(self.gpu.physical, self.surface)
        }
        .map_err(|error| format!("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: {error:?}"))?;
        let formats = unsafe {
            surface_api.get_physical_device_surface_formats(self.gpu.physical, self.surface)
        }
        .map_err(|error| format!("vkGetPhysicalDeviceSurfaceFormatsKHR failed: {error:?}"))?;
        let chosen = choose_format(&formats).ok_or("the surface reports no format")?;
        let present_modes = unsafe {
            surface_api.get_physical_device_surface_present_modes(self.gpu.physical, self.surface)
        }
        .map_err(|error| format!("vkGetPhysicalDeviceSurfacePresentModesKHR failed: {error:?}"))?;
        let present_mode = choose_present_mode(&present_modes);
        let alpha = composite_alpha(&capabilities, self.corners.is_some());

        // `0xffff_ffff` in either direction is a surface which takes its size
        // from the swapchain rather than the other way round.
        let extent = match capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: requested.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: requested.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            },
            _ => capabilities.current_extent,
        };
        if extent.width == 0 || extent.height == 0 {
            return Err("the window has no area to present into".into());
        }
        // WSI can change currentExtent without a new AppKit window size.
        // Preserve the old direct image before replacing its dimensions, not
        // just when the last requested size changed.
        if self.direct.is_some() && extent != self.extent {
            let slot = self.frame % self.frames.len();
            if matches!(self.stage_direct_snapshot(false)?, Some(true)) {
                self.gpu.wait(self.frames[slot].done)?;
                self.frames[slot].submitted = false;
                finish_source(self.frames[slot].source.take());
                self.frames[slot].backing.take();
            }
        }
        // DXVK requests the surface minimum for IMMEDIATE and min+1 for
        // queued modes. Do not expand WSI's queue to hide extra host presents.
        let count = desired_image_count(
            present_mode,
            capabilities.min_image_count,
            capabilities.max_image_count,
        );
        gpu::trace(format!(
            "window swapchain: present modes {present_modes:?}, selected {present_mode:?}, images min={} max={} requested={count}",
            capabilities.min_image_count, capabilities.max_image_count
        ));
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(count)
            .image_format(chosen.format)
            .image_color_space(chosen.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            // The frame arrives as a blit, never as an attachment: the guest
            // rendered into its own drawable long before this.
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(if self.retired {
                vk::SwapchainKHR::null()
            } else {
                self.handle
            });
        // A prior attempt retired this handle, even if a later allocation
        // failed. The driver may reject an unrelated new swapchain while the
        // retired old one still occupies the native window. Rebuild called
        // idle() first, so it is safe to destroy this old handle now.
        if self.retired && self.handle != vk::SwapchainKHR::null() {
            unsafe { api.destroy_swapchain(self.handle, None) };
            self.handle = vk::SwapchainKHR::null();
        }
        // Supplying a live oldSwapchain retires it even on creation failure.
        if self.handle != vk::SwapchainKHR::null() {
            self.retired = true;
        }
        #[cfg(test)]
        {
            self.last_create_old_swapchain = info.old_swapchain;
        }
        let wsi = self.gpu.wsi.lock().expect("Vulkan WSI");
        #[cfg(test)]
        if let Some(error) = self.injected_build_vk_error.take() {
            self.last_wsi_error = Some(error);
            return Err(format!(
                "injected vkCreateSwapchainKHR failed with {error:?}"
            ));
        }
        let handle = unsafe { api.create_swapchain(&info, None) }.map_err(|error| {
            self.last_wsi_error = Some(error);
            format!("vkCreateSwapchainKHR failed with {error:?}")
        })?;
        let handle = gpu::Guard::new(handle, |handle| {
            // SAFETY: a swapchain of this device's which was never installed,
            // so nothing has acquired an image from it.
            unsafe { api.destroy_swapchain(handle, None) };
        });
        #[cfg(test)]
        if std::mem::take(&mut self.injected_build_failure) {
            return Err("injected failure after oldSwapchain retirement".into());
        }
        // SAFETY: the swapchain was just created by this device.
        let images = unsafe { api.get_swapchain_images(*handle) }.map_err(|error| {
            self.last_wsi_error = Some(error);
            format!("vkGetSwapchainImagesKHR failed with {error:?}")
        })?;
        if crate::stats::enabled() {
            eprintln!(
                "rosemary: Metal: swapchain mode={present_mode:?} images min={} max={} requested={count} actual={}",
                capabilities.min_image_count,
                capabilities.max_image_count,
                images.len()
            );
        }
        // One semaphore per image, made before anything is installed so that a
        // failure half way through leaves neither these nor the swapchain.
        let mut blitted = gpu::GuardedVec::new(|semaphore: &vk::Semaphore| {
            // SAFETY: a semaphore of this device's, never submitted.
            unsafe { self.gpu.device.destroy_semaphore(*semaphore, None) };
        });
        for _ in &images {
            // SAFETY: a well-formed create info on this live device.
            blitted.push(
                unsafe {
                    self.gpu
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                }
                .map_err(|error| {
                    self.last_wsi_error = Some(error);
                    format!("vkCreateSemaphore failed with {error:?}")
                })?,
            );
        }
        let mut presented = gpu::GuardedVec::new(|present: &PresentFence| {
            // SAFETY: a fence of this device's, never submitted.
            unsafe { self.gpu.device.destroy_fence(present.fence, None) };
        });
        if self.gpu.has_present_fences() {
            for _ in &images {
                // Present fences must be unsignalled when passed to
                // vkQueuePresentKHR.
                let fence = unsafe {
                    self.gpu
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                }
                .map_err(|error| {
                    self.last_wsi_error = Some(error);
                    format!("vkCreateFence failed with {error:?}")
                })?;
                presented.push(PresentFence {
                    fence,
                    pending: false,
                });
            }
        }

        drop(wsi);
        // A static Metal layer cannot be redrawn by the host after a
        // resize. Preserve the window's GPU composite across the replacement.
        let keep_fifo = uses_layer_compositor(self.present_mode)
            && uses_layer_compositor(present_mode)
            && self.composite.is_some();
        let keep_direct = self.direct.is_some() && uses_layer_compositor(present_mode);
        let replace_composite = if keep_fifo {
            let old = self.composite.as_ref().expect("existing window snapshot");
            // The corners pass keeps the composite in its own format, never
            // the swapchain's, so compare with what would be made.
            let description =
                Self::composite_description(chosen.format, extent, self.corners.is_some());
            if old.width == extent.width
                && old.height == extent.height
                && old.format == description.format
            {
                None
            } else {
                let replacement = self.gpu.create_image(&description)?;
                let copied = self.gpu.one_shot(|device, command_buffer| unsafe {
                    keep_top_left(device, command_buffer, old, &replacement);
                });
                if let Err(reason) = copied {
                    self.gpu.destroy_image(replacement);
                    return Err(reason);
                }
                Some(replacement)
            }
        } else {
            None
        };

        // All fallible allocations and copies completed.
        let blitted = blitted.take();
        let presented = presented.take();
        let old = std::mem::replace(&mut self.handle, handle.take());
        if old != vk::SwapchainKHR::null() {
            // SAFETY: retired by the creation above, and every submission
            // touching its images has been waited for by the caller.
            unsafe { api.destroy_swapchain(old, None) };
        }
        // The old swapchain's images are gone with it, and so is the meaning
        // of a semaphore indexed by one.
        for semaphore in std::mem::replace(&mut self.blitted, blitted) {
            // SAFETY: `idle` waited either the presentation fences or the
            // queue before the swapchain these belonged to was destroyed.
            unsafe { self.gpu.device.destroy_semaphore(semaphore, None) };
        }
        for present in std::mem::replace(&mut self.presented, presented) {
            // SAFETY: `idle` waited every pending use of this fence.
            unsafe { self.gpu.device.destroy_fence(present.fence, None) };
        }
        self.requested = requested;
        self.acquired_unsubmitted = None;
        self.retired = false;
        self.present_mode = present_mode;
        self.composite_dirty = keep_fifo || keep_direct;
        self.composite_shape = None;
        self.composite_dirty_since = (keep_fifo || keep_direct).then(Instant::now);
        self.format = chosen.format;
        self.extent = extent;
        self.images = images;
        if !keep_fifo {
            if let Some(composite) = self.composite.take() {
                self.gpu.destroy_image(composite);
            }
        } else if let Some(replacement) = replace_composite {
            let old = self
                .composite
                .replace(replacement)
                .expect("old window snapshot");
            self.gpu.destroy_image(old);
        }
        // A surface that cannot show premultiplied alpha is presented
        // opaque, which the pass would only get wrong.
        if alpha != vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED {
            self.corners = None;
        }
        if let Some(corners) = self.corners.as_mut() {
            corners.forget_output();
        }
        Ok(())
    }

    /// Finish the source of every submitted frame whose fence has signalled,
    /// and say whether there was one.
    fn retire_copied(&mut self) -> bool {
        let mut retired = false;
        for frame in &mut self.frames {
            if (frame.source.is_none() && frame.backing.is_none()) || !frame.submitted {
                continue;
            }
            // Both an Objective-C drawable and a ref-counted store image are
            // released only after their last GPU read has completed.
            if self.gpu.fence_signaled(frame.done).unwrap_or(false) {
                finish_source(frame.source.take());
                frame.backing.take();
                retired = true;
            }
        }
        retired
    }

    /// The fences of the submitted frames still holding a source.
    fn copying(&self) -> Vec<vk::Fence> {
        self.frames
            .iter()
            .filter(|frame| frame.submitted && (frame.source.is_some() || frame.backing.is_some()))
            .map(|frame| frame.done)
            .collect()
    }

    /// Wait for every frame in flight and rebuild at `extent`.
    fn recover_unsubmitted_acquire(&mut self) -> Result<(), String> {
        let Some((semaphore, fence)) = self.acquired_unsubmitted else {
            return Ok(());
        };
        #[cfg(test)]
        if let Some(error) = self.injected_recovery_error.take() {
            return Err(format!("injected recovery submit failure: {error:?}"));
        }
        self.gpu.consume_acquired_semaphore(semaphore, fence)?;
        // A successful recovery consumed the binary semaphore, but WSI
        // still owns its acquired image until the swapchain is retired.
        self.acquired_unsubmitted = None;
        self.rebuild(self.requested)
    }

    fn rebuild(&mut self, extent: vk::Extent2D) -> Result<(), String> {
        self.idle();
        self.build(extent)
    }

    /// Block until nothing this swapchain gave the queue is still running.
    ///
    /// Presentation fences cover the presentation engine's access to the
    /// image and wait semaphore. On an older device, only making the queue
    /// idle supplies that guarantee.
    fn idle(&mut self) {
        // At least one present omitted its maintenance1 fence because an old
        // fence was still pending. Even if that *old* fence signals later, it
        // cannot cover subsequent fence-less presentations. Queue idle is
        // necessary to retire their WSI use of images and semaphores safely.
        if !self.gpu.has_present_fences() || self.stalled_present_fence {
            self.gpu.wait_queue_idle();
            self.stalled_present_fence = false;
            for present in &mut self.presented {
                present.pending = false;
            }
            for frame in &mut self.frames {
                frame.submitted = false;
                finish_source(frame.source.take());
                frame.backing.take();
            }
            return;
        }
        let submitted = self
            .frames
            .iter()
            .filter(|frame| frame.submitted)
            .map(|frame| frame.done);
        let presented = self
            .presented
            .iter()
            .filter(|present| present.pending)
            .map(|present| present.fence);
        let fences = submitted.chain(presented).collect::<Vec<_>>();
        if !fences.is_empty() {
            // SAFETY: each fence belongs to this device and is associated
            // with an operation which was successfully queued.
            let _ = unsafe { self.gpu.device.wait_for_fences(&fences, true, u64::MAX) };
        }
        for frame in &mut self.frames {
            frame.submitted = false;
            finish_source(frame.source.take());
            frame.backing.take();
        }
        for present in &mut self.presented {
            present.pending = false;
        }
    }

    /// The image layers accumulate into, made on the first frame that needs
    /// one and kept in `GENERAL` for the same reason every other image in
    /// this crate is.
    ///
    /// With the corners pass, the composite is one the pass can read.
    fn composite_description(
        format: vk::Format,
        extent: vk::Extent2D,
        corners_enabled: bool,
    ) -> ImageDescription {
        if corners_enabled {
            corners::storage(extent.width, extent.height)
        } else {
            ImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                width: extent.width,
                height: extent.height,
                depth: 1,
                levels: 1,
                layers: 1,
                samples: 1,
                usage: gpu::USAGE_RENDER_TARGET,
                cube: false,
                components: vk::ComponentMapping::default(),
            }
        }
    }

    fn composite(&mut self) -> Result<vk::Image, String> {
        if self.composite.is_none() {
            let description =
                Self::composite_description(self.format, self.extent, self.corners.is_some());
            let image = self.gpu.create_image(&description)?;
            // A fresh image's contents are undefined, and the frame that
            // brings the composite into existence has only the one layer
            // that asked for it - so everywhere the other layers have not
            // reached yet would be whatever the allocation held.
            let clear = vk::ClearColorValue { float32: [0.0; 4] };
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            self.gpu.one_shot(|device, command_buffer| {
                // SAFETY: recording into a command buffer the call owns; the
                // image is in GENERAL from creation.
                unsafe {
                    device.cmd_clear_color_image(
                        command_buffer,
                        image.image,
                        vk::ImageLayout::GENERAL,
                        &clear,
                        &[range],
                    )
                };
            })?;
            self.composite = Some(image);
        }
        Ok(self.composite.as_ref().expect("just made").image)
    }

    /// Prepare one frame and reserve its WSI semaphore before releasing the
    /// state lock for the blocking image acquisition.
    fn prepare_present(
        &mut self,
        frame: &appkit::view::Frame,
        regions: Option<&[Region]>,
        nonblocking: bool,
    ) -> Result<Option<(PreparedPresent, AcquireRequest)>, String> {
        // One layer covering the window needs no accumulator: what it draws
        // is the whole of what the window shows, so it goes straight into the
        // image the presentation engine hands back.
        // The corners pass reads the composite, so a window with it always
        // has one.
        let direct = self.layers.len() <= 1
            && (frame.covers_window() || self.direct.is_some())
            && regions.is_none()
            && self.corners.is_none();
        let composite = match direct {
            true => None,
            false => Some(self.composite()?),
        };
        if let Some(corners) = self.corners.as_mut() {
            corners.prepare(self.extent.width, self.extent.height)?;
        }

        let in_flight = frame_slots(self.present_mode);
        let slot = self.frame % in_flight;
        let this = &self.frames[slot];
        let (command_buffer, acquired, acquire_done, done) = (
            this.command_buffer,
            this.acquired,
            this.acquire_done,
            this.done,
        );
        // SAFETY: a fence of this device's, signalled at creation or by the
        // submission this slot last made.
        if this.submitted {
            let timeout = if nonblocking { 0 } else { u64::MAX };
            if !self.gpu.wait_fence_timeout(done, timeout)? {
                return Ok(None);
            }
        }
        self.frames[slot].submitted = false;
        finish_source(self.frames[slot].source.take());
        self.frames[slot].backing.take();
        if self.frames[slot].acquire_pending {
            // WSI may signal an acquisition fence after a successful
            // vkAcquireNextImageKHR returns. Never stall AppKit while it
            // waits; retry this frame slot at a subsequent pump instead.
            let timeout = if nonblocking { 0 } else { u64::MAX };
            if !self.gpu.wait_fence_timeout(acquire_done, timeout)? {
                return Ok(None);
            }
            self.frames[slot].acquire_pending = false;
            self.gpu.fence_result(
                unsafe { self.gpu.device.reset_fences(&[acquire_done]) },
                "vkResetFences",
            )?;
        }
        Ok(Some((
            PreparedPresent {
                slot,
                command_buffer,
                acquired,
                done,
                composite,
            },
            AcquireRequest {
                handle: self.handle,
                semaphore: acquired,
                fence: acquire_done,
            },
        )))
    }

    /// Finish a prepared frame after the presentation engine returns an image.
    fn finish_present(
        &mut self,
        frame: &appkit::view::Frame,
        source: PresentSource,
        regions: Option<&[Region]>,
        prepared: PreparedPresent,
        acquired: Result<Option<(u32, bool)>, vk::Result>,
    ) -> Result<Option<vk::Result>, String> {
        // An unsuccessful acquire did not submit this frame. Report its
        // raw Vulkan status to the caller so it can propagate DEVICE_LOST
        // across the logical device and mark SURFACE_LOST window-local.
        let acquired = match acquired {
            Ok(acquired) => acquired,
            Err(error) => return Ok(Some(error)),
        };
        let Some((index, _suboptimal)) = acquired else {
            // The swapchain was rebuilt underneath the acquisition. The frame
            // the guest drew is gone, which is what a resize costs; the next
            // one lands in the new swapchain.
            self.rebuild(self.requested)?;
            return Ok(None);
        };
        let PreparedPresent {
            slot,
            command_buffer,
            acquired,
            done,
            composite,
        } = prepared;
        self.frames[slot].acquire_pending = true;
        // Until a successful queue submission consumes the acquired
        // semaphore, all subsequent errors must retire the WSI swapchain.
        // This semaphore must be consumed by a queue wait even if the
        // subsequent image-copy queue submission fails.
        self.acquired_unsubmitted = Some((prepared.acquired, prepared.done));
        let target = self.images[index as usize];
        // Safe to signal: this image has just come back from the
        // presentation engine, so the presentation which last waited on this
        // semaphore is finished.
        let blitted = self.blitted[index as usize];
        let present_fence = if let Some(present) = self.presented.get_mut(index as usize) {
            #[cfg(test)]
            let forced_unsignaled = std::mem::take(&mut self.injected_unsignaled_present_fence);
            #[cfg(not(test))]
            let forced_unsignaled = false;
            if present.pending
                && (forced_unsignaled || !self.gpu.wait_fence_timeout(present.fence, 0)?)
            {
                // Successful acquisition means WSI returned this image, but
                // some implementations do not signal its old maintenance1
                // fence promptly. Do not block the compositor. Keep that
                // fence untouched and submit the new present without a fence.
                // Queue idle handles the fallback on rebuild/teardown.
                self.stalled_present_fence = true;
                #[cfg(test)]
                {
                    self.skipped_stalled_present_fence = true;
                }
                None
            } else {
                present.pending = false;
                // SAFETY: fresh or signaled; no old queued use remains.
                unsafe { self.gpu.device.reset_fences(&[present.fence]) }
                    .map_err(|error| format!("vkResetFences failed with {error:?}"))?;
                Some(present.fence)
            }
        } else {
            None
        };

        let record_timer = crate::stats::timer();
        let device = &self.gpu.device;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: this slot's fence was waited for above, so nothing is
        // reading the command buffer.
        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|error| format!("vkResetCommandBuffer failed with {error:?}"))?;
            device
                .begin_command_buffer(command_buffer, &begin)
                .map_err(|error| format!("vkBeginCommandBuffer failed with {error:?}"))?;
        }

        // SAFETY: recording into a command buffer this call owns; every image
        // named below belongs to this device.
        unsafe {
            prepare_source(device, command_buffer, source.image);
            let whole = [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ];
            // The swapchain image is display-sized even if its NSWindow is
            // not. Clear it before copying the *clipped* accumulator or layer:
            // an overlay blended on the GPU can otherwise escape the frame.
            // Transfer clear is guaranteed to cover stale pixels from prior
            // presents; the compositor's excluded region is opaque black.
            let logical = logical_image_clip(frame.window, self.extent);
            let restricted = logical != Some(whole);
            transition_to_transfer_dst(device, command_buffer, target);
            if restricted {
                let black = vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                };
                device.cmd_clear_color_image(
                    command_buffer,
                    target,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &black,
                    &[vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }],
                );
                // Transfer write -> transfer write/read within this command
                // buffer is ordered, but the barrier expresses the dependency.
                barrier(
                    device,
                    command_buffer,
                    target,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_WRITE,
                );
            }
            match composite {
                // The layer's own rectangle of the accumulator, then the
                // whole accumulator into the image being presented.
                Some(composite) => {
                    // A layer wholly off the edge of its window contributes
                    // nothing; what the composite already holds is still
                    // what the window shows, so it is presented unchanged.
                    match regions {
                        Some(regions) => {
                            for region in regions {
                                let part = appkit::view::Frame {
                                    x: region.x,
                                    y: region.y,
                                    width: region.width.max(0) as u32,
                                    height: region.height.max(0) as u32,
                                    ..*frame
                                };
                                if let Some(into) = clamp(&part, self.extent) {
                                    blit(
                                        device,
                                        command_buffer,
                                        source.image,
                                        source_of(into, frame, (source.width, source.height)),
                                        composite,
                                        vk::ImageLayout::GENERAL,
                                        into,
                                    );
                                }
                            }
                        }
                        None => {
                            if let Some(into) = clamp(frame, self.extent) {
                                blit(
                                    device,
                                    command_buffer,
                                    source.image,
                                    source_of(into, frame, (source.width, source.height)),
                                    composite,
                                    vk::ImageLayout::GENERAL,
                                    into,
                                );
                            }
                        }
                    }
                    let shown = match &self.corners {
                        Some(corners) => {
                            let shape = self.composite_shape.unwrap_or_else(|| {
                                corners::Shape::of(
                                    frame.window,
                                    self.extent.width,
                                    self.extent.height,
                                )
                            });
                            let image = self.composite.as_ref().expect("made above");
                            corners.record(command_buffer, slot, image, shape)
                        }
                        None => {
                            // The blit above must land before the one below
                            // reads it.
                            barrier(
                                device,
                                command_buffer,
                                composite,
                                vk::ImageLayout::GENERAL,
                                vk::ImageLayout::GENERAL,
                                vk::AccessFlags::TRANSFER_WRITE,
                                vk::AccessFlags::TRANSFER_READ,
                            );
                            composite
                        }
                    };
                    if let Some(clipped) = logical {
                        blit(
                            device,
                            command_buffer,
                            shown,
                            clipped,
                            target,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            clipped,
                        );
                    }
                }
                None => {
                    if let Some((from, into)) = direct_blit_to(
                        frame,
                        self.extent,
                        appkit::window::presentation_clip(frame.window),
                        (source.width, source.height),
                    ) {
                        blit(
                            device,
                            command_buffer,
                            source.image,
                            from,
                            target,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            into,
                        );
                    }
                }
            }
            transition_to_present(device, command_buffer, target);
            device
                .end_command_buffer(command_buffer)
                .map_err(|error| format!("vkEndCommandBuffer failed with {error:?}"))?;
        }
        crate::stats::finish_timer(crate::stats::Counter::PresentRecordMicros, record_timer);

        #[cfg(test)]
        let injected_submit = self.injected_queue_submit_error.take();
        #[cfg(not(test))]
        let injected_submit: Option<vk::Result> = None;
        crate::stats::timed(crate::stats::Counter::PresentSubmitMicros, || {
            if let Some(error) = injected_submit {
                return Err(format!("injected vkQueueSubmit failure: {error:?}"));
            }
            self.gpu.submit_synchronized(
                command_buffer,
                done,
                &[(acquired, vk::PipelineStageFlags::TRANSFER)],
                &[blitted],
            )
        })?;
        // The acquire semaphore is consumed only after vkQueueSubmit succeeds.
        self.acquired_unsubmitted = None;
        self.frames[slot].submitted = true;
        self.frames[slot].source = None;
        self.frames[slot].backing = None;
        self.frame = self.frame.wrapping_add(1);

        let result = crate::stats::timed(crate::stats::Counter::QueuePresentMicros, || {
            self.gpu.present(self.handle, index, blitted, present_fence)
        });
        // The fault-injection test lets real Vulkan finish its queue operation,
        // then replaces its result to exercise the post-acquire error path.
        #[cfg(test)]
        let result = self.injected_queue_present_error.take().map_or(result, Err);
        // Exclude device loss and allocation failures from present-fence
        // waits. The Gamescope WSI opt-in can also omit a signal on ordinary
        // failed presents; keep that driver quirk out of normal WSI paths.
        static GAMESCOPE_WSI: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var("ENABLE_GAMESCOPE_WSI").as_deref() == Ok("1")
        });
        if present_fence.is_some() && present_fence_will_signal(&result, *GAMESCOPE_WSI) {
            self.presented[index as usize].pending = true;
        }
        match result {
            Ok(suboptimal) => {
                if suboptimal {
                    // The image still reached the window; the next frame is
                    // the one that gets a swapchain matching it.
                    self.rebuild(self.requested)?;
                }
                Ok(None)
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.rebuild(self.requested)?;
                Ok(None)
            }
            Err(error) => {
                // The blit submission owns its source until the frame fence,
                // but the compositor worker must see terminal errors and stop.
                gpu::trace(format!("vkQueuePresentKHR failed with {error:?}"));
                Ok(Some(error))
            }
        }
    }
}

/// Bound the wait for a compositor image. A minimized X11 window may stop
/// returning images entirely; UINT64_MAX would block window teardown while
/// the worker held the presentation mutex. Timeout leaves the acquisition
/// semaphore untouched, and the dirty accumulator remains pending.
/// Waits up to a frame for an image when `wait`, and otherwise only polls:
/// the caller holds the window's presentation lock throughout.
fn acquire_compositor_image(
    gpu: &'static Gpu,
    request: AcquireRequest,
    wait: bool,
) -> CompositorAcquire {
    const TIMEOUT_NS: u64 = 20_000_000;
    let Some(api) = gpu.swapchain_api() else {
        return CompositorAcquire::Terminal(vk::Result::ERROR_DEVICE_LOST);
    };
    let timeout = if wait { TIMEOUT_NS } else { 0 };
    // SAFETY: the live window swapchain and its unused acquisition semaphore
    // remain owned by the worker while it holds the presentation mutex.
    classify_compositor_acquire(unsafe {
        api.acquire_next_image(request.handle, timeout, request.semaphore, request.fence)
    })
}

/// Copy what `old` held into `new` unscaled, from the top-left corner as the
/// window's store keeps it across a resize, and clear the rest: stretched,
/// the stale frame would be presented distorted until every layer redraws.
///
/// # Safety
///
/// Recording into `command_buffer`, with both images of `device`'s and in
/// `GENERAL`.
unsafe fn keep_top_left(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    old: &GpuImage,
    new: &GpuImage,
) {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let kept = [
        vk::Offset3D { x: 0, y: 0, z: 0 },
        vk::Offset3D {
            x: old.width.min(new.width).max(1) as i32,
            y: old.height.min(new.height).max(1) as i32,
            z: 1,
        },
    ];
    unsafe {
        device.cmd_clear_color_image(
            command_buffer,
            new.image,
            vk::ImageLayout::GENERAL,
            &vk::ClearColorValue { float32: [0.0; 4] },
            &[range],
        );
        barrier(
            device,
            command_buffer,
            new.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        blit(
            device,
            command_buffer,
            old.image,
            kept,
            new.image,
            vk::ImageLayout::GENERAL,
            kept,
        );
    }
}

/// How long the compositor worker sleeps, unlocked, after its `timeouts`th
/// acquisition in a row that found no image.
fn compositor_acquire_backoff(timeouts: u32) -> Duration {
    Duration::from_millis(1 << timeouts.saturating_sub(1).min(5))
}

fn classify_compositor_acquire(outcome: Result<(u32, bool), vk::Result>) -> CompositorAcquire {
    match outcome {
        Ok((index, suboptimal)) => CompositorAcquire::Ready(Some((index, suboptimal))),
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => CompositorAcquire::Ready(None),
        Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => CompositorAcquire::TimedOut,
        Err(error) => CompositorAcquire::Terminal(error),
    }
}

#[cfg(test)]
fn acquire_image(
    gpu: &'static Gpu,
    request: AcquireRequest,
    nonblocking: bool,
) -> Result<Option<(u32, bool)>, vk::Result> {
    let api = gpu
        .swapchain_api()
        .ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?;
    // SAFETY: a live swapchain and semaphore of this device's; no fence, so
    // the semaphore is the only signal.
    match unsafe {
        api.acquire_next_image(
            request.handle,
            if nonblocking { 0 } else { u64::MAX },
            request.semaphore,
            request.fence,
        )
    } {
        Ok((index, suboptimal)) => Ok(Some((index, suboptimal))),
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Nothing in this runtime destroys an `SDL_Window` today - `-[NSWindow
/// close]` orders it out and the window stays in the list - so a swapchain is
/// only dropped if one is taken out of the map. Teardown lives here anyway
/// rather than in a function nothing calls: whichever way a swapchain stops
/// being reachable, the surface, the semaphores and the pool go with it, and
/// the order below is the one Vulkan requires.
impl Drop for Swapchain {
    fn drop(&mut self) {
        self.idle();
        // If both queue submissions failed, the WSI acquire may still be
        // scheduling its semaphore signal. The separate acquire fence proves
        // completion of that signal; queue idle alone does not.
        if let Some((semaphore, _)) = self.acquired_unsubmitted {
            if let Some(frame) = self.frames.iter().find(|f| f.acquired == semaphore) {
                if let Err(reason) = self.gpu.wait(frame.acquire_done) {
                    // A device-lost wait counts as completed for resource
                    // lifetime under Vulkan's lost-device rules.
                    gpu::trace(format!("unconsumed WSI acquire teardown: {reason}"));
                }
            }
        }
        let device = &self.gpu.device;
        // The swapchain first: its images are what the semaphores below were
        // signalled for, and a surface outlives the swapchain made on it.
        if self.handle != vk::SwapchainKHR::null() {
            if let Some(api) = self.gpu.swapchain_api() {
                // SAFETY: teardown has finished references to this swapchain.
                unsafe { api.destroy_swapchain(self.handle, None) };
            }
        }
        for frame in &self.frames {
            // SAFETY: objects of this device's, with nothing in flight after
            // the wait above.
            unsafe {
                device.destroy_semaphore(frame.acquired, None);
                device.destroy_fence(frame.acquire_done, None);
                device.destroy_fence(frame.done, None);
            }
        }
        for semaphore in std::mem::take(&mut self.blitted) {
            // SAFETY: as above, and the swapchain whose presentations waited
            // on these is destroyed first.
            unsafe { device.destroy_semaphore(semaphore, None) };
        }
        for present in std::mem::take(&mut self.presented) {
            // SAFETY: the wait above retired every pending presentation.
            unsafe { device.destroy_fence(present.fence, None) };
        }
        // SAFETY: the pool owns the command buffers, which are freed with it.
        unsafe { device.destroy_command_pool(self.pool, None) };
        if let Some(composite) = self.composite.take() {
            self.gpu.destroy_image(composite);
        }
        if let Some(instance) = crate::vulkan::instance_handle() {
            sdl3::video::vulkan::destroy_surface(instance, self.surface.as_raw());
        }
    }
}
