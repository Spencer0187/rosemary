//! CoreMedia contiguous byte buffers. Backing storage never moves while referenced.

#![allow(non_snake_case)]

use core::ffi::c_void;
use objc::class::Class;
use objc::object::Id;
use std::sync::{Arc, Mutex, OnceLock};

const BAD_CUSTOM_SOURCE: i32 = -12702;
const BAD_OFFSET: i32 = -12703;
const BAD_LENGTH: i32 = -12704;
const BAD_POINTER: i32 = -12705;
const EMPTY_BUFFER: i32 = -12706;
const ALLOC_FAILED: i32 = -12701;
const STRUCT_ALLOC_FAILED: i32 = -12700;
const ASSURE_NOW: u32 = 1;

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(pointer: *mut c_void);
}

/// Opaque until embedded guest callbacks can be thunked by the ABI bridge.
#[repr(C)]
pub struct CustomBlockSource {
    _private: [u8; 0],
}

struct Allocation {
    memory: *mut u8,
    capacity: usize,
}

// The allocation pointer is read, allocated and freed only while its shared
// Mutex is held, so distinct buffer views can own it across threads.
unsafe impl Send for Allocation {}

impl Allocation {
    fn allocate(&mut self) -> i32 {
        if !self.memory.is_null() {
            return 0;
        }
        // SAFETY: only malloc-backed storage is supported.
        let raw = unsafe { malloc(self.capacity) };
        if raw.is_null() {
            return ALLOC_FAILED;
        }
        self.memory = raw.cast();
        0
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if !self.memory.is_null() {
            // SAFETY: the last buffer retaining the allocation owns the free.
            unsafe { free(self.memory.cast()) };
        }
    }
}

struct Storage {
    allocation: Arc<Mutex<Allocation>>,
    start: usize,
    length: usize,
}

impl Storage {
    fn allocate(&mut self) -> i32 {
        self.allocation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .allocate()
    }

    fn pointer(&mut self, offset: usize) -> Result<*mut u8, i32> {
        let mut shared = self.allocation.lock().unwrap_or_else(|e| e.into_inner());
        let status = shared.allocate();
        if status != 0 {
            return Err(status);
        }
        // SAFETY: creation validates start+length within allocation capacity;
        // all aliases hold the same allocation alive and its bytes never move.
        Ok(unsafe { shared.memory.add(self.start + offset) })
    }
}

#[repr(C)]
struct Body {
    isa: *const Class,
    contents: *mut Mutex<Storage>,
}

fn class() -> &'static Class {
    static CLASS: OnceLock<usize> = OnceLock::new();
    let pointer = *CLASS.get_or_init(|| {
        let class =
            corefoundation::base::register(c"__NSCMBlockBuffer", None, size_of::<Body>(), true);
        class.replace_method(
            objc::sel::get("dealloc"),
            dealloc as *const c_void,
            c"v16@0:8",
        );
        objc::class::ptr(class) as usize
    });
    // SAFETY: class registration persists for the life of the process.
    unsafe { objc::class::from_ptr(pointer as *const Class).expect("CMBlockBuffer class") }
}

unsafe extern "C" fn dealloc(receiver: Id, _sel: objc::sel::Sel) {
    // SAFETY: the final release owns the boxed storage exactly once.
    unsafe {
        let body = receiver as *mut Body;
        if !(*body).contents.is_null() {
            drop(Box::from_raw((*body).contents));
            (*body).contents = core::ptr::null_mut();
        }
        objc::arc::finalize(receiver);
    }
}

fn new_buffer(storage: Storage) -> *const c_void {
    // SAFETY: registered class is allocated with Body's exact layout.
    let body: *mut Body = unsafe { corefoundation::base::allocate(class()) };
    if body.is_null() {
        return core::ptr::null();
    }
    // SAFETY: the allocation's contents field is owned by the object.
    unsafe { (*body).contents = Box::into_raw(Box::new(Mutex::new(storage))) };
    body.cast()
}

/// # Safety
/// Handle must be null or a live CF object from this runtime.
unsafe fn storage<'a>(handle: *const c_void) -> Option<&'a Mutex<Storage>> {
    if handle.is_null() || !unsafe { corefoundation::base::is_kind_of(handle, class()) } {
        return None;
    }
    // SAFETY: instances of class() contain a live Mutex<Storage>.
    unsafe { (*(handle as *const Body)).contents.as_ref() }
}

pub extern "C" fn CMBlockBufferGetTypeID() -> usize {
    corefoundation::base::type_id(class())
}

/// # Safety
/// Handle must be null or a live builtin CoreFoundation object.
pub(crate) unsafe fn is_buffer(handle: *const c_void) -> bool {
    unsafe { storage(handle) }.is_some()
}

/// Creates a single-block buffer. Unsupported ambiguous raw-memory ownership
/// is rejected: caller-owned memory and callback lifetimes are unsupported.
///
/// # Safety
/// Output must be writable. Embedded callbacks are explicitly rejected
/// because guest function pointers need ABI substitution before invocation.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn CMBlockBufferCreateWithMemoryBlock(
    allocator: *const c_void,
    memory: *mut c_void,
    block_length: usize,
    block_allocator: *const c_void,
    custom_source: *const CustomBlockSource,
    offset: usize,
    length: usize,
    flags: u32,
    out: *mut *const c_void,
) -> i32 {
    if out.is_null() {
        return BAD_POINTER;
    }
    // SAFETY: out is a writable result slot.
    unsafe { *out = core::ptr::null() };
    if !corefoundation::api::supports_malloc_allocator(allocator) {
        return BAD_CUSTOM_SOURCE;
    }
    if block_length == 0 || block_length > isize::MAX as usize {
        return BAD_LENGTH;
    }
    if offset > block_length {
        return BAD_OFFSET;
    }
    if length == 0 || length > block_length - offset {
        return BAD_LENGTH;
    }
    // Guest embedded CustomBlockSource callback pointers are not ABI-thunked.
    // Reject borrowed external memory rather than guessing who releases it.
    if !memory.is_null()
        || !corefoundation::api::supports_malloc_allocator(block_allocator)
        || !custom_source.is_null()
    {
        return BAD_CUSTOM_SOURCE;
    }
    let mut storage = Storage {
        allocation: Arc::new(Mutex::new(Allocation {
            memory: core::ptr::null_mut(),
            capacity: block_length,
        })),
        start: offset,
        length,
    };
    if flags & ASSURE_NOW != 0 {
        let status = storage.allocate();
        if status != 0 {
            return status;
        }
    }
    let result = new_buffer(storage);
    if result.is_null() {
        return STRUCT_ALLOC_FAILED;
    }
    unsafe { *out = result };
    0
}

/// # Safety
/// The source must be a valid retained block buffer; out must be writable.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn CMBlockBufferCreateContiguous(
    allocator: *const c_void,
    source: *const c_void,
    block_allocator: *const c_void,
    custom_source: *const CustomBlockSource,
    offset: usize,
    length: usize,
    flags: u32,
    out: *mut *const c_void,
) -> i32 {
    if out.is_null() {
        return BAD_POINTER;
    }
    unsafe { *out = core::ptr::null() };
    if !corefoundation::api::supports_malloc_allocator(allocator) {
        return BAD_CUSTOM_SOURCE;
    }
    let Some(original) = (unsafe { storage(source) }) else {
        return BAD_POINTER;
    };
    let source_len = original.lock().unwrap_or_else(|e| e.into_inner()).length;
    if source_len == 0 {
        return EMPTY_BUFFER;
    }
    if offset >= source_len {
        return BAD_OFFSET;
    }
    let length = if length == 0 {
        source_len - offset
    } else {
        length
    };
    if length > source_len - offset {
        return BAD_OFFSET;
    }
    if !corefoundation::api::supports_malloc_allocator(block_allocator) || !custom_source.is_null()
    {
        return BAD_CUSTOM_SOURCE;
    }
    // Every represented block is already contiguous. A new view retains its
    // allocation; releasing the original cannot invalidate the view.
    let (allocation, start) = {
        let guard = original.lock().unwrap_or_else(|e| e.into_inner());
        (Arc::clone(&guard.allocation), guard.start + offset)
    };
    let mut view = Storage {
        allocation,
        start,
        length,
    };
    if flags & ASSURE_NOW != 0 {
        let status = view.allocate();
        if status != 0 {
            return status;
        }
    }
    let result = new_buffer(view);
    if result.is_null() {
        return STRUCT_ALLOC_FAILED;
    }
    unsafe { *out = result };
    0
}

/// # Safety
/// The handle must refer to a live builtin CoreFoundation object.
pub unsafe extern "C" fn CMBlockBufferGetDataLength(handle: *const c_void) -> usize {
    let Some(contents) = (unsafe { storage(handle) }) else {
        return 0;
    };
    contents.lock().unwrap_or_else(|e| e.into_inner()).length
}

/// # Safety
/// Output slots must be writable. The returned pointer is borrowed until the
/// final CF release of the buffer and must not be used after that release.
pub unsafe extern "C" fn CMBlockBufferGetDataPointer(
    handle: *const c_void,
    offset: usize,
    length_at_offset_out: *mut usize,
    total_length_out: *mut usize,
    data_pointer_out: *mut *mut u8,
) -> i32 {
    let Some(contents) = (unsafe { storage(handle) }) else {
        return BAD_POINTER;
    };
    let mut guard = contents.lock().unwrap_or_else(|e| e.into_inner());
    if guard.length == 0 {
        return EMPTY_BUFFER;
    }
    if offset >= guard.length {
        return BAD_OFFSET;
    }
    let pointer = match guard.pointer(offset) {
        Ok(pointer) => pointer,
        Err(status) => return status,
    };
    // SAFETY: caller-provided slots are valid or null.
    unsafe {
        if !length_at_offset_out.is_null() {
            *length_at_offset_out = guard.length - offset;
        }
        if !total_length_out.is_null() {
            *total_length_out = guard.length;
        }
        if !data_pointer_out.is_null() {
            *data_pointer_out = pointer;
        }
    }
    0
}

/// # Safety
/// Destination must point to at least length writable bytes.
pub unsafe extern "C" fn CMBlockBufferCopyDataBytes(
    handle: *const c_void,
    offset: usize,
    length: usize,
    destination: *mut c_void,
) -> i32 {
    let Some(contents) = (unsafe { storage(handle) }) else {
        return BAD_POINTER;
    };
    let mut guard = contents.lock().unwrap_or_else(|e| e.into_inner());
    if offset > guard.length {
        return BAD_OFFSET;
    }
    if length == 0 || length > guard.length - offset {
        return BAD_LENGTH;
    }
    if destination.is_null() {
        return BAD_POINTER;
    }
    let pointer = match guard.pointer(offset) {
        Ok(pointer) => pointer,
        Err(status) => return status,
    };
    // SAFETY: checked the ranges; copy also permits overlapping guest slices.
    unsafe { core::ptr::copy(pointer, destination.cast(), length) };
    0
}

/// # Safety
/// Source must address at least length readable bytes.
pub unsafe extern "C" fn CMBlockBufferReplaceDataBytes(
    source: *const c_void,
    handle: *const c_void,
    offset: usize,
    length: usize,
) -> i32 {
    let Some(contents) = (unsafe { storage(handle) }) else {
        return BAD_POINTER;
    };
    let mut guard = contents.lock().unwrap_or_else(|e| e.into_inner());
    if offset > guard.length {
        return BAD_OFFSET;
    }
    if length == 0 || length > guard.length - offset {
        return BAD_LENGTH;
    }
    if source.is_null() {
        return BAD_POINTER;
    }
    let pointer = match guard.pointer(offset) {
        Ok(pointer) => pointer,
        Err(status) => return status,
    };
    // SAFETY: checked destination bounds; caller supplies source range.
    unsafe { core::ptr::copy(source.cast::<u8>(), pointer, length) };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_unsupported_block_structure_allocator() {
        let mut out = core::ptr::dangling::<c_void>();
        unsafe {
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::dangling(),
                    core::ptr::null_mut(),
                    16,
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    16,
                    0,
                    &mut out,
                ),
                BAD_CUSTOM_SOURCE
            );
        }
        assert!(out.is_null());
    }

    #[test]
    fn accepts_standard_malloc_allocator_but_rejects_null_allocator() {
        use corefoundation::api::{StandardAllocator, standard_allocator};
        let mut out = core::ptr::null();
        unsafe {
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    standard_allocator(StandardAllocator::Malloc),
                    core::ptr::null_mut(),
                    4,
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    4,
                    0,
                    &mut out,
                ),
                0
            );
            assert_eq!(CMBlockBufferGetDataLength(out), 4);
            corefoundation::api::CFRelease(out);
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    standard_allocator(StandardAllocator::Null),
                    core::ptr::null_mut(),
                    4,
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    4,
                    0,
                    &mut out,
                ),
                BAD_CUSTOM_SOURCE
            );
            assert!(out.is_null());
        }
    }

    #[test]
    fn uses_standard_backing_allocator_for_allocated_and_contiguous_blocks() {
        use corefoundation::api::{StandardAllocator, standard_allocator};
        for allocator in [
            core::ptr::null(),
            standard_allocator(StandardAllocator::Malloc),
            standard_allocator(StandardAllocator::SystemDefault),
            standard_allocator(StandardAllocator::MallocZone),
        ] {
            let mut source = core::ptr::null();
            unsafe {
                assert_eq!(
                    CMBlockBufferCreateWithMemoryBlock(
                        core::ptr::null(),
                        core::ptr::null_mut(),
                        4,
                        allocator,
                        core::ptr::null(),
                        0,
                        4,
                        ASSURE_NOW,
                        &mut source
                    ),
                    0
                );
                assert_eq!(CMBlockBufferGetDataLength(source), 4);
                assert_eq!(
                    CMBlockBufferReplaceDataBytes(b"abcd".as_ptr().cast(), source, 0, 4),
                    0
                );
                let mut contiguous = core::ptr::null();
                assert_eq!(
                    CMBlockBufferCreateContiguous(
                        core::ptr::null(),
                        source,
                        allocator,
                        core::ptr::null(),
                        0,
                        4,
                        0,
                        &mut contiguous
                    ),
                    0
                );
                let mut copied = [0u8; 4];
                assert_eq!(
                    CMBlockBufferCopyDataBytes(contiguous, 0, 4, copied.as_mut_ptr().cast()),
                    0
                );
                assert_eq!(&copied, b"abcd");
                corefoundation::api::CFRelease(contiguous);
                corefoundation::api::CFRelease(source);
            }
        }
        let mut output = core::ptr::dangling::<c_void>();
        unsafe {
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    4,
                    standard_allocator(StandardAllocator::Null),
                    core::ptr::null(),
                    0,
                    4,
                    ASSURE_NOW,
                    &mut output
                ),
                BAD_CUSTOM_SOURCE
            );
            assert!(output.is_null());
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::dangling_mut(),
                    4,
                    standard_allocator(StandardAllocator::Malloc),
                    core::ptr::null(),
                    0,
                    4,
                    0,
                    &mut output
                ),
                BAD_CUSTOM_SOURCE
            );
            assert!(output.is_null());
        }
    }

    #[test]
    fn delayed_allocation_bytes_and_contiguous_copy() {
        let mut source = core::ptr::null();
        // SAFETY: out slot is writable.
        unsafe {
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    8,
                    core::ptr::null(),
                    core::ptr::null(),
                    2,
                    4,
                    0,
                    &mut source,
                ),
                0
            );
            assert_eq!(
                corefoundation::api::CFGetTypeID(source),
                CMBlockBufferGetTypeID()
            );
            assert_eq!(CMBlockBufferGetDataLength(source), 4);
            let input = *b"ABCD";
            assert_eq!(
                CMBlockBufferReplaceDataBytes(input.as_ptr().cast(), source, 0, 4),
                0
            );
            let mut ptr = core::ptr::null_mut();
            let mut available = 0;
            let mut total = 0;
            assert_eq!(
                CMBlockBufferGetDataPointer(source, 1, &mut available, &mut total, &mut ptr),
                0
            );
            assert_eq!((available, total), (3, 4));
            assert_eq!(core::slice::from_raw_parts(ptr, available), b"BCD");
            assert_eq!(
                CMBlockBufferGetDataPointer(
                    source,
                    4,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &mut ptr
                ),
                BAD_OFFSET
            );
            assert_eq!(
                CMBlockBufferCopyDataBytes(source, 3, 2, ptr.cast()),
                BAD_LENGTH
            );
            assert_eq!(
                CMBlockBufferReplaceDataBytes(b"a".as_ptr().cast(), source, 4, 1),
                BAD_LENGTH
            );
            let mut destination = [0; 4];
            assert_eq!(
                CMBlockBufferCopyDataBytes(source, 0, 4, destination.as_mut_ptr().cast()),
                0
            );
            assert_eq!(destination, input);

            let mut copy = core::ptr::null();
            assert_eq!(
                CMBlockBufferCreateContiguous(
                    core::ptr::null(),
                    source,
                    core::ptr::null(),
                    core::ptr::null(),
                    1,
                    0,
                    0,
                    &mut copy
                ),
                0
            );
            assert_eq!(CMBlockBufferGetDataLength(copy), 3);
            assert_eq!(
                CMBlockBufferReplaceDataBytes(b"xyz".as_ptr().cast(), copy, 0, 3),
                0
            );
            assert_eq!(core::slice::from_raw_parts(ptr, available), b"xyz");
            let mut bytes = [0; 3];
            assert_eq!(
                CMBlockBufferCopyDataBytes(copy, 0, 3, bytes.as_mut_ptr().cast()),
                0
            );
            assert_eq!(&bytes, b"xyz");

            corefoundation::api::CFRetain(copy);
            corefoundation::api::CFRelease(copy);
            assert_eq!(CMBlockBufferGetDataLength(copy), 3);
            corefoundation::api::CFRelease(copy);
            corefoundation::api::CFRelease(source);
        }
    }

    #[test]
    fn aliased_contiguous_views_share_bytes_and_keep_allocation_alive() {
        let mut source = core::ptr::null();
        unsafe {
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    8,
                    core::ptr::null(),
                    core::ptr::null(),
                    2,
                    4,
                    ASSURE_NOW,
                    &mut source
                ),
                0
            );
            assert_eq!(
                CMBlockBufferReplaceDataBytes(b"abcd".as_ptr().cast(), source, 0, 4),
                0
            );
            let mut full = core::ptr::null();
            assert_eq!(
                CMBlockBufferCreateContiguous(
                    core::ptr::null(),
                    source,
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    4,
                    0,
                    &mut full
                ),
                0
            );
            assert_ne!(full, source);
            let mut source_ptr = core::ptr::null_mut();
            let mut full_ptr = core::ptr::null_mut();
            assert_eq!(
                CMBlockBufferGetDataPointer(
                    source,
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &mut source_ptr
                ),
                0
            );
            assert_eq!(
                CMBlockBufferGetDataPointer(
                    full,
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &mut full_ptr
                ),
                0
            );
            assert_eq!(source_ptr, full_ptr);
            assert_eq!(
                CMBlockBufferReplaceDataBytes(b"xxxx".as_ptr().cast(), full, 0, 4),
                0
            );
            let mut readback = [0; 4];
            assert_eq!(
                CMBlockBufferCopyDataBytes(source, 0, 4, readback.as_mut_ptr().cast()),
                0
            );
            assert_eq!(&readback, b"xxxx");

            let mut slice = core::ptr::null();
            assert_eq!(
                CMBlockBufferCreateContiguous(
                    core::ptr::null(),
                    full,
                    core::ptr::null(),
                    core::ptr::null(),
                    1,
                    2,
                    0,
                    &mut slice
                ),
                0
            );
            let mut slice_ptr = core::ptr::null_mut();
            assert_eq!(
                CMBlockBufferGetDataPointer(
                    slice,
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &mut slice_ptr
                ),
                0
            );
            assert_eq!(slice_ptr, source_ptr.add(1));
            corefoundation::api::CFRelease(source);
            corefoundation::api::CFRelease(full);
            assert_eq!(
                CMBlockBufferReplaceDataBytes(b"yz".as_ptr().cast(), slice, 0, 2),
                0
            );
            let mut readback = [0; 2];
            assert_eq!(
                CMBlockBufferCopyDataBytes(slice, 0, 2, readback.as_mut_ptr().cast()),
                0
            );
            assert_eq!(&readback, b"yz");
            corefoundation::api::CFRelease(slice);
        }
    }

    #[test]
    fn zero_length_operations_and_out_of_range_classification_match_native() {
        let mut block = core::ptr::dangling::<c_void>();
        unsafe {
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    4,
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    0,
                    0,
                    &mut block
                ),
                BAD_LENGTH
            );
            assert!(block.is_null());
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    4,
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    4,
                    ASSURE_NOW,
                    &mut block
                ),
                0
            );
            let mut bytes = [0u8; 4];
            assert_eq!(
                CMBlockBufferCopyDataBytes(block, 0, 0, bytes.as_mut_ptr().cast()),
                BAD_LENGTH
            );
            assert_eq!(
                CMBlockBufferReplaceDataBytes(b"a".as_ptr().cast(), block, 0, 0),
                BAD_LENGTH
            );
            assert_eq!(
                CMBlockBufferCopyDataBytes(block, 3, 2, bytes.as_mut_ptr().cast()),
                BAD_LENGTH
            );
            assert_eq!(
                CMBlockBufferReplaceDataBytes(b"ab".as_ptr().cast(), block, 3, 2),
                BAD_LENGTH
            );
            assert_eq!(
                CMBlockBufferCopyDataBytes(block, 5, 1, bytes.as_mut_ptr().cast()),
                BAD_OFFSET
            );
            corefoundation::api::CFRelease(block);
        }
    }

    #[test]
    fn rejects_invalid_lengths_offsets_and_unsafe_ownership() {
        let mut out = core::ptr::dangling::<c_void>();
        // SAFETY: all failures are rejected before invalid pointers are used.
        unsafe {
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    0,
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    0,
                    0,
                    &mut out
                ),
                BAD_LENGTH
            );
            assert!(out.is_null());
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    4,
                    core::ptr::null(),
                    core::ptr::null(),
                    3,
                    2,
                    0,
                    &mut out
                ),
                BAD_LENGTH
            );
            let memory = [0u8; 4];
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    memory.as_ptr().cast_mut().cast(),
                    4,
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    4,
                    0,
                    &mut out
                ),
                BAD_CUSTOM_SOURCE
            );
            assert_eq!(CMBlockBufferGetDataLength(core::ptr::null()), 0);
            assert_eq!(
                CMBlockBufferGetDataPointer(
                    core::ptr::null(),
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut()
                ),
                BAD_POINTER
            );
        }
    }

    #[test]
    fn refuses_unbridged_custom_callbacks_without_dereferencing_them() {
        let fake_source = core::ptr::dangling::<CustomBlockSource>();
        let mut buffer = core::ptr::dangling::<c_void>();
        // SAFETY: custom_source is never dereferenced on the rejection path.
        unsafe {
            assert_eq!(
                CMBlockBufferCreateWithMemoryBlock(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    16,
                    core::ptr::null(),
                    fake_source,
                    0,
                    16,
                    0,
                    &mut buffer
                ),
                BAD_CUSTOM_SOURCE
            );
            assert!(buffer.is_null());
        }
    }
}

exports::exports! {
    functions {
        "CMBlockBufferGetTypeID" => CMBlockBufferGetTypeID,
        "CMBlockBufferCreateWithMemoryBlock" => CMBlockBufferCreateWithMemoryBlock,
        "CMBlockBufferCreateContiguous" => CMBlockBufferCreateContiguous,
        "CMBlockBufferGetDataPointer" => CMBlockBufferGetDataPointer,
        "CMBlockBufferGetDataLength" => CMBlockBufferGetDataLength,
        "CMBlockBufferCopyDataBytes" => CMBlockBufferCopyDataBytes,
        "CMBlockBufferReplaceDataBytes" => CMBlockBufferReplaceDataBytes,
    }
}
