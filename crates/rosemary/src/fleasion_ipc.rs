//! Opt-in, inherited local stream for read-only host memory inspection.
//!
//! There is no listener or discovery mechanism. The caller must own the socket
//! passed at exec, and only writable private mappings can be inspected.

use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::net::UnixStream;
use std::thread;

const FD_ENV: &str = "FLEASION_ROSEMARY_IPC_FD";
const VM_REGION_BASIC_INFO_64: i32 = 9;
const VM_PROT_READ: i32 = 1;
const VM_PROT_WRITE: i32 = 2;
const MAX_READ: usize = 4 * 1024 * 1024 + 64;
const MAX_REGIONS: usize = 16_384;
const HEADER_SIZE: usize = 13;

#[repr(C, packed(4))]
#[derive(Clone, Copy, Default)]
struct BasicInfo64 {
    protection: i32,
    max_protection: i32,
    inheritance: u32,
    shared: i32,
    reserved: i32,
    offset: u64,
    behavior: i32,
    user_wired_count: u16,
}

const _: () = assert!(size_of::<BasicInfo64>() == 36);

unsafe extern "C" {
    static mach_task_self_: u32;
    fn mach_vm_region(
        task: u32,
        address: *mut u64,
        size: *mut u64,
        flavor: i32,
        info: *mut i32,
        info_count: *mut u32,
        object_name: *mut u32,
    ) -> i32;
    fn mach_vm_read_overwrite(
        task: u32,
        address: u64,
        size: u64,
        data: u64,
        outsize: *mut u64,
    ) -> i32;
    fn mach_port_deallocate(task: u32, name: u32) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Region {
    start: u64,
    size: u64,
}

fn is_private_writable(info: &BasicInfo64) -> bool {
    info.protection & (VM_PROT_READ | VM_PROT_WRITE) == VM_PROT_READ | VM_PROT_WRITE
        && info.shared == 0
}

fn contains(region: Region, address: u64, length: usize) -> bool {
    length > 0
        && address >= region.start
        && address.checked_add(length as u64).is_some_and(|end| {
            region
                .start
                .checked_add(region.size)
                .is_some_and(|limit| end <= limit)
        })
}

fn next_region(address: u64) -> Option<(Region, BasicInfo64)> {
    let mut start = address;
    let mut size = 0u64;
    let mut info = BasicInfo64::default();
    let mut count = (size_of::<BasicInfo64>() / size_of::<i32>()) as u32;
    let mut object_name = 0u32;
    // SAFETY: All output storage has the Mach SDK's VM_REGION_BASIC_INFO_64 layout.
    let task = unsafe { mach_task_self_ };
    let result = unsafe {
        mach_vm_region(
            task,
            &mut start,
            &mut size,
            VM_REGION_BASIC_INFO_64,
            (&raw mut info).cast::<i32>(),
            &mut count,
            &mut object_name,
        )
    };
    if object_name != 0 {
        // SAFETY: Mach returned a send right that we own.
        unsafe { mach_port_deallocate(task, object_name) };
    }
    if result != 0 || count as usize != size_of::<BasicInfo64>() / size_of::<i32>() {
        return None;
    }
    Some((Region { start, size }, info))
}

fn list_regions() -> Option<Vec<Region>> {
    let mut current = 0;
    let mut output = Vec::new();
    for _ in 0..MAX_REGIONS {
        let Some((region, info)) = next_region(current) else {
            return Some(output);
        };
        if is_private_writable(&info) {
            output.push(region);
        }
        current = region.start.checked_add(region.size)?;
        if region.size == 0 {
            return None;
        }
    }
    None
}

fn read_region(address: u64, length: usize) -> Option<Vec<u8>> {
    if length == 0 || length > MAX_READ {
        return None;
    }
    let (region, info) = next_region(address)?;
    if !is_private_writable(&info) || !contains(region, address, length) {
        return None;
    }
    let mut data = vec![0u8; length];
    let mut read = 0u64;
    // SAFETY: The source is an allocated mapping in this task and the destination
    // buffer is writable. Mach checks access again in case the map changed.
    let result = unsafe {
        mach_vm_read_overwrite(
            mach_task_self_,
            address,
            length as u64,
            data.as_mut_ptr() as u64,
            &mut read,
        )
    };
    (result == 0 && read == length as u64).then_some(data)
}

fn reply(stream: &mut UnixStream, payload: Option<&[u8]>) -> std::io::Result<()> {
    let Some(payload) = payload else {
        return stream.write_all(&[1, 0, 0, 0, 0]);
    };
    stream.write_all(&[0])?;
    stream.write_all(&(payload.len() as u32).to_le_bytes())?;
    stream.write_all(payload)
}

fn serve(mut stream: UnixStream) -> std::io::Result<()> {
    loop {
        let mut header = [0u8; HEADER_SIZE];
        match stream.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }
        let address = u64::from_le_bytes(header[1..9].try_into().unwrap());
        let length = u32::from_le_bytes(header[9..13].try_into().unwrap()) as usize;
        match header[0] {
            b'M' if address == 0 && length == 0 => {
                let regions = list_regions();
                let payload = regions.map(|regions| {
                    let mut bytes = Vec::with_capacity(regions.len() * 16);
                    for region in regions {
                        bytes.extend_from_slice(&region.start.to_le_bytes());
                        bytes.extend_from_slice(&region.size.to_le_bytes());
                    }
                    bytes
                });
                reply(&mut stream, payload.as_deref())?;
            }
            b'R' => {
                let payload = read_region(address, length);
                reply(&mut stream, payload.as_deref())?;
            }
            _ => reply(&mut stream, None)?,
        }
    }
}

pub fn start() {
    let Some(fd_string) = std::env::var_os(FD_ENV) else {
        return;
    };
    // The FD is the capability; CLOEXEC below prevents child-process inheritance.
    let Some(fd) = fd_string
        .to_str()
        .and_then(|value| value.parse::<i32>().ok())
    else {
        eprintln!("rosemary: invalid Fleasion IPC descriptor");
        return;
    };
    if fd < 3 {
        eprintln!("rosemary: refusing Fleasion IPC over a standard descriptor");
        return;
    }
    // SAFETY: The descriptor is an explicit, owned inherited capability.
    let stream = unsafe { UnixStream::from_raw_fd(fd) };
    if stream.peer_addr().is_err() {
        eprintln!("rosemary: Fleasion IPC is not a connected Unix socket");
        return;
    }
    // Prevent the guest's spawned processes from inheriting this channel.
    // SAFETY: fcntl operates on the descriptor owned by `stream`.
    let flags = unsafe { libc::fcntl(stream.as_raw_fd(), libc::F_GETFD) };
    if flags < 0
        || unsafe { libc::fcntl(stream.as_raw_fd(), libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0
    {
        eprintln!("rosemary: could not secure Fleasion IPC descriptor");
        return;
    }
    if let Err(error) = thread::Builder::new()
        .name("rosemary-fleasion-ipc".to_string())
        .spawn(move || {
            if let Err(error) = serve(stream) {
                eprintln!("rosemary: Fleasion IPC disconnected: {error}");
            }
        })
    {
        eprintln!("rosemary: could not start Fleasion IPC: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::{BasicInfo64, Region, contains, is_private_writable};

    #[test]
    fn only_private_writable_regions_are_inspectable() {
        let mut info = BasicInfo64 {
            protection: 3,
            ..BasicInfo64::default()
        };
        assert!(is_private_writable(&info));
        info.shared = 1;
        assert!(!is_private_writable(&info));
        info.shared = 0;
        info.protection = 1;
        assert!(!is_private_writable(&info));
    }

    #[test]
    fn slice_must_be_wholly_contained_without_overflow() {
        let region = Region {
            start: 100,
            size: 20,
        };
        assert!(contains(region, 110, 10));
        assert!(!contains(region, 111, 10));
        assert!(!contains(region, 99, 1));
        assert!(!contains(region, u64::MAX, 2));
        assert!(!contains(region, 110, 0));
    }
}
