//! Interposition of the raw syscalls guest code makes on the host CPU.
//!
//! TODO (ALL): Make it so all relocations made by Rosemary that point to
//! non-Darwin implementations, AKA synthetic dylibs, point to trampolines
//! that set the appropriate SUD state, like how WINE has this exact mechanism.
//! WINE goes as follows:
//! 
//! Unix sided runtime -> Switch to guest profile -> Call PE runtime entry point
//! -> Call guest code -> Call reimplementations (PE side) -> Make a host/Unix call
//! -> Restores host profile -> Trampoline restores guest profile -> Returns to PE side

mod generic;
mod interpose;
pub mod numbers;
pub mod scope;

use std::{ffi::c_int, io, sync::LazyLock};

pub use scope::{ScopedGuestSyscallContext, ScopedHostSyscallContext};

pub fn setup_syscall_handler() -> io::Result<()> {
    static SYSCALL_INTERPOSE: LazyLock<bool> = LazyLock::new(|| {
        if let Ok(val) = std::env::var("ROSEMARY_SYSCALL_INTERPOSE") {
            val == "1"
        } else {
            false
        }
    });
    if *SYSCALL_INTERPOSE {
        interpose::install_syscall_handler()
    }
    else {
        Err(io::Error::new(io::ErrorKind::Unsupported, "syscall interposition was not enabled"))
    }
}

pub unsafe fn clock_gettime(clock: libc::clockid_t, ts: *mut libc::timespec) -> c_int {
    let _host = ScopedHostSyscallContext::new();
    // SAFETY: the caller's guarantee.
    unsafe { libc::clock_gettime(clock, ts) }
}
