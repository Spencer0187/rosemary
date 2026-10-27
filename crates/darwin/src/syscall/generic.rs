use libc::ENOSYS;

use crate::{
    errno, mach,
    syscall::numbers::{self, SYSCALL_CLASS_MACH, SYSCALL_CLASS_UNIX},
};

/// Calls a shim that reports failure C-style, `-1` with the reason in `errno`,
/// on guest registers narrowed to its parameter types, and answers as the raw
/// syscall would. `errno` itself is left alone by the caller's `errno::Held`.
macro_rules! unix_call {
    (unsafe $($function:ident)::+ ($($arg:expr),* $(,)?)) => {
        match unsafe { $($function)::+($($arg as _),*) } as i64 {
            -1 => Err(errno::guest_errno()),
            value => Ok(value as u64),
        }
    };
}

fn handle_unix_syscall(number: u32, args: [u64; 6]) -> Result<u64, i32> {
    match number {
        numbers::bsd::SYS_openat => {
            unix_call!(unsafe crate::stat::openat(args[0], args[1], args[2], args[3]))
        }
        numbers::bsd::SYS_open => {
            unix_call!(unsafe crate::stat::open(args[0], args[1], args[2]))
        }
        numbers::bsd::SYS_getpid => Ok(crate::process::getpid() as _),
        numbers::bsd::SYS_getppid => Ok(crate::process::getppid() as _),
        _ => Err(errno::from_host(ENOSYS)),
    }
}

fn handle_mach_syscall(number: u32, args: [u64; 6]) -> Result<u64, i32> {
    match number {
        numbers::mach::task_self_trap => Ok(mach::task::mach_task_self().into()),
        numbers::mach::_kernelrpc_mach_vm_allocate_trap => Ok(unsafe {
            mach::vm::vm_allocate(args[0] as _, args[1] as _, args[2], args[3] as _) as _
        }),
        numbers::mach::_kernelrpc_mach_vm_protect_trap => Ok(unsafe {
            mach::vm::vm_protect(args[0] as _, args[1], args[2], args[3] as _, args[4] as _) as _
        }),
        numbers::mach::_kernelrpc_mach_vm_deallocate_trap => {
            Ok(unsafe { mach::vm::vm_deallocate(args[0] as _, args[1], args[2]) as _ })
        }
        numbers::mach::thread_self_trap => Ok(mach::task::mach_thread_self().into()),
        numbers::mach::host_self_trap => Ok(mach::task::mach_host_self().into()),
        // FIXME (ACCCURAY): task_for_pid is the privileged variant of task_name_for_pid, they are not equal
        numbers::mach::task_name_for_pid | numbers::mach::task_for_pid => {
            Ok(unsafe { mach::tasks::task_for_pid(args[0] as _, args[1] as _, args[2] as _) } as _)
        }
        _ => Err(errno::from_host(ENOSYS)),
    }
}

/// A Darwin syscall the guest made directly: `number` as it was in `%rax`, class
/// bits included, and the six argument registers. Answers the value for `%rax`,
/// or a Darwin `errno`.
///
/// Must be async-signal-safe: it runs in the `SIGSYS` handler.
pub(crate) fn generic_syscall_entry(class: u32, number: u32, args: [u64; 6]) -> Result<u64, i32> {
    match class {
        SYSCALL_CLASS_UNIX => handle_unix_syscall(number, args),
        SYSCALL_CLASS_MACH => handle_mach_syscall(number, args),
        _ => Err(errno::from_host(ENOSYS)),
    }
}
