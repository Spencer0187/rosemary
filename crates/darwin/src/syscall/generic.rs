use libc::ENOSYS;

use crate::errno;

/// A Darwin syscall the guest made directly: `number` as it was in `%rax`, class
/// bits included, and the six argument registers. Answers the value for `%rax`,
/// or a Darwin `errno`.
///
/// Must be async-signal-safe: it runs in the `SIGSYS` handler.
pub(crate) fn generic_syscall_entry(_number: u64, _args: [u64; 6]) -> Result<u64, i32> {
    Err(errno::from_host(ENOSYS))
}
