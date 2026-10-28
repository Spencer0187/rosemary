//! CoreMedia time layout is exactly the 24-byte CMTime aggregate.

#![allow(non_snake_case)]

use core::cmp::Ordering;

pub const VALID: u32 = 1;
pub const POSITIVE_INFINITY: u32 = 4;
pub const NEGATIVE_INFINITY: u32 = 8;
pub const INDEFINITE: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CMTime {
    pub value: i64,
    pub timescale: i32,
    pub flags: u32,
    pub epoch: i64,
}

impl CMTime {
    pub const INVALID: Self = Self {
        value: 0,
        timescale: 0,
        flags: 0,
        epoch: 0,
    };

    pub fn is_numeric(self) -> bool {
        self.flags & VALID != 0
            && self.flags & (POSITIVE_INFINITY | NEGATIVE_INFINITY | INDEFINITE) == 0
            && self.timescale > 0
    }
}

/// Compare finite numeric times exactly without converting to f64.
pub(crate) fn compare_numeric(a: CMTime, b: CMTime) -> Option<Ordering> {
    (a.is_numeric() && b.is_numeric()).then(|| {
        a.epoch.cmp(&b.epoch).then_with(|| {
            (i128::from(a.value) * i128::from(b.timescale))
                .cmp(&(i128::from(b.value) * i128::from(a.timescale)))
        })
    })
}

/// Canonical nonnumeric times have zero value, scale and epoch.
fn special(flags: u32) -> CMTime {
    CMTime {
        flags: VALID | flags,
        ..CMTime::INVALID
    }
}

/// CoreMedia treats infinity and indefinite as meaningful valid durations.
/// Opposing infinities and any invalid time still yield kCMTimeInvalid.
pub(crate) fn add_time(a: CMTime, b: CMTime) -> CMTime {
    if (a.flags & VALID) == 0 || (b.flags & VALID) == 0 {
        return CMTime::INVALID;
    }
    let flags = a.flags | b.flags;
    if flags & POSITIVE_INFINITY != 0 && flags & NEGATIVE_INFINITY != 0 {
        return CMTime::INVALID;
    }
    if flags & POSITIVE_INFINITY != 0 {
        return special(POSITIVE_INFINITY);
    }
    if flags & NEGATIVE_INFINITY != 0 {
        return special(NEGATIVE_INFINITY);
    }
    if flags & INDEFINITE != 0 {
        return special(INDEFINITE);
    }
    if !a.is_numeric() || !b.is_numeric() || a.epoch != b.epoch {
        return CMTime::INVALID;
    }
    let gcd = gcd(u64::from(a.timescale as u32), u64::from(b.timescale as u32));
    let scale = (i64::from(a.timescale) / gcd as i64) * i64::from(b.timescale);
    let Ok(timescale) = i32::try_from(scale) else {
        return CMTime::INVALID;
    };
    let lhs = i128::from(a.value) * i128::from(scale / i64::from(a.timescale));
    let rhs = i128::from(b.value) * i128::from(scale / i64::from(b.timescale));
    let Ok(value) = i64::try_from(lhs + rhs) else {
        return CMTime::INVALID;
    };
    CMTime {
        value,
        timescale,
        flags: VALID | ((a.flags | b.flags) & 2),
        epoch: a.epoch,
    }
}

pub(crate) fn multiply_time(time: CMTime, count: isize) -> CMTime {
    if time.flags & VALID == 0 || count < 0 {
        return CMTime::INVALID;
    }
    if time.flags & POSITIVE_INFINITY != 0 {
        return special(POSITIVE_INFINITY);
    }
    if time.flags & NEGATIVE_INFINITY != 0 {
        return special(NEGATIVE_INFINITY);
    }
    if time.flags & INDEFINITE != 0 {
        return special(INDEFINITE);
    }
    if !time.is_numeric() {
        return CMTime::INVALID;
    }
    let Ok(value) = i64::try_from(i128::from(time.value) * count as i128) else {
        return CMTime::INVALID;
    };
    CMTime { value, ..time }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

pub extern "C" fn CMTimeMake(value: i64, timescale: i32) -> CMTime {
    if timescale <= 0 {
        return CMTime::INVALID;
    }
    CMTime {
        value,
        timescale,
        flags: VALID,
        epoch: 0,
    }
}

pub extern "C" fn CMTimeGetSeconds(time: CMTime) -> f64 {
    if time.flags & VALID == 0 {
        return f64::NAN;
    }
    if time.flags & POSITIVE_INFINITY != 0 {
        return f64::INFINITY;
    }
    if time.flags & NEGATIVE_INFINITY != 0 {
        return f64::NEG_INFINITY;
    }
    if time.flags & INDEFINITE != 0 || time.timescale <= 0 {
        return f64::NAN;
    }
    time.value as f64 / f64::from(time.timescale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_time_math_preserves_order_epoch_and_rejects_overflow() {
        use core::cmp::Ordering;
        assert_eq!(
            compare_numeric(CMTimeMake(1, 3), CMTimeMake(2, 5)),
            Some(Ordering::Less)
        );
        assert_eq!(
            add_time(CMTimeMake(1, 3), CMTimeMake(1, 6)),
            CMTimeMake(3, 6)
        );
        assert_eq!(multiply_time(CMTimeMake(2, 3), 3), CMTimeMake(6, 3));
        assert_eq!(multiply_time(CMTimeMake(i64::MAX, 1), 2), CMTime::INVALID);
        assert_eq!(
            add_time(CMTimeMake(i64::MAX, 1), CMTimeMake(1, 1)),
            CMTime::INVALID
        );
        assert_eq!(
            add_time(CMTimeMake(1, i32::MAX), CMTimeMake(1, i32::MAX - 1)),
            CMTime::INVALID
        );
        let different_epoch = CMTime {
            epoch: 1,
            ..CMTimeMake(1, 2)
        };
        assert_eq!(
            compare_numeric(different_epoch, CMTimeMake(100, 1)),
            Some(Ordering::Greater)
        );
        assert_eq!(add_time(different_epoch, CMTimeMake(1, 2)), CMTime::INVALID);
        assert!(
            CMTimeGetSeconds(CMTime {
                timescale: -1,
                ..CMTimeMake(1, 1)
            })
            .is_nan()
        );
    }

    #[test]
    fn special_durations_follow_native_video_timing_rules() {
        let indefinite = special(INDEFINITE);
        let positive = special(POSITIVE_INFINITY);
        let negative = special(NEGATIVE_INFINITY);
        let numeric = CMTimeMake(1, 30);
        for (time, expected) in [
            (indefinite, indefinite),
            (positive, positive),
            (negative, negative),
            (CMTime::INVALID, CMTime::INVALID),
        ] {
            assert_eq!(multiply_time(time, 1), expected);
            assert_eq!(multiply_time(time, 2), expected);
        }
        for (left, right, expected) in [
            (indefinite, numeric, indefinite),
            (numeric, indefinite, indefinite),
            (positive, indefinite, positive),
            (negative, indefinite, negative),
            (positive, negative, CMTime::INVALID),
            (negative, positive, CMTime::INVALID),
            (CMTime::INVALID, positive, CMTime::INVALID),
            (numeric, CMTime::INVALID, CMTime::INVALID),
            (positive, numeric, positive),
            (numeric, negative, negative),
        ] {
            assert_eq!(add_time(left, right), expected);
        }
    }

    #[test]
    fn time_layout_flags_and_seconds() {
        assert_eq!(size_of::<CMTime>(), 24);
        assert_eq!(CMTimeGetSeconds(CMTimeMake(300, 600)), 0.5);
        assert!(CMTimeGetSeconds(CMTimeMake(42, 0)).is_nan());
        assert!(CMTimeGetSeconds(CMTime::INVALID).is_nan());
        assert_eq!(
            CMTimeGetSeconds(CMTime {
                flags: VALID | POSITIVE_INFINITY,
                ..CMTime::INVALID
            }),
            f64::INFINITY
        );
    }
}

exports::exports! {
    functions {
        "CMTimeMake" => CMTimeMake,
        "CMTimeGetSeconds" => CMTimeGetSeconds,
    }
}
