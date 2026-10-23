//! H.264 software decoding without a system FFmpeg installation.
//!
//! The public boundary is native Rust buffers and timestamps, never guest
//! pointers or CoreMedia/VideoToolbox objects. Decoder output is copied before
//! advancing the native decoder, so frame storage cannot outlive its owner.

use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    I420,
    Nv12,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Plane {
    pub width_bytes: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub format: PixelFormat,
    pub pts: Option<i64>,
    pub planes: Vec<Plane>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    InvalidLengthSize,
    InvalidParameterSet,
    InvalidPacket,
    BackendUnavailable,
    Codec(i32),
    UnsupportedPixelFormat,
    InvalidImageLayout,
    Drained,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

#[repr(C)]
#[derive(Default)]
struct FrameView {
    width: i32,
    height: i32,
    format: i32,
    pts: i64,
    planes: [*const u8; 3],
    strides: [isize; 3],
}

unsafe extern "C" {
    fn rosemary_h264_new() -> *mut c_void;
    fn rosemary_h264_free(handle: *mut c_void);
    fn rosemary_h264_send(
        handle: *mut c_void,
        data: *const u8,
        length: usize,
        pts: i64,
        dts: i64,
    ) -> i32;
    fn rosemary_h264_receive(handle: *mut c_void, out: *mut FrameView) -> i32;
}

/// AVCC sample NAL widths in bytes. Apple supports 1, 2, and 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalLength {
    One,
    Two,
    Four,
}

impl TryFrom<u8> for NalLength {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Error> {
        match value {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            4 => Ok(Self::Four),
            _ => Err(Error::InvalidLengthSize),
        }
    }
}

impl NalLength {
    fn width(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
        }
    }
}

/// Converts a single AVCC access unit to Annex B, validating every length field.
/// An empty or incomplete sample must never enter the native codec.
pub fn avcc_to_annex_b(sample: &[u8], length: NalLength) -> Result<Vec<u8>, Error> {
    if sample.is_empty() || sample.len() > 64 * 1024 * 1024 {
        return Err(Error::InvalidPacket);
    }
    let mut out = Vec::with_capacity(sample.len().saturating_add(16));
    let mut remaining = sample;
    while !remaining.is_empty() {
        let width = length.width();
        let Some(prefix) = remaining.get(..width) else {
            return Err(Error::InvalidPacket);
        };
        let nal_size = prefix
            .iter()
            .fold(0usize, |value, &byte| (value << 8) | usize::from(byte));
        remaining = &remaining[width..];
        let Some(nal) = remaining.get(..nal_size).filter(|nal| !nal.is_empty()) else {
            return Err(Error::InvalidPacket);
        };
        if nal[0] & 0x1f == 0 || nal[0] & 0x80 != 0 {
            return Err(Error::InvalidPacket);
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
        remaining = &remaining[nal_size..];
    }
    Ok(out)
}

/// Stateful H.264 decoder. One owner at a time; guest frameworks decide which
/// thread submits samples and how/when decoded frames are delivered.
pub struct Decoder {
    handle: NonNull<c_void>,
    length: NalLength,
    needs_headers: bool,
    headers: Vec<u8>,
    finished: bool,
}

impl Decoder {
    /// Parameter sets are full H.264 NAL units without Annex B start codes.
    /// A new session supplies its SPS/PPS headers before the first coded access unit.
    pub fn new(parameter_sets: &[&[u8]], length: NalLength) -> Result<Self, Error> {
        if !parameter_sets
            .iter()
            .any(|n| n.first().is_some_and(|b| b & 0x1f == 7))
            || !parameter_sets
                .iter()
                .any(|n| n.first().is_some_and(|b| b & 0x1f == 8))
        {
            return Err(Error::InvalidParameterSet);
        }
        if parameter_sets.len() > 64 || parameter_sets.iter().any(|set| set.len() > 1024 * 1024) {
            return Err(Error::InvalidParameterSet);
        }
        let mut headers = Vec::new();
        for set in parameter_sets {
            if set.is_empty() || set[0] & 0x80 != 0 || !matches!(set[0] & 0x1f, 7 | 8 | 13) {
                return Err(Error::InvalidParameterSet);
            }
            headers.extend_from_slice(&[0, 0, 0, 1]);
            headers.extend_from_slice(set);
        }
        // SAFETY: the C constructor returns either a live decoder or null.
        let handle =
            NonNull::new(unsafe { rosemary_h264_new() }).ok_or(Error::BackendUnavailable)?;
        Ok(Self {
            handle,
            length,
            needs_headers: true,
            headers,
            finished: false,
        })
    }

    /// Submit one length-prefixed sample and return any frames now available.
    /// A sample may produce zero or several frames because H.264 reorders frames.
    pub fn submit_avcc(
        &mut self,
        sample: &[u8],
        pts: Option<i64>,
        dts: Option<i64>,
    ) -> Result<Vec<Frame>, Error> {
        let data = avcc_to_annex_b(sample, self.length)?;
        self.submit_annex_b(&data, pts, dts)
    }

    /// Submit one Annex B access unit, for clients that already use start codes.
    pub fn submit_annex_b(
        &mut self,
        sample: &[u8],
        pts: Option<i64>,
        dts: Option<i64>,
    ) -> Result<Vec<Frame>, Error> {
        if self.finished {
            return Err(Error::Drained);
        }
        if sample.is_empty() || sample.len() > 64 * 1024 * 1024 {
            return Err(Error::InvalidPacket);
        }
        let owned;
        let packet = if self.needs_headers {
            owned = [self.headers.as_slice(), sample].concat();
            &owned[..]
        } else {
            sample
        };
        let mut ready = Vec::new();
        let mut status = self.send(Some(packet), pts, dts);
        if status == 1 {
            ready.extend(self.receive_ready()?);
            status = self.send(Some(packet), pts, dts);
        }
        if status != 0 {
            return Err(Error::Codec(status));
        }
        self.needs_headers = false;
        ready.extend(self.receive_ready()?);
        Ok(ready)
    }

    /// Finish delayed/reordered frames. A drained decoder must be recreated.
    pub fn finish(&mut self) -> Result<Vec<Frame>, Error> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut result = self.send(None, None, None);
        let mut frames = Vec::new();
        if result == 1 {
            frames.extend(self.receive_ready()?);
            result = self.send(None, None, None);
        }
        if result != 0 {
            return Err(Error::Codec(result));
        }
        self.finished = true;
        frames.extend(self.receive_ready()?);
        Ok(frames)
    }

    fn send(&self, bytes: Option<&[u8]>, pts: Option<i64>, dts: Option<i64>) -> i32 {
        // SAFETY: owned handle remains valid until Drop and pointer/length are
        // only borrowed synchronously by the C entry point.
        unsafe {
            rosemary_h264_send(
                self.handle.as_ptr(),
                bytes.map_or(std::ptr::null(), |data| data.as_ptr()),
                bytes.map_or(0, <[u8]>::len),
                pts.unwrap_or(i64::MIN),
                dts.unwrap_or(i64::MIN),
            )
        }
    }

    fn receive_ready(&mut self) -> Result<Vec<Frame>, Error> {
        let mut frames = Vec::new();
        loop {
            let mut view = FrameView::default();
            // SAFETY: decoder handle and local output remain valid for the call.
            match unsafe { rosemary_h264_receive(self.handle.as_ptr(), &mut view) } {
                0 => frames.push(copy_frame(&view)?),
                1 | 2 => break,
                negative => return Err(Error::Codec(negative)),
            }
        }
        Ok(frames)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: unique handle was allocated by rosemary_h264_new.
        unsafe { rosemary_h264_free(self.handle.as_ptr()) };
    }
}

fn copy_frame(view: &FrameView) -> Result<Frame, Error> {
    let width = usize::try_from(view.width).map_err(|_| Error::InvalidImageLayout)?;
    let height = usize::try_from(view.height).map_err(|_| Error::InvalidImageLayout)?;
    if width == 0 || height == 0 || width > 32768 || height > 32768 {
        return Err(Error::InvalidImageLayout);
    }
    let format = match view.format {
        0 => PixelFormat::I420,
        1 => PixelFormat::Nv12,
        _ => return Err(Error::UnsupportedPixelFormat),
    };
    let uv_width = width.div_ceil(2);
    let uv_height = height.div_ceil(2);
    let dimensions = match format {
        PixelFormat::I420 => vec![
            (width, height),
            (uv_width, uv_height),
            (uv_width, uv_height),
        ],
        PixelFormat::Nv12 => vec![(width, height), (uv_width * 2, uv_height)],
    };
    let mut planes = Vec::with_capacity(dimensions.len());
    for (i, (row_width, rows)) in dimensions.into_iter().enumerate() {
        let stride = view.strides[i];
        let size = row_width
            .checked_mul(rows)
            .ok_or(Error::InvalidImageLayout)?;
        let abs_stride = stride.checked_abs().ok_or(Error::InvalidImageLayout)?;
        if size > 512 * 1024 * 1024
            || abs_stride < isize::try_from(row_width).map_err(|_| Error::InvalidImageLayout)?
            || abs_stride > 128 * 1024 * 1024
            || view.planes[i].is_null()
        {
            return Err(Error::InvalidImageLayout);
        }
        let mut data = Vec::with_capacity(size);
        for row in 0..rows {
            let row_offset = isize::try_from(row)
                .ok()
                .and_then(|r| r.checked_mul(stride))
                .ok_or(Error::InvalidImageLayout)?;
            // SAFETY: FFmpeg guarantees frame->data points to at least `rows`
            // accessible scanlines with the provided stride, including negative
            // strides. No decoder operation happens until the copy completes.
            let scan =
                unsafe { std::slice::from_raw_parts(view.planes[i].offset(row_offset), row_width) };
            data.extend_from_slice(scan);
        }
        planes.push(Plane {
            width_bytes: row_width,
            height: rows,
            data,
        });
    }
    Ok(Frame {
        width,
        height,
        format,
        pts: (view.pts != i64::MIN).then_some(view.pts),
        planes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_nal_lengths_are_rejected() {
        for size in [0, 3, 5, 255] {
            assert_eq!(NalLength::try_from(size), Err(Error::InvalidLengthSize));
        }
    }

    #[test]
    fn converts_multiple_length_prefixed_nals() {
        let sample = [0, 3, 0x67, 0x42, 0x80, 0, 2, 0x68, 0x01];
        assert_eq!(
            avcc_to_annex_b(&sample, NalLength::Two),
            Ok(vec![0, 0, 0, 1, 0x67, 0x42, 0x80, 0, 0, 0, 1, 0x68, 0x01])
        );
    }

    #[test]
    fn malformed_avcc_does_not_cross_ffi() {
        for data in [
            &[][..],
            &[0, 3, 0x65][..],
            &[0, 0][..],
            &[0, 1, 0x80][..],
            &[0, 2, 0x65, 0x01, 0][..],
        ] {
            assert_eq!(
                avcc_to_annex_b(data, NalLength::Two),
                Err(Error::InvalidPacket)
            );
        }
    }

    fn nals_from_fixture(bitstream: &[u8]) -> Vec<&[u8]> {
        let mut nals = Vec::new();
        let mut start = None;
        let mut cursor = 0;
        while cursor + 3 <= bitstream.len() {
            let separator = if bitstream[cursor..].starts_with(&[0, 0, 0, 1]) {
                4
            } else if bitstream[cursor..].starts_with(&[0, 0, 1]) {
                3
            } else {
                cursor += 1;
                continue;
            };
            if let Some(from) = start {
                nals.push(&bitstream[from..cursor]);
            }
            start = Some(cursor + separator);
            cursor += separator;
        }
        if let Some(from) = start {
            nals.push(&bitstream[from..]);
        }
        nals
    }

    fn decoder_for_fixture(bitstream: &[u8]) -> Decoder {
        let nals = nals_from_fixture(bitstream);
        let parameter_sets: Vec<&[u8]> = nals
            .into_iter()
            .filter(|nal| {
                nal.first()
                    .is_some_and(|header| matches!(header & 0x1f, 7 | 8))
            })
            .collect();
        assert_eq!(parameter_sets.len(), 2);
        Decoder::new(&parameter_sets, NalLength::Four).unwrap()
    }

    #[test]
    fn decodes_pinned_one_frame_fixture_without_system_ffmpeg() {
        let bitstream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(bitstream);
        let mut ready = decoder
            .submit_annex_b(bitstream, Some(42), Some(42))
            .unwrap();
        ready.extend(decoder.finish().unwrap());
        assert_eq!(ready.len(), 1);
        assert!(decoder.finish().unwrap().is_empty());
        assert!(matches!(
            decoder.submit_annex_b(bitstream, None, None),
            Err(Error::Drained)
        ));
        let frame = &ready[0];
        assert_eq!((frame.width, frame.height), (32, 32));
        assert_eq!(frame.format, PixelFormat::I420);
        assert_eq!(
            frame
                .planes
                .iter()
                .map(|p| p.data.len())
                .collect::<Vec<_>>(),
            vec![1024, 256, 256]
        );
        assert!(frame.planes[0].data[0] > 50); // red Y luma, not an empty/black frame
        // Output storage is owned and remains available after freeing the decoder.
        drop(decoder);
        assert_eq!(frame.planes[0].data.len(), 1024);
    }

    #[test]
    fn converts_high444_10bit_video_into_portable_i420_frames() {
        let stream = include_bytes!("../tests/fixtures/blue_high444_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut frames = decoder.submit_annex_b(stream, Some(99), Some(99)).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].format, PixelFormat::I420);
        assert_eq!((frames[0].width, frames[0].height), (32, 32));
        assert_eq!(
            frames[0].planes.iter().map(|p| p.data.len()).sum::<usize>(),
            1536
        );
        assert_ne!(frames[0].planes[0].data[0], 0);
    }

    #[test]
    fn decodes_avcc_sample_with_separate_parameter_sets() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut avcc = Vec::new();
        for nal in nals_from_fixture(stream)
            .into_iter()
            .filter(|nal| !matches!(nal[0] & 0x1f, 7 | 8))
        {
            avcc.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            avcc.extend_from_slice(nal);
        }
        let mut frames = decoder.submit_avcc(&avcc, Some(15), Some(15)).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].width, 32);
    }

    #[test]
    fn decodes_reordered_high_profile_b_frames() {
        let stream = include_bytes!("../tests/fixtures/bframes_8_64x64.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut frames = Vec::new();
        let mut sample = Vec::new();
        let mut submitted = 0;
        for nal in nals_from_fixture(stream) {
            if nal[0] & 0x1f == 9 && !sample.is_empty() {
                frames.extend(
                    decoder
                        .submit_annex_b(&sample, Some(submitted), Some(submitted))
                        .unwrap(),
                );
                submitted += 1;
                sample.clear();
            }
            sample.extend_from_slice(&[0, 0, 0, 1]);
            sample.extend_from_slice(nal);
        }
        if !sample.is_empty() {
            frames.extend(
                decoder
                    .submit_annex_b(&sample, Some(submitted), Some(submitted))
                    .unwrap(),
            );
        }
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 8);
        assert!(frames.iter().all(|f| f.width == 64 && f.height == 64));
    }

    #[test]
    fn missing_headers_do_not_create_session() {
        assert!(matches!(
            Decoder::new(&[], NalLength::Four),
            Err(Error::InvalidParameterSet)
        ));
    }
}
