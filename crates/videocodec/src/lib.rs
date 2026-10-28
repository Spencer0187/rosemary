//! H.264 software decoding without a system FFmpeg installation.
//!
//! The public boundary is native Rust buffers and timestamps, never guest
//! pointers or CoreMedia/VideoToolbox objects. Decoder output is copied before
//! advancing the native decoder, so frame storage cannot outlive its owner.

use mediabackend::{
    self as media, AccessUnit, ChromaSiting, ColorMatrix, ColorPrimaries, ColorTransfer, FrameCrop,
    FrameFieldOrder, FrameInterlaceMode, FrameLayout,
};
use std::collections::{HashMap, VecDeque};

const MAX_QUEUED_FRAMES: usize = 32;
const MAX_QUEUED_BYTES: usize = 64 * 1024 * 1024;
const MAX_QUEUED_DROPS: usize = 1024;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::time::Duration;

const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;

fn checked_packet_size(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .filter(|&size| size <= MAX_PACKET_BYTES)
        .ok_or(Error::InvalidPacket)
}

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

#[derive(Debug)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub format: PixelFormat,
    pub pts: Option<i64>,
    pub id: u64,
    pub layout: FrameLayout,
    pub source_pts: Option<i64>,
    pub source_dts: Option<i64>,
    pub duration: Option<u64>,
    pub planes: Vec<Plane>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidLengthSize,
    InvalidParameterSet,
    InvalidPacket,
    BackendUnavailable,
    Codec(i32),
    UnsupportedPixelFormat,
    InvalidImageLayout,
    Drained,
    DuplicateIdentity,
    MissingIdentity,
    InvalidTimestamp,
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
    id: u64,
    identity_valid: i32,
    color_range: i32,
    colorspace: i32,
    primaries: i32,
    transfer: i32,
    chroma_location: i32,
    aspect_num: i32,
    aspect_den: i32,
    flags: i32,
    repeat_pict: i32,
    crop_left: usize,
    crop_top: usize,
    crop_right: usize,
    crop_bottom: usize,
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
        id: u64,
    ) -> i32;
    fn rosemary_h264_receive(handle: *mut c_void, out: *mut FrameView) -> i32;
    fn rosemary_h264_token_released(handle: *mut c_void, id: u64) -> i32;
    fn rosemary_h264_forget_token(handle: *mut c_void, id: u64);
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
    if sample.is_empty() || sample.len() > MAX_PACKET_BYTES {
        return Err(Error::InvalidPacket);
    }
    let mut out = Vec::new();
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
        checked_packet_size(out.len(), checked_packet_size(4, nal.len())?)?;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
        remaining = &remaining[nal_size..];
    }
    Ok(out)
}

// Keep a leading access-unit delimiter ahead of the parameter sets. H.264
// requires an AUD, when present, to be the first NAL of an access unit.
fn prepend_headers(headers: &[u8], sample: &[u8]) -> Result<Vec<u8>, Error> {
    let total = checked_packet_size(headers.len(), sample.len())?;
    let leading_zeros = sample.iter().take_while(|&&b| b == 0).count();
    let aud_header = (leading_zeros >= 2
        && sample.get(leading_zeros) == Some(&1)
        && sample
            .get(leading_zeros + 1)
            .is_some_and(|nal| nal & 0x1f == 9 && nal & 0x80 == 0))
    .then_some(leading_zeros + 1);
    let insertion = aud_header.map_or(0, |start| {
        let remaining = &sample[start + 1..];
        remaining
            .windows(3)
            .position(|window| window == [0, 0, 1])
            .map_or(sample.len(), |next| {
                let next = if next > 0 && remaining[next - 1] == 0 {
                    next - 1
                } else {
                    next
                };
                start + 1 + next
            })
    });
    let mut packet = Vec::with_capacity(total);
    packet.extend_from_slice(&sample[..insertion]);
    packet.extend_from_slice(headers);
    packet.extend_from_slice(&sample[insertion..]);
    Ok(packet)
}

/// Stateful H.264 decoder. One owner at a time; guest frameworks decide which
/// thread submits samples and how/when decoded frames are delivered.
#[derive(Clone, Copy)]
struct SourceTiming {
    pts: Option<i64>,
    dts: Option<i64>,
    duration: Option<u64>,
}

pub struct Decoder {
    handle: NonNull<c_void>,
    length: NalLength,
    needs_headers: bool,
    headers: Vec<u8>,
    finished: bool,
    next_id: u64,
    pending: HashMap<u64, SourceTiming>,
    dropped: VecDeque<media::DroppedSample>,
    available: VecDeque<CodecFrame>,
    last_accepted_id: Option<u64>,
    queued_bytes: usize,
    last_frame_bytes: usize,
    fatal: Option<Error>,
    eof: bool,
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
            checked_packet_size(headers.len(), checked_packet_size(4, set.len())?)
                .map_err(|_| Error::InvalidParameterSet)?;
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
            next_id: 0,
            pending: HashMap::new(),
            dropped: VecDeque::new(),
            available: VecDeque::new(),
            last_accepted_id: None,
            queued_bytes: 0,
            last_frame_bytes: 0,
            fatal: None,
            eof: false,
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
        let id = self.next_id;
        let frames = self.submit_with_id(sample, id, pts, dts, None)?;
        Ok(frames)
    }

    fn submit_with_id(
        &mut self,
        sample: &[u8],
        id: u64,
        pts: Option<i64>,
        dts: Option<i64>,
        duration: Option<u64>,
    ) -> Result<Vec<Frame>, Error> {
        self.check_fatal()?;
        if self.finished {
            return Err(Error::Drained);
        }
        if id == u64::MAX || self.last_accepted_id.is_some_and(|last| id <= last) {
            return Err(Error::DuplicateIdentity);
        }
        if pts == Some(i64::MIN) || dts == Some(i64::MIN) {
            return Err(Error::InvalidTimestamp);
        }
        if sample.is_empty() || sample.len() > MAX_PACKET_BYTES {
            return Err(Error::InvalidPacket);
        }
        let owned;
        let packet = if self.needs_headers {
            owned = prepend_headers(&self.headers, sample)?;
            &owned[..]
        } else {
            sample
        };
        let mut ready = Vec::new();
        let mut status = self.send(Some(packet), pts, dts, id);
        if status == 1 {
            ready.extend(self.receive_ready()?);
            status = self.send(Some(packet), pts, dts, id);
        }
        if status != 0 {
            return Err(self.latch(Error::Codec(status)));
        }
        self.needs_headers = false;
        self.pending.insert(id, SourceTiming { pts, dts, duration });
        self.last_accepted_id = Some(id);
        self.next_id = id.saturating_add(1);
        ready.extend(self.receive_ready()?);
        Ok(ready)
    }

    /// Finish delayed/reordered frames. A drained decoder must be recreated.
    pub fn finish(&mut self) -> Result<Vec<Frame>, Error> {
        self.check_fatal()?;
        if self.finished {
            return Ok(Vec::new());
        }
        let mut result = self.send(None, None, None, 0);
        let mut frames = Vec::new();
        if result == 1 {
            frames.extend(self.receive_ready()?);
            result = self.send(None, None, None, 0);
        }
        if result != 0 {
            return Err(self.latch(Error::Codec(result)));
        }
        self.finished = true;
        frames.extend(self.receive_ready()?);
        self.eof = true;
        let mut dropped = self.pending.drain().map(|(id, _)| id).collect::<Vec<_>>();
        dropped.sort_unstable();
        for id in dropped {
            self.forget_token(id);
            self.dropped
                .push_back(media::DroppedSample { id, error: None });
        }
        Ok(frames)
    }

    fn send(&self, bytes: Option<&[u8]>, pts: Option<i64>, dts: Option<i64>, id: u64) -> i32 {
        // SAFETY: owned handle remains valid until Drop and pointer/length are
        // only borrowed synchronously by the C entry point.
        unsafe {
            rosemary_h264_send(
                self.handle.as_ptr(),
                bytes.map_or(std::ptr::null(), |data| data.as_ptr()),
                bytes.map_or(0, <[u8]>::len),
                pts.unwrap_or(i64::MIN),
                dts.unwrap_or(i64::MIN),
                id,
            )
        }
    }

    fn check_fatal(&self) -> Result<(), Error> {
        self.fatal.map_or(Ok(()), Err)
    }

    fn latch(&mut self, error: Error) -> Error {
        *self.fatal.get_or_insert(error)
    }

    fn forget_token(&mut self, id: u64) {
        // SAFETY: the decoder remains alive while its observer is removed.
        unsafe { rosemary_h264_forget_token(self.handle.as_ptr(), id) };
    }

    fn reap_dropped(&mut self) -> Result<(), Error> {
        let mut dropped = Vec::new();
        for &id in self.pending.keys() {
            // SAFETY: each pending ID has a token owned by this decoder.
            match unsafe { rosemary_h264_token_released(self.handle.as_ptr(), id) } {
                1 => dropped.push(id),
                0 => (),
                _ => return Err(self.latch(Error::MissingIdentity)),
            }
        }
        dropped.sort_unstable();
        let room = MAX_QUEUED_DROPS.saturating_sub(self.dropped.len());
        for id in dropped.into_iter().take(room) {
            self.pending.remove(&id);
            self.forget_token(id);
            self.dropped
                .push_back(media::DroppedSample { id, error: None });
        }
        Ok(())
    }

    fn receive_one(&mut self) -> Result<Option<Frame>, Error> {
        self.check_fatal()?;
        let mut view = FrameView::default();
        // SAFETY: the output view and decoder are valid for the call.
        match unsafe { rosemary_h264_receive(self.handle.as_ptr(), &mut view) } {
            0 => {
                if view.identity_valid == 0 {
                    return Err(self.latch(Error::MissingIdentity));
                }
                let mut frame = copy_frame(&view).map_err(|e| self.latch(e))?;
                let timing = self
                    .pending
                    .remove(&view.id)
                    .ok_or_else(|| self.latch(Error::MissingIdentity))?;
                self.forget_token(view.id);
                frame.source_pts = timing.pts;
                frame.source_dts = timing.dts;
                frame.duration = timing.duration;
                Ok(Some(frame))
            }
            1 => {
                self.reap_dropped()?;
                Ok(None)
            }
            2 => {
                self.eof = true;
                self.reap_dropped()?;
                Ok(None)
            }
            negative => Err(self.latch(Error::Codec(negative))),
        }
    }

    fn receive_ready(&mut self) -> Result<Vec<Frame>, Error> {
        let mut frames = Vec::new();
        while let Some(frame) = self.receive_one()? {
            frames.push(frame);
        }
        Ok(frames)
    }

    fn queue_has_room(&self) -> bool {
        self.available.len() < MAX_QUEUED_FRAMES
            && self.dropped.len() < MAX_QUEUED_DROPS
            && self
                .queued_bytes
                .saturating_add(self.last_frame_bytes.max(1))
                <= MAX_QUEUED_BYTES
    }

    fn queue_frame(&mut self, frame: Frame) -> Result<(), Error> {
        let size = frame
            .planes
            .iter()
            .try_fold(0usize, |sum, plane| sum.checked_add(plane.data.len()))
            .ok_or_else(|| self.latch(Error::InvalidImageLayout))?;
        if size > MAX_QUEUED_BYTES
            || self.available.len() >= MAX_QUEUED_FRAMES
            || self
                .queued_bytes
                .checked_add(size)
                .is_none_or(|total| total > MAX_QUEUED_BYTES)
        {
            return Err(self.latch(Error::InvalidImageLayout));
        }
        self.last_frame_bytes = size;
        self.queued_bytes += size;
        self.available.push_back(CodecFrame::from(frame));
        Ok(())
    }

    fn pump_queued(&mut self) -> Result<(), Error> {
        while self.queue_has_room() {
            let Some(frame) = self.receive_one()? else {
                break;
            };
            self.queue_frame(frame)?;
        }
        Ok(())
    }

    fn pull_queued(&mut self) -> Result<Option<CodecFrame>, Error> {
        self.check_fatal()?;
        if let Some(frame) = self.available.pop_front() {
            self.queued_bytes -= frame.bytes.len();
            return Ok(Some(frame));
        }
        Ok(self.receive_one()?.map(CodecFrame::from))
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
    let crop = (view.crop_left | view.crop_top | view.crop_right | view.crop_bottom != 0)
        .then(|| {
            Some(FrameCrop {
                x: view.crop_left,
                y: view.crop_top,
                width: width
                    .checked_sub(view.crop_left.checked_add(view.crop_right)?)
                    .filter(|&n| n > 0)?,
                height: height
                    .checked_sub(view.crop_top.checked_add(view.crop_bottom)?)
                    .filter(|&n| n > 0)?,
            })
        })
        .flatten();
    if view.crop_left | view.crop_top | view.crop_right | view.crop_bottom != 0 && crop.is_none() {
        return Err(Error::InvalidImageLayout);
    }
    let mut offsets = Vec::new();
    let mut current = 0usize;
    for plane in &planes {
        offsets.push(current);
        current = current
            .checked_add(plane.data.len())
            .ok_or(Error::InvalidImageLayout)?;
    }
    let layout = FrameLayout {
        width,
        height,
        format: match format {
            PixelFormat::I420 => media::PixelFormat::I420,
            PixelFormat::Nv12 => media::PixelFormat::Nv12,
        },
        strides: planes.iter().map(|p| p.width_bytes).collect(),
        offsets,
        crop,
        pixel_aspect_ratio: match (view.aspect_num, view.aspect_den) {
            (n, d) if n > 0 && d > 0 => (n as u32, d as u32),
            _ => (1, 1),
        },
        chroma_siting: match view.chroma_location {
            1 => ChromaSiting::Mpeg2,
            2 => ChromaSiting::Jpeg,
            3 => ChromaSiting::Cosited,
            0 => ChromaSiting::Unknown,
            other => ChromaSiting::Other(other as u32),
        },
        interlace_mode: if view.flags & (1 << 3) != 0 {
            FrameInterlaceMode::Interleaved
        } else {
            FrameInterlaceMode::Progressive
        },
        field_order: if view.flags & (1 << 3) != 0 {
            if view.flags & (1 << 4) != 0 {
                FrameFieldOrder::TopFirst
            } else {
                FrameFieldOrder::BottomFirst
            }
        } else {
            FrameFieldOrder::Unknown
        },
        repeat_first_field: view.repeat_pict > 0,
        one_field: false,
        matrix: match view.colorspace {
            0 => ColorMatrix::Rgb,
            1 => ColorMatrix::Bt709,
            4 => ColorMatrix::Fcc,
            5 | 6 => ColorMatrix::Bt601,
            7 => ColorMatrix::Smpte240m,
            9 | 10 => ColorMatrix::Bt2020,
            2 => ColorMatrix::Unknown,
            other => ColorMatrix::Other(other),
        },
        primaries: match view.primaries {
            1 => ColorPrimaries::Bt709,
            4 => ColorPrimaries::Bt470m,
            5 => ColorPrimaries::Bt470bg,
            6 => ColorPrimaries::Smpte170m,
            7 => ColorPrimaries::Smpte240m,
            8 => ColorPrimaries::Film,
            9 => ColorPrimaries::Bt2020,
            10 => ColorPrimaries::Smptest428,
            11 => ColorPrimaries::Smpterp431,
            12 => ColorPrimaries::Smpteeg432,
            22 => ColorPrimaries::Ebu3213,
            2 => ColorPrimaries::Unknown,
            other => ColorPrimaries::Other(other),
        },
        transfer: match view.transfer {
            1 => ColorTransfer::Bt709,
            4 => ColorTransfer::Gamma22,
            5 => ColorTransfer::Gamma28,
            6 => ColorTransfer::Bt601,
            7 => ColorTransfer::Smpte240m,
            9 => ColorTransfer::Log100,
            10 => ColorTransfer::Log316,
            13 => ColorTransfer::Srgb,
            14 => ColorTransfer::Bt202010,
            15 => ColorTransfer::Bt202012,
            16 => ColorTransfer::Smpte2084,
            18 => ColorTransfer::AribStdB67,
            2 => ColorTransfer::Unknown,
            other => ColorTransfer::Other(other),
        },
        full_range: match view.color_range {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        },
    };
    Ok(Frame {
        width,
        height,
        format,
        pts: (view.pts != i64::MIN).then_some(view.pts),
        id: view.id,
        layout,
        source_pts: None,
        source_dts: None,
        duration: None,
        planes,
    })
}

/// A contiguous owned decoded frame for mediabackend consumers.
/// Unlike the compatibility plane API, repeated with_bytes calls do not copy.
#[derive(Debug)]
pub struct CodecFrame {
    id: u64,
    layout: FrameLayout,
    bytes: Vec<u8>,
    pts: Option<i64>,
    source_pts: Option<i64>,
    source_dts: Option<i64>,
    duration: Option<u64>,
}

impl From<Frame> for CodecFrame {
    fn from(frame: Frame) -> Self {
        let mut bytes = Vec::with_capacity(frame.planes.iter().map(|p| p.data.len()).sum());
        for plane in frame.planes {
            bytes.extend_from_slice(&plane.data);
        }
        Self {
            id: frame.id,
            layout: frame.layout,
            bytes,
            pts: frame.pts,
            source_pts: frame.source_pts,
            source_dts: frame.source_dts,
            duration: frame.duration,
        }
    }
}

impl media::VideoFrame for CodecFrame {
    fn id(&self) -> u64 {
        self.id
    }
    fn layout(&self) -> &FrameLayout {
        &self.layout
    }
    fn pts_ns(&self) -> Option<i64> {
        self.pts
    }
    fn dts_ns(&self) -> Option<i64> {
        None
    }
    fn duration_ns(&self) -> Option<u64> {
        self.duration
    }
    fn source_pts_ns(&self) -> Option<i64> {
        self.source_pts
    }
    fn source_dts_ns(&self) -> Option<i64> {
        self.source_dts
    }
    fn with_bytes<R>(
        &self,
        read: impl FnOnce(&[u8], &FrameLayout) -> R,
    ) -> Result<R, media::Error> {
        Ok(read(&self.bytes, &self.layout))
    }
}

fn media_error(error: Error) -> media::Error {
    match error {
        Error::Drained => media::Error::EndOfStream,
        Error::InvalidLengthSize
        | Error::InvalidParameterSet
        | Error::InvalidPacket
        | Error::DuplicateIdentity
        | Error::InvalidTimestamp => media::Error::InvalidInput(error.to_string()),
        Error::BackendUnavailable => media::Error::BackendUnavailable(error.to_string()),
        _ => media::Error::Decode(error.to_string()),
    }
}

impl media::VideoDecoder for Decoder {
    type Frame = CodecFrame;

    fn codec(&self) -> media::VideoCodec {
        media::VideoCodec::H264
    }

    fn submit(&mut self, packet: AccessUnit<'_>) -> Result<(), media::Error> {
        self.check_fatal().map_err(media_error)?;
        if self.finished {
            return Err(media::Error::EndOfStream);
        }
        if !self.queue_has_room() {
            return Err(media::Error::Backpressure);
        }
        let bytes = avcc_to_annex_b(packet.bytes, self.length).map_err(media_error)?;
        let frames = self
            .submit_with_id(
                &bytes,
                packet.id,
                packet.pts_ns,
                packet.dts_ns,
                packet.duration_ns,
            )
            .map_err(media_error)?;
        for frame in frames {
            self.queue_frame(frame).map_err(media_error)?;
        }
        self.pump_queued().map_err(media_error)?;
        Ok(())
    }

    fn pull(&mut self, _timeout: Duration) -> Result<Option<Self::Frame>, media::Error> {
        self.pull_queued().map_err(media_error)
    }

    fn take_dropped(&mut self) -> Option<media::DroppedSample> {
        let reported = self.dropped.pop_front();
        if reported.is_some() && self.fatal.is_none() {
            let _ = self.reap_dropped();
        }
        reported
    }

    fn end_stream(&mut self) -> Result<(), media::Error> {
        self.check_fatal().map_err(media_error)?;
        if self.finished {
            return Ok(());
        }
        if !self.queue_has_room() {
            return Err(media::Error::Backpressure);
        }
        match self.send(None, None, None, 0) {
            0 => {
                self.finished = true;
                Ok(())
            }
            1 => Err(media::Error::Backpressure),
            error => Err(media_error(self.latch(Error::Codec(error)))),
        }
    }

    fn is_drained(&mut self) -> Result<bool, media::Error> {
        self.check_fatal().map_err(media_error)?;
        if !self.finished || !self.available.is_empty() {
            return Ok(false);
        }
        if !self.eof
            && let Some(frame) = self.receive_one().map_err(media_error)?
        {
            self.queue_frame(frame).map_err(media_error)?;
            return Ok(false);
        }
        Ok(self.eof && self.available.is_empty())
    }
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
    fn rejects_packet_expansion_above_limit() {
        assert_eq!(
            checked_packet_size(MAX_PACKET_BYTES, 1),
            Err(Error::InvalidPacket)
        );
        assert_eq!(
            checked_packet_size(usize::MAX, 1),
            Err(Error::InvalidPacket)
        );
        assert_eq!(
            checked_packet_size(MAX_PACKET_BYTES - 4, 4),
            Ok(MAX_PACKET_BYTES)
        );
        assert_eq!(
            prepend_headers(&[0; 8], &vec![0; MAX_PACKET_BYTES]),
            Err(Error::InvalidPacket)
        );
    }

    #[test]
    fn inserts_parameter_sets_after_leading_aud() {
        let headers = [0, 0, 0, 1, 0x67, 0x42, 0x80];
        for aud in [&[0, 0, 1, 9, 0xf0][..], &[0, 0, 0, 1, 9, 0xf0][..]] {
            let mut sample = aud.to_vec();
            sample.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x42]);
            let packet = prepend_headers(&headers, &sample).unwrap();
            assert!(packet.starts_with(aud));
            assert_eq!(&packet[aud.len()..aud.len() + headers.len()], headers);
            assert_eq!(&packet[aud.len() + headers.len()..], &sample[aud.len()..]);
        }
        let sample = [0, 0, 0, 0, 1, 9, 0xf0, 0, 0, 0, 1, 0x65];
        let packet = prepend_headers(&headers, &sample).unwrap();
        assert_eq!(&packet[..7], &sample[..7]);
        assert_eq!(&packet[7..7 + headers.len()], headers);
        assert_eq!(&packet[7 + headers.len()..], &sample[7..]);
        let sample = [0, 0, 0, 0, 0, 1, 9, 0xf0, 0, 0, 1, 0x65];
        let packet = prepend_headers(&headers, &sample).unwrap();
        assert_eq!(&packet[..8], &sample[..8]);
        assert_eq!(&packet[8..8 + headers.len()], headers);
        let sample = [0, 0, 0, 1, 0x65, 0x42];
        let packet = prepend_headers(&headers, &sample).unwrap();
        assert!(packet.starts_with(&headers));
        assert!(packet.ends_with(&sample));
    }

    #[test]
    fn bframe_identity_ignores_duplicate_and_missing_timestamps() {
        use mediabackend::{VideoDecoder, VideoFrame};
        let stream = include_bytes!("../tests/fixtures/bframes_8_64x64.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut access_units = Vec::new();
        let mut sample = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 0x1f == 9 && !sample.is_empty() {
                access_units.push(std::mem::take(&mut sample));
            }
            sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            sample.extend_from_slice(nal);
        }
        if !sample.is_empty() {
            access_units.push(sample);
        }
        assert_eq!(access_units.len(), 8);
        for (index, bytes) in access_units.iter().enumerate() {
            decoder
                .submit(AccessUnit {
                    id: 400 + index as u64,
                    bytes,
                    pts_ns: (index % 2 == 0).then_some(7),
                    dts_ns: None,
                    duration_ns: Some(33),
                })
                .unwrap();
        }
        decoder.end_stream().unwrap();
        let mut ids = Vec::new();
        while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
            ids.push(frame.id());
            assert_eq!(frame.duration_ns(), Some(33));
            assert_eq!(
                frame.source_pts_ns(),
                ((frame.id() - 400) % 2 == 0).then_some(7)
            );
            assert_eq!(frame.source_dts_ns(), None);
            let first = frame
                .with_bytes(|buf, layout| {
                    assert_eq!((layout.width, layout.height), (64, 64));
                    assert_eq!(buf.len(), 64 * 64 * 3 / 2);
                    buf.as_ptr()
                })
                .unwrap();
            let second = frame.with_bytes(|buf, _| buf.as_ptr()).unwrap();
            assert_eq!(first, second);
        }
        assert!(decoder.is_drained().unwrap());
        assert!(decoder.take_dropped().is_none());
        ids.sort_unstable();
        assert_eq!(ids, (400..408).collect::<Vec<_>>());
    }

    fn bframe_avcc_access_units() -> Vec<Vec<u8>> {
        let stream = include_bytes!("../tests/fixtures/bframes_8_64x64.h264");
        let mut units = Vec::new();
        let mut sample = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 0x1f == 9 && !sample.is_empty() {
                units.push(std::mem::take(&mut sample));
            }
            sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            sample.extend_from_slice(nal);
        }
        if !sample.is_empty() {
            units.push(sample);
        }
        assert_eq!(units.len(), 8);
        units
    }

    #[test]
    fn queue_backpressure_preserves_input_and_recovers_through_eos() {
        use mediabackend::{VideoDecoder, VideoFrame};
        let stream = include_bytes!("../tests/fixtures/bframes_8_64x64.h264");
        let mut decoder = decoder_for_fixture(stream);
        let units = bframe_avcc_access_units();
        let mut decoded = Vec::new();
        let mut blocked = false;
        for id in 0..96_u64 {
            let au = AccessUnit {
                id,
                bytes: &units[id as usize % units.len()],
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            };
            loop {
                match decoder.submit(au) {
                    Ok(()) => break,
                    Err(media::Error::Backpressure) => {
                        blocked = true;
                        assert_eq!(decoder.last_accepted_id, Some(id - 1));
                        assert!(decoder.available.len() <= MAX_QUEUED_FRAMES);
                        assert!(decoder.queued_bytes <= MAX_QUEUED_BYTES);
                        decoded.push(decoder.pull(Duration::ZERO).unwrap().unwrap().id());
                    }
                    Err(e) => panic!("unexpected submission error {e}"),
                }
            }
        }
        assert!(blocked, "producer never encountered backpressure");
        loop {
            match decoder.end_stream() {
                Ok(()) => break,
                Err(media::Error::Backpressure) => {
                    decoded.push(decoder.pull(Duration::ZERO).unwrap().unwrap().id());
                }
                Err(e) => panic!("unexpected EOS error {e}"),
            }
        }
        while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
            decoded.push(frame.id());
        }
        assert!(decoder.is_drained().unwrap());
        assert!(decoder.take_dropped().is_none());
        decoded.sort_unstable();
        assert_eq!(decoded, (0..96).collect::<Vec<_>>());
        assert!(decoder.pending.is_empty());
        assert_eq!(decoder.available.len(), 0);
        assert_eq!(decoder.queued_bytes, 0);
    }

    #[test]
    fn missing_initial_idr_reports_live_drops_and_recovers_at_next_gop() {
        use mediabackend::{VideoDecoder, VideoFrame};
        let stream = include_bytes!("../tests/fixtures/missing_initial_idr_64x64.h264");
        let nals = nals_from_fixture(stream);
        let sets = nals
            .iter()
            .copied()
            .filter(|nal| matches!(nal[0] & 31, 7 | 8))
            .take(2)
            .collect::<Vec<_>>();
        let mut decoder = Decoder::new(&sets, NalLength::Four).unwrap();
        let mut units = Vec::new();
        let mut sample = Vec::new();
        for nal in nals {
            if nal[0] & 31 == 9 && !sample.is_empty() {
                units.push(std::mem::take(&mut sample));
            }
            sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            sample.extend_from_slice(nal);
        }
        if !sample.is_empty() {
            units.push(sample);
        }
        assert_eq!(units.len(), 60);
        // The first packet is invalid at FFmpeg send time. It is rejected,
        // not counted as an accepted-but-dropped input. The subsequent inputs
        // are accepted even though no reference picture has arrived yet.
        let mut rejected = Decoder::new(&sets, NalLength::Four).unwrap();
        assert!(matches!(
            rejected.submit(AccessUnit {
                id: 0,
                bytes: &units[0],
                pts_ns: None,
                dts_ns: None,
                duration_ns: None
            }),
            Err(media::Error::Decode(_))
        ));
        assert!(matches!(
            rejected.submit(AccessUnit {
                id: 1,
                bytes: &units[1],
                pts_ns: None,
                dts_ns: None,
                duration_ns: None
            }),
            Err(media::Error::Decode(_))
        ));
        drop(rejected);
        let mut decoded = Vec::new();
        let mut dropped = Vec::new();
        let mut had_live_drop = false;
        for id in 1..60_u64 {
            decoder
                .submit(AccessUnit {
                    id,
                    bytes: &units[id as usize],
                    pts_ns: None,
                    dts_ns: None,
                    duration_ns: None,
                })
                .unwrap();
            while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
                decoded.push(frame.id());
            }
            while let Some(sample) = decoder.take_dropped() {
                had_live_drop = true;
                dropped.push(sample.id);
            }
        }
        assert!(
            had_live_drop,
            "pre-recovery input was never completed while streaming"
        );
        decoder.end_stream().unwrap();
        while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
            decoded.push(frame.id());
        }
        while let Some(sample) = decoder.take_dropped() {
            dropped.push(sample.id);
        }
        assert!(decoder.is_drained().unwrap());
        assert_eq!(decoded.len(), 40);
        assert_eq!(dropped.len(), 19);
        assert!(decoded.iter().all(|&id| id >= 20));
        assert!(dropped.iter().all(|&id| id < 20));
        decoded.extend(dropped);
        decoded.sort_unstable();
        assert_eq!(decoded, (1..60).collect::<Vec<_>>());
    }

    #[test]
    fn output_budget_and_crop_validation() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        assert!(decoder.queue_has_room());
        decoder.queued_bytes = MAX_QUEUED_BYTES;
        assert!(!decoder.queue_has_room());
        decoder.queued_bytes = 0;
        for id in 0..MAX_QUEUED_DROPS {
            decoder.dropped.push_back(media::DroppedSample {
                id: id as u64,
                error: None,
            });
        }
        assert!(!decoder.queue_has_room());
        assert!(decoder.dropped.pop_front().is_some());
        assert!(decoder.queue_has_room());
        decoder.dropped.clear();
        let y = [0_u8; 4];
        let u = [0_u8; 1];
        let v = [0_u8; 1];
        let mut view = FrameView {
            width: 2,
            height: 2,
            format: 0,
            planes: [y.as_ptr(), u.as_ptr(), v.as_ptr()],
            strides: [2, 1, 1],
            ..FrameView::default()
        };
        view.crop_left = 2;
        assert!(matches!(copy_frame(&view), Err(Error::InvalidImageLayout)));
        view.crop_left = 0;
        view.crop_top = 2;
        assert!(matches!(copy_frame(&view), Err(Error::InvalidImageLayout)));
        view.crop_top = 0;
        view.crop_right = 3;
        assert!(matches!(copy_frame(&view), Err(Error::InvalidImageLayout)));
    }

    #[test]
    fn fatal_decoder_state_rejects_recovery_and_drain() {
        use mediabackend::VideoDecoder;
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        assert_eq!(
            decoder.latch(Error::MissingIdentity),
            Error::MissingIdentity
        );
        assert_eq!(decoder.latch(Error::Codec(-5)), Error::MissingIdentity);
        assert!(matches!(
            decoder.submit_annex_b(stream, None, None),
            Err(Error::MissingIdentity)
        ));
        assert!(matches!(
            decoder.pull(Duration::ZERO),
            Err(media::Error::Decode(_))
        ));
        assert!(matches!(decoder.end_stream(), Err(media::Error::Decode(_))));
        assert!(matches!(decoder.is_drained(), Err(media::Error::Decode(_))));
    }

    #[test]
    fn old_ids_are_not_kept_in_a_growing_set() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        decoder.last_accepted_id = Some(1_000_000);
        assert!(matches!(
            decoder.submit_with_id(stream, 999_999, None, None, None),
            Err(Error::DuplicateIdentity)
        ));
        assert!(matches!(
            decoder.submit_with_id(stream, 1_000_000, None, None, None),
            Err(Error::DuplicateIdentity)
        ));
        assert!(
            decoder
                .submit_with_id(stream, 1_000_001, None, None, None)
                .is_ok()
        );
        assert_eq!(decoder.last_accepted_id, Some(1_000_001));
    }

    #[test]
    fn duplicate_identity_is_rejected_after_a_picture_was_delivered() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let frames = decoder
            .submit_with_id(stream, 45, None, None, None)
            .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].id, 45);
        assert!(matches!(
            decoder.submit_with_id(stream, 45, None, None, None),
            Err(Error::DuplicateIdentity)
        ));
    }

    #[test]
    fn reserved_timestamp_is_rejected_instead_of_disappearing() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        assert!(matches!(
            decoder.submit_annex_b(stream, Some(i64::MIN), None),
            Err(Error::InvalidTimestamp)
        ));
        assert!(matches!(
            decoder.submit_annex_b(stream, None, Some(i64::MIN)),
            Err(Error::InvalidTimestamp)
        ));
        let mut frames = decoder.submit_annex_b(stream, Some(-123), None).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn explicit_h264_color_range_and_transfer_are_preserved() {
        for (stream, full) in [
            (
                &include_bytes!("../tests/fixtures/color_full_64x64.h264")[..],
                true,
            ),
            (
                &include_bytes!("../tests/fixtures/color_limited_64x64.h264")[..],
                false,
            ),
        ] {
            let mut decoder = decoder_for_fixture(stream);
            let mut frames = decoder.submit_annex_b(stream, None, None).unwrap();
            frames.extend(decoder.finish().unwrap());
            assert!(!frames.is_empty());
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.layout.full_range == Some(full))
            );
        }
        let stream = include_bytes!("../tests/fixtures/explicit_bt709_color_64x64.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut frames = decoder.submit_annex_b(stream, None, None).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert!(frames.iter().all(|f| f.layout.matrix == ColorMatrix::Bt709
            && f.layout.primaries == ColorPrimaries::Bt709
            && f.layout.transfer == ColorTransfer::Bt709));
    }

    #[test]
    fn missing_headers_do_not_create_session() {
        assert!(matches!(
            Decoder::new(&[], NalLength::Four),
            Err(Error::InvalidParameterSet)
        ));
    }
}
