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
// One decoded frame that cannot fit the FIFO due to an SPS size change.
const MAX_OVERFLOW_BYTES: usize = MAX_QUEUED_BYTES;
const MAX_QUEUED_DROPS: usize = 1024;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::time::Duration;

const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;
const AV_FRAME_FLAG_CORRUPT: i32 = 1 << 0;
// Positive native frame-rejection status, distinct from fatal FFmpeg errors.
const NATIVE_PICTURE_REJECTED: i32 = 6;
#[cfg(test)]
const TEST_SWS_CONTEXT_FAILURE: i32 = 1;
#[cfg(test)]
const TEST_CONVERTED_FRAME_FAILURE: i32 = 2;

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
    Backpressure,
    UnsupportedPixelFormat,
    SourceFrameTooLarge,
    CorruptFrame,
    InvalidImageLayout,
    Drained,
    DuplicateIdentity,
    MissingIdentity,
    InvalidTimestamp,
    MixedDecoderApis,
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
    decode_error_flags: i32,
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
    fn rosemary_h264_last_send_error(handle: *mut c_void) -> i32;
    fn rosemary_h264_last_budget_id(handle: *mut c_void, id: *mut u64) -> i32;
    #[cfg(test)]
    fn rosemary_h264_test_send_enomem_after_accept(handle: *mut c_void);
    #[cfg(test)]
    fn rosemary_h264_test_fail_native_allocation(handle: *mut c_void, stage: i32);
    #[cfg(test)]
    fn rosemary_h264_test_fail_converter_allocation(handle: *mut c_void, stage: i32);
    fn rosemary_h264_token_released(handle: *mut c_void, id: u64) -> i32;
    fn rosemary_h264_forget_token(handle: *mut c_void, id: u64);
    #[cfg(test)]
    fn rosemary_h264_native_peak_bytes(handle: *mut c_void) -> usize;
    #[cfg(test)]
    fn rosemary_h264_native_live_bytes(handle: *mut c_void) -> usize;
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

// Validate framing and expanded size before allocating any converted output.
// This also permits a caller to check malformed input ahead of backpressure.
fn avcc_annex_b_len(sample: &[u8], length: NalLength) -> Result<usize, Error> {
    if sample.is_empty() || sample.len() > MAX_PACKET_BYTES {
        return Err(Error::InvalidPacket);
    }
    let mut total = 0;
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
        total = checked_packet_size(total, checked_packet_size(4, nal.len())?)?;
        remaining = &remaining[nal_size..];
    }
    Ok(total)
}

/// Converts a validated length-prefixed access unit to Annex B.
pub fn avcc_to_annex_b(sample: &[u8], length: NalLength) -> Result<Vec<u8>, Error> {
    let length_bytes = avcc_annex_b_len(sample, length)?;
    let mut out = Vec::with_capacity(length_bytes);
    let mut remaining = sample;
    while !remaining.is_empty() {
        let width = length.width();
        let nal_size = remaining[..width]
            .iter()
            .fold(0usize, |value, &byte| (value << 8) | usize::from(byte));
        remaining = &remaining[width..];
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&remaining[..nal_size]);
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
#[derive(Clone, Copy, PartialEq, Eq)]
enum DecoderApi {
    Legacy,
    Media,
}

#[derive(Clone, Copy)]
struct SourceTiming {
    pts: Option<i64>,
    dts: Option<i64>,
    duration: Option<u64>,
    decode_error: Option<Error>,
}

pub struct Decoder {
    handle: NonNull<c_void>,
    length: NalLength,
    needs_headers: bool,
    headers: Vec<u8>,
    finished: bool,
    next_id: u64,
    legacy_ready: Vec<Frame>,
    api: Option<DecoderApi>,
    pending: HashMap<u64, SourceTiming>,
    dropped: VecDeque<media::DroppedSample>,
    available: VecDeque<CodecFrame>,
    overflow: Option<CodecFrame>,
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
            legacy_ready: Vec::new(),
            api: None,
            pending: HashMap::new(),
            dropped: VecDeque::new(),
            available: VecDeque::new(),
            overflow: None,
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
        self.check_fatal()?;
        self.select_api(DecoderApi::Legacy)?;
        if self.finished {
            return Err(Error::Drained);
        }
        if self.next_id == u64::MAX {
            return Err(Error::DuplicateIdentity);
        }
        self.validate_input(self.next_id, pts, dts)?;
        let expanded = avcc_annex_b_len(sample, self.length)?;
        if self.needs_headers {
            checked_packet_size(self.headers.len(), expanded)?;
        }
        if self.dropped.len() >= MAX_QUEUED_DROPS {
            return Err(Error::Backpressure);
        }
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
        self.check_fatal()?;
        self.select_api(DecoderApi::Legacy)?;
        if self.next_id == u64::MAX {
            return Err(Error::DuplicateIdentity);
        }
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
        if self.id_in_flight(id) {
            return Err(Error::DuplicateIdentity);
        }
        if pts == Some(i64::MIN) || dts == Some(i64::MIN) {
            return Err(Error::InvalidTimestamp);
        }
        if sample.is_empty() || sample.len() > MAX_PACKET_BYTES {
            return Err(Error::InvalidPacket);
        }
        if self.dropped.len() >= MAX_QUEUED_DROPS {
            return Err(Error::Backpressure);
        }
        let owned;
        let packet = if self.needs_headers {
            owned = prepend_headers(&self.headers, sample)?;
            &owned[..]
        } else {
            sample
        };
        let mut status = self.send(Some(packet), pts, dts, id);
        if status == 1 {
            let received = self.receive_ready()?;
            self.legacy_ready.extend(received);
            if self.dropped.len() >= MAX_QUEUED_DROPS {
                return Err(Error::Backpressure);
            }
            status = self.send(Some(packet), pts, dts, id);
        }
        if status == 1 {
            return Err(Error::Backpressure);
        }
        if !matches!(status, 0 | 3 | 4 | 5) {
            return Err(self.latch(Error::Codec(status)));
        }
        let decode_error = (status == 3).then(|| self.accepted_send_error());
        self.needs_headers = false;
        self.pending.insert(
            id,
            SourceTiming {
                pts,
                dts,
                duration,
                decode_error,
            },
        );
        self.next_id = self.next_id.max(id.saturating_add(1));
        if status == 4 {
            let origin = self.budget_rejected_id()?;
            self.mark_budget_drop(origin)?;
        }
        if status == 5 {
            return Err(self.latch(self.accepted_send_error()));
        }
        let mut ready = std::mem::take(&mut self.legacy_ready);
        ready.extend(self.receive_ready()?);
        Ok(ready)
    }

    /// Finish delayed/reordered frames. A drained decoder must be recreated.
    pub fn finish(&mut self) -> Result<Vec<Frame>, Error> {
        self.check_fatal()?;
        self.select_api(DecoderApi::Legacy)?;
        if self.dropped.len() >= MAX_QUEUED_DROPS {
            return Err(Error::Backpressure);
        }
        if !self.finished {
            let mut status = self.send(None, None, None, 0);
            if status == 1 {
                let ready = self.receive_ready()?;
                self.legacy_ready.extend(ready);
                if self.dropped.len() >= MAX_QUEUED_DROPS {
                    return Err(Error::Backpressure);
                }
                status = self.send(None, None, None, 0);
            }
            if status == 1 {
                return Err(Error::Backpressure);
            }
            if status != 0 {
                return Err(self.latch(Error::Codec(status)));
            }
            self.finished = true;
        }
        if !self.eof {
            let ready = self.receive_ready()?;
            self.legacy_ready.extend(ready);
        }
        if self.dropped.len() >= MAX_QUEUED_DROPS {
            return Err(Error::Backpressure);
        }
        if !self.eof {
            // FFmpeg has not confirmed terminal EOF; keep all produced
            // frames and resume after the caller drains completions.
            return Err(Error::Backpressure);
        }
        self.reap_dropped()?;
        Ok(std::mem::take(&mut self.legacy_ready))
    }

    /// Drain completion notifications when using the legacy Annex B/AVCC API.
    /// A full completion queue makes further legacy submissions return
    /// recoverable `Error::Backpressure` until notifications are consumed.
    pub fn take_dropped(&mut self) -> Option<media::DroppedSample> {
        let reported = self.dropped.pop_front();
        if reported.is_some() && self.fatal.is_none() {
            let _ = self.reap_dropped();
        }
        reported
    }

    fn budget_rejected_id(&mut self) -> Result<u64, Error> {
        let mut id = 0;
        // SAFETY: single-owner native decoder and writable local ID.
        if unsafe { rosemary_h264_last_budget_id(self.handle.as_ptr(), &mut id) } != 1 {
            return Err(self.latch(Error::MissingIdentity));
        }
        Ok(id)
    }

    fn mark_budget_drop(&mut self, id: u64) -> Result<(), Error> {
        // A failed get_buffer2 allocation cannot produce a picture. The
        // callback identified its own access unit; complete it immediately
        // rather than waiting for FFmpeg to release decoder packet refs.
        if self.dropped.len() >= MAX_QUEUED_DROPS {
            return Err(self.latch(Error::MissingIdentity));
        }
        if self.pending.remove(&id).is_none() {
            return Err(self.latch(Error::MissingIdentity));
        }
        self.forget_token(id);
        self.dropped.push_back(media::DroppedSample {
            id,
            error: Some(media_error(Error::SourceFrameTooLarge)),
        });
        Ok(())
    }

    fn accepted_send_error(&self) -> Error {
        // SAFETY: the single-owner decoder handle is live and the preceding
        // native submission recorded its last libavcodec result.
        Error::Codec(unsafe { rosemary_h264_last_send_error(self.handle.as_ptr()) })
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

    fn id_in_flight(&self, id: u64) -> bool {
        self.pending.contains_key(&id)
            || self.available.iter().any(|frame| frame.id == id)
            || self.overflow.as_ref().is_some_and(|frame| frame.id == id)
            || self.dropped.iter().any(|drop| drop.id == id)
    }

    fn validate_input(&self, id: u64, pts: Option<i64>, dts: Option<i64>) -> Result<(), Error> {
        if self.id_in_flight(id) {
            return Err(Error::DuplicateIdentity);
        }
        if pts == Some(i64::MIN) || dts == Some(i64::MIN) {
            return Err(Error::InvalidTimestamp);
        }
        Ok(())
    }

    fn check_fatal(&self) -> Result<(), Error> {
        self.fatal.map_or(Ok(()), Err)
    }

    fn select_api(&mut self, api: DecoderApi) -> Result<(), Error> {
        match self.api {
            Some(active) if active != api => Err(Error::MixedDecoderApis),
            _ => {
                self.api = Some(api);
                Ok(())
            }
        }
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
            let timing = self.pending.remove(&id).expect("known pending identity");
            self.forget_token(id);
            self.dropped.push_back(media::DroppedSample {
                id,
                error: timing.decode_error.map(media_error),
            });
        }
        Ok(())
    }

    // An identified picture may be malformed without invalidating FFmpeg's
    // decoding state. Consume that ID once and expose the error via the drop
    // channel, not as a fatal VideoDecoder error.
    fn process_view(
        &mut self,
        view: &FrameView,
        native_error: Option<Error>,
    ) -> Result<Option<Frame>, Error> {
        if view.identity_valid == 0 {
            return Err(self.latch(Error::MissingIdentity));
        }
        if !self.pending.contains_key(&view.id) {
            return Err(self.latch(Error::MissingIdentity));
        }
        let converted = if frame_is_corrupt(view) {
            Err(Error::CorruptFrame)
        } else {
            native_error.map_or_else(|| copy_frame(view), Err)
        };
        let timing = self
            .pending
            .remove(&view.id)
            .expect("checked pending identity");
        self.forget_token(view.id);
        match converted {
            Ok(mut frame) => {
                frame.source_pts = timing.pts;
                frame.source_dts = timing.dts;
                frame.duration = timing.duration;
                Ok(Some(frame))
            }
            Err(error) => {
                self.dropped.push_back(media::DroppedSample {
                    id: view.id,
                    error: Some(media_error(error)),
                });
                Ok(None)
            }
        }
    }

    fn receive_one(&mut self) -> Result<Option<Frame>, Error> {
        self.check_fatal()?;
        loop {
            if self.dropped.len() >= MAX_QUEUED_DROPS {
                return Ok(None);
            }
            let mut view = FrameView::default();
            // SAFETY: decoder handle and output view remain valid for this call.
            match unsafe { rosemary_h264_receive(self.handle.as_ptr(), &mut view) } {
                0 => {
                    if let Some(frame) = self.process_view(&view, None)? {
                        return Ok(Some(frame));
                    }
                    // An identified picture was dropped; continue decoding.
                }
                1 => {
                    self.reap_dropped()?;
                    return Ok(None);
                }
                2 => {
                    self.eof = true;
                    self.reap_dropped()?;
                    return Ok(None);
                }
                4 => {
                    // The guarded FFmpeg allocation recorded the picture's
                    // own opaque ID, which may differ from the last input.
                    if view.identity_valid == 0 {
                        return Err(self.latch(Error::MissingIdentity));
                    }
                    self.mark_budget_drop(view.id)?;
                    self.reap_dropped()?;
                    return Ok(None);
                }
                NATIVE_PICTURE_REJECTED if view.identity_valid != 0 => {
                    self.process_view(&view, Some(Error::InvalidImageLayout))?;
                }
                NATIVE_PICTURE_REJECTED => return Err(self.latch(Error::MissingIdentity)),
                negative => return Err(self.latch(Error::Codec(negative))),
            }
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
        self.overflow.is_none()
            && self.available.len() < MAX_QUEUED_FRAMES
            && self.dropped.len() < MAX_QUEUED_DROPS
            && self
                .queued_bytes
                .saturating_add(self.last_frame_bytes.max(1))
                <= MAX_QUEUED_BYTES
    }

    // One unexpected large output is retained in a separate bounded slot.
    // It must follow every older queued frame, so it cannot enter the FIFO
    // until the consumer drains it; never reinterpret this as fatal layout.
    fn queue_frame(&mut self, frame: Frame) -> Result<(), Error> {
        let size = frame
            .planes
            .iter()
            .try_fold(0usize, |sum, plane| sum.checked_add(plane.data.len()))
            .ok_or_else(|| self.latch(Error::InvalidImageLayout))?;
        if size > MAX_OVERFLOW_BYTES {
            self.dropped.push_back(media::DroppedSample {
                id: frame.id,
                error: Some(media_error(Error::InvalidImageLayout)),
            });
            return Ok(());
        }
        self.last_frame_bytes = size;
        let frame = CodecFrame::from(frame);
        if self.overflow.is_some() {
            return Err(self.latch(Error::InvalidImageLayout));
        }
        if self.available.len() >= MAX_QUEUED_FRAMES
            || self
                .queued_bytes
                .checked_add(size)
                .is_none_or(|total| total > MAX_QUEUED_BYTES)
        {
            self.overflow = Some(frame);
        } else {
            self.queued_bytes += size;
            self.available.push_back(frame);
        }
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
        if let Some(frame) = self.overflow.take() {
            return Ok(Some(frame));
        }
        while let Some(frame) = self.receive_one()? {
            let size = frame.planes.iter().map(|p| p.data.len()).sum::<usize>();
            if size <= MAX_OVERFLOW_BYTES {
                return Ok(Some(CodecFrame::from(frame)));
            }
            self.dropped.push_back(media::DroppedSample {
                id: frame.id,
                error: Some(media_error(Error::InvalidImageLayout)),
            });
        }
        Ok(None)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: unique handle was allocated by rosemary_h264_new.
        unsafe { rosemary_h264_free(self.handle.as_ptr()) };
    }
}

fn normalized_pixel_aspect(num: i32, den: i32) -> (u32, u32) {
    if num <= 0 || den <= 0 {
        return (1, 1);
    }
    let (mut a, mut b) = (num as u32, den as u32);
    let (n, d) = (a, b);
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    (n / a, d / a)
}

fn repeat_first_field(flags: i32, repeat_pict: i32) -> bool {
    // FFmpeg repeat_pict=2/4 on progressive frames means frame doubling or
    // tripling, not a repeated interlaced field (not expressible in FrameLayout).
    flags & (1 << 3) != 0 && repeat_pict == 1
}

fn frame_is_corrupt(view: &FrameView) -> bool {
    // AV_FRAME_FLAG_CORRUPT and FF_DECODE_ERROR_* respectively.
    view.flags & AV_FRAME_FLAG_CORRUPT != 0 || view.decode_error_flags != 0
}

fn copy_frame(view: &FrameView) -> Result<Frame, Error> {
    if frame_is_corrupt(view) {
        return Err(Error::CorruptFrame);
    }
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
    let expected_bytes = dimensions
        .iter()
        .try_fold(0usize, |sum, (row, count)| {
            sum.checked_add(row.checked_mul(*count)?)
        })
        .ok_or(Error::InvalidImageLayout)?;
    if expected_bytes > MAX_OVERFLOW_BYTES {
        return Err(Error::InvalidImageLayout);
    }
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
        pixel_aspect_ratio: normalized_pixel_aspect(view.aspect_num, view.aspect_den),
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
        repeat_first_field: repeat_first_field(view.flags, view.repeat_pict),
        one_field: false,
        matrix: match view.colorspace {
            0 => ColorMatrix::Rgb,
            1 => ColorMatrix::Bt709,
            4 => ColorMatrix::Fcc,
            5 | 6 => ColorMatrix::Bt601,
            7 => ColorMatrix::Smpte240m,
            9 => ColorMatrix::Bt2020,
            10 => ColorMatrix::Bt2020Cl,
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
        | Error::InvalidTimestamp
        | Error::MixedDecoderApis => media::Error::InvalidInput(error.to_string()),
        Error::Backpressure => media::Error::Backpressure,
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
        self.select_api(DecoderApi::Media).map_err(media_error)?;
        if self.finished {
            return Err(media::Error::EndOfStream);
        }
        // Input validation precedes backpressure and never consumes the packet.
        self.validate_input(packet.id, packet.pts_ns, packet.dts_ns)
            .map_err(media_error)?;
        let expanded_bytes = avcc_annex_b_len(packet.bytes, self.length).map_err(media_error)?;
        if self.needs_headers {
            checked_packet_size(self.headers.len(), expanded_bytes).map_err(media_error)?;
        }
        if !self.queue_has_room() {
            return Err(media::Error::Backpressure);
        }
        self.pump_queued().map_err(media_error)?;
        if !self.queue_has_room() {
            return Err(media::Error::Backpressure);
        }
        let bytes = avcc_to_annex_b(packet.bytes, self.length).map_err(media_error)?;
        let prefixed;
        let raw = if self.needs_headers {
            prefixed = prepend_headers(&self.headers, &bytes).map_err(media_error)?;
            prefixed.as_slice()
        } else {
            &bytes
        };
        match self.send(Some(raw), packet.pts_ns, packet.dts_ns, packet.id) {
            status @ (0 | 3 | 4 | 5) => {
                let decode_error = (status == 3).then(|| self.accepted_send_error());
                self.pending.insert(
                    packet.id,
                    SourceTiming {
                        pts: packet.pts_ns,
                        dts: packet.dts_ns,
                        duration: packet.duration_ns,
                        decode_error,
                    },
                );
                self.needs_headers = false;
                if status == 4 {
                    let origin = self.budget_rejected_id().map_err(media_error)?;
                    self.mark_budget_drop(origin).map_err(media_error)?;
                }
                if status == 5 {
                    let error = self.accepted_send_error();
                    return Err(media_error(self.latch(error)));
                }
            }
            1 => {
                self.pump_queued().map_err(media_error)?;
                return Err(media::Error::Backpressure);
            }
            err => return Err(media_error(self.latch(Error::Codec(err)))),
        }
        self.pump_queued().map_err(media_error)?;
        Ok(())
    }

    fn pull(&mut self, _timeout: Duration) -> Result<Option<Self::Frame>, media::Error> {
        self.check_fatal().map_err(media_error)?;
        self.select_api(DecoderApi::Media).map_err(media_error)?;
        self.pull_queued().map_err(media_error)
    }

    fn take_dropped(&mut self) -> Option<media::DroppedSample> {
        Decoder::take_dropped(self)
    }

    fn end_stream(&mut self) -> Result<(), media::Error> {
        self.check_fatal().map_err(media_error)?;
        self.select_api(DecoderApi::Media).map_err(media_error)?;
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
        self.select_api(DecoderApi::Media).map_err(media_error)?;
        if !self.finished
            || !self.available.is_empty()
            || self.overflow.is_some()
            || !self.legacy_ready.is_empty()
        {
            return Ok(false);
        }
        if !self.eof
            && let Some(frame) = self.receive_one().map_err(media_error)?
        {
            self.queue_frame(frame).map_err(media_error)?;
            return Ok(false);
        }
        if self.eof && !self.pending.is_empty() {
            self.reap_dropped().map_err(media_error)?;
        }
        Ok(self.eof
            && self.available.is_empty()
            && self.overflow.is_none()
            && self.legacy_ready.is_empty()
            && self.pending.is_empty()
            && self.dropped.is_empty())
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
    fn conversion_allocation_failure_is_fatal_after_frame_identity() {
        let stream = include_bytes!("../tests/fixtures/blue_high444_32x32.h264");
        for stage in [TEST_SWS_CONTEXT_FAILURE, TEST_CONVERTED_FRAME_FAILURE] {
            let mut decoder = decoder_for_fixture(stream);
            // SAFETY: local native decoder; the hook fires only after the
            // picture has an opaque sample identity in the conversion path.
            unsafe { rosemary_h264_test_fail_converter_allocation(decoder.handle.as_ptr(), stage) };
            let outcome = decoder
                .submit_annex_b(stream, None, None)
                .and_then(|_| decoder.finish().map(|_| ()));
            assert_eq!(outcome.unwrap_err(), Error::Codec(-12), "stage {stage}");
            assert_eq!(decoder.fatal, Some(Error::Codec(-12)));
            assert!(
                decoder.pending.contains_key(&0),
                "consumed picture identity lost"
            );
            assert!(
                decoder.take_dropped().is_none(),
                "fatal ENOMEM became a sample drop"
            );
            assert_eq!(
                decoder.submit_annex_b(stream, None, None).unwrap_err(),
                Error::Codec(-12)
            );
            assert_eq!(decoder.finish().unwrap_err(), Error::Codec(-12));
        }
    }

    #[test]
    fn media_conversion_allocation_failure_is_fatal() {
        use mediabackend::VideoDecoder;
        let stream = include_bytes!("../tests/fixtures/blue_high444_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut packet = Vec::new();
        for nal in nals_from_fixture(stream) {
            packet.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            packet.extend_from_slice(nal);
        }
        // SAFETY: native decoder is exclusively owned by the test.
        unsafe {
            rosemary_h264_test_fail_converter_allocation(
                decoder.handle.as_ptr(),
                TEST_CONVERTED_FRAME_FAILURE,
            )
        };
        let sample = AccessUnit {
            id: 733,
            bytes: &packet,
            pts_ns: None,
            dts_ns: None,
            duration_ns: None,
        };
        let result = decoder.submit(sample).and_then(|_| {
            decoder.end_stream()?;
            decoder.pull(Duration::ZERO).map(|_| ())
        });
        assert!(
            matches!(result, Err(media::Error::Decode(ref message)) if message.contains("Codec(-12)"))
        );
        assert_eq!(decoder.fatal, Some(Error::Codec(-12)));
        assert!(decoder.pending.contains_key(&733));
        assert!(decoder.take_dropped().is_none());
    }

    #[test]
    fn rgb_gbrp_h264_conversion_reports_ycbcr_matrix_and_red_pixels() {
        // Encoded with libx264rgb from pure red RGB24, as AVCOL_SPC_RGB GBRP.
        let stream = include_bytes!("../tests/fixtures/rgb_gbrp_red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut frames = decoder.submit_annex_b(stream, Some(11), None).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 1);
        let frame = &frames[0];
        assert_eq!(frame.format, PixelFormat::I420);
        assert_eq!(frame.layout.matrix, ColorMatrix::Bt601);
        assert_eq!(frame.layout.full_range, Some(true));
        assert_eq!((frame.width, frame.height), (32, 32));
        // Full-range BT.601 red is approximately Y=76, U=85, V=255.
        // Check all three planes so that matrix metadata cannot mask a
        // broken RGB-to-YUV swscale conversion.
        assert!(frame.planes[0].data.iter().all(|&y| (72..=85).contains(&y)));
        assert!(frame.planes[1].data.iter().all(|&u| (79..=93).contains(&u)));
        assert!(
            frame.planes[2]
                .data
                .iter()
                .all(|&v| (245..=255).contains(&v))
        );
        assert!(decoder.take_dropped().is_none());
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
                        assert!(!decoder.id_in_flight(id));
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
        // FFmpeg can return INVALIDDATA *after* accepting an internal packet
        // reference. The caller must not retry: it is completed at release.
        decoder
            .submit(AccessUnit {
                id: 0,
                bytes: &units[0],
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            })
            .unwrap();
        assert!(decoder.fatal.is_none());
        let mut decoded = Vec::new();
        let mut dropped = Vec::new();
        let mut had_live_drop = false;
        let mut invalid_packet_completed = false;
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
                if sample.id == 0 {
                    assert!(
                        matches!(sample.error, Some(media::Error::Decode(ref message)) if message.contains("Codec("))
                    );
                    invalid_packet_completed = true;
                }
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
            if sample.id == 0 {
                assert!(
                    matches!(sample.error, Some(media::Error::Decode(ref message)) if message.contains("Codec("))
                );
                invalid_packet_completed = true;
            }
            dropped.push(sample.id);
        }
        assert!(invalid_packet_completed);
        assert!(decoder.is_drained().unwrap());
        assert_eq!(decoded.len(), 40);
        assert_eq!(dropped.len(), 20);
        assert!(decoded.iter().all(|&id| id >= 20));
        assert!(dropped.iter().all(|&id| id < 20));
        decoded.extend(dropped);
        decoded.sort_unstable();
        assert_eq!(decoded, (0..60).collect::<Vec<_>>());
    }

    #[test]
    fn real_sps_resolution_growth_preserves_overflow_picture() {
        use mediabackend::{VideoDecoder, VideoFrame};
        let stream = include_bytes!("../tests/fixtures/resolution_1024_to_4096.h264");
        let sets = nals_from_fixture(stream)
            .into_iter()
            .filter(|nal| matches!(nal[0] & 0x1f, 7 | 8))
            .take(2)
            .collect::<Vec<_>>();
        assert_eq!(sets.len(), 2);
        let mut decoder = Decoder::new(&sets, NalLength::Four).unwrap();
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
        assert_eq!(units.len(), 31);
        for (id, bytes) in units.iter().enumerate() {
            decoder
                .submit(AccessUnit {
                    id: id as u64,
                    bytes,
                    pts_ns: None,
                    dts_ns: None,
                    duration_ns: None,
                })
                .unwrap();
        }
        assert_eq!(decoder.available.len(), 30);
        assert_eq!(decoder.queued_bytes, 30 * 1024 * 1024 * 3 / 2);
        assert!(
            decoder.overflow.is_some(),
            "larger decoded frame was not retained"
        );
        assert_eq!(
            decoder.overflow.as_ref().unwrap().bytes.len(),
            4096 * 4096 * 3 / 2
        );
        assert!(matches!(
            decoder.end_stream(),
            Err(media::Error::Backpressure)
        ));
        assert!(decoder.fatal.is_none());
        let mut ids = Vec::new();
        for _ in 0..30 {
            let frame = decoder.pull(Duration::ZERO).unwrap().unwrap();
            assert_eq!((frame.layout().width, frame.layout().height), (1024, 1024));
            ids.push(frame.id());
        }
        let frame = decoder.pull(Duration::ZERO).unwrap().unwrap();
        assert_eq!((frame.layout().width, frame.layout().height), (4096, 4096));
        ids.push(frame.id());
        assert!(decoder.overflow.is_none());
        decoder.end_stream().unwrap();
        while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
            ids.push(frame.id());
        }
        assert!(decoder.is_drained().unwrap());
        assert!(decoder.take_dropped().is_none());
        assert_eq!(ids, (0..31).collect::<Vec<_>>());
    }

    #[test]
    fn identified_bad_crop_is_reported_without_terminating_the_decoder() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        decoder.pending.insert(
            50,
            SourceTiming {
                pts: None,
                dts: None,
                duration: None,
                decode_error: None,
            },
        );
        let y = [128_u8; 4];
        let chroma = [128_u8; 1];
        let view = FrameView {
            id: 50,
            identity_valid: 1,
            width: 2,
            height: 2,
            format: 0,
            planes: [y.as_ptr(), chroma.as_ptr(), chroma.as_ptr()],
            strides: [2, 1, 1],
            crop_left: 2,
            ..FrameView::default()
        };
        assert!(decoder.process_view(&view, None).unwrap().is_none());
        let error = decoder.take_dropped().unwrap();
        assert_eq!(error.id, 50);
        assert!(matches!(error.error, Some(media::Error::Decode(_))));
        assert!(decoder.fatal.is_none());
        // Native pixel-conversion failures also retain the originating ID.
        decoder.pending.insert(
            51,
            SourceTiming {
                pts: None,
                dts: None,
                duration: None,
                decode_error: None,
            },
        );
        let bad_native = FrameView {
            id: 51,
            identity_valid: 1,
            ..FrameView::default()
        };
        assert!(
            decoder
                .process_view(&bad_native, Some(Error::InvalidImageLayout))
                .unwrap()
                .is_none()
        );
        let native_drop = decoder.take_dropped().unwrap();
        assert_eq!(native_drop.id, 51);
        assert!(matches!(native_drop.error, Some(media::Error::Decode(_))));
        assert!(decoder.fatal.is_none());
        let mut frames = decoder
            .submit_with_id(stream, 49, None, None, None)
            .unwrap();
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].id, 49);
    }

    #[test]
    fn sample_aspect_ratio_is_normalized() {
        assert_eq!(normalized_pixel_aspect(2, 2), (1, 1));
        assert_eq!(normalized_pixel_aspect(20, 30), (2, 3));
        assert_eq!(normalized_pixel_aspect(65535, 21845), (3, 1));
        assert_eq!(normalized_pixel_aspect(0, 10), (1, 1));
        assert_eq!(normalized_pixel_aspect(-1, 3), (1, 1));
        assert_eq!(normalized_pixel_aspect(1, 0), (1, 1));
    }

    #[test]
    fn malformed_input_is_rejected_before_backpressure() {
        use mediabackend::VideoDecoder;
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        decoder.queued_bytes = MAX_QUEUED_BYTES;
        let invalid = [0, 0, 0, 2, 0x65];
        assert!(matches!(
            decoder.submit(AccessUnit {
                id: 9,
                bytes: &invalid,
                pts_ns: None,
                dts_ns: None,
                duration_ns: None
            }),
            Err(media::Error::InvalidInput(_))
        ));
        assert!(matches!(
            decoder.submit(AccessUnit {
                id: 9,
                bytes: &[0, 0, 0, 1, 0x65],
                pts_ns: Some(i64::MIN),
                dts_ns: None,
                duration_ns: None
            }),
            Err(media::Error::InvalidInput(_))
        ));
        assert!(matches!(
            decoder.submit(AccessUnit {
                id: 9,
                bytes: &[0, 0, 0, 1, 0x65],
                pts_ns: None,
                dts_ns: None,
                duration_ns: None
            }),
            Err(media::Error::Backpressure)
        ));
        assert!(!decoder.id_in_flight(9));
        decoder.queued_bytes = 0;
        assert!(decoder.fatal.is_none());
    }

    #[test]
    fn avcc_structure_and_expansion_are_validated_before_queue_admission() {
        use mediabackend::VideoDecoder;
        let nals = [0x67, 0x42, 0x80, 0x68, 0x01, 0x65, 0x90];
        let sample = [
            0, 0, 0, 3, nals[0], nals[1], nals[2], 0, 0, 0, 2, nals[3], nals[4], 0, 0, 0, 2,
            nals[5], nals[6],
        ];
        let length = avcc_annex_b_len(&sample, NalLength::Four).unwrap();
        assert_eq!(length, sample.len());
        assert_eq!(
            avcc_to_annex_b(&sample, NalLength::Four).unwrap().len(),
            length
        );
        for malformed in [
            &sample[..sample.len() - 1],
            &[0, 0, 0, 2, 0x80, 0x01][..],
            &[0, 0, 0, 0][..],
        ] {
            assert_eq!(
                avcc_annex_b_len(malformed, NalLength::Four),
                Err(Error::InvalidPacket)
            );
            assert_eq!(
                avcc_to_annex_b(malformed, NalLength::Four),
                Err(Error::InvalidPacket)
            );
        }
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        decoder.queued_bytes = MAX_QUEUED_BYTES;
        // A multi-megabyte valid compressed sample must be rejected at the
        // admission boundary without expanding it into a second allocation.
        let mut big = vec![0u8; 4 + 4 * 1024 * 1024];
        big[..4].copy_from_slice(&(4 * 1024 * 1024u32).to_be_bytes());
        big[4] = 0x65;
        assert_eq!(avcc_annex_b_len(&big, NalLength::Four), Ok(big.len()));
        assert!(matches!(
            decoder.submit(AccessUnit {
                id: 55,
                bytes: &big,
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            }),
            Err(media::Error::Backpressure)
        ));
        assert!(!decoder.id_in_flight(55));
        assert!(matches!(
            decoder.submit(AccessUnit {
                id: 55,
                bytes: &big[..big.len() - 1],
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            }),
            Err(media::Error::InvalidInput(_))
        ));
        assert!(!decoder.id_in_flight(55));
    }

    #[test]
    fn oversized_sps_drops_exact_origin_and_recovers_at_next_idr() {
        let huge = include_bytes!("../tests/fixtures/oversize_sps_8192.h264");
        let small = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(huge);
        // 8192² I420 would require 96 MiB: the picture must never reach the
        // default allocator, yet its token and drop reason must remain exact.
        assert!(decoder.submit_annex_b(huge, None, None).unwrap().is_empty());
        let drops = std::iter::from_fn(|| decoder.take_dropped()).collect::<Vec<_>>();
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].id, 0);
        assert!(
            matches!(drops[0].error, Some(media::Error::Decode(ref message))
            if message.contains("SourceFrameTooLarge"))
        );
        assert_eq!(
            unsafe { rosemary_h264_native_peak_bytes(decoder.handle.as_ptr()) },
            0
        );
        assert!(decoder.fatal.is_none());
        let mut recovered = decoder.submit_annex_b(small, None, None).unwrap();
        recovered.extend(decoder.finish().unwrap());
        assert_eq!(
            recovered.iter().map(|frame| frame.id).collect::<Vec<_>>(),
            vec![1]
        );
        assert!(decoder.take_dropped().is_none());
        assert!(decoder.pending.is_empty());
    }

    #[test]
    fn oversized_sps_trait_completes_one_budget_error_and_recovers() {
        use mediabackend::VideoDecoder;
        let huge = include_bytes!("../tests/fixtures/oversize_sps_8192.h264");
        let small = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(huge);
        let to_avcc = |bytes: &[u8]| {
            let mut packet = Vec::new();
            for nal in nals_from_fixture(bytes) {
                packet.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                packet.extend_from_slice(nal);
            }
            packet
        };
        let huge_packet = to_avcc(huge);
        decoder
            .submit(AccessUnit {
                id: 700,
                bytes: &huge_packet,
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            })
            .unwrap();
        assert!(decoder.pull(Duration::ZERO).unwrap().is_none());
        let budget = decoder.take_dropped().unwrap();
        assert_eq!(budget.id, 700);
        assert!(
            matches!(budget.error, Some(media::Error::Decode(ref message))
            if message.contains("SourceFrameTooLarge"))
        );
        let small_packet = to_avcc(small);
        decoder
            .submit(AccessUnit {
                id: 701,
                bytes: &small_packet,
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            })
            .unwrap();
        decoder.end_stream().unwrap();
        let frame = decoder.pull(Duration::ZERO).unwrap().unwrap();
        assert_eq!(frame.id, 701);
        assert!(decoder.pull(Duration::ZERO).unwrap().is_none());
        assert!(decoder.take_dropped().is_none());
        assert!(decoder.is_drained().unwrap());
    }

    #[test]
    fn oversized_sps_deferred_receive_attributes_drop_and_recovers() {
        let small = include_bytes!("../tests/fixtures/red_32x32.h264");
        let huge = include_bytes!("../tests/fixtures/oversize_sps_8192.h264");
        let mut decoder = decoder_for_fixture(small);
        // Leave a valid picture buffered in FFmpeg. The 8192² AU is accepted
        // but the native allocation happens only after the older picture is
        // pulled, not when that oversized AU is submitted.
        assert_eq!(decoder.send(Some(small), None, None, 101), 0);
        decoder.pending.insert(
            101,
            SourceTiming {
                pts: None,
                dts: None,
                duration: None,
                decode_error: None,
            },
        );
        assert_eq!(decoder.send(Some(huge), None, None, 202), 0);
        decoder.pending.insert(
            202,
            SourceTiming {
                pts: None,
                dts: None,
                duration: None,
                decode_error: None,
            },
        );
        assert_eq!(decoder.receive_one().unwrap().unwrap().id, 101);
        assert!(decoder.receive_one().unwrap().is_none());
        let drops = std::iter::from_fn(|| decoder.take_dropped()).collect::<Vec<_>>();
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].id, 202);
        assert!(
            matches!(drops[0].error, Some(media::Error::Decode(ref message))
            if message.contains("SourceFrameTooLarge"))
        );
        assert!(decoder.pending.is_empty());
        assert!(decoder.fatal.is_none());
        let peak = unsafe { rosemary_h264_native_peak_bytes(decoder.handle.as_ptr()) };
        assert!(peak < 64 * 1024 * 1024);
        let mut recovered = decoder
            .submit_with_id(small, 303, None, None, None)
            .unwrap();
        recovered.extend(decoder.finish().unwrap());
        assert_eq!(
            recovered.iter().map(|frame| frame.id).collect::<Vec<_>>(),
            vec![303]
        );
        assert!(decoder.take_dropped().is_none());
    }

    #[test]
    fn native_get_buffer_allocation_failure_during_send_is_fatal() {
        let small = include_bytes!("../tests/fixtures/red_32x32.h264");
        for stage in 1..=3 {
            let mut decoder = decoder_for_fixture(small);
            // SAFETY: the fault is local to this decoder and is triggered from
            // inside the get_buffer2 callback, before H.264 masks ENOMEM.
            unsafe { rosemary_h264_test_fail_native_allocation(decoder.handle.as_ptr(), stage) };
            assert_eq!(
                decoder.submit_annex_b(small, None, None).unwrap_err(),
                Error::Codec(-12),
                "allocator stage {stage}"
            );
            assert_eq!(decoder.fatal, Some(Error::Codec(-12)));
            assert!(decoder.pending.contains_key(&0));
            assert!(decoder.take_dropped().is_none());
            assert_eq!(
                decoder.submit_annex_b(small, None, None).unwrap_err(),
                Error::Codec(-12)
            );
        }
    }

    #[test]
    fn native_get_buffer_allocation_failure_in_trait_is_fatal() {
        use mediabackend::VideoDecoder;
        let small = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(small);
        let mut packet = Vec::new();
        for nal in nals_from_fixture(small) {
            packet.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            packet.extend_from_slice(nal);
        }
        // SAFETY: injected ENOMEM occurs in native callback, not after send.
        unsafe { rosemary_h264_test_fail_native_allocation(decoder.handle.as_ptr(), 1) };
        let error = decoder
            .submit(AccessUnit {
                id: 707,
                bytes: &packet,
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            })
            .unwrap_err();
        assert!(
            matches!(error, media::Error::Decode(ref message) if message.contains("Codec(-12)"))
        );
        assert_eq!(decoder.fatal, Some(Error::Codec(-12)));
        assert!(decoder.pending.contains_key(&707));
        assert!(decoder.take_dropped().is_none());
        assert!(matches!(decoder.is_drained(), Err(media::Error::Decode(_))));
    }

    #[test]
    fn native_get_buffer_allocation_failure_during_receive_is_fatal() {
        let small = include_bytes!("../tests/fixtures/red_32x32.h264");
        for stage in 1..=3 {
            let mut decoder = decoder_for_fixture(small);
            assert_eq!(decoder.send(Some(small), None, None, 101), 0);
            decoder.pending.insert(
                101,
                SourceTiming {
                    pts: None,
                    dts: None,
                    duration: None,
                    decode_error: None,
                },
            );
            assert_eq!(decoder.send(Some(small), None, None, 202), 0);
            decoder.pending.insert(
                202,
                SourceTiming {
                    pts: None,
                    dts: None,
                    duration: None,
                    decode_error: None,
                },
            );
            assert_eq!(decoder.receive_one().unwrap().unwrap().id, 101);
            // SAFETY: callback failure occurs when FFmpeg later decodes ID
            // 202, not while its compressed packet was accepted.
            unsafe { rosemary_h264_test_fail_native_allocation(decoder.handle.as_ptr(), stage) };
            assert_eq!(
                decoder.receive_one().unwrap_err(),
                Error::Codec(-12),
                "allocator stage {stage}"
            );
            assert_eq!(decoder.fatal, Some(Error::Codec(-12)));
            assert!(decoder.pending.contains_key(&202));
            assert!(!decoder.pending.contains_key(&101));
            assert!(decoder.take_dropped().is_none());
            assert_eq!(decoder.finish().unwrap_err(), Error::Codec(-12));
        }
    }

    #[test]
    fn nonbudget_native_enomem_after_acceptance_is_fatal_with_owned_identity() {
        use mediabackend::VideoDecoder;
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut packet = Vec::new();
        for nal in nals_from_fixture(stream) {
            packet.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            packet.extend_from_slice(nal);
        }
        // SAFETY: only this test owns the decoder and the hook forces a
        // failure after FFmpeg has successfully accepted a native packet.
        unsafe { rosemary_h264_test_send_enomem_after_accept(decoder.handle.as_ptr()) };
        let failed = decoder.submit(AccessUnit {
            id: 901,
            bytes: &packet,
            pts_ns: None,
            dts_ns: None,
            duration_ns: None,
        });
        assert!(
            matches!(failed, Err(media::Error::Decode(ref message)) if message.contains("Codec(-12)"))
        );
        assert_eq!(decoder.fatal, Some(Error::Codec(-12)));
        assert!(decoder.pending.contains_key(&901));
        // The observer has not been forgotten merely because the session
        // became fatal: input was consumed and cannot be submitted again.
        assert_eq!(
            unsafe { rosemary_h264_token_released(decoder.handle.as_ptr(), 901) },
            0
        );
        assert!(matches!(
            decoder.submit(AccessUnit {
                id: 902,
                bytes: &packet,
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            }),
            Err(media::Error::Decode(_))
        ));
        assert!(matches!(
            decoder.pull(Duration::ZERO),
            Err(media::Error::Decode(_))
        ));
        assert!(matches!(decoder.end_stream(), Err(media::Error::Decode(_))));
    }

    #[test]
    fn legacy_consumed_enomem_remains_fatal() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        // SAFETY: inject after native acceptance in this private test decoder.
        unsafe { rosemary_h264_test_send_enomem_after_accept(decoder.handle.as_ptr()) };
        assert_eq!(
            decoder.submit_annex_b(stream, None, None).unwrap_err(),
            Error::Codec(-12)
        );
        assert_eq!(decoder.fatal, Some(Error::Codec(-12)));
        assert!(decoder.pending.contains_key(&0));
        assert!(matches!(
            decoder.submit_annex_b(stream, None, None),
            Err(Error::Codec(-12))
        ));
    }

    #[test]
    fn budget_failure_on_native_receive_marks_the_original_packet_id() {
        let small = include_bytes!("../tests/fixtures/red_32x32.h264");
        let big = include_bytes!("../tests/fixtures/high444_10bit_4096x4096.h264");
        let mut decoder = decoder_for_fixture(small);
        // Keep FFmpeg's first decoded picture in its native buffer_frame so
        // the next packet is admitted *without* decoding the large picture.
        assert_eq!(decoder.send(Some(small), None, None, 101), 0);
        decoder.pending.insert(
            101,
            SourceTiming {
                pts: None,
                dts: None,
                duration: None,
                decode_error: None,
            },
        );
        assert_eq!(decoder.send(Some(big), None, None, 202), 0);
        decoder.pending.insert(
            202,
            SourceTiming {
                pts: None,
                dts: None,
                duration: None,
                decode_error: None,
            },
        );
        let mut view = FrameView::default();
        // SAFETY: native decoder and frame view are live for each call.
        assert_eq!(
            unsafe { rosemary_h264_receive(decoder.handle.as_ptr(), &mut view) },
            0
        );
        assert_eq!(view.id, 101);
        decoder.process_view(&view, None).unwrap().unwrap();
        let mut refused = FrameView::default();
        assert_eq!(
            unsafe { rosemary_h264_receive(decoder.handle.as_ptr(), &mut refused) },
            4
        );
        assert_eq!(refused.identity_valid, 1);
        assert_eq!(refused.id, 202);
        decoder.mark_budget_drop(refused.id).unwrap();
        // A rejected native allocation can be completed immediately even if
        // FFmpeg retains its packet reference until a later IDR or EOS.
        let drop = decoder
            .take_dropped()
            .expect("rejected picture must complete");
        assert_eq!(drop.id, 202);
        assert!(
            matches!(drop.error, Some(media::Error::Decode(ref msg)) if msg.contains("SourceFrameTooLarge"))
        );
        assert!(decoder.pending.is_empty());
        assert!(decoder.fatal.is_none());
    }

    #[test]
    fn source_budget_drops_high444_picture_and_recovers_at_next_idr() {
        use mediabackend::{VideoDecoder, VideoFrame};
        let big = include_bytes!("../tests/fixtures/high444_10bit_4096x4096.h264");
        let small = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(big);
        let to_avcc = |data: &[u8]| {
            let mut packet = Vec::new();
            for nal in nals_from_fixture(data) {
                packet.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                packet.extend_from_slice(nal);
            }
            packet
        };
        decoder
            .submit(AccessUnit {
                id: 777,
                bytes: &to_avcc(big),
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            })
            .unwrap();
        assert!(decoder.fatal.is_none());
        // A later low-resolution IDR and its new SPS/PPS must be admissible
        // without a fresh decoder session or any unbounded picture allocation.
        decoder
            .submit(AccessUnit {
                id: 778,
                bytes: &to_avcc(small),
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            })
            .unwrap();
        decoder.end_stream().unwrap();
        let mut frames = Vec::new();
        while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
            frames.push(frame);
        }
        let mut dropped = Vec::new();
        while let Some(sample) = decoder.take_dropped() {
            dropped.push(sample);
        }
        assert!(decoder.is_drained().unwrap());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].id(), 778);
        assert_eq!(
            (frames[0].layout().width, frames[0].layout().height),
            (32, 32)
        );
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].id, 777);
        assert!(
            matches!(dropped[0].error, Some(media::Error::Decode(ref message)) if message.contains("SourceFrameTooLarge"))
        );
        // SAFETY: the native decoder handle is live.
        assert!(
            unsafe { rosemary_h264_native_peak_bytes(decoder.handle.as_ptr()) } < 64 * 1024 * 1024
        );
    }

    #[test]
    fn corrupt_flags_and_decode_error_flags_always_reject_frame_copy() {
        for (flags, decode_error_flags) in [
            (AV_FRAME_FLAG_CORRUPT, 0),
            (0, 1), // FF_DECODE_ERROR_INVALID_BITSTREAM
            (0, 2), // FF_DECODE_ERROR_MISSING_REFERENCE
            (0, 4), // FF_DECODE_ERROR_CONCEALMENT_ACTIVE
        ] {
            let view = FrameView {
                flags,
                decode_error_flags,
                ..FrameView::default()
            };
            assert_eq!(copy_frame(&view).unwrap_err(), Error::CorruptFrame);
        }
    }

    #[test]
    fn damaged_reference_slice_is_not_delivered_as_clean_picture() {
        use mediabackend::{VideoDecoder, VideoFrame};
        // P slice truncated mid-macroblock; FFmpeg identifies corrupted
        // frames while still returning recoverable later pictures.
        let stream = include_bytes!("../tests/fixtures/corrupt_p_slice_8_64x64.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut units = Vec::new();
        let mut unit = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 31 == 9 && !unit.is_empty() {
                units.push(std::mem::take(&mut unit));
            }
            unit.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            unit.extend_from_slice(nal);
        }
        if !unit.is_empty() {
            units.push(unit);
        }
        assert_eq!(units.len(), 8);
        let mut presented = Vec::new();
        let mut dropped = Vec::new();
        for (i, bytes) in units.iter().enumerate() {
            decoder
                .submit(AccessUnit {
                    id: i as u64,
                    bytes,
                    pts_ns: None,
                    dts_ns: None,
                    duration_ns: None,
                })
                .unwrap();
            while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
                presented.push(frame.id());
            }
            while let Some(sample) = decoder.take_dropped() {
                dropped.push(sample);
            }
        }
        decoder.end_stream().unwrap();
        while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
            presented.push(frame.id());
        }
        while let Some(sample) = decoder.take_dropped() {
            dropped.push(sample);
        }
        assert!(decoder.is_drained().unwrap());
        assert!(!presented.is_empty());
        assert!(
            dropped.iter().any(|sample| matches!(sample.error,
                Some(media::Error::Decode(ref message)) if message.contains("CorruptFrame"))),
            "FFmpeg marked no output picture corrupt"
        );
        let mut ids = presented;
        ids.extend(dropped.iter().map(|s| s.id));
        ids.sort_unstable();
        assert_eq!(ids, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn interlaced_high_chroma_is_rejected_without_mixing_fields() {
        let stream = include_bytes!("../tests/fixtures/interlaced_422_64x64.h264");
        let sets = nals_from_fixture(stream)
            .into_iter()
            .filter(|nal| matches!(nal[0] & 31, 7 | 8))
            .take(2)
            .collect::<Vec<_>>();
        let mut decoder = Decoder::new(&sets, NalLength::Four).unwrap();
        let mut units = Vec::new();
        let mut unit = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 31 == 9 && !unit.is_empty() {
                units.push(std::mem::take(&mut unit));
            }
            unit.extend_from_slice(&[0, 0, 0, 1]);
            unit.extend_from_slice(nal);
        }
        if !unit.is_empty() {
            units.push(unit);
        }
        assert_eq!(units.len(), 2);
        let mut frames = Vec::new();
        for unit in &units {
            frames.extend(decoder.submit_annex_b(unit, None, None).unwrap());
        }
        frames.extend(decoder.finish().unwrap());
        assert!(frames.is_empty());
        let dropped = (0..2)
            .map(|_| decoder.take_dropped().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(dropped.iter().map(|x| x.id).collect::<Vec<_>>(), vec![0, 1]);
        assert!(dropped.iter().all(|x| x.error.is_some()));
        assert!(decoder.take_dropped().is_none());
        assert!(decoder.fatal.is_none());
    }

    #[test]
    fn legacy_avcc_validates_before_drop_queue_admission() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        for id in 10000..10000 + MAX_QUEUED_DROPS as u64 {
            decoder
                .dropped
                .push_back(media::DroppedSample { id, error: None });
        }
        let mut large = vec![0; 2 * 1024 * 1024 + 4];
        large[..4].copy_from_slice(&(2 * 1024 * 1024_u32).to_be_bytes());
        large[4] = 0x65;
        assert_eq!(
            decoder.submit_avcc(&large, None, None).unwrap_err(),
            Error::Backpressure
        );
        assert_eq!(
            decoder
                .submit_avcc(&large[..large.len() - 1], None, None)
                .unwrap_err(),
            Error::InvalidPacket
        );
        assert_eq!(decoder.next_id, 0);
        assert!(decoder.pending.is_empty());
    }

    #[test]
    fn native_source_budget_counts_format_and_reference_buffers() {
        use mediabackend::VideoDecoder;
        let stream = include_bytes!("../tests/fixtures/resolution_1024_to_4096.h264");
        let sets = nals_from_fixture(stream)
            .into_iter()
            .filter(|nal| matches!(nal[0] & 31, 7 | 8))
            .take(2)
            .collect::<Vec<_>>();
        let mut decoder = Decoder::new(&sets, NalLength::Four).unwrap();
        let mut access_units = Vec::new();
        let mut unit = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 31 == 9 && !unit.is_empty() {
                access_units.push(std::mem::take(&mut unit));
            }
            unit.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            unit.extend_from_slice(nal);
        }
        if !unit.is_empty() {
            access_units.push(unit);
        }
        for (i, bytes) in access_units.iter().enumerate() {
            decoder
                .submit(AccessUnit {
                    id: i as u64,
                    bytes,
                    pts_ns: None,
                    dts_ns: None,
                    duration_ns: None,
                })
                .unwrap();
        }
        // SAFETY: decoder and native context remain live.
        let peak = unsafe { rosemary_h264_native_peak_bytes(decoder.handle.as_ptr()) };
        let live = unsafe { rosemary_h264_native_live_bytes(decoder.handle.as_ptr()) };
        assert!(peak >= 4096 * 4096 * 3 / 2, "native guard wasn't exercised");
        assert!(peak <= 256 * 1024 * 1024);
        assert!(live <= peak);
    }

    #[test]
    fn eos_with_nearly_full_notification_queue_waits_for_pending_identities() {
        use mediabackend::VideoDecoder;
        let stream = include_bytes!("../tests/fixtures/missing_initial_idr_64x64.h264");
        let sets = nals_from_fixture(stream)
            .into_iter()
            .filter(|nal| matches!(nal[0] & 31, 7 | 8))
            .take(2)
            .collect::<Vec<_>>();
        let mut decoder = Decoder::new(&sets, NalLength::Four).unwrap();
        let mut units = Vec::new();
        let mut unit = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 31 == 9 && !unit.is_empty() {
                units.push(std::mem::take(&mut unit));
            }
            unit.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            unit.extend_from_slice(nal);
        }
        if !unit.is_empty() {
            units.push(unit);
        }
        assert_eq!(units.len(), 60);
        let mut decoded = Vec::new();
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
                decoded.push(frame.id);
            }
        }
        let mut known = Vec::new();
        while let Some(drop) = decoder.take_dropped() {
            known.push(drop.id);
        }
        assert_eq!(known.len(), 19);
        // Saturate the completion notification channel after actual decoding;
        // the remaining terminal state must never be reported as drained.
        for id in 10000..10000 + MAX_QUEUED_DROPS as u64 - 1 {
            decoder
                .dropped
                .push_back(media::DroppedSample { id, error: None });
        }
        // The final real accepted pictures remain inside FFmpeg's reorder
        // buffer at EOF; no synthetic native identity is introduced.
        assert!(!decoder.pending.is_empty());
        decoder.end_stream().unwrap();
        assert!(!decoder.is_drained().unwrap());
        let mut final_frames = Vec::new();
        // Drain the deliberately full completion channel while pulling the
        // final two real frames from FFmpeg in their original output order.
        for _ in 0..MAX_QUEUED_DROPS {
            while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
                final_frames.push(frame.id);
            }
            if decoder.pending.is_empty() {
                break;
            }
            assert!(decoder.take_dropped().is_some());
        }
        assert!(decoder.pending.is_empty());
        assert!(!final_frames.is_empty());
        decoded.extend(final_frames);
        assert!(!decoder.is_drained().unwrap());
        let mut notifications = Vec::new();
        while let Some(drop) = decoder.take_dropped() {
            notifications.push(drop.id);
        }
        assert_eq!(notifications.len(), MAX_QUEUED_DROPS - 1);
        assert!(decoder.is_drained().unwrap());
        assert!(notifications.iter().all(|&id| id >= 10000));
        known.extend(decoded);
        known.sort_unstable();
        assert_eq!(known, (1..60).collect::<Vec<_>>());
    }

    #[test]
    fn progressive_frame_repetition_is_not_field_repetition() {
        // In H.264 picture-timing SEI, pic_struct=7/8 maps to FFmpeg's
        // progressive repeat_pict=2/4 (frame doubling/tripling).
        assert!(!repeat_first_field(0, 2));
        assert!(!repeat_first_field(0, 4));
        assert!(!repeat_first_field(0, 1));
        assert!(repeat_first_field(1 << 3, 1));
        assert!(!repeat_first_field(1 << 3, 2));
        assert!(!repeat_first_field(1 << 3, 0));
    }

    #[test]
    fn progressive_h264_picture_timing_sei_does_not_set_repeat_field() {
        // The eight actual H.264 access units contain pic_struct=7 SEI,
        // which FFmpeg reports as progressive repeat_pict=2 (frame doubling).
        let stream = include_bytes!("../tests/fixtures/pic_timing_frame_doubling_64x64.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut frames = Vec::new();
        let mut unit = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 31 == 9 && !unit.is_empty() {
                frames.extend(decoder.submit_annex_b(&unit, None, None).unwrap());
                unit.clear();
            }
            unit.extend_from_slice(&[0, 0, 0, 1]);
            unit.extend_from_slice(nal);
        }
        if !unit.is_empty() {
            frames.extend(decoder.submit_annex_b(&unit, None, None).unwrap());
        }
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 8);
        assert!(frames.iter().all(|frame| {
            frame.layout.interlace_mode == FrameInterlaceMode::Progressive
                && !frame.layout.repeat_first_field
        }));
    }

    #[test]
    fn full_width_opaque_id_is_valid_without_a_sentinel() {
        use mediabackend::{VideoDecoder, VideoFrame};
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut avcc = Vec::new();
        for nal in nals_from_fixture(stream) {
            avcc.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            avcc.extend_from_slice(nal);
        }
        decoder
            .submit(AccessUnit {
                id: u64::MAX,
                bytes: &avcc,
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            })
            .unwrap();
        decoder.end_stream().unwrap();
        assert_eq!(
            decoder.pull(Duration::ZERO).unwrap().unwrap().id(),
            u64::MAX
        );
        assert!(decoder.is_drained().unwrap());
        assert!(decoder.take_dropped().is_none());
    }

    #[test]
    fn legacy_backpressure_can_be_drained_and_retried() {
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        for id in 0..MAX_QUEUED_DROPS {
            decoder.dropped.push_back(media::DroppedSample {
                id: id as u64 + 1000,
                error: None,
            });
        }
        assert!(matches!(
            decoder.submit_annex_b(stream, None, None),
            Err(Error::Backpressure)
        ));
        assert!(decoder.fatal.is_none());
        assert!(matches!(decoder.finish(), Err(Error::Backpressure)));
        while decoder.take_dropped().is_some() {}
        let mut frames = decoder.submit_annex_b(stream, None, None).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 1);
        assert!(decoder.fatal.is_none());
    }

    #[test]
    fn legacy_missing_idr_discard_pressure_does_not_poison_session() {
        let stream = include_bytes!("../tests/fixtures/missing_initial_idr_64x64.h264");
        let sets = nals_from_fixture(stream)
            .into_iter()
            .filter(|nal| matches!(nal[0] & 31, 7 | 8))
            .take(2)
            .collect::<Vec<_>>();
        let mut decoder = Decoder::new(&sets, NalLength::Four).unwrap();
        let mut units = Vec::new();
        let mut unit = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 31 == 9 && !unit.is_empty() {
                units.push(std::mem::take(&mut unit));
            }
            unit.extend_from_slice(&[0, 0, 0, 1]);
            unit.extend_from_slice(nal);
        }
        if !unit.is_empty() {
            units.push(unit);
        }
        assert_eq!(units.len(), 60);
        let mut frames = Vec::new();
        let mut drops = Vec::new();
        for (i, bytes) in units.iter().enumerate().skip(1) {
            if i == 30 {
                for n in 0..MAX_QUEUED_DROPS {
                    decoder.dropped.push_back(media::DroppedSample {
                        id: 10000 + n as u64,
                        error: None,
                    });
                }
                assert!(matches!(
                    decoder.submit_annex_b(bytes, None, None),
                    Err(Error::Backpressure)
                ));
                assert!(decoder.fatal.is_none());
                while let Some(completion) = decoder.take_dropped() {
                    assert!(completion.id >= 10000);
                }
            }
            frames.extend(decoder.submit_annex_b(bytes, None, None).unwrap());
            while let Some(completion) = decoder.take_dropped() {
                drops.push(completion.id);
            }
        }
        frames.extend(decoder.finish().unwrap());
        while let Some(completion) = decoder.take_dropped() {
            drops.push(completion.id);
        }
        assert!(decoder.fatal.is_none());
        assert_eq!(frames.len(), 40);
        assert_eq!(drops.len(), 19);
        let mut ids: Vec<_> = frames
            .into_iter()
            .map(|frame| frame.id)
            .chain(drops)
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..59).collect::<Vec<_>>());
    }

    #[test]
    fn legacy_finish_backpressure_cannot_silently_drain_through_media_api() {
        use mediabackend::VideoDecoder;
        let stream = include_bytes!("../tests/fixtures/bframes_8_64x64.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut unit = Vec::new();
        let mut frames = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 31 == 9 && !unit.is_empty() {
                frames.extend(decoder.submit_annex_b(&unit, None, None).unwrap());
                unit.clear();
            }
            unit.extend_from_slice(&[0, 0, 0, 1]);
            unit.extend_from_slice(nal);
        }
        if !unit.is_empty() {
            frames.extend(decoder.submit_annex_b(&unit, None, None).unwrap());
        }
        let saved = frames.pop().expect("fixture must emit output before EOS");
        decoder.legacy_ready.push(saved);
        for id in 10000..10000 + MAX_QUEUED_DROPS as u64 {
            decoder
                .dropped
                .push_back(media::DroppedSample { id, error: None });
        }
        assert_eq!(decoder.finish().unwrap_err(), Error::Backpressure);
        for _ in 0..MAX_QUEUED_DROPS {
            assert!(decoder.take_dropped().is_some());
        }
        assert!(
            matches!(decoder.is_drained(), Err(media::Error::InvalidInput(ref reason)) if reason.contains("MixedDecoderApis"))
        );
        assert!(matches!(
            decoder.pull(Duration::ZERO),
            Err(media::Error::InvalidInput(_))
        ));
        assert!(matches!(
            decoder.end_stream(),
            Err(media::Error::InvalidInput(_))
        ));
        assert!(!decoder.legacy_ready.is_empty());
        frames.extend(decoder.finish().unwrap());
        assert!(decoder.legacy_ready.is_empty());
        frames.sort_by_key(|frame| frame.id);
        assert_eq!(
            frames.iter().map(|f| f.id).collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn trait_decoder_rejects_switching_to_legacy_submission() {
        use mediabackend::VideoDecoder;
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut packet = Vec::new();
        for nal in nals_from_fixture(stream) {
            packet.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            packet.extend_from_slice(nal);
        }
        decoder
            .submit(AccessUnit {
                id: 88,
                bytes: &packet,
                pts_ns: None,
                dts_ns: None,
                duration_ns: None,
            })
            .unwrap();
        assert_eq!(
            decoder.submit_annex_b(stream, None, None).unwrap_err(),
            Error::MixedDecoderApis
        );
        assert_eq!(decoder.finish().unwrap_err(), Error::MixedDecoderApis);
        decoder.end_stream().unwrap();
        assert_eq!(decoder.pull(Duration::ZERO).unwrap().unwrap().id, 88);
        assert!(decoder.is_drained().unwrap());
    }

    #[test]
    fn legacy_finish_retries_after_full_drop_queue_without_stranding_frames() {
        let stream = include_bytes!("../tests/fixtures/bframes_8_64x64.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut units = Vec::new();
        let mut unit = Vec::new();
        for nal in nals_from_fixture(stream) {
            if nal[0] & 31 == 9 && !unit.is_empty() {
                units.push(std::mem::take(&mut unit));
            }
            unit.extend_from_slice(&[0, 0, 0, 1]);
            unit.extend_from_slice(nal);
        }
        if !unit.is_empty() {
            units.push(unit);
        }
        assert_eq!(units.len(), 8);
        let mut ids = Vec::new();
        for unit in &units {
            ids.extend(
                decoder
                    .submit_annex_b(unit, None, None)
                    .unwrap()
                    .into_iter()
                    .map(|f| f.id),
            );
        }
        assert!(ids.len() < 8, "fixture must have delayed reordered output");
        // Send the native flush, but do not receive its remaining pictures.
        // This is exactly the state after finish() successfully sent EOS but
        // a frame-local failure filled the notification queue during receive.
        assert_eq!(decoder.send(None, None, None, 0), 0);
        decoder.finished = true;
        for id in 10000..10000 + (MAX_QUEUED_DROPS - 1) as u64 {
            decoder
                .dropped
                .push_back(media::DroppedSample { id, error: None });
        }
        decoder.pending.insert(
            9000,
            SourceTiming {
                pts: None,
                dts: None,
                duration: None,
                decode_error: None,
            },
        );
        let y = [0_u8; 4];
        let c = [0_u8; 1];
        let malformed = FrameView {
            id: 9000,
            identity_valid: 1,
            width: 2,
            height: 2,
            format: 0,
            crop_left: 2,
            planes: [y.as_ptr(), c.as_ptr(), c.as_ptr()],
            strides: [2, 1, 1],
            ..FrameView::default()
        };
        assert!(decoder.process_view(&malformed, None).unwrap().is_none());
        assert_eq!(decoder.dropped.len(), MAX_QUEUED_DROPS);
        assert!(!decoder.eof);
        assert!(matches!(decoder.finish(), Err(Error::Backpressure)));
        assert!(!decoder.eof);
        let dropped = (0..MAX_QUEUED_DROPS)
            .map(|_| decoder.take_dropped().unwrap())
            .collect::<Vec<_>>();
        assert!(
            dropped
                .iter()
                .any(|sample| sample.id == 9000 && sample.error.is_some())
        );
        ids.extend(decoder.finish().unwrap().into_iter().map(|f| f.id));
        assert!(decoder.eof);
        assert!(decoder.finish().unwrap().is_empty());
        ids.sort_unstable();
        assert_eq!(ids, (0..8).collect::<Vec<_>>());
        assert!(decoder.pending.is_empty());
        assert!(decoder.fatal.is_none());
    }

    #[test]
    fn bt2020_constant_luminance_is_preserved_or_explicitly_rejected() {
        let direct = include_bytes!("../tests/fixtures/i420_bt2020cl_32x32.h264");
        let mut decoder = decoder_for_fixture(direct);
        let mut frames = decoder.submit_annex_b(direct, None, None).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].layout.matrix, ColorMatrix::Bt2020Cl);
        assert!(decoder.take_dropped().is_none());

        let unsupported = include_bytes!("../tests/fixtures/high444_bt2020cl_32x32.h264");
        let mut decoder = decoder_for_fixture(unsupported);
        let mut frames = decoder.submit_annex_b(unsupported, None, None).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert!(frames.is_empty());
        let dropped = decoder.take_dropped().unwrap();
        assert_eq!(dropped.id, 0);
        assert!(matches!(dropped.error, Some(media::Error::Decode(_))));
        assert!(decoder.fatal.is_none());
        assert!(decoder.take_dropped().is_none());
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
    fn unordered_ids_are_accepted_and_inflight_duplicates_rejected() {
        use mediabackend::{VideoDecoder, VideoFrame};
        let stream = include_bytes!("../tests/fixtures/red_32x32.h264");
        let mut decoder = decoder_for_fixture(stream);
        let mut bytes = Vec::new();
        for nal in nals_from_fixture(stream) {
            bytes.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            bytes.extend_from_slice(nal);
        }
        let au = |id| AccessUnit {
            id,
            bytes: &bytes,
            pts_ns: None,
            dts_ns: None,
            duration_ns: None,
        };
        decoder.submit(au(200)).unwrap();
        assert!(matches!(
            decoder.submit(au(200)),
            Err(media::Error::InvalidInput(_))
        ));
        decoder.submit(au(100)).unwrap();
        assert!(matches!(
            decoder.submit(au(100)),
            Err(media::Error::InvalidInput(_))
        ));
        decoder.end_stream().unwrap();
        let mut ids = Vec::new();
        while let Some(frame) = decoder.pull(Duration::ZERO).unwrap() {
            ids.push(frame.id());
        }
        ids.sort_unstable();
        assert_eq!(ids, [100, 200]);
        assert!(decoder.is_drained().unwrap());
        assert!(decoder.pending.is_empty());
        assert!(!decoder.id_in_flight(200));
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
