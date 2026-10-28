# Portable H.264 decoder backend

Rosemary embeds a software H.264 decoder behind a Rust API. It does **not** load
or require a system-installed FFmpeg library at runtime. The crate builds the
necessary FFmpeg libraries from the exact source revision identified in
`build.rs`, and links them statically.

Build prerequisites are a C compiler, GNU make, Git, and an internet connection
for the first source retrieval in a normal Cargo build. Offline/Nix builders
can prefetch that exact source revision and set `ROSEMARY_FFMPEG_SOURCE` to its
read-only directory before invoking Cargo. A supplied directory must either be
an unmodified Git checkout at the pinned commit or exactly match the SHA-256
file manifest of the official pinned codeload archive. The archive fingerprint
is verified before cached static FFmpeg libraries are considered reusable.
A source path, fingerprint, target, or build-recipe change causes a full
out-of-tree FFmpeg rebuild rather than linking stale native archives.

The current bundled-FFmpeg builder supports **native builds only**: it rejects
`HOST != TARGET` explicitly rather than compiling FFmpeg for the host and the
Rust shim for another architecture. Cross-compilation requires an independent,
verified FFmpeg target-compiler configuration before that restriction is lifted.
The external x86 assembler is optional; the
source build uses FFmpeg's portable fallback if NASM is missing. The source tree
lives only in Cargo's build output, not the upstream repository.

The decoder API uses H.264 access units, SPS/PPS parameter sets, timestamps,
and **owned** decoded planes. It intentionally knows nothing about Darwin guest
memory or CoreMedia types. VideoToolbox is responsible for session state,
callback delivery, and converting the result into CoreVideo objects.

FFmpeg licensing and source-distribution obligations apply to binary packages
that statically link these libraries. Packagers must retain corresponding
source information and the applicable FFmpeg license notices. The build enables
neither FFmpeg's optional GPL components nor nonfree components.

The software fallback produces 8-bit I420 or NV12. Uncommon H.264 source
chroma formats and bit depths are converted to 8-bit I420 through libswscale.
This is a deliberate display-oriented output contract, not a lossless
representation of High 10/4:4:4 source precision. High-bit-depth native output,
HDR metadata, hardware decoding, and end-to-end guest playback belong to later
integration work.

The `mediabackend::VideoDecoder` implementation consumes AVCC access units
with stable `AccessUnit.id` values, emits owned contiguous `CodecFrame` buffers,
and exposes per-frame color matrix, primaries, transfer, chroma siting and range.
FFmpeg's packet opaque-reference propagation carries the ID even across
B-frame reordering and missing/duplicate timestamps. A retained observer
reference allows the decoder to report accepted access units through
`take_dropped` as soon as FFmpeg releases their identity, including during
normal streaming and missing-IDR recovery, without relying on PTS/order.
Terminal EOS reports any remaining unreturned IDs.

The trait implementation has an output limit of 32 owned frames and a 64 MiB
FIFO byte budget. A single separately bounded 64 MiB overflow-frame slot
preserves output when an SPS resolution change produces a picture larger than
the space available in the FIFO. No more input is accepted while that slot is
occupied; the older queued frames must be pulled before the overflow picture.
A picture individually exceeding the 64 MiB slot is reported as a frame-local
drop with an error. Full queues return recoverable `Backpressure` without
consuming the caller's access unit; after `pull`, retry the same ID.
The dropped-completion queue is separately bounded to 1024 entries; poll
`take_dropped` alongside `pull`. Delayed EOS output is pulled incrementally,
and successful `is_drained()` requires every accepted sample ID to be resolved
and every completion notification consumed. `pull` is synchronous and ignores
its timeout. An identified frame-local
layout/conversion error is reported as `DroppedSample.error`, while decoder
errors that invalidate the session or lose identity are latched as fatal.

The `AccessUnit.id` is an opaque sample identity: it need not be numerically
ordered, but the caller must provide distinct IDs for the life of a session.
The decoder checks collisions among in-flight outputs, pending inputs, and
unconsumed dropped notifications without retaining an unbounded set of all
historical IDs. Malformed inputs are validated before checking backpressure.
The legacy `Decoder::submit_annex_b` and `submit_avcc` methods remain available
with automatically assigned IDs and plane-based `Frame` output. When the
completion queue stalls, these methods and `finish()` return recoverable
`videocodec::Error::Backpressure`; consume notifications using the inherent
`Decoder::take_dropped()` and retry the same unaccepted input. The legacy
automatic-ID counter never emits the reserved exhaustion value, but the
`mediabackend` trait accepts the entire opaque `u64` identity domain.

The pinned FFmpeg H.264 decoder rejects SPS dimensions above 4096² pixels
before a picture allocation, and its guarded `get_buffer2` enforces a 64 MiB
source-frame request limit with a 256 MiB budget for live native reference
buffers. The source-format and bit depth determine these charges (not the
converted I420 bytes); single-frame decoding threads avoid multiplying the
reference budget by frame-thread count. Converted buffers and Rust's bounded
output FIFO are separate allocations. These safeguards limit decoded-picture
allocations, not compressed packet storage or all FFmpeg parser scratch memory.

FFmpeg `repeat_pict` represents frame doubling/tripling for progressive H.264
picture-timing SEI. Only interlaced first-field repetition maps to
`FrameLayout.repeat_first_field`; the shared layout does not represent
progressive frame repetition, which is left unset rather than misrepresented.

FFmpeg reserves `i64::MIN` for absent timestamps, so callers supplying that
exact number receive `InvalidTimestamp` rather than silent corruption.

The FFmpeg build-recipe fingerprint includes the resolved C compiler and
version, compiler arguments, detected assembler version, source provenance,
target, and native-build environment variables. Toolchain changes therefore
invalidate cached native FFmpeg archives before the shim is rebuilt.
