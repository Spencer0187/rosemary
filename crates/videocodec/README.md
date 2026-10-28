# Portable H.264 decoder backend

Rosemary embeds a software H.264 decoder behind a Rust API. It does **not** load
or require a system-installed FFmpeg library at runtime. The crate builds the
necessary FFmpeg libraries from the exact source revision identified in
`build.rs`, and links them statically.

Build prerequisites are a C compiler, GNU make, Git, `patch`, and an internet connection
for the first source retrieval in a normal Cargo build. Offline/Nix builders
can prefetch that exact source revision and set `ROSEMARY_FFMPEG_SOURCE` to its
read-only directory before invoking Cargo. A supplied directory must match the SHA-256
file manifest of the official pinned codeload archive **regardless of Git
history**. Git checkouts also require the exact pinned HEAD and clean tracked
files. Ignored generated files (such as `config.h` and codec lists) are included
in the content manifest: an extra or modified file is rejected before staging,
including if `git status` reports clean. The staged copy is checked against
that same manifest before the pinned source patch is applied. The content
fingerprint is checked before cached static FFmpeg libraries can be reused.
A source path, fingerprint, target, build-recipe or bundled FFmpeg patch
change causes a full out-of-tree rebuild rather than linking stale native
archives. After verifying the original pinned upstream source, the builder
copies it into Cargo's build output and applies the exact-version source patch
`patches/ffmpeg-h264-memory-guards.patch`. The original source is never
modified. Redistributors must also provide this patch alongside the
corresponding FFmpeg source and license information.

## Release integration gate

This crate is currently **standalone and unshipped**. The `rosemary` runtime
binary does not depend on `videocodec`. The root workspace explicitly lists
its other 36 crates as `default-members`, so a plain root `cargo build` or
`cargo test` does not fetch or build FFmpeg. Decoder tests are still selected
explicitly with `cargo test -p videocodec` (including `--test source_integrity`)
or as part of `cargo test --workspace`. A dedicated decoder CI job fetches
the immutable FFmpeg source before Cargo builds, checks default-member
selection, and runs all decoder tests and strict Clippy natively on Linux
x86-64 and macOS ARM64. The CI membership assertion requires every workspace
crate other than `videocodec` to remain part of default builds and tests. The existing Nix package selects
`--package rosemary --bin rosemary`, while the Arch recipe builds a prior
release. Adding this crate to the workspace does not enable guest VideoToolbox
or include a decoder in either release artifact.

**Before any shipped Rosemary binary depends on this crate**, the release
recipes must inject the checksum-verified pinned FFmpeg source, rather than
allowing `build.rs` to clone a Git repository inside a sandboxed Cargo build.
For Nix, make the pinned source a fixed-output derivation and export
`ROSEMARY_FFMPEG_SOURCE` into the decoder build. For Arch, prefetch the same
revision in `source=()`, verify it with `sha256sums`, and pass its unpacked path
through `ROSEMARY_FFMPEG_SOURCE`. Both recipes must also distribute the
corresponding modified FFmpeg source, checked-in source patch and required
license notices, and be verified from a clean offline/sandboxed build.
The local offline workspace test with a prepopulated FFmpeg source directory
does not establish that release packaging gate.

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
layout/unsupported-conversion error is reported as `DroppedSample.error`.
Unexpected native conversion failures, including swscale context creation and
converted-frame allocation `ENOMEM` after FFmpeg identified a picture, latch
fatal decoder failure without discarding the accepted sample identity.

Every submitted access unit also has a 1024-NAL upper bound, independent of
the 64 MiB compressed packet limit. FFmpeg allocates per-NAL parser descriptors
and skipped-byte position arrays that are not charged to native decoded-frame
buffers. Both AVCC and Annex B input are checked before FFmpeg sees the packet;
initially prepended SPS/PPS headers count toward the limit. AVCC validation
also counts embedded Annex B start codes inside unescaped NAL payloads, because
they become additional parser boundaries after conversion. An over-limit packet
returns `InvalidPacket` without consuming the sample identity or allocating
FFmpeg parser metadata.

The `AccessUnit.id` is an opaque sample identity: it need not be numerically
ordered, but the caller must provide distinct IDs for the life of a session.
The decoder checks collisions among in-flight outputs, pending inputs, and
unconsumed dropped notifications without retaining an unbounded set of all
historical IDs. AVCC framing, NAL headers and expanded packet size are
validated before checking backpressure, but Annex B output allocation and
copy occur only after the packet can enter the decoder, including in the
legacy AVCC API.
The legacy `Decoder::submit_annex_b`, `submit_avcc`, and `finish` methods
remain available with automatically assigned IDs and plane-based `Frame`
output. A session must use either the legacy API or the
`mediabackend::VideoDecoder` trait API; attempts to switch interfaces after
selection return `MixedDecoderApis` (`InvalidInput` through the trait).
`take_dropped` is a shared notification accessor. The trait's drain predicate
also checks retained legacy frames defensively, so a mixed interface cannot
report EOS with inaccessible frames. When the
completion queue stalls, these methods and `finish()` return recoverable
`videocodec::Error::Backpressure`; consume notifications using the inherent
`Decoder::take_dropped()` and retry the same unaccepted input.
The legacy `finish()` resumes FFmpeg's terminal receive loop after
notification backpressure; success requires a native EOF, not merely a
full completion queue. Previously received frames remain buffered for the
successful retry. The legacy
automatic-ID counter never emits the reserved exhaustion value, but the
`mediabackend` trait accepts the entire opaque `u64` identity domain.

The source patch rejects SPS dimensions exceeding 4096² pixels before
FFmpeg initializes dimension-dependent H.264 macroblock tables. The guarded
`get_buffer2` independently rejects such pictures before decoded-frame storage, alongside its 64 MiB source-frame
request limit and 256 MiB budget for live native reference buffers. FFmpeg's
`max_pixels` is not used as the 4096² guard because it rejects frames before
the callback receives their opaque source IDs. FFmpeg's independent generic
image-size and overflow validity checks remain enabled; inputs so extreme
that FFmpeg rejects them before invoking `get_buffer2` fail as native decoder
errors rather than receiving an invented picture identity. For attributable
oversize H.264 pictures, either the early SPS check or guarded frame-allocation
callback reports `SourceFrameTooLarge` using the actual packet identity,
including when decoding happens during deferred receive. A rejected SPS does
not by itself imply that the packet's picture was rejected: FFmpeg can recover
using later valid parameter sets or a previously decoded SPS. Native successful
frame output takes precedence over a stale metadata-rejection flag. If an
oversized SPS for one pending access-unit ID is observed while a different
reordered picture succeeds, the SPS ID retains a provisional
`SourceFrameTooLarge` error. That error is reported only if FFmpeg releases the
access unit without producing its frame; successful output for the same ID
supersedes the provisional error. Each accepted identity is completed exactly
once. The source-format and bit depth determine these charges (not the
converted I420 bytes); single-frame decoding threads avoid multiplying the
reference budget by frame-thread count. Converted buffers and Rust's bounded
output FIFO are separate allocations. These safeguards limit decoded-picture
allocations, not compressed packet storage or all FFmpeg parser scratch memory.

FFmpeg `repeat_pict` represents frame doubling/tripling for progressive H.264
picture-timing SEI. Only interlaced first-field repetition maps to
`FrameLayout.repeat_first_field`; the shared layout does not represent
progressive frame repetition, which is left unset rather than misrepresented.

A decode error from `avcodec_send_packet` does not imply a rejected
input: FFmpeg may have retained the opaque packet after accepting it.
`EAGAIN` and pre-call failures do not consume the input. Once FFmpeg may
have consumed it, `AVERROR_INVALIDDATA` is acknowledged as accepted;
its opaque ID is retained and completed as a dropped sample with a decode
error if no output picture arrives. Clients must not resubmit it. This
allows the same session to recover at the next IDR without losing ownership
of a corrupted access unit. Other negative results from the native send
call (including non-budget `ENOMEM` and unexpected decoder-state errors)
remain **fatal**, even when FFmpeg may have consumed the packet. Native
`get_buffer2` allocation errors are captured at the callback before H.264
can rewrite them to `AVERROR_INVALIDDATA`; a real host allocator failure
always latches the original fatal error, never a recoverable bitstream drop. Their
sample identity is retained for session teardown; they are never represented
as a successful recoverable submission.

Native source-frame admission is limited by the 4096² SPS-pixel bound,
64 MiB per-picture source allocation estimate and 256 MiB tracked live
reference allocation budget. A predictable frame-budget rejection is
reported as `SourceFrameTooLarge` on that sample's dropped completion,
not as a fatal decoder state. A later supported IDR may still recover.
When source allocation is refused, the guarded FFmpeg callback records
the **originating picture's** opaque sample ID. That picture is completed
immediately as a frame-local error, including when allocation occurs during
`avcodec_receive_frame` after a later packet was submitted. A missing or
unknown picture ID is fatal, not an error-free discard. Arbitrary FFmpeg
`ENOMEM` or unexpected decoder errors are not recoverable.

FFmpeg frames marked with `AV_FRAME_FLAG_CORRUPT` or nonzero
`decode_error_flags` are not returned as clean decoded output. Identified
corrupt pictures complete through `DroppedSample.error` (`CorruptFrame`),
consistent with the other media decoder's corrupt-frame discard policy.
This includes error-concealed output that FFmpeg could otherwise render.

Interlaced H.264 pictures needing a high-chroma or high-bit-depth swscale
conversion are rejected as identified frame-local errors: whole-picture
vertical resampling would mix chroma between fields. Native supported
8-bit 4:2:0 interlaced frames are left untouched. Field-aware conversion
is not implemented in this backend.

BT.2020 constant-luminance signals map to the distinct shared
`ColorMatrix::Bt2020Cl` instead of non-constant-luminance `Bt2020`.
Native I420/NV12 output retains its signaled metadata. Converting an
unsupported bit depth/chroma format using BT.2020 constant-luminance
coefficients fails for that specific picture, rather than silently
applying BT.601 conversion coefficients. For other uncommon input formats,
swscale's selected YCbCr coefficients are also checked against the signaled
matrix; unsupported matrices are rejected as frame-local conversion errors.
Decoded planar RGB (GBRP) H.264 uses an explicit BT.601 conversion, and
its I420 output reports BT.601 rather than the original RGB matrix.

Arch release packaging for this standalone decoder is deferred until a
Rosemary release actually includes `videocodec`. The existing PKGBUILD still
builds Rosemary v0.2.1 and is left unchanged by this decoder contribution.
For offline builds of this crate, `ROSEMARY_FFMPEG_SOURCE` accepts a pinned,
independently verified FFmpeg source tree.

FFmpeg reserves `i64::MIN` for absent timestamps, so callers supplying that
exact number receive `InvalidTimestamp` rather than silent corruption.

The FFmpeg build-recipe fingerprint includes the resolved C compiler and
version, compiler arguments, detected assembler version, source provenance,
target, and native-build environment variables. The resolved compiler flags
are also passed to FFmpeg's C and linker invocations, not only fingerprinted. Toolchain changes therefore
invalidate cached native FFmpeg archives before the shim is rebuilt.
