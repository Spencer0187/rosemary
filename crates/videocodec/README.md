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
B-frame reordering and missing/duplicate timestamps. Unreturned accepted
IDs are reported by `take_dropped` on terminal EOS; no premature drop is
inferred from output order. `pull` is synchronous and ignores its timeout.
The legacy `Decoder::submit_annex_b` and `submit_avcc` methods remain available
with automatically assigned IDs and plane-based `Frame` output.

FFmpeg reserves `i64::MIN` for absent timestamps, so callers supplying that
exact number receive `InvalidTimestamp` rather than silent corruption.
