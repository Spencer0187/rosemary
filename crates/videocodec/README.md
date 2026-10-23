# Portable H.264 decoder backend

Rosemary embeds a software H.264 decoder behind a Rust API. It does **not** load
or require a system-installed FFmpeg library at runtime. The crate builds the
necessary FFmpeg libraries from the exact source revision identified in
`build.rs`, and links them statically.

Build prerequisites are a C compiler, GNU make, Git, and an internet connection
for the first source retrieval in a normal Cargo build. Offline/Nix builders
can prefetch that exact source revision and set `ROSEMARY_FFMPEG_SOURCE` to its
read-only directory before invoking Cargo. The external x86 assembler is optional; the
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
