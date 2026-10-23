use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// Versioned upstream source, not a moving release branch.
const FFMPEG_REF: &str = "n9.0.2";
const FFMPEG_COMMIT: &str = "946fcce07b6dcd0331c8cc609192aeff5e1924f8";

fn command(cmd: &mut Command) {
    let result = cmd.status().expect("failed to spawn build tool");
    assert!(result.success(), "build tool returned {result}");
}

fn source_dir(out: &Path) -> PathBuf {
    // Nix and offline builders can supply an already-fetched, read-only tree.
    // Its version/revision must be pinned by the caller's source fetch rule.
    if let Some(prefetched) = env::var_os("ROSEMARY_FFMPEG_SOURCE") {
        let source = PathBuf::from(prefetched);
        assert!(source.join("configure").is_file(), "invalid FFmpeg source");
        return source;
    }
    let src = out.join("ffmpeg-source");
    if !src.join("configure").exists() {
        command(
            Command::new("git")
                .arg("clone")
                .args(["--depth", "1", "--branch", FFMPEG_REF, "--single-branch"])
                .arg("https://github.com/FFmpeg/FFmpeg.git")
                .arg(&src),
        );
    }
    let actual = Command::new("git")
        .arg("-C")
        .arg(&src)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("cannot verify bundled FFmpeg revision");
    assert!(actual.status.success());
    assert_eq!(
        String::from_utf8_lossy(&actual.stdout).trim(),
        FFMPEG_COMMIT
    );
    src
}

fn main() {
    println!("cargo:rerun-if-changed=src/ffmpeg_shim.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROSEMARY_FFMPEG_SOURCE");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let install = out.join("ffmpeg-install");
    let archive = install.join("lib/libavcodec.a");
    if !archive.exists() || !install.join("lib/libswscale.a").exists() {
        let source = source_dir(&out);
        let build = out.join("ffmpeg-build");
        fs::create_dir_all(&build).expect("cannot create FFmpeg build directory");
        let mut configure = Command::new(source.join("configure"));
        configure
            .current_dir(&build)
            .arg(format!("--prefix={}", install.display()))
            .args([
                "--disable-programs",
                "--disable-doc",
                "--disable-autodetect",
                "--disable-network",
                "--disable-everything",
                "--enable-decoder=h264",
                "--enable-parser=h264",
                "--disable-avformat",
                "--disable-avdevice",
                "--disable-avfilter",
                "--enable-swscale",
                "--disable-swresample",
                "--enable-static",
                "--disable-shared",
                "--enable-pic",
            ]);
        // The reference decoder has C fallbacks where the optional external assembler
        // is unavailable. These builds are slower, but still portable.
        if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
            && Command::new("nasm").arg("--version").output().is_err()
        {
            configure.arg("--disable-x86asm");
        }
        command(&mut configure);
        let workers = std::thread::available_parallelism().map_or(2, |n| n.get());
        command(
            Command::new("make")
                .current_dir(&build)
                .arg(format!("-j{workers}")),
        );
        command(Command::new("make").current_dir(&build).arg("install"));
    }
    assert!(archive.exists(), "bundled libavcodec is absent");
    cc::Build::new()
        .file("src/ffmpeg_shim.c")
        .include(install.join("include"))
        .warnings(true)
        .compile("rosemary_video_codec_shim");
    println!(
        "cargo:rustc-link-search=native={}",
        install.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=avcodec");
    println!("cargo:rustc-link-lib=static=swscale");
    println!("cargo:rustc-link-lib=static=avutil");
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }
    // FFmpeg source is kept only under Cargo's target output, never in the repo.
}
