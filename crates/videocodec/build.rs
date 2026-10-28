use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// Versioned upstream source, not a moving release branch.
const FFMPEG_REF: &str = "n9.0.2";
const FFMPEG_COMMIT: &str = "946fcce07b6dcd0331c8cc609192aeff5e1924f8";
// All 10,424 non-.git files of FFmpeg's commit-pinned codeload archive, in
// lexical relative-path order; hash each `path + NUL + SHA256(file)`, then hash
// the concatenation. A source tree without Git history must match exactly.
const FFMPEG_TREE_SHA256: &str = "ac162b65212c0ef5f5a45aa11874036ca2bcd63a964a7815c234c8010b352a19";

fn command(cmd: &mut Command) {
    let result = cmd.status().expect("failed to spawn build tool");
    assert!(result.success(), "build tool returned {result}");
}

fn tree_hash(root: &Path) -> String {
    fn visit(dir: &Path, root: &Path, names: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("cannot enumerate FFmpeg source") {
            let path = entry.expect("invalid FFmpeg directory entry").path();
            if path.file_name().is_some_and(|name| name == ".git") {
                continue;
            }
            let ty = fs::symlink_metadata(&path)
                .expect("cannot stat FFmpeg source")
                .file_type();
            assert!(!ty.is_symlink(), "FFmpeg source must not contain symlinks");
            if ty.is_dir() {
                visit(&path, root, names);
            } else if ty.is_file() {
                names.push(path.strip_prefix(root).unwrap().to_owned());
            }
        }
    }
    let mut names = Vec::new();
    visit(root, root, &mut names);
    names.sort();
    let mut digest = Sha256::new();
    for relative in names {
        let name = relative.to_str().expect("non-UTF8 FFmpeg filename");
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update(Sha256::digest(
            fs::read(root.join(relative)).expect("cannot hash FFmpeg file"),
        ));
    }
    format!("{:x}", digest.finalize())
}

fn source_dir(out: &Path) -> (PathBuf, String) {
    let source = if let Some(prefetched) = env::var_os("ROSEMARY_FFMPEG_SOURCE") {
        PathBuf::from(prefetched)
    } else {
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
        src
    };
    assert!(source.join("configure").is_file(), "invalid FFmpeg source");
    let fingerprint = if source.join(".git").exists() {
        let revision = Command::new("git")
            .arg("-C")
            .arg(&source)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("cannot read FFmpeg revision");
        assert!(revision.status.success());
        let head = String::from_utf8_lossy(&revision.stdout);
        assert_eq!(
            head.trim(),
            FFMPEG_COMMIT,
            "incorrect FFmpeg source revision"
        );
        let status = Command::new("git")
            .arg("-C")
            .arg(&source)
            .args(["status", "--porcelain", "--untracked-files=all"])
            .output()
            .expect("cannot verify FFmpeg working tree");
        assert!(
            status.status.success() && status.stdout.is_empty(),
            "modified FFmpeg source"
        );
        format!("git:{FFMPEG_COMMIT}")
    } else {
        let digest = tree_hash(&source);
        assert_eq!(digest, FFMPEG_TREE_SHA256, "unverified FFmpeg source tree");
        format!("tree:{digest}")
    };
    println!("cargo:rerun-if-changed={}", source.display());
    (source, fingerprint)
}

fn main() {
    println!("cargo:rerun-if-changed=src/ffmpeg_shim.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROSEMARY_FFMPEG_SOURCE");
    let host = env::var("HOST").expect("HOST");
    let target = env::var("TARGET").expect("TARGET");
    assert_eq!(
        host, target,
        "cross-compiling the bundled FFmpeg decoder is unsupported: HOST={host}, TARGET={target}; build on the target platform instead"
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let (source, fingerprint) = source_dir(&out);
    let install = out.join("ffmpeg-install");
    let archive = install.join("lib/libavcodec.a");
    let stamp = out.join("ffmpeg-source-recipe.sha256");
    let mut recipe = Sha256::new();
    recipe.update(include_bytes!("build.rs"));
    recipe.update(target.as_bytes());
    recipe.update(
        source
            .canonicalize()
            .expect("FFmpeg source path")
            .to_string_lossy()
            .as_bytes(),
    );
    recipe.update(fingerprint.as_bytes());
    let recipe = format!("{:x}", recipe.finalize());
    let stale = fs::read_to_string(&stamp).ok().as_deref() != Some(recipe.as_str());
    if stale {
        let _ = fs::remove_dir_all(&install);
        let _ = fs::remove_dir_all(out.join("ffmpeg-build"));
        let _ = fs::remove_file(&stamp);
    }
    if stale || !archive.exists() || !install.join("lib/libswscale.a").exists() {
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
        fs::write(&stamp, &recipe).expect("cannot persist FFmpeg source stamp");
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
