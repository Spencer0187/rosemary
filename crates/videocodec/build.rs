use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

// Versioned upstream source, not a moving release branch.
const FFMPEG_REF: &str = "n9.0.2";
const FFMPEG_COMMIT: &str = "946fcce07b6dcd0331c8cc609192aeff5e1924f8";
// All 10,424 non-.git files in both the pinned Git checkout and codeload
// archive. Hash each `path + NUL + SHA256(file)` in lexical order, then hash
// the concatenation. Nothing outside this manifest may enter the build.
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

// A clean Git status is insufficient: generated FFmpeg headers, codec lists,
// and other files ignored by Git can still be selected by configure/make.
// Verify every staged file, including ignored files, regardless of origin.
fn verify_source(source: &Path, expected_commit: &str, expected_tree_hash: &str) -> String {
    assert!(source.join("configure").is_file(), "invalid FFmpeg source");
    if source.join(".git").exists() {
        let revision = Command::new("git")
            .arg("-C")
            .arg(source)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("cannot read FFmpeg revision");
        assert!(revision.status.success());
        let head = String::from_utf8_lossy(&revision.stdout);
        assert_eq!(
            head.trim(),
            expected_commit,
            "incorrect FFmpeg source revision"
        );
        let status = Command::new("git")
            .arg("-C")
            .arg(source)
            .args(["status", "--porcelain", "--untracked-files=all"])
            .output()
            .expect("cannot verify FFmpeg working tree");
        assert!(
            status.status.success() && status.stdout.is_empty(),
            "modified FFmpeg source"
        );
    }
    let digest = tree_hash(source);
    assert_eq!(digest, expected_tree_hash, "unverified FFmpeg source tree");
    format!("tree:{digest}")
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
    let fingerprint = verify_source(&source, FFMPEG_COMMIT, FFMPEG_TREE_SHA256);
    println!("cargo:rerun-if-changed={}", source.display());
    (source, fingerprint)
}

// Stage a patched, immutable-source-derived copy. Never patch user-supplied
// or cached pinned source in place; the verified input remains reproducible.
fn staged_ffmpeg_source(source: &Path, out: &Path, patch: &Path) -> PathBuf {
    fn copy_tree(from: &Path, to: &Path) {
        fs::create_dir_all(to).expect("cannot stage FFmpeg source directory");
        for entry in fs::read_dir(from).expect("cannot enumerate pinned FFmpeg") {
            let entry = entry.expect("invalid FFmpeg directory entry");
            let path = entry.path();
            if entry.file_name() == ".git" {
                continue;
            }
            let target = to.join(entry.file_name());
            let ty = entry.file_type().expect("cannot stat pinned FFmpeg");
            if ty.is_dir() {
                copy_tree(&path, &target);
            } else if ty.is_file() {
                fs::copy(path, target).expect("cannot copy pinned FFmpeg source");
            } else {
                panic!("unexpected file in verified FFmpeg source");
            }
        }
    }
    let staged = out.join("ffmpeg-patched-source");
    if staged.exists() {
        fs::remove_dir_all(&staged).expect("cannot replace stale FFmpeg stage");
    }
    copy_tree(source, &staged);
    // Detect changes between the original verification and staging before
    // applying the intentional, checked-in patch to the copied source.
    assert_eq!(
        tree_hash(&staged),
        FFMPEG_TREE_SHA256,
        "staged FFmpeg source does not match the verified input"
    );
    for file in [
        "libavutil/mem.c",
        "libavcodec/h264dec.c",
        "libavcodec/h264_slice.c",
        "libavcodec/h2645_parse.c",
        "libavcodec/h264_ps.c",
    ] {
        let path = staged.join(file);
        let mut permissions = fs::metadata(&path)
            .expect("FFmpeg stage metadata")
            .permissions();
        permissions.set_mode(permissions.mode() | 0o200);
        fs::set_permissions(path, permissions).expect("FFmpeg stage permissions");
    }
    command(
        Command::new("patch")
            .current_dir(&staged)
            .args(["--batch", "--fuzz=0", "-p1", "-i"])
            .arg(patch),
    );
    staged
}

fn main() {
    println!("cargo:rerun-if-changed=src/ffmpeg_shim.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=patches/ffmpeg-h264-memory-guards.patch");
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
    let shim_compiler = cc::Build::new().file("src/ffmpeg_shim.c").get_compiler();
    let compiler_path = shim_compiler.path().to_string_lossy().into_owned();
    let compiler_version = Command::new(shim_compiler.path())
        .arg("--version")
        .output()
        .expect("cannot identify native C compiler");
    assert!(
        compiler_version.status.success(),
        "native C compiler cannot report its version"
    );
    let nasm_version = Command::new("nasm")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success());
    let mut recipe = Sha256::new();
    recipe.update(include_bytes!("build.rs"));
    recipe.update(include_bytes!("patches/ffmpeg-h264-memory-guards.patch"));
    recipe.update(compiler_path.as_bytes());
    recipe.update(&compiler_version.stdout);
    recipe.update(&compiler_version.stderr);
    let compiler_cflags = shim_compiler
        .args()
        .iter()
        .map(|arg| {
            let arg = arg.to_str().expect("non-UTF8 native compiler argument");
            assert!(
                !arg.chars().any(char::is_whitespace),
                "FFmpeg cannot safely pass compiler arguments containing whitespace: {arg:?}"
            );
            arg
        })
        .collect::<Vec<_>>()
        .join(" ");
    recipe.update(compiler_cflags.as_bytes());
    if let Some(nasm) = &nasm_version {
        recipe.update(&nasm.stdout);
        recipe.update(&nasm.stderr);
    } else {
        recipe.update(b"no-nasm");
    }
    for key in [
        "CC",
        "CFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "AS",
        "ASFLAGS",
        "AR",
        "ARFLAGS",
        "NASM",
        "NASMFLAGS",
        "PATH",
        "PKG_CONFIG",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_CFG_TARGET_ARCH",
        "CARGO_CFG_TARGET_OS",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        recipe.update(key.as_bytes());
        recipe.update(
            env::var_os(key)
                .unwrap_or_default()
                .to_string_lossy()
                .as_bytes(),
        );
        recipe.update([0]);
    }
    for prefix in ["CC", "CFLAGS", "AR", "AS"] {
        for suffix in [target.as_str(), &target.replace('-', "_")] {
            let key = format!("{prefix}_{suffix}");
            println!("cargo:rerun-if-env-changed={key}");
            recipe.update(key.as_bytes());
            recipe.update(
                env::var_os(&key)
                    .unwrap_or_default()
                    .to_string_lossy()
                    .as_bytes(),
            );
            recipe.update([0]);
        }
    }
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
        let _ = fs::remove_dir_all(out.join("ffmpeg-patched-source"));
        let _ = fs::remove_file(&stamp);
    }
    if stale || !archive.exists() || !install.join("lib/libswscale.a").exists() {
        let patch =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("patches/ffmpeg-h264-memory-guards.patch");
        let patched = staged_ffmpeg_source(&source, &out, &patch);
        let build = out.join("ffmpeg-build");
        fs::create_dir_all(&build).expect("cannot create FFmpeg build directory");
        let mut configure = Command::new(patched.join("configure"));
        configure
            .current_dir(&build)
            .arg(format!("--prefix={}", install.display()))
            .arg(format!("--cc={compiler_path}"))
            .arg(format!("--extra-cflags={compiler_cflags}"))
            // FFmpeg links configure probes through the C driver. Keep ABI,
            // target and sysroot arguments in those invocations as well.
            .arg(format!("--extra-ldflags={compiler_cflags}"))
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
        if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") && nasm_version.is_none() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Fixture(PathBuf);

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn pinned_shape_fixture() -> (Fixture, String, String) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "rosemary-ffmpeg-source-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let fixture = Fixture(root.clone());
        fs::create_dir_all(root.join("libavcodec")).unwrap();
        fs::write(root.join("configure"), b"#!/bin/sh\n").unwrap();
        fs::write(
            root.join("libavcodec/h264dec.c"),
            b"pinned decoder source\n",
        )
        .unwrap();
        fs::write(
            root.join(".gitignore"),
            b"/config.h\n/libavcodec/codec_list.c\n",
        )
        .unwrap();
        command(Command::new("git").current_dir(&root).args(["init", "-q"]));
        command(Command::new("git").current_dir(&root).args(["add", "."]));
        command(Command::new("git").current_dir(&root).args([
            "-c",
            "user.name=Source Test",
            "-c",
            "user.email=source-test@example.invalid",
            "commit",
            "-qm",
            "pinned fixture",
        ]));
        let rev = Command::new("git")
            .current_dir(&root)
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap();
        assert!(rev.status.success());
        let commit = String::from_utf8(rev.stdout).unwrap().trim().to_owned();
        let hash = tree_hash(&root);
        (fixture, commit, hash)
    }

    #[test]
    fn git_checkout_and_archive_use_the_same_content_manifest() {
        let (fixture, commit, hash) = pinned_shape_fixture();
        assert_eq!(
            verify_source(&fixture.0, &commit, &hash),
            format!("tree:{hash}")
        );
        let archive = Fixture(fixture.0.with_extension("archive"));
        fs::create_dir_all(archive.0.join("libavcodec")).unwrap();
        for name in ["configure", ".gitignore", "libavcodec/h264dec.c"] {
            fs::copy(fixture.0.join(name), archive.0.join(name)).unwrap();
        }
        assert_eq!(
            verify_source(&archive.0, &commit, &hash),
            format!("tree:{hash}")
        );
    }

    #[test]
    fn ignored_generated_files_in_clean_git_checkout_are_rejected() {
        let (fixture, commit, hash) = pinned_shape_fixture();
        for ignored in ["config.h", "libavcodec/codec_list.c"] {
            fs::write(fixture.0.join(ignored), b"unexpected generated header\n").unwrap();
            let status = Command::new("git")
                .current_dir(&fixture.0)
                .args(["status", "--porcelain", "--untracked-files=all"])
                .output()
                .unwrap();
            assert!(status.status.success() && status.stdout.is_empty());
            let result = std::panic::catch_unwind(|| verify_source(&fixture.0, &commit, &hash));
            assert!(
                result.is_err(),
                "ignored {ignored} bypassed source manifest"
            );
            fs::remove_file(fixture.0.join(ignored)).unwrap();
            assert_eq!(
                verify_source(&fixture.0, &commit, &hash),
                format!("tree:{hash}")
            );
        }
    }
}
