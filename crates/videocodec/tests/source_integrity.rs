// Compile the production build-script verifier into a normal Cargo test
// target so source provenance checks do not rely on manual inspection.
#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;
