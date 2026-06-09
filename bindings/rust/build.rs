use std::path::PathBuf;

// Link the libmlxforge.dylib produced by the repo's CMake build, and add an
// rpath to it so the test/example binaries find it at runtime. A published
// crate would instead ship/locate a prebuilt dylib.
fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let build_dir = manifest.join("../../build");
    let build_dir = build_dir.canonicalize().unwrap_or(build_dir);
    let dir = build_dir.display();

    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=mlxforge");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    println!("cargo:rerun-if-changed=build.rs");
}
