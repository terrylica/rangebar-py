//! build.rs - Optimization enforcement at compile time
//!
//! This build script enforces that release builds use proper optimization settings.
//! It prevents accidental builds with disabled optimizations (e.g., via RUSTFLAGS).
//!
//! CRITICAL: This is essential for PyO3 projects where performance is critical
//! and where panic=abort cannot be used (PyO3 requires catch_unwind).

fn main() {
    // Re-run if these environment variables change
    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=OPT_LEVEL");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");

    let profile = std::env::var("PROFILE").unwrap_or_default();
    let opt_level = std::env::var("OPT_LEVEL").unwrap_or_default();

    // Block release builds with disabled optimizations
    assert!(
        !(profile == "release" && opt_level == "0"),
        "\n\
        ============================================================\n\
        ERROR: Release build with opt-level=0 is forbidden!\n\
        ============================================================\n\
        \n\
        This typically means RUSTFLAGS is overriding Cargo.toml.\n\
        \n\
        Fix: unset RUSTFLAGS && maturin build --release\n\
        \n\
        Or check your environment for:\n\
        - RUSTFLAGS=-C opt-level=0\n\
        - .cargo/config.toml overrides\n\
        \n"
    );

    // Detect problematic RUSTFLAGS
    if let Ok(rustflags) = std::env::var("RUSTFLAGS") {
        let bad_flags = ["-C opt-level=0", "-C lto=off", "-C lto=no"];
        for flag in bad_flags {
            assert!(
                !rustflags.contains(flag),
                "\n\
                ============================================================\n\
                ERROR: RUSTFLAGS contains optimization-disabling flag!\n\
                ============================================================\n\
                \n\
                Detected: {flag}\n\
                RUSTFLAGS: {rustflags}\n\
                \n\
                Fix: unset RUSTFLAGS\n\
                \n\
                This project requires optimizations for acceptable performance.\n\
                See Cargo.toml [profile.release] for required settings.\n\
                \n"
            );
        }
    }

    // PyO3 extension module configuration
    pyo3_build_config::add_extension_module_link_args();
}
