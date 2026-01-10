//! Validates README.md version matches Cargo.toml
//! Catches stale documentation before release
//!
//! SSoT chain: lib.rs → cargo-rdme → README.md → crates.io

#[test]
fn test_readme_version_matches_cargo_toml() {
    version_sync::assert_markdown_deps_updated!("../../README.md");
}

#[test]
fn test_html_root_url() {
    version_sync::assert_html_root_url_updated!("src/lib.rs");
}
