//! Helpers shared by the runtime-facing integration tests (E1 here, G5 later).
#![allow(dead_code)] // each test crate uses a different subset

use std::path::PathBuf;

/// Pure core of [`require_or_skip`]: `require` is the value of `FERRUM_REQUIRE_RUNTIMES=1`.
pub fn require_or_skip_with<T>(tool: Result<T, String>, require: bool) -> Option<T> {
    match tool {
        Ok(t) => Some(t),
        Err(e) if require => panic!("FERRUM_REQUIRE_RUNTIMES=1 but a runtime is missing: {e}"),
        Err(e) => {
            eprintln!("skip: {e}");
            None
        }
    }
}

/// `Some(tool)` when found. When missing: print `skip: …` and return `None`,
/// or panic if `FERRUM_REQUIRE_RUNTIMES=1`.
pub fn require_or_skip<T>(tool: Result<T, String>) -> Option<T> {
    let v = std::env::var("FERRUM_REQUIRE_RUNTIMES").ok();
    require_or_skip_with(tool, is_required(v.as_deref()))
}

/// Whether the value of `FERRUM_REQUIRE_RUNTIMES` turns a skip into a failure.
pub fn is_required(v: Option<&str>) -> bool {
    v == Some("1")
}

/// Locate a tool: `$env`, then `PATH`, then `~/<home_rel>`.
pub fn tool(env: &str, name: &str, home_rel: &str) -> Result<PathBuf, String> {
    let mut cands: Vec<PathBuf> = std::env::var_os(env)
        .map(PathBuf::from)
        .into_iter()
        .collect();
    if let Some(p) = std::env::var_os("PATH") {
        cands.extend(std::env::split_paths(&p).map(|d| d.join(name)));
    }
    if let Some(h) = std::env::var_os("HOME") {
        cands.push(PathBuf::from(h).join(home_rel));
    }
    cands
        .into_iter()
        .find(|c| c.is_file())
        .ok_or_else(|| format!("{name} not found (${env}, PATH, ~/{home_rel})"))
}
