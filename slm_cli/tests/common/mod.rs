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

/// Held while a test touches Ollama, LM Studio or their ports. Serialises threads of this
/// process (mutex) *and* concurrent test processes (advisory `flock` on a file in the temp
/// dir), so two `cargo test` runs never fight over the same runtime. Released on drop.
pub struct RuntimeLock {
    _thread: std::sync::MutexGuard<'static, ()>,
    _file: std::fs::File,
}

static THREAD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn runtime_lock_path() -> PathBuf {
    std::env::temp_dir().join("ferrum-runtime-tests.lock")
}

fn open_lock_file() -> std::fs::File {
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(runtime_lock_path())
        .expect("open runtime lock file")
}

/// Block until this process owns the runtime lock.
pub fn runtime_lock() -> RuntimeLock {
    let thread = THREAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let file = open_lock_file();
    file.lock().expect("flock runtime lock"); // blocks; the OS drops it if we die
    RuntimeLock {
        _thread: thread,
        _file: file,
    }
}

/// `true` if some other holder (this or another process) currently owns the lock.
pub fn runtime_lock_is_held_elsewhere() -> bool {
    matches!(
        open_lock_file().try_lock(),
        Err(std::fs::TryLockError::WouldBlock)
    )
}
