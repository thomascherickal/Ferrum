//! Global verbose-tracing flag for ferrum_core diagnostics.
//!
//! Call `set_verbose(true)` once at program start to enable detailed
//! `println!` output throughout every module. When verbose is **off**
//! (the default) the only overhead is a single `AtomicBool` load per
//! call-site — branch-predicted away to nothing.

use std::sync::atomic::{AtomicBool, Ordering};

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Enable or disable verbose diagnostic output for all ferrum_core operations.
pub fn set_verbose(on: bool) {
    VERBOSE.store(on, Ordering::Relaxed);
    if on {
        println!("[ferrum_core::verbose] Verbose tracing ENABLED");
    }
}

/// Returns `true` if verbose tracing is currently enabled.
#[inline(always)]
pub fn is_verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

/// Convenience macro — prints only when verbose mode is active.
///
/// Usage: `vprintln!("message {}", value);`
#[macro_export]
macro_rules! vprintln {
    ($($arg:tt)*) => {
        if $crate::verbose::is_verbose() {
            println!($($arg)*);
        }
    };
}

/// Helper: compute basic statistics (min, max, mean) of a float slice.
/// Returns `(min, max, mean)`. For empty slices returns `(0, 0, 0)`.
pub fn stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in data {
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v as f64;
    }
    (min, max, (sum / data.len() as f64) as f32)
}

/// Helper: check for any NaN or Inf values and print a warning.
/// Returns `true` if any bad values are found.
pub fn check_nan_inf(data: &[f32], label: &str) -> bool {
    let nan_count = data.iter().filter(|v| v.is_nan()).count();
    let inf_count = data.iter().filter(|v| v.is_infinite()).count();
    if nan_count > 0 || inf_count > 0 {
        println!(
            "[ferrum_core::WARN] ⚠️  {} contains {} NaN, {} Inf out of {} elements!",
            label, nan_count, inf_count, data.len()
        );
        return true;
    }
    false
}
