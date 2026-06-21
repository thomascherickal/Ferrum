//! Global verbose-tracing flag for ferrum_core diagnostics.
//!
//! Call `set_verbose(true)` once at program start to enable detailed
//! `println!` output throughout every module. When verbose is **off**
//! (the default) the only overhead is a single `AtomicBool` load per
//! call-site — branch-predicted away to nothing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Optional consumer of every diagnostic line. When set (e.g. by a GUI), each
/// line produced by [`vprintln!`] / [`log_line`] is forwarded here *in addition*
/// to being printed to stdout, so an embedding application can mirror the engine
/// trace into its own console without scraping the process's stdout.
type Sink = Arc<dyn Fn(&str) + Send + Sync>;
static SINK: Mutex<Option<Sink>> = Mutex::new(None);

/// Enable or disable verbose diagnostic output for all ferrum_core operations.
pub fn set_verbose(on: bool) {
    VERBOSE.store(on, Ordering::Relaxed);
    if on {
        log_line("[ferrum_core::verbose] Verbose tracing ENABLED");
    }
}

/// Returns `true` if verbose tracing is currently enabled.
#[inline(always)]
pub fn is_verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

/// Install a sink that receives every diagnostic line (see [`Sink`]). Replaces
/// any previously installed sink. Pass the result of [`clear_log_sink`] usage,
/// or call [`clear_log_sink`] to remove it.
pub fn set_log_sink<F>(f: F)
where
    F: Fn(&str) + Send + Sync + 'static,
{
    *SINK.lock().unwrap() = Some(Arc::new(f));
}

/// Remove any installed diagnostic sink.
pub fn clear_log_sink() {
    *SINK.lock().unwrap() = None;
}

/// Print one diagnostic line to stdout and forward it to the installed sink (if
/// any). This is the single choke-point behind [`vprintln!`]; callers that want
/// a line captured by an embedding GUI should route it through here (or the
/// macro) rather than calling [`println!`] directly.
pub fn log_line(line: &str) {
    println!("{line}");
    // Clone the Arc out under the lock, then release it before invoking the
    // sink so the callback can never deadlock against this mutex.
    let sink = SINK.lock().unwrap().clone();
    if let Some(sink) = sink {
        sink(line);
    }
}

/// Convenience macro — prints (and forwards to any sink) only when verbose mode
/// is active.
///
/// Usage: `vprintln!("message {}", value);`
#[macro_export]
macro_rules! vprintln {
    ($($arg:tt)*) => {
        if $crate::verbose::is_verbose() {
            $crate::verbose::log_line(&format!($($arg)*));
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
        log_line(&format!(
            "[ferrum_core::WARN] ⚠️  {} contains {} NaN, {} Inf out of {} elements!",
            label, nan_count, inf_count, data.len()
        ));
        return true;
    }
    false
}
