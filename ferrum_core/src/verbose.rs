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

#[cfg(test)]
mod tests {
    use super::*;

    // Serializes the tests that mutate the process-global VERBOSE/SINK state so
    // they cannot observe each other's writes when the harness runs in parallel.
    static GUARD: Mutex<()> = Mutex::new(());

    #[test]
    fn stats_handles_empty_and_values() {
        assert_eq!(stats(&[]), (0.0, 0.0, 0.0));
        let (mn, mx, mean) = stats(&[1.0, -2.0, 3.0, 4.0]);
        assert_eq!(mn, -2.0);
        assert_eq!(mx, 4.0);
        assert!((mean - 1.5).abs() < 1e-6);
    }

    #[test]
    fn check_nan_inf_flags_bad_and_passes_clean() {
        assert!(!check_nan_inf(&[1.0, 2.0, 3.0], "clean"));
        assert!(check_nan_inf(&[1.0, f32::NAN, 3.0], "has-nan"));
        assert!(check_nan_inf(&[1.0, f32::INFINITY], "has-inf"));
    }

    #[test]
    fn verbose_flag_toggles() {
        let _g = GUARD.lock().unwrap();
        let prev = is_verbose();
        set_verbose(false);
        assert!(!is_verbose());
        set_verbose(true);
        assert!(is_verbose());
        set_verbose(prev);
    }

    #[test]
    fn sink_receives_logged_lines_then_clears() {
        let _g = GUARD.lock().unwrap();
        let captured = Arc::new(Mutex::new(Vec::<String>::new()));
        let c2 = Arc::clone(&captured);
        set_log_sink(move |line| c2.lock().unwrap().push(line.to_string()));
        log_line("hello-sink");
        clear_log_sink();
        // After clearing, further lines are not captured.
        log_line("after-clear");
        let got = captured.lock().unwrap();
        assert!(got.iter().any(|l| l == "hello-sink"));
        assert!(!got.iter().any(|l| l == "after-clear"));
    }

    #[test]
    fn vprintln_macro_only_emits_when_verbose() {
        let _g = GUARD.lock().unwrap();
        let captured = Arc::new(Mutex::new(Vec::<String>::new()));
        let c2 = Arc::clone(&captured);
        set_log_sink(move |line| c2.lock().unwrap().push(line.to_string()));

        let prev = is_verbose();
        set_verbose(false);
        vprintln!("should-not-appear {}", 1);
        set_verbose(true);
        vprintln!("should-appear {}", 2);
        set_verbose(prev);
        clear_log_sink();

        let got = captured.lock().unwrap();
        assert!(got.iter().any(|l| l == "should-appear 2"));
        assert!(!got.iter().any(|l| l.contains("should-not-appear")));
    }
}
