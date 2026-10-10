//! Command-line parsing and the serve loop behind `ferrum-serve`.

use crate::{app, Limits};
use std::future::Future;
use std::str::FromStr;

pub const USAGE: &str =
    "usage: ferrum-serve --model <model.gguf> [--host 127.0.0.1] [--port 8080] \
[--max-batch 64] [--max-chars 65536] [--max-body 8388608] [--max-tokens 4096] [--help]";

const KNOWN: &[&str] = &[
    "--model",
    "--host",
    "--port",
    "--max-batch",
    "--max-chars",
    "--max-body",
    "--max-tokens",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub limits: Limits,
}

fn parse_value<T: FromStr>(name: &str, v: &str) -> Result<T, String> {
    v.parse().map_err(|_| format!("{name}: cannot parse '{v}'"))
}

/// What the command line asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parsed {
    Help,
    Run(Config),
}

/// Parse `--flag value` pairs (program name already stripped). A flag given
/// twice takes its last value. `--help`/`-h` anywhere in flag position wins.
pub fn parse_args(args: &[String]) -> Result<Parsed, String> {
    let mut model = String::new();
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut limits = Limits::default();
    let mut it = args.iter();
    while let Some(name) = it.next() {
        if name == "--help" || name == "-h" {
            return Ok(Parsed::Help);
        }
        if !name.starts_with("--") {
            return Err(format!("unexpected argument '{name}'"));
        }
        if !KNOWN.contains(&name.as_str()) {
            return Err(format!("unknown flag {name} (valid: {})", KNOWN.join(" ")));
        }
        let v = it
            .next()
            .filter(|v| !v.starts_with("--"))
            .ok_or(format!("{name} needs a value"))?;
        match name.as_str() {
            "--model" => model = v.clone(),
            "--host" => host = v.clone(),
            "--port" => port = parse_value(name, v)?,
            "--max-batch" => limits.max_batch = parse_value(name, v)?,
            "--max-chars" => limits.max_chars = parse_value(name, v)?,
            "--max-body" => limits.max_body = parse_value(name, v)?,
            _ => limits.max_tokens = parse_value(name, v)?,
        }
    }
    if model.is_empty() {
        return Err(USAGE.into());
    }
    Ok(Parsed::Run(Config {
        model,
        host,
        port,
        limits,
    }))
}

/// Entry point behind `main`: parse `args`, print usage for `--help`, else serve.
pub async fn start(
    args: &[String],
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), String> {
    match parse_args(args)? {
        Parsed::Help => {
            println!("{USAGE}");
            Ok(())
        }
        Parsed::Run(cfg) => run(cfg, shutdown).await,
    }
}

/// Load the model, bind, and serve until `shutdown` resolves.
pub async fn run(
    cfg: Config,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), String> {
    let bytes = std::fs::read(&cfg.model).map_err(|e| format!("cannot read {}: {e}", cfg.model))?;
    let (router, info) = app(bytes, cfg.limits)?;
    let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port))
        .await
        .map_err(|e| format!("cannot bind {}:{}: {e}", cfg.host, cfg.port))?;
    eprintln!(
        "ferrum-serve: {} ({} params, context {}, vocab {}) on http://{}",
        cfg.model,
        info.params,
        info.context_len,
        info.vocab,
        listener.local_addr().map_err(|e| e.to_string())?
    );
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

    /// Config-returning view of the real `parse_args` (`--help` becomes Err("help")).
    fn parse_args(args: &[String]) -> Result<Config, String> {
        match super::parse_args(args)? {
            Parsed::Run(c) => Ok(c),
            Parsed::Help => Err("help".into()),
        }
    }

    fn a(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn defaults_with_only_model() {
        let c = parse_args(&a(&["--model", "m.gguf"])).unwrap();
        assert_eq!(
            c,
            Config {
                model: "m.gguf".into(),
                host: "127.0.0.1".into(),
                port: 8080,
                limits: Limits::default()
            }
        );
    }

    #[test]
    fn all_flags() {
        let c = parse_args(&a(&[
            "--model",
            "m",
            "--host",
            "0.0.0.0",
            "--port",
            "9",
            "--max-batch",
            "2",
            "--max-chars",
            "3",
            "--max-body",
            "4",
            "--max-tokens",
            "5",
        ]))
        .unwrap();
        assert_eq!((c.host.as_str(), c.port), ("0.0.0.0", 9));
        assert_eq!(
            c.limits,
            Limits {
                max_batch: 2,
                max_chars: 3,
                max_body: 4,
                max_tokens: 5
            }
        );
    }

    #[test]
    fn missing_model_is_usage() {
        assert_eq!(parse_args(&[]).unwrap_err(), USAGE);
        assert_eq!(parse_args(&a(&["--port", "1"])).unwrap_err(), USAGE);
    }

    #[test]
    fn unknown_flag() {
        let e = parse_args(&a(&["--model", "m", "--bogus", "1"])).unwrap_err();
        assert!(
            e.contains("unknown flag --bogus") && e.contains("--max-body"),
            "{e}"
        );
    }

    #[test]
    fn stray_positional() {
        let e = parse_args(&a(&["model.gguf"])).unwrap_err();
        assert_eq!(e, "unexpected argument 'model.gguf'");
    }

    #[test]
    fn missing_value() {
        assert_eq!(
            parse_args(&a(&["--model", "m", "--port"])).unwrap_err(),
            "--port needs a value"
        );
    }

    #[test]
    fn bad_value_for_each_numeric_flag() {
        for f in [
            "--port",
            "--max-batch",
            "--max-chars",
            "--max-body",
            "--max-tokens",
        ] {
            let e = parse_args(&a(&["--model", "m", f, "x"])).unwrap_err();
            assert_eq!(e, format!("{f}: cannot parse 'x'"));
        }
        // out of range for u16
        assert!(parse_args(&a(&["--model", "m", "--port", "70000"])).is_err());
    }

    #[test]
    fn flag_value_starting_with_dashes_is_missing_value() {
        let e = parse_args(&a(&["--model", "--port", "1"])).unwrap_err();
        assert_eq!(e, "--model needs a value");
    }

    #[test]
    fn help_flags() {
        for h in ["--help", "-h"] {
            assert_eq!(super::parse_args(&a(&[h])), Ok(Parsed::Help));
            assert_eq!(
                super::parse_args(&a(&["--model", "m", h])),
                Ok(Parsed::Help)
            );
        }
        assert_eq!(parse_args(&a(&["-h"])).unwrap_err(), "help");
        // a lone `-x` is not a flag
        assert!(parse_args(&a(&["-x"])).unwrap_err().contains("unexpected"));
    }

    #[test]
    fn duplicate_flag_last_wins() {
        let c = parse_args(&a(&[
            "--model", "a", "--model", "b", "--port", "1", "--port", "2",
        ]))
        .unwrap();
        assert_eq!((c.model.as_str(), c.port), ("b", 2));
    }

    #[tokio::test]
    async fn start_help_and_errors_and_serve() {
        assert_eq!(start(&a(&["--help"]), async {}).await, Ok(()));
        assert!(start(&a(&["--bogus", "1"]), async {}).await.is_err());
        let path = model_file("start");
        let r = start(&a(&["--model", &path, "--port", "0"]), async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await
        })
        .await;
        std::fs::remove_file(path).ok();
        assert_eq!(r, Ok(()));
    }

    fn model_file(name: &str) -> String {
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size: 0,
            ..TransformerConfig::default()
        };
        let m = GenerativeSLM::train_transformer_config(
            &"abc ".repeat(40),
            &cfg,
            &mut Rng::new(1),
            |_, _| {},
        )
        .unwrap();
        let p = std::env::temp_dir().join(format!("ferrum-serve-{}-{name}", std::process::id()));
        std::fs::write(&p, m.to_bytes_quantized().unwrap()).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn cfg(model: String, host: &str, port: u16) -> Config {
        Config {
            model,
            host: host.into(),
            port,
            limits: Limits::default(),
        }
    }

    #[tokio::test]
    async fn run_serves_until_shutdown() {
        let path = model_file("ok");
        // Ephemeral port; shutdown fires once the server has had a moment to start.
        let r = run(cfg(path.clone(), "127.0.0.1", 0), async {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await
        })
        .await;
        std::fs::remove_file(path).ok();
        assert_eq!(r, Ok(()));
    }

    #[tokio::test]
    async fn run_missing_model_file() {
        let e = run(cfg("/nonexistent/m.gguf".into(), "127.0.0.1", 0), async {})
            .await
            .unwrap_err();
        assert!(e.starts_with("cannot read /nonexistent/m.gguf"), "{e}");
    }

    #[tokio::test]
    async fn run_corrupt_model_file() {
        let p = std::env::temp_dir().join(format!("ferrum-serve-{}-bad", std::process::id()));
        std::fs::write(&p, b"garbage").unwrap();
        let e = run(cfg(p.to_string_lossy().into(), "127.0.0.1", 0), async {})
            .await
            .unwrap_err();
        std::fs::remove_file(p).ok();
        assert!(!e.is_empty());
    }

    #[tokio::test]
    async fn run_bind_failure() {
        let path = model_file("bind");
        let e = run(cfg(path.clone(), "256.256.256.256", 0), async {})
            .await
            .unwrap_err();
        std::fs::remove_file(path).ok();
        assert!(e.starts_with("cannot bind 256.256.256.256:0"), "{e}");
    }
}
