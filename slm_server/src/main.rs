//! `ferrum-serve --model <path> [--host 127.0.0.1] [--port 8080]
//!   [--max-batch 64] [--max-chars 65536] [--max-body 8388608] [--max-tokens 4096]`
#![forbid(unsafe_code)]

use slm_server::cli::start;

// Thin process wrapper: all logic lives in (tested) `slm_server::cli`.
#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = start(&args, async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await;
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
