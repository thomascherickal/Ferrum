//! Ferrum SLM Studio — Tauri backend.
//!
//! The frontend (vanilla HTML/CSS/JS in `ui/`) talks to the `ferrum_core`
//! engine through the `#[tauri::command]` functions in [`commands`]. Long
//! operations run on a blocking task and stream progress/log/output back to the
//! webview as Tauri events:
//!
//! | event           | payload                              | source                |
//! |-----------------|--------------------------------------|-----------------------|
//! | `engine-log`    | `String` (one diagnostic line)       | `--verbose` sink      |
//! | `train-progress`| `{ epoch, total, loss }`             | training callback     |
//! | `train-done`    | [`commands::TrainResult`]            | training completion   |
//! | `finetune-progress` | `{ epoch, total, loss, ppl }`    | GGUF fine-tune epoch  |
//! | `finetune-done` | [`commands::FinetuneResult`]         | GGUF fine-tune done   |
//! | `gen-fragment`  | `String` (streamed text fragment)    | streaming generation  |
//! | `term-output`   | `{ line, stream }`                   | interactive terminal  |

mod commands;
mod datasets;
mod capable;

use std::path::PathBuf;
use std::sync::Mutex;
use sysinfo::System;
use tauri::Emitter;

/// Process-wide state shared across commands.
pub struct AppState {
    /// Reused `sysinfo` handle so CPU usage deltas are meaningful between polls.
    pub sys: Mutex<System>,
    /// Working directory for the embedded shell (so `cd` persists).
    pub cwd: Mutex<PathBuf>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            sys: Mutex::new(System::new_all()),
            cwd: Mutex::new(cwd),
        })
        .setup(|app| {
            // Mirror every ferrum_core `--verbose` line into the GUI terminal.
            let handle = app.handle().clone();
            ferrum_core::set_log_sink(move |line| {
                let _ = handle.emit("engine-log", line.to_string());
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::download_text,
            commands::clean_text,
            commands::save_corpus,
            commands::read_text_file,
            commands::train_slm,
            commands::generate_slm,
            commands::evaluate_slm,
            commands::model_info,
            commands::gguf_info,
            commands::run_gguf,
            commands::finetune_gguf,
            commands::run_terminal,
            commands::term_cwd,
            commands::system_stats,
            capable::capability_report,
            datasets::list_datasets,
            datasets::download_dataset,
            datasets::download_hf_file,
            datasets::download_kaggle_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Ferrum SLM Studio");
}
