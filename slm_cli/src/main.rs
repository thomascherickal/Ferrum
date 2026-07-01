//! `train_transformer` — corpus → causal transformer SLM trainer & generator.
//!
//! Trains a character-level decoder-only **causal Transformer** Small
//! Language Model from any raw text corpus — entirely on CPU, zero external
//! dependencies. Training is int8 quantization-aware (QAT) and models are
//! always exported as int8-quantized FINF v5 (≈4× smaller than f32).
//!
//! Trained weights are cached on disk: if the model file already exists,
//! `train` and `run` load it instead of retraining (use `--force` to retrain).
//!
//! ## Usage
//!
//! ```text
//! train_transformer train    <corpus.txt> <model.bin> [options]
//! train_transformer run      <corpus.txt> <model.bin> <seed text> [options]
//! train_transformer generate <model.bin>  <seed text> [options]
//! train_transformer run-gguf  <model.gguf> [prompt] [--resume ckpt.flck] [options]
//! train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> [options]
//! train_transformer export-gguf <in.gguf> <out.gguf> [--quant q8_0|q4_0|q4_k|…] [--resume ckpt.flck]
//! train_transformer info     <model.bin>
//! ```
//!
//! ### `train` / `run` options
//! ```text
//!   --context <N>   context window in characters   (default 16)
//!   --embed   <N>   embedding dimension            (default 32)
//!   --heads   <N>   attention heads                (default 4)
//!   --blocks  <N>   transformer blocks             (default 2)
//!   --hidden  <N>   FFN hidden width               (default 64)
//!   --epochs  <N>   training epochs                (default 100)
//!   --lr      <F>   Adam learning rate             (default 0.01)
//!   --batch   <N>   minibatch size                 (default 16)
//!   --vocab   <N>   BPE vocab size (0 = char-level)(default 512)
//!   --seed    <N>   RNG seed                       (default 1337)
//!   --threads <N>   data-parallel worker threads   (default 0 = auto)
//!   --force         retrain even if the model file exists
//!   --sample        print a short sample after training
//!   --verbose | -v  print all engine internals
//! ```
//!
//! ### `generate` / `run` options
//! ```text
//!   --chars <N>  characters to generate            (default 200)
//!   --temp  <F>  sampling temperature              (default 0.8)
//!   --gen-seed <N>  generation RNG seed            (default time-based)
//!   --stream     print the completion live as it is generated
//! ```

use ferrum_core::{GenerativeSLM, Rng, TaskType, TransformerConfig};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Minimal flag parser over a positional/`--flag value` argument vector.
struct Args {
    positional: Vec<String>,
    verbose: bool,
    flags: std::collections::HashMap<String, String>,
    bools: std::collections::HashSet<String>,
}

impl Args {
    fn parse(raw: &[String]) -> Self {
        let mut positional = Vec::new();
        let mut flags = std::collections::HashMap::new();
        let mut bools = std::collections::HashSet::new();
        let mut verbose = false;
        // Flags that take a value rather than acting as a boolean switch.
        const VALUE_FLAGS: &[&str] = &[
            "context",
            "embed",
            "heads",
            "blocks",
            "hidden",
            "epochs",
            "lr",
            "batch",
            "seed",
            "chars",
            "temp",
            "gen-seed",
            "vocab",
            "threads",
            // AdamW / regularization (exposing the engine's existing knobs).
            "weight_decay",
            "dropout",
            // run-gguf options.
            "quant",
            "max",
            "ids",
            // finetune-gguf options.
            "seq",
            "warmup",
            "clip",
            "resume",
        ];
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if a == "--verbose" || a == "-v" {
                verbose = true;
            } else if let Some(name) = a.strip_prefix("--") {
                if VALUE_FLAGS.contains(&name) {
                    if let Some(val) = raw.get(i + 1) {
                        flags.insert(name.to_string(), val.clone());
                        i += 1;
                    }
                } else {
                    bools.insert(name.to_string());
                }
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Args {
            positional,
            verbose,
            flags,
            bools,
        }
    }

    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.flags
            .get(key)
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }
    fn has(&self, key: &str) -> bool {
        self.bools.contains(key)
    }

    fn config(&self) -> TransformerConfig {
        let d = TransformerConfig::default();
        TransformerConfig {
            context_len: self.get("context", d.context_len),
            embed_dim: self.get("embed", d.embed_dim),
            num_heads: self.get("heads", d.num_heads),
            num_blocks: self.get("blocks", d.num_blocks),
            hidden_dim: self.get("hidden", d.hidden_dim),
            epochs: self.get("epochs", d.epochs),
            lr: self.get("lr", d.lr),
            batch_size: self.get("batch", d.batch_size),
            vocab_size: self.get("vocab", d.vocab_size),
            weight_decay: self.get("weight_decay", d.weight_decay),
            dropout: self.get("dropout", d.dropout),
        }
    }
}

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
        | 1
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    let cmd = argv[1].clone();
    let args = Args::parse(&argv[2..]);
    if args.verbose {
        ferrum_core::set_verbose(true);
    }
    match cmd.as_str() {
        "train" => cmd_train(&args),
        "run" => cmd_run(&args),
        "generate" | "gen" => cmd_generate(&args),
        "run-gguf" | "gguf" => cmd_run_gguf(&args),
        "finetune-gguf" | "finetune" => cmd_finetune_gguf(&args),
        "export-gguf" | "export" => cmd_export_gguf(&args),
        "eval" => cmd_eval(&args),
        "info" => cmd_info(&args),
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        other => {
            eprintln!("unknown command: {other}\n");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
        "train_transformer — ferrum causal transformer SLM trainer & generator\n\
         (int8 quantization-aware training; weights cached on disk)\n\n\
         USAGE:\n\
         \x20 train_transformer train    <corpus.txt> <model.bin> [options]\n\
         \x20 train_transformer run      <corpus.txt> <model.bin> <seed text> [options]\n\
         \x20 train_transformer generate <model.bin>  <seed text> [options]\n\
         \x20 train_transformer run-gguf <model.gguf> [prompt] [options]\n\
         \x20 train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> [options]\n\
         \x20 train_transformer export-gguf <in.gguf> <out.gguf> [options]\n\
         \x20 train_transformer eval     <model.bin>  <heldout.txt>\n\
         \x20 train_transformer info     <model.bin>\n\n\
         TRAIN / RUN options:\n\
         \x20 --context N  --embed N  --heads N  --blocks N  --hidden N\n\
         \x20 --epochs N   --lr F     --batch N  --vocab N  --seed N\n\
         \x20 --weight_decay F  --dropout F   (AdamW decay + FFN dropout; default 0)\n\
         \x20 --threads N  --force    --sample   --verbose|-v\n\
         \x20 (--vocab 0 = character-level; >=256 = byte-level BPE, default 512)\n\
         \x20 (--threads 0 = auto-detect cores [default]; 1 = serial training)\n\n\
         GENERATE / RUN options:\n\
         \x20 --chars N    --temp F   --gen-seed N   --stream\n\
         \x20 (--stream prints the completion live as it is generated)\n\n\
         RUN-GGUF options (import & run a llama/qwen2 GGUF checkpoint):\n\
         \x20 --quant int4|int8|f32  (in-memory precision; default int4)\n\
         \x20 --max N      --temp F  --gen-seed N\n\
         \x20 --ids \"1 2 3\"  (raw prompt token IDs; required if the file has no tokenizer)\n\
         \x20 --force        (load even if the memory estimate exceeds available RAM)\n\
         \x20 --resume ckpt.flck  (overlay fine-tuned weights; forces f32 load)\n\
         \x20 NOTE: only F32/F16/Q8_0/Q8_1/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K GGUFs; on CPU\n\
         \x20 a 1B model decodes at only a few tokens/sec (see ferrum_review.md §4).\n\n\
         FINETUNE-GGUF options (AdamW fine-tune an imported GGUF; f32 masters):\n\
         \x20 --epochs N  --lr F  --batch N  --seq N   (window length; capped at ctx)\n\
         \x20 --warmup N  --clip F  --weight_decay F  --dropout F   (schedule/regularization)\n\
         \x20 --qat          (int8 quantization-aware fine-tuning)\n\
         \x20 --threads N  --seed N  --resume ckpt.flck  --sample\n\
         \x20 writes a .flck checkpoint; apply it with  run-gguf ... --resume out.flck\n\n\
         EXPORT-GGUF options (re-quantize / export an imported GGUF to disk):\n\
         \x20 --quant q8_0|q4_0|q4_1|q8_1|q4_k|q5_k|q6_k|f16|f32  (output precision; default q8_0)\n\
         \x20 --resume ckpt.flck  (apply a fine-tune checkpoint's f32 masters before export)\n\
         \x20 --force        (write even if the memory estimate exceeds available RAM)\n\n\
         If <model.bin> already exists, train/run load the saved weights from\n\
         disk instead of retraining. Pass --force to retrain from scratch.\n\n\
         EXAMPLES:\n\
         \x20 train_transformer train corpus.txt model.bin --epochs 200\n\
         \x20 train_transformer run   corpus.txt model.bin \"Once upon a time\" --chars 300\n\
         \x20 train_transformer generate model.bin \"Once upon a time\" --temp 0.7"
    );
}

fn read_corpus(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let corpus =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read corpus {path}: {e}"))?;
    if corpus.trim().is_empty() {
        return Err(format!("corpus {path} is empty").into());
    }
    Ok(corpus)
}

/// Train a transformer SLM on `corpus` (QAT, int8) and save it to
/// `model_path`. Returns the trained model.
fn train_and_save(
    corpus: &str,
    corpus_path: &str,
    model_path: &str,
    args: &Args,
) -> Result<GenerativeSLM, Box<dyn std::error::Error>> {
    let cfg = args.config();
    let seed: u64 = args.get("seed", 1337);
    // 0 = auto-detect the machine's parallelism; 1 = serial.
    let threads: usize = args.get("threads", 0);
    let mut rng = Rng::new(seed);
    let chars = corpus.chars().filter(|&c| c != '\r').count();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("  ferrum transformer SLM trainer (int8 QAT)");
    println!("  Corpus  : {corpus_path}  ({chars} chars)");
    let tok_desc = if cfg.vocab_size == 0 {
        "character-level".to_string()
    } else {
        format!("byte-level BPE (vocab {})", cfg.vocab_size)
    };
    println!(
        "  Context : {}   Embed: {}   Hidden: {}",
        cfg.context_len, cfg.embed_dim, cfg.hidden_dim
    );
    println!("  Heads   : {}   Blocks: {}", cfg.num_heads, cfg.num_blocks);
    println!("  Tokenizer: {tok_desc}");
    println!(
        "  Epochs  : {}   LR: {}   Batch: {}   Seed: {seed}",
        cfg.epochs, cfg.lr, cfg.batch_size
    );
    let resolved_threads = if threads == 0 {
        ferrum_core::num_threads()
    } else {
        threads
    };
    println!("  Threads : {resolved_threads} (data-parallel minibatch training)");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let epochs = cfg.epochs;
    let report_every = (epochs / 20).max(1);
    let progress = |ep: usize, loss: f32| {
        if ep == 1 || ep.is_multiple_of(report_every) || ep == epochs {
            println!("  epoch {ep:>5}/{epochs}   loss = {loss:.6}");
        }
    };

    let slm = GenerativeSLM::train_transformer_config_threaded(
        corpus, &cfg, threads, &mut rng, progress,
    )?;

    println!("\nTrained in {:.2}s.", t0.elapsed().as_secs_f32());
    let vocab_kind = if slm.meta.tokenizer_state.is_empty() {
        "chars"
    } else {
        "BPE tokens"
    };
    println!(
        "  vocab = {} {}   context = {}   output_dim = {}",
        slm.meta.output_dim, vocab_kind, slm.meta.input_dim, slm.meta.output_dim
    );

    slm.save(model_path)?;
    let size = std::fs::metadata(model_path)?.len();
    println!("Saved {size} bytes → {model_path} (int8-quantized FINF v5)");

    // Verify the file roundtrips.
    let reloaded = GenerativeSLM::load(model_path)?;
    println!("Reload check: OK ({} layers).", reloaded.model.len());
    Ok(reloaded)
}

fn print_sample(slm: &GenerativeSLM, corpus: &str) -> Result<(), Box<dyn std::error::Error>> {
    let context_len = slm.meta.input_dim;
    let seed_text: String = corpus
        .chars()
        .filter(|&c| c != '\r')
        .take(context_len)
        .collect();
    let mut g_rng = Rng::new(time_seed());
    let sample = slm.generate(&seed_text, 120, 0.7, &mut g_rng)?;
    println!("\n── sample ──\n{sample}\n────────────");
    Ok(())
}

fn cmd_train(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err("usage: train_transformer train <corpus.txt> <model.bin> [options]".into());
    }
    let corpus_path = &args.positional[0];
    let model_path = &args.positional[1];
    let corpus = read_corpus(corpus_path)?;

    let slm = if std::path::Path::new(model_path).exists() && !args.has("force") {
        let slm = GenerativeSLM::load(model_path)?;
        println!(
            "Model {model_path} already exists — loaded saved weights ({} layers, vocab {}).",
            slm.model.len(),
            slm.meta.output_dim
        );
        println!("Skipping training. Pass --force to retrain from scratch.");
        slm
    } else {
        train_and_save(&corpus, corpus_path, model_path, args)?
    };

    if args.has("sample") {
        print_sample(&slm, &corpus)?;
    }
    Ok(())
}

fn cmd_run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 3 {
        return Err(
            "usage: train_transformer run <corpus.txt> <model.bin> <seed text> [options]".into(),
        );
    }
    let corpus_path = &args.positional[0];
    let model_path = &args.positional[1];
    let seed_text = args.positional[2..].join(" ");
    let corpus = read_corpus(corpus_path)?;

    let slm = if std::path::Path::new(model_path).exists() && !args.has("force") {
        let slm = GenerativeSLM::load(model_path)?;
        println!("Loaded saved weights from {model_path} — no retraining needed.\n");
        slm
    } else {
        train_and_save(&corpus, corpus_path, model_path, args)?
    };

    let out = generate_text(&slm, &seed_text, args)?;
    if !args.has("stream") {
        println!("{out}");
    }
    Ok(())
}

fn cmd_generate(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err("usage: train_transformer generate <model.bin> <seed text> [options]".into());
    }
    let model_path = &args.positional[0];
    // Join remaining positionals so multi-word seeds work without quoting.
    let seed_text = args.positional[1..].join(" ");

    let slm = GenerativeSLM::load(model_path)
        .map_err(|e| format!("cannot load model {model_path}: {e}"))?;
    let out = generate_text(&slm, &seed_text, args)?;
    // Streaming already printed the text live; avoid printing it twice.
    if !args.has("stream") {
        println!("{out}");
    }
    Ok(())
}

fn generate_text(
    slm: &GenerativeSLM,
    seed_text: &str,
    args: &Args,
) -> Result<String, Box<dyn std::error::Error>> {
    let num_chars: usize = args.get("chars", 200);
    let temp: f32 = args.get("temp", 0.8);
    let seed: u64 = args.get("gen-seed", time_seed());

    // BPE models tokenize the seed and left-pad short prompts, so the
    // char-count check only applies to character-level models.
    if slm.meta.tokenizer_state.is_empty() {
        let context_len = if slm.meta.task == TaskType::TransformerSLM {
            slm.meta.input_dim
        } else {
            slm.meta.input_dim / slm.meta.output_dim.max(1)
        };
        let seed_chars = seed_text.chars().count();
        if seed_chars < context_len {
            return Err(format!(
                "seed text is {seed_chars} chars but the model needs at least {context_len} \
                 (context window). Provide a longer seed."
            )
            .into());
        }
    }

    let mut rng = Rng::new(seed);
    if args.has("stream") {
        // Print the seed, then each generated fragment as it lands, flushing so
        // the terminal shows the completion appearing live.
        use std::io::Write;
        print!("{seed_text}");
        let _ = std::io::stdout().flush();
        let out = slm.generate_stream(seed_text, num_chars, temp, &mut rng, |frag| {
            print!("{frag}");
            let _ = std::io::stdout().flush();
        })?;
        println!();
        Ok(out)
    } else {
        Ok(slm.generate(seed_text, num_chars, temp, &mut rng)?)
    }
}

fn cmd_eval(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err("usage: train_transformer eval <model.bin> <heldout.txt>".into());
    }
    let model_path = &args.positional[0];
    let text_path = &args.positional[1];

    let slm = GenerativeSLM::load(model_path)
        .map_err(|e| format!("cannot load model {model_path}: {e}"))?;
    let text = read_corpus(text_path)?;

    let eval = slm.evaluate(&text)?;
    println!("Model        : {model_path}");
    println!(
        "Held-out text: {text_path}  ({} chars)",
        text.chars().count()
    );
    println!("Predictions  : {}", eval.num_predictions);
    println!("Cross-entropy: {:.4} nats/token", eval.cross_entropy);
    println!("Bits/token   : {:.4}", eval.bits_per_token);
    println!("Perplexity   : {:.4}", eval.perplexity);
    Ok(())
}

fn cmd_info(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.is_empty() {
        return Err("usage: train_transformer info <model.bin>".into());
    }
    let model_path = &args.positional[0];
    let bytes =
        std::fs::read(model_path).map_err(|e| format!("cannot read model {model_path}: {e}"))?;
    let version = if bytes.len() >= 8 {
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
    } else {
        0
    };
    let slm = GenerativeSLM::from_bytes(&bytes)?;
    let m = &slm.meta;
    println!("Model     : {model_path}  ({} bytes)", bytes.len());
    println!(
        "Format    : FINF v{version}{}",
        if version == 5 {
            " (int8-quantized)"
        } else {
            " (f32)"
        }
    );
    println!("Name      : {}", m.dataset_name);
    println!("Task      : {:?}", m.task);
    println!("Input dim : {}", m.input_dim);
    println!("Output dim: {}", m.output_dim);
    if m.tokenizer_state.is_empty() {
        println!(
            "Tokenizer : character-level ({} chars)",
            m.class_names.len()
        );
    } else {
        let merges = m
            .tokenizer_state
            .split(';')
            .filter(|s| !s.is_empty())
            .count();
        println!(
            "Tokenizer : byte-level BPE ({} tokens, {} merges)",
            m.output_dim, merges
        );
    }
    println!("Layers    : {}", slm.model.len());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// run-gguf: import & run an external llama/qwen2 GGUF checkpoint (G-CLI)
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the resident memory a loaded model will need, from the GGUF tensor
/// directory and the chosen in-memory precision. The token embedding is kept
/// f32 (so it is the single largest array for big-vocab models); every other
/// weight packs to int4/int8/f32. A pre-load guard, not an exact figure.
fn estimate_resident_bytes(g: &ferrum_core::Gguf, prec: Option<ferrum_core::QKind>) -> usize {
    use ferrum_core::QKind;
    let mut total = 0usize;
    for t in &g.tensors {
        let n = t.num_elements();
        let bytes = if t.name == "token_embd.weight" {
            n * 4 // kept f32 in LlamaModel
        } else {
            match prec {
                Some(QKind::Int4) => n.div_ceil(2),
                Some(QKind::Int8) => n,
                None => n * 4,
            }
        };
        total = total.saturating_add(bytes);
    }
    total
}

/// Best-effort available RAM in bytes (Linux `/proc/meminfo` `MemAvailable`);
/// `None` where it cannot be determined, in which case the guard is skipped.
fn available_memory_bytes() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in info.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn cmd_run_gguf(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use ferrum_core::{Gguf, GgufTokenizer, QKind, Rng, SamplingParams};

    if args.positional.is_empty() {
        return Err("usage: train_transformer run-gguf <model.gguf> [prompt] \
                    [--quant int4|int8|f32] [--max N] [--temp F] [--ids \"1 2 3\"]"
            .into());
    }
    let path = &args.positional[0];
    let prompt_text = args.positional[1..].join(" ");

    // A fine-tune checkpoint holds f32 weights, so applying one forces an f32
    // load regardless of --quant.
    let resume = args.flags.get("resume").cloned();
    let quant = args
        .flags
        .get("quant")
        .map(String::as_str)
        .unwrap_or("int4");
    let prec = if resume.is_some() {
        if quant != "int4" && quant != "f32" && quant != "none" {
            eprintln!("note: --resume applies f32 fine-tuned weights; ignoring --quant {quant}");
        }
        None
    } else {
        match quant {
            "int4" | "q4" => Some(QKind::Int4),
            "int8" | "q8" => Some(QKind::Int8),
            "f32" | "none" => None,
            other => return Err(format!("--quant must be int4|int8|f32 (got '{other}')").into()),
        }
    };

    // Streamed open: parse the header without reading the whole file.
    println!("Opening {path} (streamed)…");
    let g = Gguf::open(path).map_err(|e| format!("cannot open GGUF {path}: {e}"))?;
    println!(
        "  GGUF v{}   architecture = {}",
        g.version,
        g.architecture().unwrap_or("?")
    );

    // Memory guard before the (potentially multi-GB) load.
    let est = estimate_resident_bytes(&g, prec);
    println!(
        "  estimated resident ≈ {:.2} GB  (--quant {quant})",
        est as f64 / 1e9
    );
    if let Some(avail) = available_memory_bytes() {
        println!("  available memory   ≈ {:.2} GB", avail as f64 / 1e9);
        if (est as f64) > 0.9 * avail as f64 && !args.has("force") {
            return Err(format!(
                "estimated resident memory ({:.2} GB) exceeds 90% of available ({:.2} GB) — \
                 try a smaller --quant, or pass --force to attempt it anyway.",
                est as f64 / 1e9,
                avail as f64 / 1e9
            )
            .into());
        }
    }

    // Tokenizer is optional; without one, the prompt must be raw token IDs.
    let tok = GgufTokenizer::from_gguf(&g).ok();
    match &tok {
        Some(t) => println!("  tokenizer = {:?}  (vocab {})", t.model(), t.vocab_size()),
        None => println!("  tokenizer = none in file — supply --ids \"<token ids>\""),
    }

    println!("Loading weights…");
    let t0 = Instant::now();
    let mut model = g
        .load_llama_prec(prec)
        .map_err(|e| format!("cannot load model: {e}"))?;
    println!(
        "  loaded in {:.2}s: {} layers, dim {}, vocab {}, ctx {}",
        t0.elapsed().as_secs_f32(),
        model.cfg.n_layers,
        model.cfg.model_dim,
        model.cfg.vocab_size,
        model.cfg.context_len,
    );

    // Overlay fine-tuned weights from a checkpoint, if requested.
    if let Some(ckpt) = &resume {
        use ferrum_core::LlamaTrainer;
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        let mut tr = LlamaTrainer::new(model).map_err(|e| format!("cannot wrap model: {e}"))?;
        tr.load_checkpoint_into(&bytes)
            .map_err(|e| format!("cannot apply checkpoint {ckpt}: {e}"))?;
        model = tr.model;
        println!("  applied fine-tuned checkpoint {ckpt}");
    }

    // Build the prompt token IDs (explicit --ids win; else encode the text).
    let prompt_ids: Vec<usize> = if let Some(ids) = args.flags.get("ids") {
        ids.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect()
    } else if let Some(t) = &tok {
        let mut ids = Vec::new();
        if let Some(bos) = t.bos() {
            ids.push(bos);
        }
        ids.extend(t.encode(&prompt_text));
        ids
    } else {
        return Err(
            "this GGUF has no tokenizer; pass --ids \"<space-separated token ids>\"".into(),
        );
    };
    if prompt_ids.is_empty() {
        return Err("empty prompt — provide text (with a tokenizer) or --ids".into());
    }

    let max_new: usize = args.get("max", 64);
    let temp: f32 = args.get("temp", 0.8);
    let gen_seed: u64 = args.get("gen-seed", time_seed());
    let eos = tok.as_ref().and_then(GgufTokenizer::eos);
    let params = SamplingParams::with_temperature(temp);
    let mut rng = Rng::new(gen_seed);

    println!(
        "\nPrefilling {} prompt tokens, generating up to {max_new} (temp {temp})…",
        prompt_ids.len()
    );
    let t1 = Instant::now();
    let out_ids = model.generate(&prompt_ids, max_new, &params, eos, &mut rng)?;
    let dt = t1.elapsed().as_secs_f32();

    println!("\n── output ──");
    match &tok {
        Some(t) => println!("{prompt_text}{}", t.decode(&out_ids)),
        None => {
            let ids: Vec<String> = out_ids.iter().map(usize::to_string).collect();
            println!("token ids: {}", ids.join(" "));
        }
    }
    println!("────────────");
    println!(
        "[{} tokens in {:.2}s = {:.1} tok/s]",
        out_ids.len(),
        dt,
        out_ids.len() as f32 / dt.max(1e-6)
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// finetune-gguf: fine-tune an imported llama/qwen2 GGUF on a text corpus
// ─────────────────────────────────────────────────────────────────────────────

/// Fine-tune an imported GGUF (f32) with the full AdamW stack and write a
/// checkpoint that `run-gguf --resume` (or a later `finetune-gguf --resume`) can
/// apply back over the base GGUF.
fn cmd_finetune_gguf(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use ferrum_core::{Adam, Gguf, GgufTokenizer, LlamaTrainer, LrSchedule, Rng, SamplingParams};

    if args.positional.len() < 3 {
        return Err(
            "usage: train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> \
                    [--epochs N] [--lr F] [--batch N] [--seq N] [--warmup N] [--clip F] \
                    [--weight_decay F] [--dropout F] [--threads N] [--qat] [--seed N] \
                    [--resume ckpt.flck] [--sample]"
                .into(),
        );
    }
    let gguf_path = &args.positional[0];
    let corpus_path = &args.positional[1];
    let out_path = &args.positional[2];

    // 1. Open + f32-load the base model (training needs full-precision masters).
    println!("Opening {gguf_path} (streamed)…");
    let g = Gguf::open(gguf_path).map_err(|e| format!("cannot open GGUF {gguf_path}: {e}"))?;
    println!(
        "  GGUF v{}   architecture = {}",
        g.version,
        g.architecture().unwrap_or("?")
    );
    let est = estimate_resident_bytes(&g, None); // f32
    println!("  estimated resident (f32) ≈ {:.2} GB", est as f64 / 1e9);
    if let Some(avail) = available_memory_bytes() {
        println!("  available memory         ≈ {:.2} GB", avail as f64 / 1e9);
        // Training also needs grads + 2 Adam moments (≈4× the f32 weights again).
        let train_est = est.saturating_mul(4);
        println!(
            "  estimated training RAM   ≈ {:.2} GB (weights + grad + Adam m/v)",
            train_est as f64 / 1e9
        );
        if (train_est as f64) > 0.9 * avail as f64 && !args.has("force") {
            return Err(format!(
                "estimated training memory ({:.2} GB) exceeds 90% of available ({:.2} GB) — \
                 fine-tune a smaller model, or pass --force to attempt it anyway.",
                train_est as f64 / 1e9,
                avail as f64 / 1e9
            )
            .into());
        }
    }

    let tok = GgufTokenizer::from_gguf(&g)
        .map_err(|_| "this GGUF has no tokenizer; text fine-tuning needs one")?;
    println!(
        "  tokenizer = {:?}  (vocab {})",
        tok.model(),
        tok.vocab_size()
    );

    println!("Loading weights (f32)…");
    let t0 = Instant::now();
    let model = g
        .load_llama_prec(None)
        .map_err(|e| format!("cannot load model: {e}"))?;
    println!(
        "  loaded in {:.2}s: {} layers, dim {}, vocab {}, ctx {}",
        t0.elapsed().as_secs_f32(),
        model.cfg.n_layers,
        model.cfg.model_dim,
        model.cfg.vocab_size,
        model.cfg.context_len,
    );

    // 2. Read + tokenize the corpus.
    let text = std::fs::read_to_string(corpus_path)
        .map_err(|e| format!("cannot read corpus {corpus_path}: {e}"))?;
    let mut tokens: Vec<usize> = Vec::new();
    if let Some(bos) = tok.bos() {
        tokens.push(bos);
    }
    tokens.extend(tok.encode(&text));
    println!("  corpus: {} chars → {} tokens", text.len(), tokens.len());

    // 3. Hyperparameters.
    let epochs: usize = args.get("epochs", 3);
    let lr: f32 = args.get("lr", 1e-4);
    let batch: usize = args.get("batch", 8);
    let seq: usize = args.get("seq", 64).min(model.cfg.context_len).max(2);
    let weight_decay: f32 = args.get("weight_decay", 0.0);
    let dropout: f32 = args.get("dropout", 0.0);
    let clip: f32 = args.get("clip", 1.0);
    let warmup: u64 = args.get("warmup", 0);
    let qat = args.has("qat");
    let seed: u64 = args.get("seed", 1337);
    let threads_req: usize = args.get("threads", 0);
    let threads = if threads_req == 0 {
        ferrum_core::num_threads()
    } else {
        threads_req
    };

    if tokens.len() < seq {
        return Err(format!(
            "corpus has {} tokens but --seq is {seq}; supply more text or a smaller --seq",
            tokens.len()
        )
        .into());
    }

    // 4. Build + configure the trainer.
    let mut tr = LlamaTrainer::new(model).map_err(|e| format!("cannot build trainer: {e}"))?;
    tr.set_optimizer(Adam::new(lr));
    tr.set_weight_decay(weight_decay);
    tr.set_dropout(dropout);
    tr.set_grad_clip(if clip > 0.0 { Some(clip) } else { None });
    tr.set_qat(qat);

    let num_windows = tokens.len() - seq + 1;
    let steps_per_epoch = num_windows.div_ceil(batch.max(1)) as u64;

    // 5. Resume optimizer state if requested (continues where a prior run stopped).
    let mut rng = if let Some(ckpt) = args.flags.get("resume") {
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        let r = tr
            .load_checkpoint_into(&bytes)
            .map_err(|e| format!("cannot resume from {ckpt}: {e}"))?;
        println!("  resumed from {ckpt} at step {}", tr.step_count());
        r
    } else {
        Rng::new(seed)
    };

    // The schedule spans the *cumulative* step timeline (after any resume), so a
    // resumed run does not start past total_steps — which would pin the LR at 0.
    if warmup > 0 {
        let total = tr.step_count() + steps_per_epoch * epochs as u64;
        tr.set_lr_schedule(Some(LrSchedule::warmup_cosine(
            lr,
            warmup,
            total.max(warmup + 1),
        )));
    }

    println!(
        "\nFine-tuning: {epochs} epochs · seq {seq} · batch {batch} · lr {lr} · \
         wd {weight_decay} · dropout {dropout} · clip {clip} · {}{} · {threads} threads",
        if warmup > 0 {
            format!("warmup {warmup} ")
        } else {
            String::new()
        },
        if qat { "QAT int8" } else { "f32" },
    );
    println!("  {num_windows} windows · {steps_per_epoch} steps/epoch");

    // 6. Train.
    for e in 0..epochs {
        let te = Instant::now();
        let loss = tr.finetune_epoch_threaded(&tokens, seq, batch, &mut rng, threads)?;
        let secs = te.elapsed().as_secs_f32();
        let toks = (num_windows * seq) as f32;
        println!(
            "  epoch {:>3}/{epochs}  loss {loss:.4}  ppl {:.2}  [{:.1}s, {:.0} tok/s, lr {:.2e}]",
            e + 1,
            loss.exp(),
            secs,
            toks / secs.max(1e-6),
            tr.lr_schedule()
                .map(|s| s.lr_at(tr.step_count()))
                .unwrap_or(lr),
        );
    }

    // 7. Save the checkpoint (weights + optimizer moments + RNG + step).
    let bytes = tr.save_checkpoint(&rng);
    std::fs::write(out_path, &bytes).map_err(|e| format!("cannot write {out_path}: {e}"))?;
    println!(
        "\nSaved fine-tune checkpoint → {out_path} ({:.2} MB)\n  \
         run it with:  train_transformer run-gguf {gguf_path} \"<prompt>\" --resume {out_path}",
        bytes.len() as f64 / 1e6
    );

    // 8. Optional sample generation from the fine-tuned model.
    if args.has("sample") {
        let prompt = tok.bos().into_iter().collect::<Vec<_>>();
        let prompt = if prompt.is_empty() {
            vec![tokens[0]]
        } else {
            prompt
        };
        let params = SamplingParams::with_temperature(args.get("temp", 0.8));
        let eos = tok.eos();
        let mut grng = Rng::new(time_seed());
        let out = tr
            .model
            .generate(&prompt, args.get("max", 48), &params, eos, &mut grng)?;
        println!("\n── sample ──\n{}\n────────────", tok.decode(&out));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// export-gguf: re-quantize / export an imported llama/qwen2 GGUF checkpoint
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_export_gguf(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use ferrum_core::{Gguf, GgufQuant, LlamaTrainer};

    if args.positional.len() < 2 {
        return Err("usage: train_transformer export-gguf <in.gguf> <out.gguf> \
                    [--quant q8_0|q4_0|q4_1|q8_1|q4_k|q5_k|q6_k|f16|f32] \
                    [--resume tuned.flck] [--force]"
            .into());
    }
    let in_path = &args.positional[0];
    let out_path = &args.positional[1];
    let quant_name = args
        .flags
        .get("quant")
        .map(String::as_str)
        .unwrap_or("q8_0");
    let quant =
        GgufQuant::from_str(quant_name).ok_or_else(|| format!("unknown --quant '{quant_name}'"))?;

    println!("Opening {in_path} (streamed)…");
    let g = Gguf::open(in_path).map_err(|e| format!("cannot open GGUF {in_path}: {e}"))?;
    println!(
        "  GGUF v{}   architecture = {}",
        g.version,
        g.architecture().unwrap_or("?")
    );

    // Export re-quantizes from f32, so guard for an f32-sized load.
    let est = estimate_resident_bytes(&g, None);
    println!("  estimated resident (f32) ≈ {:.2} GB", est as f64 / 1e9);
    if let Some(avail) = available_memory_bytes() {
        if (est as f64) > 0.9 * avail as f64 && !args.has("force") {
            return Err(format!(
                "estimated resident memory ({:.2} GB) exceeds 90% of available ({:.2} GB) — \
                 pass --force to attempt it anyway.",
                est as f64 / 1e9,
                avail as f64 / 1e9
            )
            .into());
        }
    }

    println!("Loading weights (f32)…");
    let mut model = g
        .load_llama_prec(None)
        .map_err(|e| format!("cannot load model: {e}"))?;

    // Optional: apply a fine-tune checkpoint's f32 masters before export.
    if let Some(ckpt) = args.flags.get("resume") {
        println!("Applying fine-tune checkpoint {ckpt}…");
        let bytes = std::fs::read(ckpt)?;
        let mut trainer = LlamaTrainer::new(model)?;
        trainer.load_checkpoint_into(&bytes)?;
        model = trainer.model;
    }

    println!("Writing {out_path}  (--quant {quant_name})…");
    ferrum_core::write_llama_gguf(&model, &g, quant, out_path)
        .map_err(|e| format!("export failed: {e}"))?;
    let sz = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    println!("Done: {out_path}  ({:.2} MB)", sz as f64 / 1e6);
    Ok(())
}
