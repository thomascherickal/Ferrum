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
            "context", "embed", "heads", "blocks", "hidden", "epochs",
            "lr", "batch", "seed", "chars", "temp", "gen-seed", "vocab",
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
        Args { positional, verbose, flags, bools }
    }

    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.flags.get(key).and_then(|s| s.parse().ok()).unwrap_or(default)
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
         \x20 train_transformer info     <model.bin>\n\n\
         TRAIN / RUN options:\n\
         \x20 --context N  --embed N  --heads N  --blocks N  --hidden N\n\
         \x20 --epochs N   --lr F     --batch N  --vocab N  --seed N\n\
         \x20 --force      --sample   --verbose|-v\n\
         \x20 (--vocab 0 = character-level; >=256 = byte-level BPE, default 512)\n\n\
         GENERATE / RUN options:\n\
         \x20 --chars N    --temp F   --gen-seed N\n\n\
         If <model.bin> already exists, train/run load the saved weights from\n\
         disk instead of retraining. Pass --force to retrain from scratch.\n\n\
         EXAMPLES:\n\
         \x20 train_transformer train corpus.txt model.bin --epochs 200\n\
         \x20 train_transformer run   corpus.txt model.bin \"Once upon a time\" --chars 300\n\
         \x20 train_transformer generate model.bin \"Once upon a time\" --temp 0.7"
    );
}

fn read_corpus(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let corpus = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read corpus {path}: {e}"))?;
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
    println!("  Context : {}   Embed: {}   Hidden: {}", cfg.context_len, cfg.embed_dim, cfg.hidden_dim);
    println!("  Heads   : {}   Blocks: {}", cfg.num_heads, cfg.num_blocks);
    println!("  Tokenizer: {tok_desc}");
    println!("  Epochs  : {}   LR: {}   Batch: {}   Seed: {seed}", cfg.epochs, cfg.lr, cfg.batch_size);
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let epochs = cfg.epochs;
    let report_every = (epochs / 20).max(1);
    let progress = |ep: usize, loss: f32| {
        if ep == 1 || ep % report_every == 0 || ep == epochs {
            println!("  epoch {ep:>5}/{epochs}   loss = {loss:.6}");
        }
    };

    let slm = GenerativeSLM::train_transformer_config(corpus, &cfg, &mut rng, progress)?;

    println!("\nTrained in {:.2}s.", t0.elapsed().as_secs_f32());
    let vocab_kind = if slm.meta.tokenizer_state.is_empty() { "chars" } else { "BPE tokens" };
    println!(
        "  vocab = {} {}   context = {}   output_dim = {}",
        slm.meta.output_dim,
        vocab_kind,
        slm.meta.input_dim,
        slm.meta.output_dim
    );

    slm.save(model_path)?;
    let size = std::fs::metadata(model_path)?.len();
    println!("Saved {size} bytes → {model_path} (int8-quantized FINF v5)");

    // Verify the file roundtrips.
    let reloaded = GenerativeSLM::load(model_path)?;
    println!("Reload check: OK ({} layers).", reloaded.model.len());
    Ok(reloaded)
}

fn print_sample(
    slm: &GenerativeSLM,
    corpus: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let context_len = slm.meta.input_dim;
    let seed_text: String = corpus.chars().filter(|&c| c != '\r').take(context_len).collect();
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
    println!("{out}");
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
    println!("{out}");
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
    Ok(slm.generate(seed_text, num_chars, temp, &mut rng)?)
}

fn cmd_info(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.is_empty() {
        return Err("usage: train_transformer info <model.bin>".into());
    }
    let model_path = &args.positional[0];
    let bytes = std::fs::read(model_path)
        .map_err(|e| format!("cannot read model {model_path}: {e}"))?;
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
        if version == 5 { " (int8-quantized)" } else { " (f32)" }
    );
    println!("Name      : {}", m.dataset_name);
    println!("Task      : {:?}", m.task);
    println!("Input dim : {}", m.input_dim);
    println!("Output dim: {}", m.output_dim);
    if m.tokenizer_state.is_empty() {
        println!("Tokenizer : character-level ({} chars)", m.class_names.len());
    } else {
        let merges = m.tokenizer_state.split(';').filter(|s| !s.is_empty()).count();
        println!("Tokenizer : byte-level BPE ({} tokens, {} merges)", m.output_dim, merges);
    }
    println!("Layers    : {}", slm.model.len());
    Ok(())
}
