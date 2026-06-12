//! `train_transformer` — corpus → Small Language Model trainer & generator.
//!
//! This is the companion to `train_cli` (which handles tabular CSV models).
//! It trains a character-level causal Small Language Model from any raw text
//! corpus and exports a self-contained FINF model, then lets you generate text
//! from it — entirely on CPU, zero external dependencies.
//!
//! ## Usage
//!
//! ```text
//! train_transformer train    <corpus.txt> <model.bin> [options]
//! train_transformer generate <model.bin>  <seed text> [options]
//! train_transformer info     <model.bin>
//! ```
//!
//! ### `train` options
//! ```text
//!   --arch <transformer|embedded|mlp>  architecture            (default transformer)
//!   --context <N>   context window in characters               (default 16)
//!   --embed   <N>   embedding dimension (transformer/embedded)  (default 32)
//!   --heads   <N>   attention heads (transformer)               (default 4)
//!   --blocks  <N>   transformer blocks (transformer)            (default 2)
//!   --hidden  <N>   FFN / hidden width                          (default 64)
//!   --epochs  <N>   training epochs                             (default 100)
//!   --lr      <F>   learning rate                  (default 0.01 tf / 0.05 mlp)
//!   --momentum<F>   SGD momentum (embedded/mlp)                 (default 0.9)
//!   --batch   <N>   minibatch size                             (default 16)
//!   --seed    <N>   RNG seed                                   (default 1337)
//!   --quantize      export int8-quantised FINF (≈4× smaller)
//!   --sample        print a short sample after training
//!   --verbose | -v  print all engine internals
//! ```
//!
//! ### `generate` options
//! ```text
//!   --chars <N>  characters to generate            (default 200)
//!   --temp  <F>  sampling temperature              (default 0.8)
//!   --seed  <N>  RNG seed                          (default time-based)
//!   --verbose | -v
//! ```

use ferrum_core::{GenerativeSLM, Rng, TaskType};
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
            "arch", "context", "embed", "heads", "blocks", "hidden", "epochs",
            "lr", "momentum", "batch", "seed", "chars", "temp",
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
    fn get_str<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.flags.get(key).map(|s| s.as_str()).unwrap_or(default)
    }
    fn has(&self, key: &str) -> bool {
        self.bools.contains(key)
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
        "train_transformer — ferrum SLM trainer & generator\n\n\
         USAGE:\n\
         \x20 train_transformer train    <corpus.txt> <model.bin> [options]\n\
         \x20 train_transformer generate <model.bin>  <seed text> [options]\n\
         \x20 train_transformer info     <model.bin>\n\n\
         TRAIN options:\n\
         \x20 --arch <transformer|embedded|mlp>  (default transformer)\n\
         \x20 --context N  --embed N  --heads N  --blocks N  --hidden N\n\
         \x20 --epochs N   --lr F     --momentum F --batch N  --seed N\n\
         \x20 --quantize   --sample   --verbose|-v\n\n\
         GENERATE options:\n\
         \x20 --chars N    --temp F   --seed N    --verbose|-v\n\n\
         EXAMPLES:\n\
         \x20 train_transformer train corpus.txt model.bin --arch transformer --epochs 200\n\
         \x20 train_transformer generate model.bin \"Once upon a time\" --chars 300 --temp 0.7"
    );
}

fn cmd_train(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err("usage: train_transformer train <corpus.txt> <model.bin> [options]".into());
    }
    let corpus_path = &args.positional[0];
    let model_path = &args.positional[1];

    let corpus = std::fs::read_to_string(corpus_path)
        .map_err(|e| format!("cannot read corpus {corpus_path}: {e}"))?;
    if corpus.trim().is_empty() {
        return Err(format!("corpus {corpus_path} is empty").into());
    }

    let arch = args.get_str("arch", "transformer");
    let context: usize = args.get("context", 16);
    let embed: usize = args.get("embed", 32);
    let heads: usize = args.get("heads", 4);
    let blocks: usize = args.get("blocks", 2);
    let hidden: usize = args.get("hidden", 64);
    let epochs: usize = args.get("epochs", 100);
    let momentum: f32 = args.get("momentum", 0.9);
    let batch: usize = args.get("batch", 16);
    let seed: u64 = args.get("seed", 1337);
    let default_lr = if arch == "mlp" || arch == "embedded" { 0.05 } else { 0.01 };
    let lr: f32 = args.get("lr", default_lr);

    let mut rng = Rng::new(seed);
    let chars = corpus.chars().filter(|&c| c != '\r').count();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("  ferrum SLM trainer");
    println!("  Corpus  : {corpus_path}  ({chars} chars)");
    println!("  Arch    : {arch}");
    println!("  Context : {context}   Embed: {embed}   Hidden: {hidden}");
    if arch == "transformer" {
        println!("  Heads   : {heads}   Blocks: {blocks}");
    }
    println!("  Epochs  : {epochs}   LR: {lr}   Batch: {batch}   Seed: {seed}");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let report_every = (epochs / 20).max(1);
    let progress = |ep: usize, loss: f32| {
        if ep == 1 || ep % report_every == 0 || ep == epochs {
            println!("  epoch {ep:>5}/{epochs}   loss = {loss:.6}");
        }
    };

    let slm = match arch {
        "transformer" => GenerativeSLM::train_transformer_with_callback(
            &corpus, context, embed, heads, blocks, hidden, epochs, lr, batch, &mut rng, progress,
        )?,
        "embedded" => GenerativeSLM::train_embedded_with_callback(
            &corpus, context, embed, hidden, epochs, lr, momentum, batch, &mut rng, progress,
        )?,
        "mlp" => GenerativeSLM::train_with_callback(
            &corpus, context, hidden, epochs, lr, momentum, batch, &mut rng, progress,
        )?,
        other => {
            return Err(format!("unknown --arch '{other}' (expected transformer|embedded|mlp)").into())
        }
    };

    println!("\nTrained in {:.2}s.", t0.elapsed().as_secs_f32());
    println!(
        "  vocab = {} chars   input_dim = {}   output_dim = {}",
        slm.meta.class_names.len(),
        slm.meta.input_dim,
        slm.meta.output_dim
    );

    let bytes = if args.has("quantize") {
        slm.to_bytes_quantized()?
    } else {
        slm.to_bytes()?
    };
    if let Some(parent) = std::path::Path::new(model_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(model_path, &bytes)?;
    println!(
        "Saved {} bytes → {model_path}{}",
        bytes.len(),
        if args.has("quantize") { " (int8-quantised)" } else { "" }
    );

    // Verify the file roundtrips, then optionally print a sample continuation.
    let reloaded = GenerativeSLM::from_bytes(&bytes)?;
    println!("Reload check: OK ({} layers).", reloaded.model.len());

    if args.has("sample") {
        let seed_text: String = corpus.chars().filter(|&c| c != '\r').take(context).collect();
        let mut g_rng = Rng::new(time_seed());
        let sample = reloaded.generate(&seed_text, 120, 0.7, &mut g_rng)?;
        println!("\n── sample ──\n{sample}\n────────────");
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

    let bytes = std::fs::read(model_path)
        .map_err(|e| format!("cannot read model {model_path}: {e}"))?;
    let slm = GenerativeSLM::from_bytes(&bytes)?;

    let num_chars: usize = args.get("chars", 200);
    let temp: f32 = args.get("temp", 0.8);
    let seed: u64 = args.get("seed", time_seed());

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

    let mut rng = Rng::new(seed);
    let out = slm.generate(&seed_text, num_chars, temp, &mut rng)?;
    println!("{out}");
    Ok(())
}

fn cmd_info(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.is_empty() {
        return Err("usage: train_transformer info <model.bin>".into());
    }
    let model_path = &args.positional[0];
    let bytes = std::fs::read(model_path)
        .map_err(|e| format!("cannot read model {model_path}: {e}"))?;
    let slm = GenerativeSLM::from_bytes(&bytes)?;
    let m = &slm.meta;
    println!("Model     : {model_path}  ({} bytes)", bytes.len());
    println!("Name      : {}", m.dataset_name);
    println!("Task      : {:?}", m.task);
    println!("Input dim : {}", m.input_dim);
    println!("Output dim: {}", m.output_dim);
    println!("Vocab     : {} characters", m.class_names.len());
    println!("Layers    : {}", slm.model.len());
    Ok(())
}
