# 🧬 How-To-Use Tutorial: Building Custom Edge SLMs

This tutorial walks you through training a custom Causal Small Language Model (SLM) on your own raw text dataset, exporting the model to a standalone `.bin` file, and integrating it into an interactive browser-side WASM playground.

---

## Step 1: Prepare Your Custom Corpus

Your corpus should be a raw text file containing the domain style you want the edge model to replicate (e.g. fantasy planet names, code snippets, chord progressions). Keep the corpus under **20 KB** for fast CPU training convergence.

Create a parent folder structure and a raw text corpus:
```rust
// my_slm/src/main.rs
const CORPUS: &str = "\
valinor: deep green elven forests
gondor: white stone towers
rohan: grassy green fields
mordor: red volcanic ash
";
```

---

## Step 2: Write Your Training script

Use the unified `GenerativeSLM` library module from `ferrum_core` to compile your dataset and train the neural network in just a few lines of code:

```rust
use ferrum_core::{slm::GenerativeSLM, Rng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(42);
    let context_len = 4; // Sliding character-level context
    let hidden_size = 64;
    let epochs = 300;

    println!("Training custom Generative SLM...");
    let slm = GenerativeSLM::train(
        CORPUS,
        context_len,
        hidden_size,
        epochs,
        0.08, // learning rate
        0.9,  // SGD momentum
        16,   // minibatch size
        &mut rng,
    )?;

    // Export to standalone binary format
    let model_path = "fantasy_world.bin";
    let bytes = slm.to_bytes()?;
    std::fs::write(model_path, &bytes)?;
    println!("Saved self-contained model to {model_path}!");
    
    Ok(())
}
```

Compile and run your script:
```bash
cargo run --release
```

---

## Step 3: Integrate with the WASM Web Playground

To run your custom model directly in the browser via WebAssembly:

1. **Host Directory Structure**: Create a dedicated subdirectory in your hosted web workspace:
   ```bash
   mkdir -p web/datasets/fantasy_world/
   cp fantasy_world.bin web/datasets/fantasy_world/model.bin
   ```

2. **Load inside Javascript**: Import our universal WASM loader and start streaming predictions autoregressively:

```html
<!-- index.html -->
<script type="module">
    import { loadModel, generateAutoregressive } from '../shared/engine.js';

    async function runGenerator() {
        // Load WASM and fetch model.bin
        const slm = await loadModel('../datasets/fantasy_world/model.bin');
        
        const seed = "vali";
        console.log(`Seed prompt: [${seed}]`);

        // Generate 30 characters autoregressively
        await generateAutoregressive(
            slm,
            seed,
            30,
            0.15, // temperature
            (nextChar) => {
                // Stream character to console or DOM element
                process.stdout.write(nextChar);
            }
        );
    }
    
    runGenerator();
</script>
```

3. **Serve**: Start a local web server to check your web implementation:
   ```bash
   python3 -m http.server 8080 --directory web
   # Open browser at http://localhost:8080
   ```
