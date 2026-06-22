# 5. Using the GUI — Ferrum SLM Studio, Step by Step

> **Who this is for:** anyone who would rather click buttons than type commands.
> This page is a complete, beginner-friendly walkthrough of **Ferrum SLM
> Studio**, the desktop app that wraps the whole engine in one window. You can do
> everything here without touching the command line.

If you prefer the command line, the project's `instructions.md` and `howtouse.md`
cover it; this page is the visual route.

---

## 5.1 What the GUI is

Ferrum SLM Studio is a **cross-platform desktop (and mobile) app** built with a
toolkit called **Tauri 2**. Its interface is plain HTML/CSS/JavaScript — no heavy
frameworks — and every button calls straight into the Rust engine you read about
on [page 3](03-the-ferrum-engine-and-its-capabilities.md). It puts the *entire*
project in one place: fetching and cleaning data, all three training paths,
streaming generation, perplexity evaluation, model inspection, the tabular tool,
a live terminal, and a system monitor.

---

## 5.2 Getting it running (one-time setup)

> **Heads-up, stated honestly by the project itself:** the GUI was authored in an
> environment *without* the system graphics libraries, so it has not been
> compiled there. The engine pieces it relies on are fully tested, but you may be
> the first to compile the GUI on your machine — install the prerequisites below,
> then report any compile errors.

**You need:**
1. **Rust** and **Cargo** (the Rust toolchain).
2. The **Tauri CLI**: `cargo install tauri-cli --version "^2"`
3. **System WebView libraries** for your OS:
   - **Linux (Debian/Ubuntu):** `libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev build-essential curl wget file libssl-dev`
   - **macOS:** Xcode command-line tools.
   - **Windows:** the WebView2 runtime (already on Windows 11) + MSVC build tools.

**To launch it**, from inside the `ferrum_gui` folder:

```bash
cd ferrum_gui
cargo tauri dev      # opens the app in development mode
# or
cargo tauri build    # produces an installer/binary for your OS
```

There is **no Node.js/npm build step** — the interface is static files served
directly.

> **Linux + Snap gotcha:** if you launch from a terminal *inside* a Snap app
> (like the VS Code snap), the app may crash with a `__libc_pthread_init` symbol
> error. The fix (a one-liner that strips the polluting environment variables) is
> in `ferrum_gui/README.md`. A normal login shell doesn't need it.

---

## 5.3 The lay of the land

Across the top is a row of **tabs**. Along the bottom is a **docked terminal**
that's always visible. Up in the top bar are **mini gauges** showing live CPU and
memory use.

The tabs, left to right:

| Tab | What you do there |
|-----|-------------------|
| **Datasets** | Download and clean the text your model will learn from. |
| **Train** | Build a model (the three training paths live here). |
| **Generate** | Type a prompt and watch the model write, live. |
| **Evaluate** | Score a model's quality on held-out text (perplexity). |
| **Models** | Inspect / reload a saved `.bin` model file. |
| **Tabular (CLI)** | Train a spreadsheet/CSV model via `train_cli`. |
| **System** | A fuller view of CPU and memory load. |
| **Terminal** (docked) | A real shell *and* the engine's live `--verbose` log. |

A friendly design touch: every form gives **clear, plain-language error
messages** when you mistype something, and if you run the app somewhere the Rust
backend can't operate (like a plain web preview), a banner tells you so instead
of failing silently.

---

## 5.4 A complete first run, click by click

Here's the natural order. Follow it once and you'll have trained and used your
own model entirely with the mouse.

### Step 1 — Datasets: get some text

A model learns from text, so first you need a **corpus** (a body of text).

1. Go to the **Datasets** tab.
2. Either paste a URL of a plain-text file (e.g. a Project Gutenberg `.txt`) and
   click **Download**, *or* click **Browse… → Load file** to use a text file you
   already have.
3. Tick the **cleaning options** you want (strip Project Gutenberg boilerplate,
   lowercase, collapse whitespace, normalise punctuation, remove control
   characters), then click **Clean & preview**. You'll see statistics —
   characters, words, lines, unique characters — and a preview of the result.
4. Click **Save corpus** to write the cleaned text to a file. This file is your
   training input.

> The cleaning is done by the engine's real `clean_corpus` function, so the GUI
> and command line produce identical corpora.

### Step 2 — Train: build the model

1. Go to the **Train** tab.
2. Pick a **method**: *transformer* (best quality — recommended),
   *embedded* (small & fast), or *one-hot* (simplest).
3. **Browse…** to your saved corpus, and choose an output path for the model
   (e.g. `model.bin`).
4. Set the knobs. Sensible beginner values:
   - **context** (field of view): 12–16
   - **embed** (must divide evenly by *heads*): 32
   - **heads**: 4
   - **blocks**: 2
   - **hidden** (FFN width): 64
   - **epochs**: 60–200
   - **learning rate (lr)**: 0.01
   - **batch**: 16
   - **vocab**: `0` for character-level, or `512` for byte-level BPE (recommended)
   - **threads**: 0 (auto-detect all cores)
   - **verbose**: tick it to watch the engine's inner log in the terminal
5. Click **Train**. A live progress bar and a falling **loss** number appear (the
   model getting less wrong, exactly as on [page 1](01-generative-ai-slms-and-transformers.md)).
   When it finishes you'll see how long it took, the final loss, the model size,
   and the tokenizer it used.

> If you enter something invalid — say an embedding size not divisible by the
> number of heads — the app refuses with a clear message instead of crashing.

### Step 3 — Generate: make it write

1. Go to the **Generate** tab.
2. **Browse…** to your trained `model.bin`.
3. Type a **seed/prompt** to start from.
4. Set **chars** (how many characters to produce) and **temperature** (creativity:
   0.2 = safe and repetitive, 0.7 = varied, 1.0+ = wild).
5. Tick **stream** to watch the text appear fragment-by-fragment, live.
6. Click **Generate**.

> For character-level models, your prompt must be at least as long as the context
> window — the app tells you the exact minimum if it's too short. BPE models don't
> have this restriction.

### Step 4 — Evaluate: is it actually any good?

1. Go to the **Evaluate** tab.
2. **Browse…** to your model, then provide **held-out text** (text the model was
   *not* trained on) by pasting it or loading a file.
3. Click **Evaluate & add row**. A table row appears with **perplexity**,
   cross-entropy, bits/token, and the "learned-nothing" baseline for comparison.

Lower perplexity is better (1.0 is perfect). Compare held-out perplexity against
the baseline (≈ vocabulary size) to confirm the model learned real structure
rather than just memorising — the honesty check from
[page 3, §3.9](03-the-ferrum-engine-and-its-capabilities.md).

### Step 5 — Models: inspect any saved file

Go to the **Models** tab, **Browse…** to any `.bin`, and click **Inspect /
reload** to see its format (FINF v4 or v5), task type, input/output dimensions,
tokenizer (character-level or BPE, with merge count), and layer count. Handy for
checking a file someone handed you.

---

## 5.5 The other tabs

### Tabular (CLI) — models from spreadsheets

Not all AI is text. The **Tabular** tab drives `train_cli`, which turns a **CSV
file** into a classifier or regressor (it auto-detects which). Point it at your
CSV, name the target, set a couple of sizes, and **Run in terminal**. Great for
"predict a number/category from columns of data" tasks — fraud scores, quality
checks, sensor readings — deployable to devices that can't run Python.

### System — watch the load

The **System** tab (and the top-bar gauges) show live **CPU usage per core** and
**memory use**, plus how many threads the Ferrum engine is currently using.
Useful for understanding why training is fast or slow on your hardware.

### Terminal — a real shell + the engine's diary

The docked **Terminal** at the bottom is two things at once:
- a genuine interactive **shell** (with a built-in `cd` that remembers where you
  are between commands), and
- the destination for the engine's **verbose log** — when you tick *verbose* on
  Train or Generate, the engine's internal trace streams here line by line.

(The interactive shell isn't available on mobile or in a web preview — there's no
shell to talk to — and the app says so clearly when that's the case.)

---

## 5.6 Cross-platform reality, stated honestly

One codebase targets desktop (Linux/macOS/Windows), mobile (Android/iOS), and —
with extra work — the web. But the platforms differ, so **features degrade
gracefully rather than break**:

| Feature | Desktop | Mobile | Web |
|---------|:------:|:------:|:---:|
| Datasets, Train, Generate, Evaluate, Models | ✅ | ✅ | ❌ |
| Interactive shell terminal | ✅ | ⛔ (no shell) | ⛔ |
| System monitor | ✅ | partial | ❌ |

On the web, Tauri's Rust backend doesn't run in a plain browser, so the
backend-powered features are unavailable and a banner explains why. (Running a
*model* in the browser is a separate path — the WASM build from
[page 2, §2.5](02-rust-and-why-it-matters.md).)

---

## 5.7 What you now know

- Ferrum SLM Studio puts the **whole project in one window**, with a tab for each
  task and a live terminal + system monitor.
- The natural workflow is **Datasets → Train → Generate → Evaluate → Models**, and
  you can do all of it by clicking.
- The app is **honest about its limits**: clear error messages, graceful
  degradation across platforms, and an upfront note that you may be the first to
  compile the GUI on your system.

You've now toured the whole project. For the unvarnished limits, read
[`07-critique.md`](07-critique.md); for whether CPU-bound inference suits your real
task, read [`08-applications.md`](08-applications.md).

---

## Sources

- Ferrum project docs in this repository: `ferrum_gui/README.md`,
  `ferrum_gui/src/commands.rs`, `ferrum_gui/ui/index.html`
- Tauri (the GUI toolkit used) — https://tauri.app/
