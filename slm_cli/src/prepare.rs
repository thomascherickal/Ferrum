//! `train_transformer prepare`: stream raw text, clean it per document, drop
//! duplicates and out-of-range lengths, split train/val/test by a seeded
//! content hash, and write a manifest that pins every input and output by
//! sha256 — a deterministic, DVC-friendly data stage.

use ferrum_core::{
    clean_corpus, gutenberg_bodies, normalize_nfkc, squeeze_spaces, strip_invisible, CleanOptions,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const JOIN: &str = "\n<|endoftext|>\n";

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

pub struct PrepareOptions {
    pub inputs: Vec<String>,
    pub out: String,
    pub doc_sep: String,
    pub lossy: bool,
    pub nfkc: bool,
    pub lowercase: bool,
    pub strip_gutenberg: bool,
    pub min_chars: usize,
    pub max_chars: usize,
    pub dedup: bool,
    pub val_frac: f64,
    pub test_frac: f64,
    pub seed: u64,
    /// The raw argument list, recorded in the manifest.
    pub args: Vec<String>,
}

struct Split {
    path: PathBuf,
    w: BufWriter<std::fs::File>,
    hash: Sha256,
    bytes: u64,
    docs: u64,
    chars: u64,
}

impl Split {
    fn create(dir: &Path, name: &str) -> std::io::Result<Self> {
        let path = dir.join(format!("{name}.txt"));
        Ok(Split {
            w: BufWriter::new(std::fs::File::create(&path)?),
            path,
            hash: Sha256::new(),
            bytes: 0,
            docs: 0,
            chars: 0,
        })
    }
    fn write(&mut self, doc: &str) -> std::io::Result<()> {
        if self.docs > 0 {
            self.put(JOIN.as_bytes())?;
        }
        self.put(doc.as_bytes())?;
        self.docs += 1;
        self.chars += doc.chars().count() as u64;
        Ok(())
    }
    fn put(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.w.write_all(b)?;
        self.hash.update(b);
        self.bytes += b.len() as u64;
        Ok(())
    }
    fn finish(mut self) -> std::io::Result<Value> {
        self.w.flush()?;
        Ok(json!({
            "path": self.path.to_string_lossy(),
            "bytes": self.bytes,
            "sha256": hex(&self.hash.finalize()),
            "documents": self.docs,
            "chars": self.chars,
        }))
    }
}

/// Every regular, non-hidden file under each input (directories walked
/// recursively), sorted by path.
fn expand_inputs(inputs: &[String]) -> std::io::Result<Vec<PathBuf>> {
    fn walk(p: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if p.is_dir() {
            for e in std::fs::read_dir(p)? {
                let e = e?.path();
                if !e
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                {
                    walk(&e, out)?;
                }
            }
        } else {
            out.push(p.to_path_buf());
        }
        Ok(())
    }
    let mut out = Vec::new();
    for i in inputs {
        let p = Path::new(i);
        if !p.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("input {i} does not exist"),
            ));
        }
        walk(p, &mut out)?;
    }
    out.sort();
    Ok(out)
}

/// Stream `path` as raw documents split on `sep` (byte-exact). An empty
/// separator yields the whole file as one document. Calls `f(bytes, offset)`.
fn for_each_raw_doc(
    path: &Path,
    sep: &str,
    hasher: &mut Sha256,
    total: &mut u64,
    mut f: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<(), String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, file);
    let sep = sep.as_bytes();
    let mut doc: Vec<u8> = Vec::new();
    let mut doc_start = 0u64;
    let mut line = Vec::new();
    loop {
        line.clear();
        let n = r.read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&line);
        *total += n as u64;
        doc.extend_from_slice(&line);
        if !sep.is_empty() {
            // A separator can only complete inside the bytes just appended.
            let mut from = doc.len().saturating_sub(line.len() + sep.len() - 1);
            while let Some(i) = doc[from..]
                .windows(sep.len())
                .position(|w| w == sep)
                .map(|i| from + i)
            {
                f(&doc[..i], doc_start)?;
                doc_start += (i + sep.len()) as u64;
                doc.drain(..i + sep.len());
                from = 0; // what remains is the short tail of this line
            }
        }
    }
    f(&doc, doc_start)
}

pub fn run(o: &PrepareOptions) -> Result<(), Box<dyn std::error::Error>> {
    if !(0.0..1.0).contains(&o.val_frac)
        || !(0.0..1.0).contains(&o.test_frac)
        || o.val_frac + o.test_frac >= 1.0
    {
        return Err("--val-frac and --test-frac must be in [0, 1) and sum to less than 1".into());
    }
    let out = Path::new(&o.out);
    std::fs::create_dir_all(out)?;
    // A failed run must not leave a stale manifest beside truncated splits.
    match std::fs::remove_file(out.join("manifest.json")) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
        _ => {}
    }
    let mut splits = [
        Split::create(out, "train")?,
        Split::create(out, "val")?,
        Split::create(out, "test")?,
    ];
    let val_cut = (o.val_frac * 10_000.0).round() as u64;
    let test_cut = val_cut + (o.test_frac * 10_000.0).round() as u64;
    let clean_opts = CleanOptions {
        strip_gutenberg: false,
        lowercase: o.lowercase,
        collapse_whitespace: true,
        normalize_punctuation: true,
        strip_control_chars: true,
        max_chars: None,
    };
    let mut rejected: BTreeMap<&str, u64> = [
        "duplicate",
        "empty",
        "invalid_utf8",
        "too_long",
        "too_short",
    ]
    .map(|k| (k, 0))
    .into();
    let mut seen: HashSet<[u8; 16]> = HashSet::new();
    let mut uniq: HashSet<char> = HashSet::new();
    let (mut lines, mut words) = (0u64, 0u64);
    let mut inputs = Vec::new();

    for path in expand_inputs(&o.inputs)? {
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        for_each_raw_doc(&path, &o.doc_sep, &mut hasher, &mut total, |raw, offset| {
            let text = match std::str::from_utf8(raw) {
                Ok(s) => std::borrow::Cow::Borrowed(s),
                Err(_) if o.lossy => {
                    *rejected.get_mut("invalid_utf8").unwrap() += 1;
                    String::from_utf8_lossy(raw)
                }
                Err(e) => {
                    return Err(format!(
                        "{}: invalid UTF-8 at byte {} (pass --lossy to replace invalid bytes)",
                        path.display(),
                        offset + e.valid_up_to() as u64
                    ))
                }
            };
            let books = if o.strip_gutenberg {
                gutenberg_bodies(&text)
            } else {
                vec![text.as_ref()]
            };
            for book in books {
                let mut doc = strip_invisible(book);
                if o.nfkc {
                    doc = normalize_nfkc(&doc);
                }
                let doc = squeeze_spaces(&clean_corpus(&doc, &clean_opts));
                let doc = doc.trim();
                let n = doc.chars().count();
                let reason = if n == 0 {
                    Some("empty")
                } else if n < o.min_chars {
                    Some("too_short")
                } else if o.max_chars > 0 && n > o.max_chars {
                    Some("too_long")
                } else {
                    None
                };
                if let Some(r) = reason {
                    *rejected.get_mut(r).unwrap() += 1;
                    continue;
                }
                if o.dedup {
                    let key: [u8; 16] = Sha256::digest(doc.as_bytes())[..16].try_into().unwrap();
                    if !seen.insert(key) {
                        *rejected.get_mut("duplicate").unwrap() += 1;
                        continue;
                    }
                }
                let mut h = Sha256::new();
                h.update(o.seed.to_le_bytes());
                h.update(doc.as_bytes());
                let bucket = u64::from_be_bytes(h.finalize()[..8].try_into().unwrap()) % 10_000;
                let which = if bucket < val_cut {
                    1
                } else if bucket < test_cut {
                    2
                } else {
                    0
                };
                if which == 0 {
                    uniq.extend(doc.chars());
                    lines += doc.lines().count() as u64;
                    words += doc.split_whitespace().count() as u64;
                }
                splits[which].write(doc).map_err(|e| e.to_string())?;
            }
            Ok(())
        })?;
        inputs.push(json!({"path": path.to_string_lossy(), "bytes": total, "sha256": hex(&hasher.finalize())}));
    }

    let [train, val, test] = splits;
    let train = train.finish()?;
    let train_stats = json!({
        "bytes": train["bytes"], "chars": train["chars"], "lines": lines,
        "unique_chars": uniq.len(), "words": words,
    });
    let mut outputs = Map::new();
    outputs.insert("test".into(), test.finish()?);
    outputs.insert("train".into(), train);
    outputs.insert("val".into(), val.finish()?);
    let manifest = json!({
        "args": o.args,
        "ferrum_version": env!("CARGO_PKG_VERSION"),
        "inputs": inputs,
        "outputs": outputs,
        "rejected": rejected,
        "train_stats": train_stats,
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    let docs: Vec<String> = ["train", "val", "test"]
        .iter()
        .map(|s| format!("{s}={}", manifest["outputs"][s]["documents"]))
        .collect();
    println!(
        "prepared {} → {}  ({})  rejected {:?}",
        o.inputs.join(" "),
        o.out,
        docs.join(" "),
        manifest["rejected"]
    );
    Ok(())
}
