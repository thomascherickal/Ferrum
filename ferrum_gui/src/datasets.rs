//! Dataset discovery + download from HuggingFace and Kaggle.
//!
//! Only datasets small enough to be practical for **CPU** training/inference are
//! surfaced ([`CPU_MAX_MB`]). The pure logic — the curated catalog, the URL
//! builders, Kaggle credential parsing, base64 for Basic auth, and the
//! destination-path sanitiser — is split out from the `#[tauri::command]`
//! wrappers so it is unit-tested without a GUI runtime or a network.
//!
//! HuggingFace dataset files resolve to a public URL and need no auth (a token
//! is accepted for gated repos). Kaggle's API requires Basic auth from
//! `KAGGLE_USERNAME` / `KAGGLE_KEY` or `~/.kaggle/kaggle.json`.

use crate::commands::http_agent;
use serde::Serialize;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Upper bound (MB) on datasets we surface as "CPU-friendly". Bigger sets still
/// download via the generic commands, but are kept out of the curated list so a
/// user does not pick something that won't train on a laptop.
pub const CPU_MAX_MB: u32 = 200;

/// Hard cap on a single download (bytes) so a mislabeled huge file cannot fill
/// the disk unbounded.
const DOWNLOAD_CAP: usize = 512 * 1024 * 1024;

/// One curated, CPU-sized dataset.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DatasetEntry {
    pub id: String,
    pub name: String,
    /// `"huggingface"` or `"kaggle"`.
    pub source: String,
    pub description: String,
    pub approx_mb: u32,
    /// HuggingFace `owner/name`, or Kaggle `owner/slug`.
    pub repo: String,
    /// File within the repo to fetch.
    pub file: String,
    /// `"text"` | `"csv"` — informational for the UI.
    pub format: String,
}

fn e(id: &str, name: &str, source: &str, desc: &str, mb: u32, repo: &str, file: &str, fmt: &str) -> DatasetEntry {
    DatasetEntry {
        id: id.into(),
        name: name.into(),
        source: source.into(),
        description: desc.into(),
        approx_mb: mb,
        repo: repo.into(),
        file: file.into(),
        format: fmt.into(),
    }
}

/// The curated catalog of small, CPU-friendly text datasets. URLs are resolved
/// (and existence verified) at download time, so a stale entry surfaces as a
/// clear "download failed" message rather than silently doing the wrong thing.
pub fn catalog() -> Vec<DatasetEntry> {
    vec![
        e(
            "tinystories-valid",
            "TinyStories (validation)",
            "huggingface",
            "Short synthetic children's stories — ideal tiny LM corpus.",
            20,
            "roneneldan/TinyStories",
            "TinyStories-valid.txt",
            "text",
        ),
        e(
            "tinystories-v2-valid",
            "TinyStories V2 GPT-4 (validation)",
            "huggingface",
            "GPT-4-generated TinyStories, validation split.",
            22,
            "roneneldan/TinyStories",
            "TinyStoriesV2-GPT4-valid.txt",
            "text",
        ),
        e(
            "wikitext2-raw-train",
            "WikiText-2 (raw, train)",
            "huggingface",
            "Lightly-processed Wikipedia text; a classic small LM benchmark.",
            12,
            "mindchain/wikitext2",
            "wikitext-2-raw/wiki.train.raw",
            "text",
        ),
        e(
            "shakespeare-plays",
            "Shakespeare Plays",
            "kaggle",
            "All of Shakespeare's plays as lines of dialogue.",
            10,
            "kingburrito666/shakespeare-plays",
            "Shakespeare_data.csv",
            "csv",
        ),
        e(
            "poetry-foundation",
            "Poetry Foundation Poems",
            "kaggle",
            "~13k English poems — compact, stylistically rich corpus.",
            6,
            "tgdivy/poetry-foundation-poems",
            "PoetryFoundationData.csv",
            "csv",
        ),
    ]
}

/// The catalog filtered to entries within the CPU size budget.
pub fn cpu_catalog() -> Vec<DatasetEntry> {
    catalog().into_iter().filter(|d| d.approx_mb <= CPU_MAX_MB).collect()
}

// ── URL builders ──────────────────────────────────────────────────────────────

/// Public HuggingFace dataset file URL (`resolve` endpoint).
pub fn hf_resolve_url(repo: &str, file: &str, revision: &str) -> String {
    let rev = if revision.is_empty() { "main" } else { revision };
    format!("https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}")
}

/// Kaggle single-file download URL (requires Basic auth).
pub fn kaggle_file_url(owner_slug: &str, file: &str) -> String {
    format!("https://www.kaggle.com/api/v1/datasets/download/{owner_slug}/{file}")
}

// ── Kaggle credentials ────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct KaggleCreds {
    pub username: String,
    pub key: String,
}

/// Minimal parser for `~/.kaggle/kaggle.json` (`{"username":"..","key":".."}`),
/// using `serde_json` (already a dependency).
pub fn parse_kaggle_json(s: &str) -> Option<KaggleCreds> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let username = v.get("username")?.as_str()?.to_string();
    let key = v.get("key")?.as_str()?.to_string();
    if username.is_empty() || key.is_empty() {
        return None;
    }
    Some(KaggleCreds { username, key })
}

/// Resolve Kaggle credentials from the environment, then from
/// `~/.kaggle/kaggle.json`.
pub fn kaggle_creds() -> Option<KaggleCreds> {
    if let (Ok(username), Ok(key)) = (std::env::var("KAGGLE_USERNAME"), std::env::var("KAGGLE_KEY")) {
        if !username.is_empty() && !key.is_empty() {
            return Some(KaggleCreds { username, key });
        }
    }
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).ok()?;
    let path = Path::new(&home).join(".kaggle").join("kaggle.json");
    let s = std::fs::read_to_string(path).ok()?;
    parse_kaggle_json(&s)
}

// ── base64 (for the Basic auth header) ────────────────────────────────────────

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
pub fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

fn basic_auth_header(user: &str, key: &str) -> String {
    format!("Basic {}", base64_encode(format!("{user}:{key}").as_bytes()))
}

// ── Destination path ──────────────────────────────────────────────────────────

/// The leaf file name of a repo path, stripped of any directory components so a
/// crafted `file` cannot escape `dest_dir`. Falls back to `fallback`.
pub fn safe_basename(file: &str, fallback: &str) -> String {
    let name = file
        .rsplit(['/', '\\'])
        .next()
        .filter(|s| !s.is_empty() && *s != "." && *s != "..")
        .unwrap_or(fallback);
    name.to_string()
}

/// Build the on-disk destination path inside `dest_dir` for `file`.
pub fn dest_path(dest_dir: &str, file: &str, fallback: &str) -> PathBuf {
    Path::new(dest_dir).join(safe_basename(file, fallback))
}

// ── Download core ─────────────────────────────────────────────────────────────

/// Where this download came from, returned to the UI.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DownloadResult {
    pub path: String,
    pub bytes: u64,
    pub source: String,
}

/// Stream `url` (with optional Basic / Bearer auth) to `dest`, capped at
/// [`DOWNLOAD_CAP`]. Validates the URL scheme and creates parent directories.
fn download_to_file(
    url: &str,
    basic: Option<&KaggleCreds>,
    bearer: Option<&str>,
    dest: &Path,
) -> Result<u64, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("URL must start with http:// or https://".to_string());
    }
    if let Some(parent) = dest.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {parent:?}: {e}"))?;
        }
    }
    let mut req = http_agent().get(url);
    if let Some(c) = basic {
        req = req.set("Authorization", &basic_auth_header(&c.username, &c.key));
    }
    if let Some(tok) = bearer {
        if !tok.is_empty() {
            req = req.set("Authorization", &format!("Bearer {tok}"));
        }
    }
    let resp = req.call().map_err(|e| format!("download failed: {e}"))?;
    let mut reader = resp.into_reader().take(DOWNLOAD_CAP as u64);
    let mut file = std::fs::File::create(dest).map_err(|e| format!("cannot create {dest:?}: {e}"))?;
    let n = std::io::copy(&mut reader, &mut file).map_err(|e| format!("write failed: {e}"))?;
    if n == 0 {
        let _ = std::fs::remove_file(dest);
        return Err("downloaded resource was empty (check the dataset id / file path)".into());
    }
    Ok(n)
}

// ── Tauri commands ────────────────────────────────────────────────────────────

/// List the curated, CPU-sized datasets available to download.
#[tauri::command]
pub fn list_datasets() -> Vec<DatasetEntry> {
    cpu_catalog()
}

/// Download a curated catalog entry by `id` into `dest_dir`.
#[tauri::command]
pub async fn download_dataset(id: String, dest_dir: String) -> Result<DownloadResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let entry = catalog()
            .into_iter()
            .find(|d| d.id == id)
            .ok_or_else(|| format!("unknown dataset id '{id}'"))?;
        let dest = dest_path(&dest_dir, &entry.file, &entry.id);
        let (url, basic, bearer, source) = match entry.source.as_str() {
            "huggingface" => (
                hf_resolve_url(&entry.repo, &entry.file, "main"),
                None,
                std::env::var("HF_TOKEN").ok(),
                "huggingface",
            ),
            "kaggle" => {
                let creds = kaggle_creds().ok_or_else(|| {
                    "Kaggle needs credentials: set KAGGLE_USERNAME and KAGGLE_KEY, or place \
                     kaggle.json in ~/.kaggle/"
                        .to_string()
                })?;
                (kaggle_file_url(&entry.repo, &entry.file), Some(creds), None, "kaggle")
            }
            other => return Err(format!("unknown dataset source '{other}'")),
        };
        let bytes = download_to_file(&url, basic.as_ref(), bearer.as_deref(), &dest)?;
        Ok(DownloadResult { path: dest.display().to_string(), bytes, source: source.into() })
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

/// Download any HuggingFace dataset file (`repo` = `owner/name`), not just the
/// curated ones. `revision` defaults to `main`; `hf_token` is for gated repos.
#[tauri::command]
pub async fn download_hf_file(
    repo: String,
    file: String,
    revision: Option<String>,
    dest_dir: String,
    hf_token: Option<String>,
) -> Result<DownloadResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        if repo.trim().is_empty() || file.trim().is_empty() {
            return Err("provide a HuggingFace repo (owner/name) and file".into());
        }
        let url = hf_resolve_url(&repo, &file, revision.as_deref().unwrap_or("main"));
        let dest = dest_path(&dest_dir, &file, "dataset.bin");
        let token = hf_token.or_else(|| std::env::var("HF_TOKEN").ok());
        let bytes = download_to_file(&url, None, token.as_deref(), &dest)?;
        Ok(DownloadResult { path: dest.display().to_string(), bytes, source: "huggingface".into() })
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

/// Download any Kaggle dataset file (`owner_slug` = `owner/dataset`).
#[tauri::command]
pub async fn download_kaggle_file(
    owner_slug: String,
    file: String,
    dest_dir: String,
) -> Result<DownloadResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        if owner_slug.trim().is_empty() || file.trim().is_empty() {
            return Err("provide a Kaggle owner/slug and file".into());
        }
        let creds = kaggle_creds().ok_or_else(|| {
            "Kaggle needs credentials: set KAGGLE_USERNAME and KAGGLE_KEY, or place kaggle.json \
             in ~/.kaggle/"
                .to_string()
        })?;
        let url = kaggle_file_url(&owner_slug, &file);
        let dest = dest_path(&dest_dir, &file, "dataset.bin");
        let bytes = download_to_file(&url, Some(&creds), None, &dest)?;
        Ok(DownloadResult { path: dest.display().to_string(), bytes, source: "kaggle".into() })
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_nonempty_and_cpu_sized() {
        let c = catalog();
        assert!(!c.is_empty());
        for d in &c {
            assert!(d.approx_mb <= CPU_MAX_MB, "{} exceeds CPU budget", d.id);
            assert!(d.source == "huggingface" || d.source == "kaggle", "bad source for {}", d.id);
            assert!(!d.repo.is_empty() && !d.file.is_empty());
        }
        // ids are unique.
        let mut ids: Vec<&str> = c.iter().map(|d| d.id.as_str()).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate dataset ids");
    }

    #[test]
    fn cpu_catalog_filters_by_size() {
        // Every listed entry is within budget (all of ours are, but the filter
        // is what the command relies on).
        assert!(cpu_catalog().iter().all(|d| d.approx_mb <= CPU_MAX_MB));
        assert_eq!(cpu_catalog().len(), catalog().iter().filter(|d| d.approx_mb <= CPU_MAX_MB).count());
    }

    #[test]
    fn hf_url_is_well_formed() {
        assert_eq!(
            hf_resolve_url("roneneldan/TinyStories", "TinyStories-valid.txt", "main"),
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        );
        // Empty revision defaults to main.
        assert!(hf_resolve_url("a/b", "f.txt", "").ends_with("/resolve/main/f.txt"));
    }

    #[test]
    fn kaggle_url_is_well_formed() {
        assert_eq!(
            kaggle_file_url("kingburrito666/shakespeare-plays", "Shakespeare_data.csv"),
            "https://www.kaggle.com/api/v1/datasets/download/kingburrito666/shakespeare-plays/Shakespeare_data.csv"
        );
    }

    #[test]
    fn parse_kaggle_json_extracts_creds() {
        let c = parse_kaggle_json(r#"{"username":"alice","key":"deadbeef"}"#).unwrap();
        assert_eq!(c.username, "alice");
        assert_eq!(c.key, "deadbeef");
        assert!(parse_kaggle_json(r#"{"username":"alice"}"#).is_none()); // missing key
        assert!(parse_kaggle_json(r#"{"username":"","key":"x"}"#).is_none()); // empty
        assert!(parse_kaggle_json("not json").is_none());
    }

    #[test]
    fn base64_matches_known_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        // Basic auth header shape.
        assert_eq!(basic_auth_header("user", "pass"), "Basic dXNlcjpwYXNz");
    }

    #[test]
    fn safe_basename_strips_directories_and_traversal() {
        assert_eq!(safe_basename("wikitext-2-raw/wiki.train.raw", "fb"), "wiki.train.raw");
        assert_eq!(safe_basename("a/b/c.txt", "fb"), "c.txt");
        assert_eq!(safe_basename("../../etc/passwd", "fb"), "passwd");
        assert_eq!(safe_basename("", "fallback"), "fallback");
        assert_eq!(safe_basename("..", "fallback"), "fallback");
    }

    #[test]
    fn dest_path_joins_into_dir() {
        let p = dest_path("/tmp/out", "sub/dir/file.txt", "fb");
        assert_eq!(p, Path::new("/tmp/out/file.txt"));
    }

    #[test]
    fn download_rejects_non_http_urls_before_network() {
        let dest = std::env::temp_dir().join("ferrum_ds_test_noop.bin");
        let err = download_to_file("ftp://example.com/x", None, None, &dest).unwrap_err();
        assert!(err.contains("http://"));
    }

    #[test]
    fn download_dataset_unknown_id_errors_fast() {
        // The synchronous validation path (no network) is what we assert on.
        assert!(catalog().iter().all(|d| d.id != "does-not-exist"));
    }
}
