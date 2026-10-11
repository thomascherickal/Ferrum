use std::path::{Path, PathBuf};
use std::process::Command;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_prep_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn prepare(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .arg("prepare")
        .args(args)
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout),
    )
}

fn stories(n: usize) -> String {
    (0..n)
        .map(|i| format!("Story number {i}. Tom had a red ball."))
        .collect::<Vec<_>>()
        .join("\n<|endoftext|>\n")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap()
}

fn docs(p: &Path) -> Vec<String> {
    let t = read(p);
    if t.is_empty() {
        return vec![];
    }
    t.split("\n<|endoftext|>\n").map(String::from).collect()
}

#[test]
fn prepare_writes_splits_and_manifest() {
    let d = tmp("basic");
    let input = d.join("raw.txt");
    std::fs::write(&input, stories(400)).unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&[
        "--input",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(ok, "{log}");
    let n: usize = ["train", "val", "test"]
        .iter()
        .map(|s| docs(&out.join(format!("{s}.txt"))).len())
        .sum();
    assert_eq!(n, 400);
    let val = docs(&out.join("val.txt")).len();
    assert!((5..=40).contains(&val), "val docs {val}");
    let m: serde_json::Value = serde_json::from_str(&read(&out.join("manifest.json"))).unwrap();
    let train_bytes = std::fs::read(out.join("train.txt")).unwrap();
    use sha2::Digest;
    let hex = sha2::Sha256::digest(&train_bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(m["outputs"]["train"]["sha256"], hex);
    assert_eq!(m["outputs"]["train"]["bytes"], train_bytes.len());
}

#[test]
fn prepare_is_deterministic() {
    let d = tmp("det");
    let input = d.join("raw.txt");
    std::fs::write(&input, stories(200)).unwrap();
    for o in ["a", "b"] {
        let (ok, log) = prepare(&[
            "--input",
            input.to_str().unwrap(),
            "--out",
            d.join(o).to_str().unwrap(),
            "--dedup",
            "--nfkc",
        ]);
        assert!(ok, "{log}");
    }
    for f in ["train.txt", "val.txt", "test.txt"] {
        assert_eq!(
            std::fs::read(d.join("a").join(f)).unwrap(),
            std::fs::read(d.join("b").join(f)).unwrap()
        );
    }
    // Manifests differ only in the --out argument.
    let ma = read(&d.join("a/manifest.json")).replace(d.join("a").to_str().unwrap(), "OUT");
    let mb = read(&d.join("b/manifest.json")).replace(d.join("b").to_str().unwrap(), "OUT");
    assert_eq!(ma, mb);
}

#[test]
fn splits_are_stable_when_documents_are_appended() {
    let d = tmp("stable");
    let input = d.join("raw.txt");
    std::fs::write(&input, stories(300)).unwrap();
    prepare(&[
        "--input",
        input.to_str().unwrap(),
        "--out",
        d.join("a").to_str().unwrap(),
    ]);
    std::fs::write(
        &input,
        stories(300)
            + "\n<|endoftext|>\n"
            + &(300..400)
                .map(|i| format!("New story {i}."))
                .collect::<Vec<_>>()
                .join("\n<|endoftext|>\n"),
    )
    .unwrap();
    prepare(&[
        "--input",
        input.to_str().unwrap(),
        "--out",
        d.join("b").to_str().unwrap(),
    ]);
    for s in ["train", "val", "test"] {
        let before = docs(&d.join("a").join(format!("{s}.txt")));
        let after = docs(&d.join("b").join(format!("{s}.txt")));
        assert!(
            before.iter().all(|x| after.contains(x)),
            "{s}: a document changed split"
        );
    }
}

#[test]
fn prepare_cleans_bom_crlf_and_zero_width_and_dedups() {
    let d = tmp("clean");
    let input = d.join("raw.txt");
    std::fs::write(
        &input,
        "\u{FEFF}Hello\u{200B} world.\r\n<|endoftext|>\r\nHello world.\n<|endoftext|>\nBye  now.",
    )
    .unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&[
        "--input",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--dedup",
        "--val-frac",
        "0",
        "--test-frac",
        "0",
    ]);
    assert!(ok, "{log}");
    let train = docs(&out.join("train.txt"));
    assert_eq!(
        train,
        vec!["Hello world.".to_string(), "Bye now.".to_string()]
    );
    let m: serde_json::Value = serde_json::from_str(&read(&out.join("manifest.json"))).unwrap();
    assert_eq!(m["rejected"]["duplicate"], 1);
}

#[test]
fn invalid_utf8_errors_unless_lossy() {
    let d = tmp("utf8");
    let input = d.join("raw.bin");
    std::fs::write(&input, b"good text\n<|endoftext|>\nbad \xff\xfe bytes").unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&[
        "--input",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(!ok);
    assert!(
        log.contains("invalid UTF-8") && log.contains("byte"),
        "{log}"
    );
    let (ok, log) = prepare(&[
        "--input",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--lossy",
        "--val-frac",
        "0",
        "--test-frac",
        "0",
    ]);
    assert!(ok, "{log}");
    assert!(read(&out.join("train.txt")).contains("bad \u{FFFD}"));
}

#[test]
fn length_filters_and_directory_input() {
    let d = tmp("dir");
    let inp = d.join("in");
    std::fs::create_dir_all(&inp).unwrap();
    std::fs::write(inp.join("b.txt"), "tiny").unwrap();
    std::fs::write(inp.join("a.txt"), "a long enough document here").unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&[
        "--input",
        inp.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--min-chars",
        "10",
        "--val-frac",
        "0",
        "--test-frac",
        "0",
    ]);
    assert!(ok, "{log}");
    assert_eq!(
        docs(&out.join("train.txt")),
        vec!["a long enough document here".to_string()]
    );
    let m: serde_json::Value = serde_json::from_str(&read(&out.join("manifest.json"))).unwrap();
    assert_eq!(m["rejected"]["too_short"], 1);
    assert!(
        m["inputs"][0]["path"].as_str().unwrap().ends_with("a.txt"),
        "inputs sorted"
    );
}

#[test]
fn eval_writes_json_metrics() {
    let d = tmp("evaljson");
    let corpus = d.join("c.txt");
    std::fs::write(&corpus, "the cat sat on the mat. ".repeat(60)).unwrap();
    let model = d.join("m.gguf");
    let st = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args([
            "train",
            corpus.to_str().unwrap(),
            model.to_str().unwrap(),
            "--context",
            "8",
            "--embed",
            "16",
            "--heads",
            "2",
            "--blocks",
            "1",
            "--hidden",
            "32",
            "--epochs",
            "1",
            "--vocab",
            "270",
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let json = d.join("m.json");
    let st = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args([
            "eval",
            model.to_str().unwrap(),
            corpus.to_str().unwrap(),
            "--json",
            json.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let v: serde_json::Value = serde_json::from_str(&read(&json)).unwrap();
    for k in [
        "perplexity",
        "cross_entropy_nats",
        "bits_per_token",
        "bits_per_byte",
        "tokens",
        "bytes",
        "seconds",
    ] {
        assert!(v.get(k).is_some(), "missing {k}");
    }
    let bpb = v["bits_per_byte"].as_f64().unwrap();
    let want = v["cross_entropy_nats"].as_f64().unwrap() * v["tokens"].as_f64().unwrap()
        / std::f64::consts::LN_2
        / v["bytes"].as_f64().unwrap();
    assert!((bpb - want).abs() < 1e-6);
}

/// Run prepare on `raw` with `--doc-sep sep`, no val/test split, no length
/// floor; returns (ok, log, train docs, manifest).
fn prep_sep(
    name: &str,
    raw: &[u8],
    sep: &str,
    extra: &[&str],
) -> (bool, String, Vec<String>, serde_json::Value) {
    let d = tmp(name);
    let input = d.join("raw.txt");
    std::fs::write(&input, raw).unwrap();
    let out = d.join("out");
    let mut a = vec![
        "--input",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--doc-sep",
        sep,
        "--val-frac",
        "0",
        "--test-frac",
        "0",
        "--min-chars",
        "1",
    ];
    a.extend_from_slice(extra);
    let (ok, log) = prepare(&a);
    let m = std::fs::read_to_string(out.join("manifest.json"))
        .map(|t| serde_json::from_str(&t).unwrap())
        .unwrap_or(serde_json::Value::Null);
    let docs = if ok {
        docs(&out.join("train.txt"))
    } else {
        vec![]
    };
    (ok, log, docs, m)
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn separator_at_eof_and_start_and_many_per_line() {
    let (ok, log, d, m) = prep_sep("sep_eof", b"alpha|beta|", "|", &[]);
    assert!(ok, "{log}");
    assert_eq!(d, strs(&["alpha", "beta"]));
    assert_eq!(m["rejected"]["empty"], 1, "trailing empty doc rejected");
    let (ok, _, d, _) = prep_sep("sep_many", b"a1|b2||c3|d4\ne5|f6\n", "|", &[]);
    assert!(ok);
    assert_eq!(d, strs(&["a1", "b2", "c3", "d4\ne5", "f6"]));
    let (_, _, d, m) = prep_sep("sep_start", b"|x1|y2", "|", &[]);
    assert_eq!(d, strs(&["x1", "y2"]));
    assert_eq!(m["rejected"]["empty"], 1);
}

#[test]
fn multi_line_separator_split_across_reads() {
    let (ok, log, d, _) = prep_sep("sep_ml", b"one a\ntwo\n\nthree\n\n\nfour\n\n", "\n\n", &[]);
    assert!(ok, "{log}");
    assert_eq!(d, strs(&["one a\ntwo", "three", "four"]));
    let (_, _, d, _) = prep_sep("sep_ml2", b"aa\nbb\n--\n--\ncc", "\n--\n--\n", &[]);
    assert_eq!(d, strs(&["aa\nbb", "cc"]));
}

#[test]
fn empty_separator_is_one_document() {
    let (ok, log, d, m) = prep_sep("sep_empty", b"line one\nline two|x\n", "", &[]);
    assert!(ok, "{log}");
    assert_eq!(d.len(), 1);
    assert!(d[0].contains("line one") && d[0].contains("line two|x"));
    assert_eq!(m["outputs"]["train"]["documents"], 1);
}

#[test]
fn invalid_utf8_error_reports_exact_byte_offset() {
    // doc 1 = "good text\n" (10) + sep (13); doc 2 starts at 23, "\nbad " = 5 valid bytes.
    let raw = b"good text\n<|endoftext|>\nbad \xff\xfe bytes";
    let (ok, log, _, _) = prep_sep("off1", raw, "<|endoftext|>", &[]);
    assert!(!ok);
    assert!(log.contains("invalid UTF-8 at byte 28"), "{log}");
    let (ok, log, _, _) = prep_sep("off0", b"ab\xffcd", "<|endoftext|>", &[]);
    assert!(!ok);
    assert!(log.contains("invalid UTF-8 at byte 2"), "{log}");
}

#[test]
fn lossy_counts_documents_needing_replacement() {
    let raw = b"fine doc\n<|endoftext|>\nbad \xff one\n<|endoftext|>\nbad \xfe two\n<|endoftext|>\nfine 2";
    let (ok, log, d, m) = prep_sep("lossy_cnt", raw, "<|endoftext|>", &["--lossy"]);
    assert!(ok, "{log}");
    assert_eq!(d.len(), 4);
    assert_eq!(m["rejected"]["invalid_utf8"], 2);
    let (_, _, _, m) = prep_sep("lossy_none", b"all good", "<|endoftext|>", &["--lossy"]);
    assert_eq!(m["rejected"]["invalid_utf8"], 0);
}

#[test]
fn failed_run_removes_stale_manifest() {
    let d = tmp("stale");
    let good = d.join("good.txt");
    std::fs::write(&good, "a perfectly fine document").unwrap();
    let out = d.join("out");
    let (g, o) = (good.to_str().unwrap(), out.to_str().unwrap());
    let (ok, log) = prepare(&["--input", g, "--out", o]);
    assert!(ok, "{log}");
    assert!(out.join("manifest.json").exists());
    let bad = d.join("bad.txt");
    std::fs::write(&bad, b"oops \xff").unwrap();
    let (ok, _) = prepare(&["--input", bad.to_str().unwrap(), "--out", o]);
    assert!(!ok);
    assert!(!out.join("manifest.json").exists(), "stale manifest left");
    std::fs::write(out.join("manifest.json"), "{}").unwrap();
    let (ok, _) = prepare(&["--input", d.join("nope").to_str().unwrap(), "--out", o]);
    assert!(!ok);
    assert!(!out.join("manifest.json").exists());
}

fn fetch(url: &str, sha: &str, out: &Path) -> std::process::Output {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/fetch.sh");
    Command::new("bash")
        .arg(script)
        .args([url, sha, out.to_str().unwrap()])
        .output()
        .unwrap()
}

#[test]
fn fetch_sh_leaves_nothing_behind_on_failure_and_keeps_good_download() {
    if Command::new("curl").arg("--version").output().is_err() {
        eprintln!("curl absent; skipping");
        return;
    }
    let d = tmp("fetch");
    let out = d.join("sub/data.txt");
    let part = d.join("sub/data.txt.part");
    let r = fetch("http://127.0.0.1:9/x", &"0".repeat(64), &out);
    assert!(!r.status.success());
    assert!(
        !out.exists() && !part.exists(),
        "leftovers after curl failure"
    );
    let src = d.join("src.txt");
    std::fs::write(&src, "hello\n").unwrap();
    let url = format!("file://{}", src.display());
    let r = fetch(&url, &"0".repeat(64), &out);
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("sha256 mismatch"));
    assert!(!out.exists() && !part.exists(), "leftovers after mismatch");
    let want = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
    let r = fetch(&url, want, &out);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(read(&out), "hello\n");
    assert!(!part.exists());
}

#[test]
fn fetch_sh_removes_partial_file_when_curl_dies_midway() {
    // Fake curl: writes a partial body to the -o target, then fails.
    let d = tmp("fakecurl");
    let bin = d.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(
        bin.join("curl"),
        "#!/bin/bash\nwhile [ $# -gt 0 ]; do [ \"$1\" = -o ] && echo partial > \"$2\"; shift; done\nexit 18\n",
    )
    .unwrap();
    std::process::Command::new("chmod")
        .arg("+x")
        .arg(bin.join("curl"))
        .status()
        .unwrap();
    let out = d.join("data.txt");
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/fetch.sh");
    let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap());
    let r = Command::new("bash")
        .arg(script)
        .args(["http://x/y", &"0".repeat(64), out.to_str().unwrap()])
        .env("PATH", path)
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert!(!out.exists());
    assert!(
        !d.join("data.txt.part").exists(),
        "partial download left behind"
    );
}

#[test]
fn bad_fractions_rejected_and_nothing_written() {
    let d = tmp("frac");
    let input = d.join("a.txt");
    std::fs::write(&input, "some document").unwrap();
    let out = d.join("out");
    for (v, t) in [("0.7", "0.4"), ("1.0", "0"), ("-0.1", "0")] {
        let (ok, log) = prepare(&[
            "--input",
            input.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--val-frac",
            v,
            "--test-frac",
            t,
        ]);
        assert!(!ok, "{v} {t}");
        assert!(log.contains("--val-frac"), "{log}");
        assert!(!out.exists(), "output dir created despite bad args");
    }
}

#[test]
fn too_long_nfkc_lowercase_gutenberg_and_no_dedup() {
    let raw = "junk\n*** START OF THE BOOK ***\nHello \u{FB01}ne WORLD\n*** END OF THE BOOK ***\nfooter\n\
               <|endoftext|>\n*** START OF X ***\nHello \u{FB01}ne WORLD\n*** END OF X ***\n\
               <|endoftext|>\nthis document is much too long for the cap";
    let (ok, log, d, m) = prep_sep(
        "flags",
        raw.as_bytes(),
        "<|endoftext|>",
        &[
            "--strip-gutenberg",
            "--nfkc",
            "--lowercase",
            "--max-chars",
            "20",
        ],
    );
    assert!(ok, "{log}");
    assert_eq!(d, strs(&["hello fine world", "hello fine world"]));
    assert_eq!(m["rejected"]["too_long"], 1);
    assert_eq!(m["rejected"]["duplicate"], 0, "dedup is opt-in");
    let (_, _, d, m) = prep_sep(
        "flags2",
        raw.as_bytes(),
        "<|endoftext|>",
        &[
            "--strip-gutenberg",
            "--nfkc",
            "--lowercase",
            "--max-chars",
            "20",
            "--dedup",
        ],
    );
    assert_eq!(d, strs(&["hello fine world"]));
    assert_eq!(m["rejected"]["duplicate"], 1);
}

#[test]
fn directory_walk_skips_hidden_and_recurses() {
    let d = tmp("hidden");
    let inp = d.join("in");
    std::fs::create_dir_all(inp.join("sub")).unwrap();
    std::fs::create_dir_all(inp.join(".git")).unwrap();
    std::fs::write(inp.join("sub/deep.txt"), "deep document").unwrap();
    std::fs::write(inp.join(".hidden"), "hidden document").unwrap();
    std::fs::write(inp.join(".git/obj"), "git document").unwrap();
    let out = d.join("out");
    let (ok, log) = prepare(&[
        "--input",
        inp.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--val-frac",
        "0",
        "--test-frac",
        "0",
    ]);
    assert!(ok, "{log}");
    assert_eq!(
        docs(&out.join("train.txt")),
        vec!["deep document".to_string()]
    );
}
