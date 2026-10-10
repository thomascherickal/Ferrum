//! Repository invariant: no `unsafe` code in any Ferrum crate.
//!
//! The scanner blanks out comments and string/char literals first, so docs such
//! as "no `unsafe`" and strings mentioning it are not flagged.

use std::path::Path;

/// Replace comments (`//`, nested `/* */`), string literals (plain and raw) and
/// char literals with spaces, keeping newlines so line numbers stay valid.
fn blank_non_code(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let blank = |c: char| if c == '\n' { '\n' } else { ' ' };
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < b.len() && b[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            let mut depth = 0;
            while i < b.len() {
                if b[i] == '/' && b.get(i + 1) == Some(&'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if b[i] == '*' && b.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(b[i]));
                    i += 1;
                }
            }
        } else if c == '"' || (c == 'r' && is_raw_start(&b, i)) {
            // Plain or raw string; raw has `r`, N `#`s, then `"`.
            let mut hashes = 0;
            let raw = c == 'r';
            if raw {
                out.push(' ');
                i += 1;
                while b[i] == '#' {
                    hashes += 1;
                    out.push(' ');
                    i += 1;
                }
            }
            out.push(' '); // opening quote
            i += 1;
            while i < b.len() {
                if !raw && b[i] == '\\' {
                    out.push(blank(b[i]));
                    if let Some(&n) = b.get(i + 1) {
                        out.push(blank(n));
                    }
                    i += 2;
                } else if b[i] == '"' && (0..hashes).all(|k| b.get(i + 1 + k) == Some(&'#')) {
                    for _ in 0..=hashes {
                        out.push(' ');
                    }
                    i += 1 + hashes;
                    break;
                } else {
                    out.push(blank(b[i]));
                    i += 1;
                }
            }
        } else if c == '\'' && is_char_literal(&b, i) {
            // 'x' or '\n' / '\u{..}'; a lifetime like 'a is not a literal.
            out.push(' ');
            i += 1;
            while i < b.len() && b[i] != '\'' {
                if b[i] == '\\' {
                    out.push(' ');
                    i += 1;
                }
                out.push(' ');
                i += 1;
            }
            out.push(' ');
            i += 1;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// `r"`, `r#"`, `r##"` ... and not part of an identifier (`for"`, `bar`).
fn is_raw_start(b: &[char], i: usize) -> bool {
    if i > 0 && (b[i - 1].is_alphanumeric() || b[i - 1] == '_') {
        return false;
    }
    let mut j = i + 1;
    while b.get(j) == Some(&'#') {
        j += 1;
    }
    b.get(j) == Some(&'"')
}

fn is_char_literal(b: &[char], i: usize) -> bool {
    matches!(
        (b.get(i + 1), b.get(i + 2)),
        (Some('\\'), _) | (Some(_), Some('\''))
    )
}

/// Lines (1-based) whose code contains the `unsafe` keyword.
fn unsafe_lines(src: &str) -> Vec<usize> {
    let is_id = |c: char| c.is_alphanumeric() || c == '_';
    blank_non_code(src)
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            line.match_indices("unsafe").any(|(p, _)| {
                let before = line[..p].chars().next_back();
                let after = line[p + 6..].chars().next();
                !before.is_some_and(is_id) && !after.is_some_and(is_id)
            })
        })
        .map(|(i, _)| i + 1)
        .collect()
}

#[test]
fn no_unsafe_blocks_or_impls_in_sources() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let mut bad = Vec::new();
    let mut scanned = 0;
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                if !["target", ".git", "node_modules", "gen"].contains(&name.as_str()) {
                    stack.push(p);
                }
            } else if name.ends_with(".rs") && name != "test_no_unsafe.rs" {
                scanned += 1;
                let src = std::fs::read_to_string(&p).unwrap();
                let lines: Vec<&str> = src.lines().collect();
                for n in unsafe_lines(&src) {
                    bad.push(format!("{}:{}: {}", p.display(), n, lines[n - 1].trim()));
                }
            }
        }
    }
    assert!(scanned > 20, "scanner found only {scanned} .rs files");
    assert!(bad.is_empty(), "unsafe code found:\n{}", bad.join("\n"));
}

#[test]
fn flags_real_unsafe() {
    assert_eq!(unsafe_lines("fn f() {\n    unsafe { x() }\n}"), vec![2]);
    assert_eq!(unsafe_lines("unsafe{ x() }"), vec![1]);
    assert_eq!(unsafe_lines("unsafe impl Send for T {}"), vec![1]);
    assert_eq!(unsafe_lines("pub unsafe fn f() {}"), vec![1]);
    assert_eq!(unsafe_lines("let a = 1; unsafe { }\nlet b = 2;"), vec![1]);
    // Code after a closed block comment / string on the same line still counts.
    assert_eq!(unsafe_lines("/* c */ unsafe { }"), vec![1]);
    assert_eq!(unsafe_lines("let s = \"x\"; unsafe { }"), vec![1]);
    assert_eq!(unsafe_lines("let c = 'a'; unsafe { }"), vec![1]);
    assert_eq!(unsafe_lines("fn f<'a>(x: &'a u8) { unsafe { } }"), vec![1]);
}

#[test]
fn ignores_comments() {
    assert!(unsafe_lines("// unsafe { }").is_empty());
    assert!(unsafe_lines("/// no `unsafe` here").is_empty());
    assert!(unsafe_lines("//! and **no `unsafe`**:").is_empty());
    assert!(unsafe_lines("let x = 1; // unsafe {").is_empty());
    assert!(unsafe_lines("/* unsafe { } */").is_empty());
    assert!(unsafe_lines("/* a\n unsafe\n b */ let x = 1;").is_empty());
    assert!(unsafe_lines("/* outer /* nested unsafe */ still unsafe */ let x = 1;").is_empty());
    // A block comment ends: a later line is code again.
    assert_eq!(unsafe_lines("/* a\n b */\nunsafe { }"), vec![3]);
}

#[test]
fn ignores_strings_and_chars() {
    assert!(unsafe_lines("let s = \"unsafe { }\";").is_empty());
    assert!(unsafe_lines("let s = \"a \\\" unsafe \\\" b\";").is_empty());
    assert!(unsafe_lines("let s = \"line1\nunsafe line2\";").is_empty());
    assert!(unsafe_lines("let s = r\"unsafe\";").is_empty());
    assert!(unsafe_lines("let s = r#\"say \"unsafe\" // x\"#;").is_empty());
    assert!(unsafe_lines("let s = r##\"a \"# unsafe \"##;").is_empty());
    assert!(unsafe_lines("let s = \"// not a comment\"; let t = 1;").is_empty());
    assert!(unsafe_lines("let c = '\"'; let t = 1;").is_empty());
    assert!(unsafe_lines("let c = '\\'';").is_empty());
    assert!(unsafe_lines("let c = '\\u{1F600}';").is_empty());
    // A quote char literal must not open a string that swallows later code.
    assert_eq!(unsafe_lines("let c = '\"';\nunsafe { }"), vec![2]);
}

#[test]
fn ignores_identifiers_containing_the_word() {
    assert!(unsafe_lines("#![forbid(unsafe_code)]").is_empty());
    assert!(unsafe_lines("#![deny(unsafe_code)]").is_empty());
    assert!(unsafe_lines("let not_unsafe = 1; let unsafely = 2;").is_empty());
    assert!(unsafe_lines("").is_empty());
}

#[test]
fn raw_string_prefix_requires_a_token_boundary() {
    // `bar"` is an identifier char then a plain string, not a raw string.
    assert!(unsafe_lines("let s = bar\"unsafe\";").is_empty());
    // A lone `r` identifier is code, and `r#ident` raw identifiers don't start strings.
    assert_eq!(unsafe_lines("let r = 1; unsafe { }"), vec![1]);
    assert_eq!(unsafe_lines("let r#type = 1; unsafe { }"), vec![1]);
}

#[test]
fn every_crate_root_forbids_or_denies_unsafe_code() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    for rel in [
        "ferrum_core/src/lib.rs",
        "ferrum_core/benches/gemm.rs",
        "slm_cli/src/main.rs",
        "slm_wasm/src/lib.rs",
        "slm_server/src/lib.rs",
        "slm_server/src/main.rs",
        "ferrum_gui/src/lib.rs",
        "ferrum_gui/src/main.rs",
        "ferrum_gui/build.rs",
    ] {
        let src = std::fs::read_to_string(root.join(rel)).unwrap();
        let code = blank_non_code(&src);
        assert!(
            code.contains("#![forbid(unsafe_code)]") || code.contains("#![deny(unsafe_code)]"),
            "{rel} lacks #![forbid(unsafe_code)] / #![deny(unsafe_code)]"
        );
    }
}
