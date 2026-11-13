//! Generates `mklight_types.wgsl` from the C headers of `mklight-gpu`
//! (`include/*.h`), so WGSL kernels share every `MKL_*` constant and packed
//! struct with the Metal shaders and the Rust mirror without a hand copy:
//!
//! * integer `#define MKL_*` lines (`123u`, `0xABCDu`, `(1u << 5)`, `2`)
//!   become `const MKL_*: u32`;
//! * `struct Mkl* { … };` blocks become WGSL structs. Their fields are 4-byte
//!   scalars (`MKL_U32`/`uint`/`uint32_t`, `MKL_F32`/`float`), fixed arrays of
//!   them and nested `struct Mkl*` members, so the WGSL storage layout equals
//!   the C layout (alignment 4, array stride 4, no implicit padding).
//!
//! Uniform buffers need 16-byte array strides, so kernels bind structs with
//! array members as read-only storage.

use std::fmt::Write as _;
use std::{env, fs, path::PathBuf};

fn parse_value(v: &str) -> Option<u64> {
    let v = v.trim();
    let num = |s: &str| -> Option<u64> {
        let s = s.trim().trim_end_matches(['u', 'U']);
        if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            u64::from_str_radix(h, 16).ok()
        } else {
            s.parse().ok()
        }
    };
    if let Some(inner) = v.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
        let (a, b) = inner.split_once("<<")?;
        return Some(num(a)? << num(b)?);
    }
    num(v)
}

/// WGSL keywords and reserved words that appear as C field names.
const RESERVED: &[&str] = &[
    "sample",
    "filter",
    "target",
    "precise",
    "smooth",
    "mod",
    "module",
    "self",
    "type",
    "ref",
    "enum",
    "public",
    "private",
    "static",
    "unsafe",
    "use",
    "where",
    "yield",
    "layout",
    "active",
    "as",
    "async",
    "attribute",
    "become",
    "binding_array",
    "cast",
    "catch",
    "class",
    "co_await",
    "common",
    "concept",
    "const_cast",
    "consteval",
    "constexpr",
    "constinit",
    "crate",
    "debugger",
    "decltype",
    "delete",
    "demote",
    "do",
    "dynamic_cast",
    "explicit",
    "export",
    "extends",
    "extern",
    "external",
    "fallthrough",
    "final",
    "finally",
    "friend",
    "from",
    "fxgroup",
    "get",
    "goto",
    "groupshared",
    "highp",
    "impl",
    "implements",
    "import",
    "inline",
    "instanceof",
    "interface",
    "lowp",
    "macro",
    "macro_rules",
    "match",
    "mediump",
    "meta",
    "mutable",
    "namespace",
    "new",
    "nil",
    "noexcept",
    "noinline",
    "nointerpolation",
    "non_coherent",
    "noncoherent",
    "noperspective",
    "null",
    "nullptr",
    "of",
    "operator",
    "package",
    "packoffset",
    "partition",
    "pass",
    "patch",
    "pixelfragment",
    "premerge",
    "priv",
    "protected",
    "pub",
    "readonly",
    "regardless",
    "register",
    "reinterpret_cast",
    "require",
    "resource",
    "restrict",
    "set",
    "shared",
    "sizeof",
    "snorm",
    "static_assert",
    "static_cast",
    "std",
    "subroutine",
    "super",
    "template",
    "this",
    "thread_local",
    "throw",
    "trait",
    "try",
    "typedef",
    "typeid",
    "typename",
    "typeof",
    "union",
    "unless",
    "unorm",
    "unsized",
    "using",
    "varying",
    "virtual",
    "volatile",
    "wgsl",
    "with",
    "writeonly",
    "alias",
    "break",
    "case",
    "const",
    "const_assert",
    "continue",
    "continuing",
    "default",
    "diagnostic",
    "discard",
    "else",
    "enable",
    "false",
    "fn",
    "for",
    "if",
    "let",
    "loop",
    "override",
    "requires",
    "return",
    "struct",
    "switch",
    "true",
    "var",
    "while",
    "f32",
    "u32",
    "i32",
    "bool",
];

fn field_name(n: &str) -> String {
    if RESERVED.contains(&n) {
        format!("{n}_")
    } else {
        n.to_string()
    }
}

fn scalar(t: &str) -> Option<&'static str> {
    match t {
        "MKL_U32" | "uint" | "uint32_t" => Some("u32"),
        "MKL_F32" | "float" => Some("f32"),
        "int" | "int32_t" => Some("i32"),
        // MSL vector members (8- and 16-byte aligned there and in WGSL).
        "float2" => Some("vec2<f32>"),
        "float4" => Some("vec4<f32>"),
        "uint2" => Some("vec2<u32>"),
        "uint4" => Some("vec4<u32>"),
        _ => None,
    }
}

fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("/*") {
        out.push_str(&rest[..i]);
        match rest[i..].find("*/") {
            Some(j) => rest = &rest[i + j + 2..],
            None => {
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    let include = PathBuf::from(mklight_gpu_include());
    println!("cargo:rerun-if-changed={}", include.display());
    let mut headers: Vec<_> = fs::read_dir(&include)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "h"))
        .collect();
    // The umbrella header first, then the sidecars in name order.
    headers.sort_by_key(|p| {
        (
            p.file_name().unwrap() != "mklight_gpu_types.h",
            p.file_name().unwrap().to_owned(),
        )
    });
    let mut out =
        String::from("// @generated by mklight-webgpu/build.rs from mklight-gpu/include/*.h\n");
    let mut seen_const = std::collections::HashSet::new();
    let mut seen_struct = std::collections::HashSet::new();
    let mut skipped = std::collections::HashSet::<String>::new();
    for path in headers {
        println!("cargo:rerun-if-changed={}", path.display());
        let text = typedef_structs(&strip_comments(&fs::read_to_string(&path).unwrap()));
        writeln!(out, "\n// {}", path.file_name().unwrap().to_string_lossy()).unwrap();
        for line in text.lines() {
            let Some(rest) = line.trim().strip_prefix("#define ") else {
                continue;
            };
            let mut it = rest.splitn(2, char::is_whitespace);
            let (Some(name), Some(value)) = (it.next(), it.next()) else {
                continue;
            };
            if !name.starts_with("MKL_") || name.contains('(') {
                continue;
            }
            if let Some(v) = parse_value(value)
                && v <= u32::MAX as u64
                && seen_const.insert(name.to_string())
            {
                writeln!(out, "const {name}: u32 = {v:#x}u;").unwrap();
            }
        }
        // Structs: `struct Name {` … `};`
        let mut rest = text.as_str();
        while let Some(i) = rest.find("struct Mkl") {
            let after = &rest[i + "struct ".len()..];
            let name_end = after
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(after.len());
            let name = &after[..name_end];
            let tail = after[name_end..].trim_start();
            if !tail.starts_with('{') {
                // A use (`struct MklX member;`, sizeof), not a definition.
                rest = &after[name_end..];
                continue;
            }
            let body_start = after[name_end..].find('{').unwrap() + name_end + 1;
            let body_end = after[body_start..].find('}').unwrap() + body_start;
            let body = &after[body_start..body_end];
            rest = &after[body_end..];
            if !seen_struct.insert(name.to_string()) {
                continue;
            }
            let mut fields = Vec::new();
            let mut unknown = None;
            for decl in body.split(';') {
                let decl = decl.trim().replace("unsigned int ", "uint ");
                if decl.is_empty() {
                    continue;
                }
                let (ty, names) = if let Some(d) = decl.strip_prefix("struct ") {
                    let (t, n) = d.split_once(char::is_whitespace).unwrap();
                    if !seen_struct.contains(t) || skipped.contains(t) {
                        unknown = Some(t.to_string());
                    }
                    (t.to_string(), n.to_string())
                } else {
                    let (t, n) = decl.split_once(char::is_whitespace).unwrap();
                    match scalar(t) {
                        Some(s) => (s.to_string(), n.to_string()),
                        None if seen_struct.contains(t) && !skipped.contains(t) => {
                            (t.to_string(), n.to_string())
                        }
                        None => {
                            unknown = Some(t.to_string());
                            (String::new(), n.to_string())
                        }
                    }
                };
                for n in names.split(',') {
                    let n = n.trim();
                    if let Some((base, dims)) = n.split_once('[') {
                        // `x[a][b]` is an array of a arrays of b.
                        let dims: Vec<&str> = dims
                            .split('[')
                            .map(|d| d.trim().trim_end_matches(']').trim())
                            .collect();
                        let mut t = ty.clone();
                        for d in dims.iter().rev() {
                            t = format!("array<{t}, {d}>");
                        }
                        fields.push(format!("    {}: {t},", field_name(base.trim())));
                    } else {
                        fields.push(format!("    {}: {ty},", field_name(n)));
                    }
                }
            }
            if let Some(t) = unknown {
                writeln!(out, "// {name}: not generated (field type {t})").unwrap();
                skipped.insert(name.to_string());
                continue;
            }
            writeln!(out, "struct {name} {{").unwrap();
            for f in fields {
                writeln!(out, "{f}").unwrap();
            }
            writeln!(out, "}}").unwrap();
        }
    }
    let dest = PathBuf::from(env::var("OUT_DIR").unwrap()).join("mklight_types.wgsl");
    fs::write(dest, out).unwrap();
    trace_kernel();
}

/// `typedef struct { … } MklName;` as `struct MklName { … };`, the form the
/// struct parser reads.
fn typedef_structs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("typedef struct {") {
        let body_start = i + "typedef struct ".len();
        let Some(close) = rest[body_start..].find('}').map(|j| j + body_start) else {
            break;
        };
        let Some(semi) = rest[close..].find(';').map(|j| j + close) else {
            break;
        };
        let name = rest[close + 1..semi].trim();
        out.push_str(&rest[..i]);
        if name.starts_with("Mkl") && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            write!(out, "struct {name} {}}};", &rest[body_start..close]).unwrap();
        } else {
            out.push_str(&rest[i..=semi]);
        }
        rest = &rest[semi + 1..];
    }
    out.push_str(rest);
    out
}

/// Replace whole-word occurrences of `word` (identifier characters around
/// it end the match).
fn replace_word(s: &str, word: &str, by: &str) -> String {
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(word) {
        let before = rest[..i].chars().next_back();
        let after = rest[i + word.len()..].chars().next();
        out.push_str(&rest[..i]);
        if before.is_some_and(ident) || after.is_some_and(ident) {
            out.push_str(word);
        } else {
            out.push_str(by);
        }
        rest = &rest[i + word.len()..];
    }
    out.push_str(rest);
    out
}

/// `trace.wgsl` from `kernels/trace/`: `head.wgsl` (types, helpers and the
/// scalar primitives) with the correctly rounded division and square root of
/// `rn.wgsl` inserted before its primitives; the surface math of
/// `generic.wgsl` written once over a real type `R_` (jet `J_`, sag `S_`,
/// coordinates `U_`, function suffix `@`) and expanded for f32 (`_f`) and
/// double-float (`_d`); then the trace itself (`tail.wgsl`). Every division
/// and square root outside `rn.wgsl` becomes its round-to-nearest variant,
/// so the kernel is bitwise the CPU-f32 emulation on hardware whose `/` and
/// `sqrt` are approximations (GPU.md §11).
fn trace_kernel() {
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels/trace");
    println!("cargo:rerun-if-changed={}", dir.display());
    let read = |f: &str| {
        let p = dir.join(f);
        println!("cargo:rerun-if-changed={}", p.display());
        fs::read_to_string(p).unwrap()
    };
    let rounded = |s: &str| {
        let mut s = s.to_string();
        for (a, b) in [
            ("mkl_div(", "mkl_div_rn("),
            ("mkl_sqrt(", "mkl_sqrt_rn("),
            ("mkl_df_div(", "mkl_df_div_rn("),
            ("mkl_df_sqrt(", "mkl_df_sqrt_rn("),
            ("mkl_hypot3(", "mkl_hypot3_rn("),
        ] {
            s = s.replace(a, b);
        }
        s
    };
    let head = read("head.wgsl");
    let cut = head
        .find("// ---------------------------------------------------------------------------\n// Scalar primitives")
        .expect("head.wgsl: scalar primitives marker");
    let rn = read("rn.wgsl");
    let mut out = format!(
        "{}{}\n{}",
        &head[..cut],
        rn.trim_start_matches('\n'),
        rounded(&head[cut..])
    );
    let generic = rounded(&read("generic.wgsl"));
    for (suffix, r, j, s, u, title) in [
        (
            "_f",
            "f32",
            "MklJetF",
            "MklSagF",
            "MklUvsF",
            "R = f32 (suffix _f) ",
        ),
        (
            "_d",
            "vec2<f32>",
            "MklJetD",
            "MklSagD",
            "MklUvsD",
            "R = double-float (suffix _d) ",
        ),
    ] {
        out.push_str(&format!(
            "\n// ---- {title}{}\n",
            "-".repeat(if suffix == "_f" { 50 } else { 41 })
        ));
        let mut g = generic.replace('@', suffix);
        for (w, by) in [("R_", r), ("J_", j), ("S_", s), ("U_", u)] {
            g = replace_word(&g, w, by);
        }
        out.push_str(&g);
    }
    out.push_str(&rounded(&read("tail.wgsl")));
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("trace.wgsl"),
        out,
    )
    .unwrap();
}

fn mklight_gpu_include() -> String {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest
        .join("../mklight-gpu/include")
        .canonicalize()
        .unwrap()
        .display()
        .to_string()
}
