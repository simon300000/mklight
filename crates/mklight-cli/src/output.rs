//! Output formatting: deterministic JSON documents and plain-text tables.
//!
//! JSON follows the engine conventions (`docs/FFI.md`): `±∞` are the strings
//! `"inf"`/`"-inf"`, NaN is `null`. Documents are pretty-printed with a
//! trailing newline; object keys come out in a fixed order, so two runs on
//! the same inputs print identical bytes apart from the timing fields that
//! `--no-timing` removes (`TIMING_KEYS`).

use serde_json::{Value, json};

/// Keys that carry wall-clock data. `--no-timing` removes them anywhere in
/// a document (`run_info.started_unix_ms`, `run_info.duration_ms`,
/// `status.elapsed_ms`, tolerancing `monte_carlo.seconds_per_trial`), plus
/// `created`/`modified` inside `metadata` objects.
pub const TIMING_KEYS: &[&str] = &[
    "started_unix_ms",
    "duration_ms",
    "elapsed_ms",
    "seconds_per_trial",
];
const METADATA_TIME_KEYS: &[&str] = &["created", "modified"];

/// Remove the timing fields from `v` in place (see `TIMING_KEYS`).
pub fn strip_timing(v: &mut Value) {
    match v {
        Value::Object(map) => {
            for k in TIMING_KEYS {
                map.remove(*k);
            }
            for (k, child) in map.iter_mut() {
                if k == "metadata"
                    && let Value::Object(m) = child
                {
                    for t in METADATA_TIME_KEYS {
                        m.remove(*t);
                    }
                }
                strip_timing(child);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(strip_timing),
        _ => {}
    }
}

/// Pretty JSON with a trailing newline.
pub fn to_pretty(v: &Value) -> String {
    let mut s = serde_json::to_string_pretty(v).unwrap_or_else(|_| "null".into());
    s.push('\n');
    s
}

/// A number as JSON: `±∞` as `"inf"`/`"-inf"`, NaN as `null`.
pub fn num(x: f64) -> Value {
    if x.is_nan() {
        Value::Null
    } else if x.is_infinite() {
        json!(if x > 0.0 { "inf" } else { "-inf" })
    } else {
        json!(x)
    }
}

/// A number for a table: fixed `prec` decimals, `inf`/`-inf`, `-` for NaN.
/// Negative zero prints as zero.
pub fn fmt_f(x: f64, prec: usize) -> String {
    if x.is_nan() {
        "-".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        let s = format!("{x:.prec$}");
        // "-0.000" → "0.000"
        if s.starts_with('-') && s[1..].chars().all(|c| c == '0' || c == '.') {
            s[1..].to_string()
        } else {
            s
        }
    }
}

/// A number for a table with ~`sig` significant digits: fixed notation for
/// moderate magnitudes, exponent notation otherwise.
/// Trailing zeros after the decimal point are dropped (`5`, `1.4`).
pub fn fmt_g(x: f64, sig: usize) -> String {
    if !x.is_finite() || x == 0.0 {
        return fmt_f(x, 0);
    }
    let e = x.abs().log10().floor() as i32;
    if (-4..=7).contains(&e) {
        let prec = (sig as i32 - 1 - e).max(0) as usize;
        trim_zeros(fmt_f(x, prec))
    } else {
        let s = format!("{x:.p$e}", p = sig.saturating_sub(1));
        match s.split_once('e') {
            Some((m, ex)) => format!("{}e{ex}", trim_zeros(m.to_string())),
            None => s,
        }
    }
}

fn trim_zeros(s: String) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// A JSON value for a table cell: strings bare, numbers via `fmt_g`.
pub fn cell(v: &Value) -> String {
    match v {
        Value::Null => "-".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => n.as_f64().map_or_else(|| n.to_string(), |x| fmt_g(x, 6)),
        Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

/// Column alignment.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

/// A plain-text table with a header row and a rule.
pub struct Table {
    header: Vec<String>,
    align: Vec<Align>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(cols: &[(&str, Align)]) -> Self {
        Table {
            header: cols.iter().map(|c| c.0.to_string()).collect(),
            align: cols.iter().map(|c| c.1).collect(),
            rows: Vec::new(),
        }
    }
    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }
    pub fn render(&self) -> String {
        let n = self.header.len();
        let mut w: Vec<usize> = self.header.iter().map(|h| h.chars().count()).collect();
        for r in &self.rows {
            for (i, c) in r.iter().enumerate().take(n) {
                w[i] = w[i].max(c.chars().count());
            }
        }
        let line = |cells: &[String]| {
            let mut s = String::new();
            for (i, wi) in w.iter().enumerate() {
                let c = cells.get(i).map_or("", String::as_str);
                let pad = wi - c.chars().count();
                if i > 0 {
                    s.push_str("  ");
                }
                match self.align[i] {
                    Align::Left => {
                        s.push_str(c);
                        if i + 1 < n {
                            s.push_str(&" ".repeat(pad));
                        }
                    }
                    Align::Right => {
                        s.push_str(&" ".repeat(pad));
                        s.push_str(c);
                    }
                }
            }
            s.trim_end().to_string() + "\n"
        };
        let mut out = line(&self.header);
        let total: usize = w.iter().sum::<usize>() + 2 * n.saturating_sub(1);
        out.push_str(&"-".repeat(total));
        out.push('\n');
        for r in &self.rows {
            out.push_str(&line(r));
        }
        out
    }
}

/// `key: value` lines with the keys padded to one width.
pub fn key_values(pairs: &[(&str, String)]) -> String {
    let w = pairs.iter().map(|p| p.0.chars().count()).max().unwrap_or(0);
    let mut s = String::new();
    for (k, v) in pairs {
        s.push_str(&format!("{k:<w$}  {v}\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_timing_removes_only_timing_keys() {
        let mut v = json!({
            "run_info": {"started_unix_ms": 1, "duration_ms": 2.0, "seed": 3},
            "status": [{"elapsed_ms": 4.0, "progress": 0.5}],
            "metadata": {"created": "x", "modified": "y", "author": "a"},
            "result": {"created": "kept outside metadata", "seconds_per_trial": 0.2}
        });
        strip_timing(&mut v);
        assert_eq!(
            v,
            json!({
                "run_info": {"seed": 3},
                "status": [{"progress": 0.5}],
                "metadata": {"author": "a"},
                "result": {"created": "kept outside metadata"}
            })
        );
    }

    #[test]
    fn numbers_format_stably() {
        assert_eq!(num(f64::INFINITY), json!("inf"));
        assert_eq!(num(f64::NEG_INFINITY), json!("-inf"));
        assert_eq!(num(f64::NAN), Value::Null);
        assert_eq!(fmt_f(-0.0001, 3), "0.000");
        assert_eq!(fmt_f(-1.25, 1), "-1.2");
        assert_eq!(fmt_f(f64::INFINITY, 3), "inf");
        assert_eq!(fmt_g(100.0123456, 6), "100.012");
        assert_eq!(fmt_g(5.0, 6), "5");
        assert_eq!(fmt_g(1.4, 6), "1.4");
        assert_eq!(fmt_g(1.5e-7, 3), "1.5e-7");
        assert_eq!(fmt_g(0.0, 6), "0");
        assert_eq!(cell(&json!(3)), "3");
        assert_eq!(cell(&json!(2.5)), "2.5");
    }

    #[test]
    fn table_aligns_columns() {
        let mut t = Table::new(&[("a", Align::Left), ("value", Align::Right)]);
        t.row(vec!["long name".into(), "1".into()]);
        t.row(vec!["x".into(), "12.5".into()]);
        let s = t.render();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0], "a          value");
        assert_eq!(lines[1], "----------------");
        assert_eq!(lines[2], "long name      1");
        assert_eq!(lines[3], "x           12.5");
    }
}
