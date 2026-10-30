//! Settings and result schemas.
//!
//! Every analysis publishes two JSON Schemas (the subset `settings`
//! documents, plus `anyOf`, `additionalProperties`, `minItems`/`maxItems`):
//!
//! * a **settings schema** with defaults. For analyses whose settings are a
//!   serde struct of a solver crate, [`settings_from_default`] derives it
//!   from the struct's `Default` value (types and defaults) and merges
//!   per-key annotations (`enum`, bounds, units, descriptions), so the
//!   engine's defaults can never drift from the crate's;
//! * a **result schema**: the stable, versioned shape of `result` in the
//!   envelope (`docs/RESULTS.md`). Its `x-buffers` list declares the typed
//!   buffers that travel next to the JSON: `{name, dtype, shape, unit,
//!   description}`, with `shape` a list of axis names.
//!
//! [`validate`] checks a value against such a schema. In strict mode an
//! object schema with `properties` rejects keys it does not list (unless it
//! says `"additionalProperties": true`), which is how the golden tests make
//! sure every result field is documented.

use serde::Serialize;
use serde_json::{Map, Value, json};

/// A number that may be non-finite or undefined (CONVENTIONS §11): a JSON
/// number, `null` (NaN, "undefined"), or one of `"inf"`, `"-inf"`, `"nan"`.
pub fn nf() -> Value {
    json!({"anyOf": [{"type": ["number", "null"]}, {"enum": ["inf", "-inf", "nan"]}]})
}

/// `nf` with a description and unit.
pub fn nf_u(unit: &str, description: &str) -> Value {
    let mut v = nf();
    annotate(&mut v, unit, description);
    v
}

pub fn num(unit: &str, description: &str) -> Value {
    let mut v = json!({"type": "number"});
    annotate(&mut v, unit, description);
    v
}

pub fn int(description: &str) -> Value {
    let mut v = json!({"type": "integer", "minimum": 0});
    annotate(&mut v, "", description);
    v
}

pub fn string(description: &str) -> Value {
    let mut v = json!({"type": "string"});
    annotate(&mut v, "", description);
    v
}

pub fn boolean(description: &str) -> Value {
    let mut v = json!({"type": "boolean"});
    annotate(&mut v, "", description);
    v
}

/// One of a fixed set of strings.
pub fn enumeration(values: &[&str], description: &str) -> Value {
    let mut v = json!({"type": "string", "enum": values});
    annotate(&mut v, "", description);
    v
}

/// An array whose elements all match `items`.
pub fn arr(items: Value, description: &str) -> Value {
    let mut v = json!({"type": "array", "items": items});
    annotate(&mut v, "", description);
    v
}

/// A fixed-length array of `n` elements matching `items` (e.g. `[x, y]`).
pub fn tuple(items: Value, n: usize, description: &str) -> Value {
    let mut v = json!({"type": "array", "items": items, "minItems": n, "maxItems": n});
    annotate(&mut v, "", description);
    v
}

/// `v` or `null`.
pub fn nullable(v: Value) -> Value {
    json!({"anyOf": [v, {"type": "null"}]})
}

/// An object whose listed properties are all required (strict: no others).
pub fn obj(props: &[(&str, Value)]) -> Value {
    let mut p = Map::new();
    for (k, v) in props {
        p.insert((*k).to_string(), v.clone());
    }
    let req: Vec<&str> = props.iter().map(|(k, _)| *k).collect();
    json!({"type": "object", "properties": p, "required": req})
}

/// As [`obj`], but `optional` keys may be absent.
pub fn obj_opt(props: &[(&str, Value)], optional: &[&str]) -> Value {
    let mut v = obj(props);
    if let Some(Value::Array(r)) = v.get_mut("required") {
        r.retain(|k| !optional.contains(&k.as_str().unwrap_or("")));
    }
    v
}

/// An object documented elsewhere (a solver crate's serde type), checked
/// only for being an object.
pub fn opaque(description: &str) -> Value {
    let mut v = json!({"type": "object", "additionalProperties": true});
    annotate(&mut v, "", description);
    v
}

/// Any JSON value (documented in the description).
pub fn any(description: &str) -> Value {
    let mut v = json!({});
    annotate(&mut v, "", description);
    v
}

fn annotate(v: &mut Value, unit: &str, description: &str) {
    if let Value::Object(m) = v {
        if !unit.is_empty() {
            m.insert("unit".into(), json!(unit));
        }
        if !description.is_empty() {
            m.insert("description".into(), json!(description));
        }
    }
}

/// Add a description to a schema.
pub fn describe(mut v: Value, description: &str) -> Value {
    annotate(&mut v, "", description);
    v
}

/// One declared result buffer: `name`, dtype (`f64`, `f32`, `i32`, `u32`,
/// `i64`, `u64`, `u8`), axis names, unit and meaning.
pub fn buffer(name: &str, dtype: &str, shape: &[&str], unit: &str, description: &str) -> Value {
    json!({"name": name, "dtype": dtype, "shape": shape, "unit": unit,
           "description": description})
}

/// A result schema: a strict object with `x-buffers`.
pub fn result(props: &[(&str, Value)], buffers: Vec<Value>) -> Value {
    let mut v = obj(props);
    v["x-buffers"] = Value::Array(buffers);
    v
}

/// Settings schema derived from a serde settings struct's `Default`: every
/// key of the default object becomes a property with that `default` and
/// the JSON type of its value; `annotations` (an object of per-key schema
/// fragments) adds titles, units, bounds and enums, and supplies the type
/// of keys whose default is `null` (those must be annotated). Annotating a
/// key the struct does not have panics: it is a bug in the engine.
pub fn settings_from_default<S: Serialize + Default>(annotations: Value) -> Value {
    let d = serde_json::to_value(S::default()).expect("settings serialise");
    let Value::Object(defaults) = d else {
        panic!("settings must serialise to an object");
    };
    let ann = match annotations {
        Value::Object(m) => m,
        Value::Null => Map::new(),
        _ => panic!("annotations must be an object"),
    };
    for k in ann.keys() {
        assert!(
            defaults.contains_key(k),
            "annotation for unknown setting '{k}'"
        );
    }
    let mut props = Map::new();
    for (k, v) in defaults {
        let ty = match &v {
            Value::Null => None,
            Value::Bool(_) => Some("boolean"),
            Value::Number(n) if n.is_f64() => Some("number"),
            Value::Number(_) => Some("integer"),
            Value::String(_) => Some("string"),
            Value::Array(_) => Some("array"),
            Value::Object(_) => Some("object"),
        };
        let mut p = Map::new();
        if let Some(t) = ty {
            p.insert("type".into(), json!(t));
        }
        if let Some(Value::Object(a)) = ann.get(&k) {
            for (ak, av) in a {
                p.insert(ak.clone(), av.clone());
            }
        }
        assert!(
            p.contains_key("type") || p.contains_key("anyOf"),
            "setting '{k}' defaults to null and needs an annotated type"
        );
        p.insert("default".into(), v);
        props.insert(k, Value::Object(p));
    }
    json!({"type": "object", "properties": props})
}

// ---------------------------------------------------------------------------
// Validation

fn type_matches(t: &str, v: &Value) -> bool {
    match t {
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "null" => v.is_null(),
        _ => false,
    }
}

/// Validate `value` against `schema` (module docs). Returns the first
/// violation as `"<path>: <problem>"`.
pub fn validate(schema: &Value, value: &Value, strict: bool) -> Result<(), String> {
    check(schema, value, "$", strict)
}

fn check(schema: &Value, v: &Value, path: &str, strict: bool) -> Result<(), String> {
    let fail = |m: String| Err(format!("{path}: {m}"));
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array)
        && !any_of.iter().any(|s| check(s, v, path, strict).is_ok())
    {
        let first = any_of
            .iter()
            .map(|s| check(s, v, path, strict).unwrap_err())
            .collect::<Vec<_>>()
            .join(" | ");
        return fail(format!("matches no alternative ({first})"));
    }
    if let Some(t) = schema.get("type") {
        let ok = match t {
            Value::String(s) => type_matches(s, v),
            Value::Array(ts) => ts
                .iter()
                .filter_map(Value::as_str)
                .any(|s| type_matches(s, v)),
            _ => true,
        };
        if !ok {
            return fail(format!("expected type {t}, got {}", short(v)));
        }
    }
    if let Some(Value::Array(allowed)) = schema.get("enum")
        && !allowed.contains(v)
    {
        return fail(format!("{} is not one of {}", short(v), json!(allowed)));
    }
    if let Some(x) = v.as_f64() {
        if let Some(lo) = schema.get("minimum").and_then(Value::as_f64)
            && x < lo
        {
            return fail(format!("{x} < minimum {lo}"));
        }
        if let Some(hi) = schema.get("maximum").and_then(Value::as_f64)
            && x > hi
        {
            return fail(format!("{x} > maximum {hi}"));
        }
    }
    if let Value::Array(a) = v {
        if let Some(n) = schema.get("minItems").and_then(Value::as_u64)
            && (a.len() as u64) < n
        {
            return fail(format!("{} items < minItems {n}", a.len()));
        }
        if let Some(n) = schema.get("maxItems").and_then(Value::as_u64)
            && (a.len() as u64) > n
        {
            return fail(format!("{} items > maxItems {n}", a.len()));
        }
        if let Some(items) = schema.get("items") {
            for (i, e) in a.iter().enumerate() {
                check(items, e, &format!("{path}[{i}]"), strict)?;
            }
        }
    }
    if let Value::Object(o) = v {
        let props = schema.get("properties").and_then(Value::as_object);
        if let Some(req) = schema.get("required").and_then(Value::as_array) {
            for k in req.iter().filter_map(Value::as_str) {
                if !o.contains_key(k) {
                    return fail(format!("missing required key '{k}'"));
                }
            }
        }
        if let Some(props) = props {
            let open = schema.get("additionalProperties") == Some(&Value::Bool(true));
            for (k, e) in o {
                match props.get(k) {
                    Some(s) => check(s, e, &format!("{path}.{k}"), strict)?,
                    None if strict && !open => {
                        return fail(format!("undocumented key '{k}'"));
                    }
                    None => {}
                }
            }
        }
    }
    Ok(())
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 60 {
        format!("{}…", s.chars().take(60).collect::<String>())
    } else {
        s
    }
}

/// Check that `buffers` (name, dtype, shape) match the `x-buffers`
/// declaration of a result schema: every buffer is declared, with the
/// declared dtype and number of axes. Optional buffers may be absent.
pub fn check_buffers(schema: &Value, buffers: &[(String, String, usize)]) -> Result<(), String> {
    let declared = schema
        .get("x-buffers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for (name, dtype, ndim) in buffers {
        // A declared name may end in `*` (one buffer per path, e.g. `map_*`).
        let d = declared.iter().find(|d| {
            let n = d["name"].as_str().unwrap_or("");
            n == name || n.strip_suffix('*').is_some_and(|p| name.starts_with(p))
        });
        let Some(d) = d else {
            return Err(format!("buffer '{name}' is not declared"));
        };
        if d["dtype"] != dtype.as_str() {
            return Err(format!(
                "buffer '{name}': dtype {dtype}, declared {}",
                d["dtype"]
            ));
        }
        let nd = d["shape"].as_array().map_or(0, Vec::len);
        if nd != *ndim {
            return Err(format!("buffer '{name}': {ndim} axes, declared {nd}"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Markdown reference (docs/RESULTS.md)

/// Short type label of a schema: `integer`, `number|null`, `nf` (see
/// [`nf`]), `array<…>`, `[T; n]`, `object`, `T?` for `T` or null.
pub fn type_label(s: &Value) -> String {
    if s == &nf() || s.get("anyOf").is_some_and(|a| a.get(0) == Some(&nf()["anyOf"][0])) {
        return "nf".into();
    }
    if let Some(alts) = s.get("anyOf").and_then(Value::as_array) {
        let labels: Vec<String> = alts.iter().map(type_label).collect();
        if labels.len() == 2 && labels[1] == "null" {
            return format!("{}?", labels[0]);
        }
        return labels.join("|");
    }
    let base = match s.get("type") {
        Some(Value::String(t)) => t.clone(),
        Some(Value::Array(ts)) => ts
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("|"),
        _ if s.get("enum").is_some() => "string".into(),
        _ => "any".into(),
    };
    if base == "array"
        && let Some(items) = s.get("items")
    {
        let it = type_label(items);
        return match (s.get("minItems"), s.get("maxItems")) {
            (Some(a), Some(b)) if a == b => format!("[{it}; {a}]"),
            _ => format!("array<{it}>"),
        };
    }
    base
}

fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn notes(s: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(e) = s.get("enum").and_then(Value::as_array) {
        let v: Vec<String> = e.iter().map(|x| format!("`{x}`")).collect();
        parts.push(format!("one of {}", v.join(", ")));
    }
    match (s.get("minimum"), s.get("maximum")) {
        (Some(a), Some(b)) => parts.push(format!("{a} … {b}")),
        (Some(a), None) => parts.push(format!("≥ {a}")),
        (None, Some(b)) => parts.push(format!("≤ {b}")),
        _ => {}
    }
    if let Some(d) = s.get("description").and_then(Value::as_str) {
        parts.push(d.to_string());
    }
    cell(&parts.join("; "))
}

fn unit(s: &Value) -> String {
    s.get("unit")
        .and_then(Value::as_str)
        .map_or(String::new(), String::from)
}

/// Rows `path | type | unit | description` of a result schema, nested
/// objects and arrays of objects flattened (`fields[].rms_radius`).
fn result_rows(path: &str, s: &Value, required: bool, out: &mut Vec<String>) {
    let inner = s
        .get("anyOf")
        .and_then(Value::as_array)
        .and_then(|a| a.iter().find(|x| x.get("properties").is_some()))
        .unwrap_or(s);
    if !path.is_empty() {
        let opt = if required { "" } else { " (optional)" };
        out.push(format!(
            "| `{path}`{opt} | {} | {} | {} |",
            cell(&type_label(s)),
            cell(&unit(s)),
            notes(s)
        ));
    }
    let props = inner.get("properties").and_then(Value::as_object);
    let item_props = inner
        .get("items")
        .and_then(|i| i.get("properties").map(|_| i));
    if let Some(p) = props {
        let req: Vec<&str> = inner
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        for (k, v) in p {
            let sub = if path.is_empty() {
                k.clone()
            } else {
                format!("{path}.{k}")
            };
            result_rows(&sub, v, req.contains(&k.as_str()), out);
        }
    } else if let Some(items) = item_props {
        result_rows_items(&format!("{path}[]"), items, out);
    }
}

fn result_rows_items(path: &str, items: &Value, out: &mut Vec<String>) {
    let req: Vec<&str> = items
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if let Some(p) = items.get("properties").and_then(Value::as_object) {
        for (k, v) in p {
            result_rows(&format!("{path}.{k}"), v, req.contains(&k.as_str()), out);
        }
    }
}

/// The reference section of one analysis for docs/RESULTS.md: metadata,
/// settings (with defaults), result fields and buffers.
pub fn render_markdown(info: &super::AnalysisInfo) -> String {
    let mut o = String::new();
    o.push_str(&format!("### `{}` — {}\n\n", info.name, info.title));
    o.push_str(&format!(
        "Category `{}`, result version {}{}{}.\n\n",
        info.category,
        info.result_version,
        if info.gpu_path { ", **GPU path**" } else { "" },
        if info.available {
            String::new()
        } else {
            format!(
                ", **not available**: {}",
                info.unavailable_reason.clone().unwrap_or_default()
            )
        }
    ));
    if !info.description.is_empty() {
        o.push_str(&format!("{}\n\n", info.description.replace("  ", " ")));
    }
    let props = info.settings_schema["properties"].as_object();
    match props {
        Some(p) if !p.is_empty() => {
            let req: Vec<&str> = info.settings_schema["required"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            o.push_str("Settings:\n\n| key | type | default | unit | notes |\n|---|---|---|---|---|\n");
            for (k, s) in p {
                let default = match s.get("default") {
                    Some(d) => format!("`{d}`"),
                    None if req.contains(&k.as_str()) => "**required**".into(),
                    None => String::new(),
                };
                o.push_str(&format!(
                    "| `{k}` | {} | {} | {} | {} |\n",
                    cell(&type_label(s)),
                    cell(&default),
                    cell(&unit(s)),
                    notes(s)
                ));
            }
            o.push('\n');
        }
        _ => o.push_str("Settings: none.\n\n"),
    }
    let mut rows = Vec::new();
    result_rows("", &info.result_schema, true, &mut rows);
    if rows.is_empty() {
        o.push_str("Result: an object (not documented field by field).\n\n");
    } else {
        o.push_str("Result (`result` in the envelope):\n\n| field | type | unit | notes |\n|---|---|---|---|\n");
        for r in rows {
            o.push_str(&r);
            o.push('\n');
        }
        o.push('\n');
    }
    let bufs = info.result_schema["x-buffers"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if !bufs.is_empty() {
        o.push_str("Buffers:\n\n| name | dtype | shape | unit | notes |\n|---|---|---|---|---|\n");
        for b in bufs {
            let shape: Vec<String> = b["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect())
                .unwrap_or_default();
            o.push_str(&format!(
                "| `{}` | {} | [{}] | {} | {} |\n",
                b["name"].as_str().unwrap_or(""),
                b["dtype"].as_str().unwrap_or(""),
                shape.join(", "),
                cell(b["unit"].as_str().unwrap_or("")),
                cell(b["description"].as_str().unwrap_or(""))
            ));
        }
        o.push('\n');
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, Default)]
    #[serde(default)]
    struct S {
        n: usize,
        x: f64,
        on: bool,
        opt: Option<f64>,
    }

    #[test]
    fn derived_settings_schema() {
        let s = settings_from_default::<S>(json!({
            "n": {"minimum": 1},
            "opt": {"type": ["number", "null"], "unit": "mm"}
        }));
        assert_eq!(s["properties"]["n"]["type"], "integer");
        assert_eq!(s["properties"]["n"]["minimum"], 1);
        assert_eq!(s["properties"]["x"]["type"], "number");
        assert_eq!(s["properties"]["on"]["default"], false);
        assert_eq!(s["properties"]["opt"]["default"], Value::Null);
        let d = crate::settings::defaults(&s).unwrap();
        assert_eq!(d, json!({"n": 0, "x": 0.0, "on": false, "opt": null}));
    }

    #[test]
    #[should_panic]
    fn unknown_annotation_panics() {
        let _ = settings_from_default::<S>(json!({"bogus": {}}));
    }

    #[test]
    #[should_panic]
    fn null_default_needs_a_type() {
        let _ = settings_from_default::<S>(Value::Null);
    }

    #[test]
    fn validator() {
        let s = obj(&[
            ("a", nf()),
            ("b", arr(int(""), "")),
            ("c", nullable(string(""))),
            ("d", tuple(json!({"type": "number"}), 2, "")),
            ("e", enumeration(&["x", "y"], "")),
        ]);
        let good = json!({"a": "inf", "b": [1, 2], "c": null, "d": [1.0, 2.0], "e": "x"});
        assert!(validate(&s, &good, true).is_ok());
        for (k, bad) in [
            ("a", json!("big")),
            ("b", json!([1.5])),
            ("c", json!(3)),
            ("d", json!([1.0])),
            ("e", json!("z")),
        ] {
            let mut v = good.clone();
            v[k] = bad;
            let e = validate(&s, &v, true).unwrap_err();
            assert!(e.contains(&format!("$.{k}")), "{e}");
        }
        let mut v = good.clone();
        v.as_object_mut().unwrap().remove("a");
        assert!(validate(&s, &v, false).unwrap_err().contains("'a'"));
        let mut v = good.clone();
        v["extra"] = json!(1);
        assert!(validate(&s, &v, false).is_ok());
        assert!(validate(&s, &v, true).unwrap_err().contains("extra"));
        let o = obj_opt(&[("a", nf()), ("b", nf())], &["b"]);
        assert!(validate(&o, &json!({"a": 1}), true).is_ok());
    }

    #[test]
    fn buffer_declarations() {
        let s = result(
            &[],
            vec![
                buffer("pts", "f64", &["n", "2"], "mm", ""),
                buffer("map_*", "f64", &["ny", "nx"], "", ""),
            ],
        );
        let b = |n: &str, d: &str, k| (n.to_string(), d.to_string(), k);
        assert!(check_buffers(&s, &[b("pts", "f64", 2), b("map_3", "f64", 2)]).is_ok());
        assert!(check_buffers(&s, &[b("pts", "f32", 2)]).is_err());
        assert!(check_buffers(&s, &[b("pts", "f64", 1)]).is_err());
        assert!(check_buffers(&s, &[b("other", "f64", 1)]).is_err());
    }
}
