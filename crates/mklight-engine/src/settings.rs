//! Analysis settings schemas.
//!
//! Each analysis publishes its settings as a small subset of JSON Schema, so
//! front ends can build forms without hard-coding anything:
//!
//! ```json
//! { "type": "object",
//!   "properties": {
//!     "count": { "type": "integer", "default": 1000, "minimum": 1,
//!                "title": "Samples", "description": "…" },
//!     "catalog": { "type": ["string", "null"], "default": null },
//!     "mode": { "type": "string", "enum": ["a", "b"], "default": "a" } },
//!   "required": ["name"] }
//! ```
//!
//! Supported keywords: `type` (a name or a list of names among `number`,
//! `integer`, `string`, `boolean`, `array`, `object`, `null`), `default`,
//! `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `enum`, `items` (one schema for every array element),
//! `required`, plus the descriptive `title`/`description`/`unit`.
//!
//! `normalize` validates user settings against such a schema, fills in
//! defaults and rejects unknown keys. The normalised object is what the
//! analysis receives and what `RunInfo.settings` records, so a stored result
//! always carries every effective setting.

use serde_json::{Map, Value};

use crate::error::{EngineError, EngineResult};

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

fn check(path: &str, schema: &Value, v: &Value) -> EngineResult<()> {
    let err = |m: String| Err(EngineError::InvalidSettings(format!("'{path}': {m}")));
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
            return err(format!("expected type {t}, got {v}"));
        }
    }
    if let Some(x) = v.as_f64() {
        if !x.is_finite() {
            return err("must be finite".into());
        }
        if let Some(lo) = schema.get("minimum").and_then(Value::as_f64)
            && x < lo
        {
            return err(format!("{x} is below the minimum {lo}"));
        }
        if let Some(hi) = schema.get("maximum").and_then(Value::as_f64)
            && x > hi
        {
            return err(format!("{x} is above the maximum {hi}"));
        }
        for (key, below) in [("exclusiveMinimum", true), ("exclusiveMaximum", false)] {
            if let Some(bound) = schema.get(key).and_then(Value::as_f64)
                && if below { x <= bound } else { x >= bound }
            {
                return err(format!("{x} violates {key} {bound}"));
            }
        }
    }
    if let Some(Value::Array(allowed)) = schema.get("enum")
        && !allowed.contains(v)
    {
        return err(format!(
            "{v} is not one of {}",
            Value::Array(allowed.clone())
        ));
    }
    if let (Some(items), Value::Array(arr)) = (schema.get("items"), v) {
        for (i, e) in arr.iter().enumerate() {
            check(&format!("{path}[{i}]"), items, e)?;
        }
    }
    Ok(())
}

/// Validate `input` (an object, or null for "all defaults") against `schema`
/// and return the complete settings object.
pub fn normalize(schema: &Value, input: &Value) -> EngineResult<Value> {
    let empty = Map::new();
    let props = schema
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let given = match input {
        Value::Null => &empty,
        Value::Object(o) => o,
        other => {
            return Err(EngineError::InvalidSettings(format!(
                "settings must be a JSON object, got {other}"
            )));
        }
    };
    if let Some(k) = given.keys().find(|k| !props.contains_key(*k)) {
        let known: Vec<&str> = props.keys().map(String::as_str).collect();
        return Err(EngineError::InvalidSettings(format!(
            "unknown setting '{k}' (known: {})",
            known.join(", ")
        )));
    }
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut out = Map::new();
    for (k, s) in props {
        match given.get(k) {
            Some(v) => {
                check(k, s, v)?;
                out.insert(k.clone(), v.clone());
            }
            None => {
                if required.contains(&k.as_str()) {
                    return Err(EngineError::InvalidSettings(format!(
                        "missing required setting '{k}'"
                    )));
                }
                if let Some(d) = s.get("default") {
                    out.insert(k.clone(), d.clone());
                }
            }
        }
    }
    Ok(Value::Object(out))
}

/// The defaults of a schema (settings with nothing given). Fails only when
/// the schema has required keys.
pub fn defaults(schema: &Value) -> EngineResult<Value> {
    normalize(schema, &Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "scale": {"type": "number", "default": 1.5},
                "catalog": {"type": ["string", "null"], "default": null},
                "mode": {"type": "string", "enum": ["a", "b"], "default": "a"},
                "wl": {"type": "array", "items": {"type": "number", "minimum": 0.1}},
                "name": {"type": "string"}
            },
            "required": ["name"]
        })
    }

    #[test]
    fn exclusive_bounds_and_array_items() {
        let schema = json!({"properties": {
            "step": {"type": ["number", "null"], "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            "values": {"type": "array", "items": {"type": "number", "exclusiveMaximum": 2}}
        }});
        for x in [-0.1, 0.0, 1.0, 1.1] {
            assert!(normalize(&schema, &json!({"step": x})).is_err());
        }
        for x in [json!(null), json!(0.5)] {
            assert!(normalize(&schema, &json!({"step": x})).is_ok());
        }
        assert!(normalize(&schema, &json!({"values": [1, 2]})).is_err());
        assert!(normalize(&schema, &json!({"values": [1, 1.5]})).is_ok());
    }

    #[test]
    fn fills_defaults_and_keeps_given() {
        let n = normalize(&schema(), &json!({"name": "x", "count": 5})).unwrap();
        assert_eq!(
            n,
            json!({"count": 5, "scale": 1.5, "catalog": null, "mode": "a", "name": "x"})
        );
    }

    #[test]
    fn rejects_bad_input() {
        let s = schema();
        for bad in [
            json!({"name": "x", "count": 0}),
            json!({"name": "x", "count": 101}),
            json!({"name": "x", "count": 2.5}),
            json!({"name": "x", "mode": "c"}),
            json!({"name": "x", "wl": [0.5, 0.05]}),
            json!({"name": "x", "bogus": 1}),
            json!({"count": 3}),
            json!([1]),
            json!({"name": 3}),
        ] {
            assert!(normalize(&s, &bad).is_err(), "accepted {bad}");
        }
        assert!(normalize(&s, &json!({"name": "x", "catalog": "CAT"})).is_ok());
        assert!(defaults(&s).is_err());
    }
}
