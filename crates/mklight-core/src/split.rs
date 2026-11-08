//! Work-split variants: named alternative divisions of an analysis between
//! the CPU and the GPU, kept side by side while they are measured.
//!
//! A study key (for example `geometric`) selects one variant by name; `None`
//! is the key's default design. The benchmark harness selects variants
//! in-process with [`set`]; otherwise the environment variable
//! `MKLIGHT_SPLIT_<KEY>` (key upper-cased, `default` or empty meaning the
//! default design) applies. Code reads the selection with [`variant`] once
//! per run, where the designs diverge, and names it in the run's backend
//! report so a measurement can show which design ran.
//!
//! This is measurement scaffolding: once a study is decided, its key and the
//! variants that lost are removed.

use std::collections::HashMap;
use std::sync::RwLock;

type Overrides = Option<HashMap<String, Option<String>>>;

static OVERRIDES: RwLock<Overrides> = RwLock::new(None);

/// The variant selected for `key`, or `None` for the default design.
pub fn variant(key: &str) -> Option<String> {
    if let Some(map) = OVERRIDES.read().unwrap_or_else(|e| e.into_inner()).as_ref()
        && let Some(v) = map.get(key)
    {
        return v.clone();
    }
    std::env::var(format!("MKLIGHT_SPLIT_{}", key.to_ascii_uppercase()))
        .ok()
        .filter(|s| !s.is_empty() && s != "default")
}

/// Whether `key` selects the variant `name`.
pub fn is(key: &str, name: &str) -> bool {
    variant(key).as_deref() == Some(name)
}

/// Selects `value` for `key` in this process, overriding the environment;
/// `None` selects the default design.
pub fn set(key: &str, value: Option<&str>) {
    OVERRIDES
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(key.to_string(), value.map(String::from));
}

/// Removes every in-process selection: the environment applies again.
pub fn reset() {
    *OVERRIDES.write().unwrap_or_else(|e| e.into_inner()) = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_process_selection_overrides_and_resets() {
        let key = "split_selftest";
        assert_eq!(variant(key), None);
        set(key, Some("pipelined"));
        assert!(is(key, "pipelined"));
        set(key, None);
        assert_eq!(variant(key), None);
        reset();
        assert_eq!(variant(key), None);
    }
}
