//! Route options: named alternative divisions of an analysis between the
//! CPU and the GPU. The default division of each analysis was measured on
//! one machine; a key selects a GPU-heavier alternative that a machine with
//! more GPU cores per CPU core may prefer ([`KEYS`], docs/GPU.md §9.3). They
//! are the hook for letting users test and choose routes on their own
//! machine.
//!
//! A key selects one variant by name; `None` is the key's default design.
//! The benchmark harness selects variants in-process with [`set`]; otherwise
//! the environment variable `MKLIGHT_SPLIT_<KEY>` (key upper-cased,
//! `default` or empty meaning the default design) applies. Code reads the
//! selection with [`variant`] once per run, where the designs diverge, and
//! names it in the run's backend report so a measurement can show which
//! design ran. The selection is process-wide: results computed under
//! different selections differ, so anything that caches results by content
//! includes [`selection`] in its key.

use std::collections::HashMap;
use std::sync::RwLock;

type Overrides = Option<HashMap<String, Option<String>>>;

static OVERRIDES: RwLock<Overrides> = RwLock::new(None);

/// Every route option. A key outside this list selects nothing, so [`set`]
/// and [`variant`] refuse it rather than run the default design under a
/// variant's name.
pub const KEYS: [&str; 5] = ["aiming", "fieldmaps", "stray", "stray_share", "stray_sag"];

fn known(key: &str) {
    assert!(
        KEYS.contains(&key),
        "unknown route option '{key}' (known: {})",
        KEYS.join(", ")
    );
}

/// The variant selected for `key`, or `None` for the default design.
///
/// # Panics
/// If `key` is not one of [`KEYS`].
pub fn variant(key: &str) -> Option<String> {
    known(key);
    if let Some(map) = OVERRIDES.read().unwrap_or_else(|e| e.into_inner()).as_ref()
        && let Some(v) = map.get(key)
    {
        return v.clone();
    }
    std::env::var(format!("MKLIGHT_SPLIT_{}", key.to_ascii_uppercase()))
        .ok()
        .filter(|s| !s.is_empty() && s != "default")
}

/// The non-default selections in effect, as `(key, variant)` in [`KEYS`]
/// order: the part of the process state that a result depends on.
pub fn selection() -> Vec<(&'static str, String)> {
    KEYS.iter()
        .filter_map(|&key| variant(key).map(|v| (key, v)))
        .collect()
}

/// Selects `value` for `key` in this process, overriding the environment;
/// `None` selects the default design.
///
/// # Panics
/// If `key` is not one of [`KEYS`].
pub fn set(key: &str, value: Option<&str>) {
    known(key);
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

    // One test: the selection is process-wide.
    #[test]
    fn in_process_selection_overrides_and_resets() {
        let key = "stray_sag";
        assert_eq!(variant(key), None);
        set(key, Some("f32-grazing"));
        assert_eq!(variant(key).as_deref(), Some("f32-grazing"));
        assert_eq!(selection(), [(key, "f32-grazing".to_string())]);
        set(key, None);
        assert_eq!(variant(key), None);
        set(key, Some("f32-grazing"));
        reset();
        assert_eq!(variant(key), None);
        assert!(selection().is_empty());
        assert!(std::panic::catch_unwind(|| set("geometric", Some("cpu-aim"))).is_err());
        assert!(std::panic::catch_unwind(|| variant("stray_asphere")).is_err());
    }
}
