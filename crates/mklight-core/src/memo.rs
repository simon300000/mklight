//! Shared intermediate results between analyses (the engine's analysis
//! dependency graph, `docs/ARCHITECTURE.md`).
//!
//! A solver that can reuse an expensive intermediate (a traced pupil, a
//! resampled pupil grid) asks a [`Memo`] for it by **content key**: a
//! BLAKE3 digest of everything the value is computed from — system content,
//! tracer settings, field, wavelength, sampling settings, precision and an
//! algorithm version ([`KeyHasher`]). A memo returns the stored value when
//! the key is present and otherwise runs the build closure. The contract:
//!
//! * equal keys mean bit-identical values, so a hit can never be stale: an
//!   edit changes the content and therefore the key;
//! * a failed or cancelled build publishes nothing;
//! * the store is bounded; an entry may be evicted at any time, so callers
//!   must always be able to build;
//! * build closures must not wait on other nodes of the same memo.
//!
//! The keys a solver builds cover the solver's own inputs; an engine memo
//! additionally scopes them to what the solver cannot see (the material
//! library the system was resolved with). Without a memo (`None`) every
//! value is built, which is the reference behaviour.

use std::any::Any;
use std::sync::Arc;

use crate::{JobContext, MkResult};

/// Content key of a node.
pub type NodeKey = [u8; 32];

/// A shared value.
pub type NodeValue = Arc<dyn Any + Send + Sync>;

/// A freshly built value and the bytes it holds (for the memory budget).
pub struct Built {
    pub value: NodeValue,
    pub bytes: usize,
}

impl Built {
    pub fn new<T: Any + Send + Sync>(value: T, bytes: usize) -> Self {
        Built {
            value: Arc::new(value),
            bytes,
        }
    }
}

/// A store of shared nodes (module docs).
pub trait Memo: Send + Sync {
    /// The nodes `keys` of `kind`, in order. Missing ones are built with
    /// `build(i)` (in parallel where possible); a key that another caller
    /// is building is waited for instead of built twice when that is safe.
    /// Each entry is the node or its build error; errors are not stored.
    fn nodes(
        &self,
        kind: &'static str,
        keys: &[NodeKey],
        ctx: &JobContext,
        build: &(dyn Fn(usize) -> MkResult<Built> + Sync),
    ) -> Vec<MkResult<NodeValue>>;
}

/// Downcast a node built as `T`.
pub fn downcast<T: Any + Send + Sync>(v: NodeValue) -> Arc<T> {
    v.downcast::<T>()
        .unwrap_or_else(|_| panic!("memo node of the wrong type"))
}

/// Incremental BLAKE3 key: length-prefixed parts, floats by their bits.
pub struct KeyHasher(blake3::Hasher);

impl KeyHasher {
    /// A key of the node family `domain` (name and algorithm version).
    pub fn new(domain: &str) -> Self {
        let mut h = KeyHasher(blake3::Hasher::new());
        h.str(domain);
        h
    }
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.0.update(&(b.len() as u64).to_le_bytes());
        self.0.update(b);
        self
    }
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.bytes(s.as_bytes())
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.update(&v.to_le_bytes());
        self
    }
    pub fn usize(&mut self, v: usize) -> &mut Self {
        self.u64(v as u64)
    }
    pub fn bool(&mut self, v: bool) -> &mut Self {
        self.u64(v as u64)
    }
    /// Exact: distinct bit patterns (including `-0.0` and NaN payloads)
    /// give distinct keys.
    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.u64(v.to_bits())
    }
    pub fn f64s(&mut self, v: &[f64]) -> &mut Self {
        self.usize(v.len());
        for &x in v {
            self.f64(x);
        }
        self
    }
    pub fn key(&mut self, k: &NodeKey) -> &mut Self {
        self.0.update(k);
        self
    }
    /// Canonical JSON of a value without non-finite floats (settings,
    /// enums). Use [`Self::f64`] for anything that may be non-finite.
    pub fn json(&mut self, v: &impl serde::Serialize) -> &mut Self {
        let bytes = serde_json::to_vec(v).expect("serialisable key part");
        self.bytes(&bytes)
    }
    pub fn finish(&self) -> NodeKey {
        *self.0.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_exact_and_length_prefixed() {
        let k = |f: &dyn Fn(&mut KeyHasher)| {
            let mut h = KeyHasher::new("t/1");
            f(&mut h);
            h.finish()
        };
        assert_eq!(
            k(&|h| {
                h.f64(1.0);
            }),
            k(&|h| {
                h.f64(1.0);
            })
        );
        assert_ne!(
            k(&|h| {
                h.f64(0.0);
            }),
            k(&|h| {
                h.f64(-0.0);
            })
        );
        assert_ne!(
            k(&|h| {
                h.str("ab").str("c");
            }),
            k(&|h| {
                h.str("a").str("bc");
            })
        );
        assert_ne!(
            KeyHasher::new("t/1").finish(),
            KeyHasher::new("t/2").finish()
        );
    }
}
