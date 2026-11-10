//! Bounded immutable system-set cache (the prepared-systems category). Live
//! handles independently pin entries.
use super::immutable::{CacheCategory, CacheCategoryStats};
use super::{ResidentCacheStats, SystemCacheKey, bad};
use crate::{PackedSystem, consts::MKL_LAYOUT_VERSION};
use mklight_core::MkResult;
use std::{collections::VecDeque, sync::Arc};

pub struct SystemCache<T> {
    entries: VecDeque<(SystemCacheKey, blake3::Hash, usize, Arc<T>)>,
    bytes: usize,
    limit: usize,
    stats: ResidentCacheStats,
    category: CacheCategoryStats,
}
impl<T> Default for SystemCache<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> SystemCache<T> {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            limit: CacheCategory::PreparedSystems.budget().0,
            stats: Default::default(),
            category: Default::default(),
        }
    }
    pub fn get(&mut self, key: &SystemCacheKey, digest: blake3::Hash) -> Option<Arc<T>> {
        let Some(i) = self
            .entries
            .iter()
            .position(|(k, h, _, _)| k == key && *h == digest)
        else {
            self.category.misses += 1;
            return None;
        };
        let entry = self.entries.remove(i).unwrap();
        let value = entry.3.clone();
        self.entries.push_back(entry);
        self.stats.system_hits += 1;
        self.category.hits += 1;
        Some(value)
    }
    pub fn insert(
        &mut self,
        key: SystemCacheKey,
        digest: blake3::Hash,
        bytes: usize,
        value: Arc<T>,
    ) {
        self.stats.system_uploads += 1;
        self.stats.uploaded_bytes += bytes as u64;
        self.category.inserts += 1;
        if bytes > self.limit {
            self.category.rejected += 1;
            return;
        }
        let (_, max_entries) = CacheCategory::PreparedSystems.budget();
        while self.bytes + bytes > self.limit || self.entries.len() >= max_entries {
            let (_, _, n, _) = self.entries.pop_front().unwrap();
            self.bytes -= n;
            self.category.evictions += 1;
            self.category.evicted_bytes += n as u64;
        }
        self.bytes += bytes;
        self.entries.push_back((key, digest, bytes, value));
    }
    pub fn stats(&self) -> ResidentCacheStats {
        let mut s = ResidentCacheStats {
            system_bytes: self.bytes as u64,
            ..self.stats.clone()
        };
        s.categories.insert(
            CacheCategory::PreparedSystems,
            CacheCategoryStats {
                bytes: self.bytes as u64,
                entries: self.entries.len() as u64,
                limit_bytes: self.limit as u64,
                limit_entries: CacheCategory::PreparedSystems.budget().1 as u64,
                ..self.category.clone()
            },
        );
        s
    }
}
/// Digest of packed numeric system content (headers, surfaces, coefficients):
/// the identity of an immutable upload, independent of handles and keys.
pub fn content_digest(systems: &[PackedSystem]) -> blake3::Hash {
    let mut h = blake3::Hasher::new();
    h.update(&(systems.len() as u64).to_le_bytes());
    for s in systems {
        h.update(bytemuck::bytes_of(&s.header));
        h.update(bytemuck::cast_slice(&s.surfaces));
        h.update(bytemuck::cast_slice(&s.coefficients));
    }
    h.finalize()
}
pub fn fingerprint(
    key: &SystemCacheKey,
    systems: &[PackedSystem],
    build: &str,
) -> MkResult<blake3::Hash> {
    if key.layout != MKL_LAYOUT_VERSION || key.kernel_build != build {
        return Err(bad("system cache layout/build mismatch"));
    }
    Ok(content_digest(systems))
}
