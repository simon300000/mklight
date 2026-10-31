//! Bounded immutable system-set cache. Live handles independently pin entries.
use super::{ResidentCacheStats, SystemCacheKey, bad};
use crate::{PackedSystem, consts::MKL_LAYOUT_VERSION};
use mklight_core::MkResult;
use std::{collections::VecDeque, sync::Arc};

pub struct SystemCache<T> {
    entries: VecDeque<(SystemCacheKey, blake3::Hash, usize, Arc<T>)>,
    bytes: usize,
    limit: usize,
    stats: ResidentCacheStats,
}
impl<T> SystemCache<T> {
    pub fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            limit,
            stats: Default::default(),
        }
    }
    pub fn get(&mut self, key: &SystemCacheKey, digest: blake3::Hash) -> Option<Arc<T>> {
        let i = self
            .entries
            .iter()
            .position(|(k, h, _, _)| k == key && *h == digest)?;
        let entry = self.entries.remove(i).unwrap();
        let value = entry.3.clone();
        self.entries.push_back(entry);
        self.stats.system_hits += 1;
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
        if bytes > self.limit {
            return;
        }
        while self.bytes + bytes > self.limit || self.entries.len() >= 64 {
            let (_, _, n, _) = self.entries.pop_front().unwrap();
            self.bytes -= n;
        }
        self.bytes += bytes;
        self.entries.push_back((key, digest, bytes, value));
    }
    pub fn stats(&self) -> ResidentCacheStats {
        ResidentCacheStats {
            system_bytes: self.bytes as u64,
            ..self.stats.clone()
        }
    }
}
pub fn fingerprint(
    key: &SystemCacheKey,
    systems: &[PackedSystem],
    build: &str,
) -> MkResult<blake3::Hash> {
    if key.layout != MKL_LAYOUT_VERSION || key.kernel_build != build {
        return Err(bad("system cache layout/build mismatch"));
    }
    let mut h = blake3::Hasher::new();
    h.update(&(systems.len() as u64).to_le_bytes());
    for s in systems {
        h.update(bytemuck::bytes_of(&s.header));
        h.update(bytemuck::cast_slice(&s.surfaces));
        h.update(bytemuck::cast_slice(&s.coefficients));
    }
    Ok(h.finalize())
}
