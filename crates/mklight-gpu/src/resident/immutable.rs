//! Bounded immutable per-operation state. Chief sources, seed tables, traced
//! pupils, transform plans and small metadata have independent byte/entry
//! budgets, so one category never evicts another. Within a category entries
//! are retained by measured rebuild cost per byte with recency aging
//! (GreedyDual-Size): an entry's priority is the category clock plus its cost
//! per byte, refreshed on every hit, and the lowest priority is evicted first,
//! advancing the clock to it. Active jobs hold their own Arc leases.
use super::{SystemCacheKey, bad};
use mklight_core::MkResult;
use mklight_core::platform::Instant;
use serde::{Deserialize, Serialize};
use std::{any::Any, collections::BTreeMap, sync::Arc};

/// Independently budgeted cache categories. Prepared (uploaded) systems live
/// in the backend's system cache; the others in [`ImmutableCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheCategory {
    PreparedSystems,
    /// Chief rays, sources and Jacobians, one row per field and wavelength.
    ChiefSources,
    /// Aim seed tables per field row.
    SeedTables,
    /// Traced or resampled pupils.
    TracedPupils,
    /// Transform recipes, twiddles, chirps and immutable spectra.
    TransformPlans,
    /// Small parameter blocks such as path programs.
    Metadata,
}
impl CacheCategory {
    pub const OPERATION: [Self; 5] = [
        Self::ChiefSources,
        Self::SeedTables,
        Self::TracedPupils,
        Self::TransformPlans,
        Self::Metadata,
    ];
    /// Byte and entry budget of the category.
    pub fn budget(self) -> (usize, usize) {
        match self {
            Self::PreparedSystems => (64 << 20, 64),
            Self::ChiefSources => (8 << 20, 1024),
            Self::SeedTables => (64 << 20, 256),
            Self::TracedPupils => (128 << 20, 64),
            Self::TransformPlans => (64 << 20, 256),
            Self::Metadata => (4 << 20, 512),
        }
    }
}

/// Counters of one category. Plan statistics hold the activity of one plan;
/// backend statistics accumulate it and add the current occupancy.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheCategoryStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    /// Entries larger than the whole category budget (never retained).
    pub rejected: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    /// Measured build cost of inserted entries, seconds.
    pub rebuild_seconds: f64,
    /// Build cost that hits did not repeat, seconds.
    pub saved_seconds: f64,
    /// Build cost of evicted entries, seconds.
    pub evicted_seconds: f64,
    /// Occupancy and budget (backend statistics only).
    pub bytes: u64,
    pub entries: u64,
    pub limit_bytes: u64,
    pub limit_entries: u64,
}
impl CacheCategoryStats {
    /// Add another record's activity counters (occupancy is not summed).
    pub fn add(&mut self, o: &Self) {
        self.hits += o.hits;
        self.misses += o.misses;
        self.inserts += o.inserts;
        self.rejected += o.rejected;
        self.evictions += o.evictions;
        self.evicted_bytes += o.evicted_bytes;
        self.rebuild_seconds += o.rebuild_seconds;
        self.saved_seconds += o.saved_seconds;
        self.evicted_seconds += o.evicted_seconds;
    }
}
pub type CacheStats = BTreeMap<CacheCategory, CacheCategoryStats>;
pub fn merge_cache_stats(a: &mut CacheStats, b: &CacheStats) {
    for (k, v) in b {
        a.entry(*k).or_default().add(v);
    }
}

/// How the build cost of an entry is known when it is published.
#[derive(Clone, Copy, Debug)]
pub enum BuildCost {
    Seconds(f64),
    /// Wall time from this instant to publication; device builds publish when
    /// their command completes, so fused later work can only overstate it.
    Since(Instant),
}
impl BuildCost {
    pub fn seconds(self) -> f64 {
        match self {
            Self::Seconds(s) => s,
            Self::Since(t) => t.elapsed().as_secs_f64(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperationCacheKey {
    pub system: SystemCacheKey,
    pub category: CacheCategory,
    pub operation: String,
    pub plan_version: u32,
}
type Value = Arc<dyn Any + Send + Sync>;
struct Entry {
    key: OperationCacheKey,
    bytes: usize,
    cost: f64,
    priority: f64,
    used: u64,
    value: Value,
}
#[derive(Default)]
struct Category {
    entries: Vec<Entry>,
    bytes: usize,
    clock: f64,
    stats: CacheCategoryStats,
}
pub struct ImmutableCache {
    categories: BTreeMap<CacheCategory, Category>,
    tick: u64,
}
impl Default for ImmutableCache {
    fn default() -> Self {
        Self::new()
    }
}
/// Cost per byte, the GreedyDual-Size credit of an entry.
fn credit(cost: f64, bytes: usize) -> f64 {
    cost.max(0.) / bytes.max(1) as f64
}
impl ImmutableCache {
    pub fn new() -> Self {
        Self {
            categories: BTreeMap::new(),
            tick: 0,
        }
    }
    pub fn bytes(&self) -> usize {
        self.categories.values().map(|c| c.bytes).sum()
    }
    pub fn stats(&self) -> CacheStats {
        CacheCategory::OPERATION
            .iter()
            .map(|&k| {
                let (limit_bytes, limit_entries) = k.budget();
                let mut s = self
                    .categories
                    .get(&k)
                    .map(|c| {
                        let mut s = c.stats.clone();
                        s.bytes = c.bytes as u64;
                        s.entries = c.entries.len() as u64;
                        s
                    })
                    .unwrap_or_default();
                s.limit_bytes = limit_bytes as u64;
                s.limit_entries = limit_entries as u64;
                (k, s)
            })
            .collect()
    }
    /// The entry and the build cost it saves; hits and misses are counted.
    pub fn get<T: Any + Send + Sync>(
        &mut self,
        key: &OperationCacheKey,
    ) -> MkResult<Option<(Arc<T>, f64)>> {
        self.tick += 1;
        let tick = self.tick;
        let c = self.categories.entry(key.category).or_default();
        let Some(e) = c.entries.iter_mut().find(|e| &e.key == key) else {
            c.stats.misses += 1;
            return Ok(None);
        };
        let value = e
            .value
            .clone()
            .downcast::<T>()
            .map_err(|_| bad("operation cache type collision"))?;
        e.priority = c.clock + credit(e.cost, e.bytes);
        e.used = tick;
        c.stats.hits += 1;
        c.stats.saved_seconds += e.cost;
        Ok(Some((value, e.cost)))
    }
    /// Insert (or replace) an entry; returns this insertion's activity.
    pub fn insert(
        &mut self,
        key: OperationCacheKey,
        bytes: usize,
        cost: f64,
        value: Value,
    ) -> CacheCategoryStats {
        self.tick += 1;
        let tick = self.tick;
        let (limit, max_entries) = key.category.budget();
        let c = self.categories.entry(key.category).or_default();
        let mut delta = CacheCategoryStats {
            inserts: 1,
            rebuild_seconds: cost,
            ..Default::default()
        };
        if bytes > limit {
            delta.rejected = 1;
            c.stats.add(&delta);
            return delta;
        }
        if let Some(i) = c.entries.iter().position(|e| e.key == key) {
            c.bytes -= c.entries.swap_remove(i).bytes;
        }
        while c.bytes + bytes > limit || c.entries.len() >= max_entries {
            // Lowest priority first; among equals, the least recently used.
            let i = (0..c.entries.len())
                .min_by(|&a, &b| {
                    let (x, y) = (&c.entries[a], &c.entries[b]);
                    x.priority.total_cmp(&y.priority).then(x.used.cmp(&y.used))
                })
                .unwrap();
            let e = c.entries.swap_remove(i);
            c.clock = c.clock.max(e.priority);
            c.bytes -= e.bytes;
            delta.evictions += 1;
            delta.evicted_bytes += e.bytes as u64;
            delta.evicted_seconds += e.cost;
        }
        c.bytes += bytes;
        c.entries.push(Entry {
            key,
            bytes,
            cost,
            priority: c.clock + credit(cost, bytes),
            used: tick,
            value,
        });
        c.stats.add(&delta);
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(category: CacheCategory, op: &str) -> OperationCacheKey {
        OperationCacheKey {
            system: SystemCacheKey {
                snapshot: String::new(),
                library: String::new(),
                configuration: String::new(),
                spectrum: String::new(),
                layout: 0,
                kernel_build: String::new(),
                operation_settings: String::new(),
            },
            category,
            operation: op.into(),
            plan_version: 1,
        }
    }
    #[test]
    fn categories_do_not_evict_each_other() {
        let mut c = ImmutableCache::new();
        c.insert(
            key(CacheCategory::SeedTables, "table"),
            1 << 20,
            0.5,
            Arc::new(1u8),
        );
        let (_, n) = CacheCategory::TransformPlans.budget();
        for i in 0..4 * n {
            c.insert(
                key(CacheCategory::TransformPlans, &i.to_string()),
                64,
                1e-4,
                Arc::new(i),
            );
        }
        let (v, cost) = c
            .get::<u8>(&key(CacheCategory::SeedTables, "table"))
            .unwrap()
            .unwrap();
        assert_eq!((*v, cost), (1, 0.5));
        let s = c.stats();
        assert_eq!(s[&CacheCategory::TransformPlans].entries, n as u64);
        assert_eq!(s[&CacheCategory::TransformPlans].evictions, 3 * n as u64);
        assert_eq!(s[&CacheCategory::SeedTables].evictions, 0);
        assert_eq!(s[&CacheCategory::SeedTables].hits, 1);
        assert_eq!(s[&CacheCategory::SeedTables].saved_seconds, 0.5);
    }
    #[test]
    fn expensive_entries_outlive_cheap_recent_ones() {
        let mut c = ImmutableCache::new();
        let (limit, _) = CacheCategory::SeedTables.budget();
        let size = limit / 4;
        c.insert(
            key(CacheCategory::SeedTables, "slow"),
            size,
            1.0,
            Arc::new(0u32),
        );
        for i in 0..12 {
            let d = c.insert(
                key(CacheCategory::SeedTables, &format!("fast{i}")),
                size,
                1e-3,
                Arc::new(i as u32),
            );
            assert!(d.evictions <= 1);
        }
        assert!(
            c.get::<u32>(&key(CacheCategory::SeedTables, "slow"))
                .unwrap()
                .is_some()
        );
        assert!(
            c.get::<u32>(&key(CacheCategory::SeedTables, "fast0"))
                .unwrap()
                .is_none()
        );
        assert!(
            c.get::<u32>(&key(CacheCategory::SeedTables, "fast11"))
                .unwrap()
                .is_some()
        );
        // Aging: enough expensive-enough newcomers eventually displace it.
        for i in 0..2000 {
            c.insert(
                key(CacheCategory::SeedTables, &format!("mid{i}")),
                size,
                0.1,
                Arc::new(0u32),
            );
        }
        assert!(
            c.get::<u32>(&key(CacheCategory::SeedTables, "slow"))
                .unwrap()
                .is_none()
        );
        let s = &c.stats()[&CacheCategory::SeedTables];
        assert!(s.bytes <= limit as u64 && s.evicted_bytes == s.evictions * size as u64);
        assert!(s.evicted_seconds >= 1.0);
    }
    #[test]
    fn oversized_entries_are_rejected_and_replacement_keeps_bytes() {
        let mut c = ImmutableCache::new();
        let (limit, _) = CacheCategory::Metadata.budget();
        let d = c.insert(
            key(CacheCategory::Metadata, "big"),
            limit + 1,
            0.,
            Arc::new(0u8),
        );
        assert_eq!(d.rejected, 1);
        c.insert(key(CacheCategory::Metadata, "a"), 100, 0., Arc::new(0u8));
        c.insert(key(CacheCategory::Metadata, "a"), 60, 0., Arc::new(1u8));
        assert_eq!(c.bytes(), 60);
        assert!(c.get::<u16>(&key(CacheCategory::Metadata, "a")).is_err());
        assert_eq!(c.stats()[&CacheCategory::Metadata].misses, 0);
    }
}
