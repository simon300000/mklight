//! A small bounded least-recently-used cache. Engine capacities are tiny (a
//! handful of resolved systems), so a linear scan on eviction is cheaper and
//! simpler than an intrusive list.

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug)]
pub struct LruCache<K, V> {
    capacity: usize,
    tick: u64,
    map: HashMap<K, (u64, V)>,
}

impl<K: Eq + Hash + Clone, V: Clone> LruCache<K, V> {
    /// A cache holding at most `capacity` entries (at least 1).
    pub fn new(capacity: usize) -> Self {
        LruCache {
            capacity: capacity.max(1),
            tick: 0,
            map: HashMap::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// Look up `k` and mark it most recently used.
    pub fn get(&mut self, k: &K) -> Option<V> {
        self.tick += 1;
        let t = self.tick;
        self.map.get_mut(k).map(|e| {
            e.0 = t;
            e.1.clone()
        })
    }
    pub fn contains(&self, k: &K) -> bool {
        self.map.contains_key(k)
    }
    /// Insert or replace, evicting the least recently used entry when full.
    pub fn insert(&mut self, k: K, v: V) {
        self.tick += 1;
        if !self.map.contains_key(&k) && self.map.len() >= self.capacity {
            let oldest = self
                .map
                .iter()
                .min_by_key(|(_, (t, _))| *t)
                .map(|(k, _)| k.clone());
            if let Some(o) = oldest {
                self.map.remove(&o);
            }
        }
        self.map.insert(k, (self.tick, v));
    }
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_least_recently_used() {
        let mut c = LruCache::new(2);
        c.insert(1, "a");
        c.insert(2, "b");
        assert_eq!(c.get(&1), Some("a")); // 2 is now the oldest
        c.insert(3, "c");
        assert!(c.contains(&1) && c.contains(&3) && !c.contains(&2));
        c.insert(3, "c2"); // replacing does not evict
        assert_eq!(c.len(), 2);
        assert_eq!(c.get(&3), Some("c2"));
        assert_eq!(c.get(&2), None);
    }
}
