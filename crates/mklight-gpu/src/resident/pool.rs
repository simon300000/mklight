//! Exact-size scratch reuse. Only idle allocations count toward the bound.
//! Reuse prefers the most recently returned allocation; eviction the oldest.
use mklight_core::MkResult;
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

pub struct Pool<T> {
    idle: Vec<(usize, T)>,
    bytes: usize,
    limit: usize,
}
impl<T> Pool<T> {
    pub fn new(limit: usize) -> Self {
        Self {
            idle: vec![],
            bytes: 0,
            limit,
        }
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn take(
        pool: &Arc<Mutex<Self>>,
        words: usize,
        allocate: impl FnOnce(usize) -> MkResult<T>,
    ) -> MkResult<(Allocation<T>, bool)> {
        let cached = {
            let mut p = pool.lock().unwrap_or_else(|e| e.into_inner());
            p.idle.iter().rposition(|(n, _)| *n == words).map(|i| {
                p.bytes -= words.max(4) * 4;
                p.idle.remove(i).1
            })
        };
        let reused = cached.is_some();
        let value = match cached {
            Some(v) => v,
            None => allocate(words)?,
        };
        Ok((
            Allocation {
                value: Some(value),
                words,
                pool: Some(pool.clone()),
            },
            reused,
        ))
    }
}
pub struct Allocation<T> {
    value: Option<T>,
    words: usize,
    pool: Option<Arc<Mutex<Pool<T>>>>,
}
impl<T> Allocation<T> {
    /// Externally owned storage is released, never recycled into the scratch pool.
    pub fn unpooled(value: T, words: usize) -> Self {
        Self {
            value: Some(value),
            words,
            pool: None,
        }
    }
    pub fn words(&self) -> usize {
        self.words
    }
}
impl<T> Deref for Allocation<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().unwrap()
    }
}
impl<T> DerefMut for Allocation<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().unwrap()
    }
}
impl<T> Drop for Allocation<T> {
    fn drop(&mut self) {
        let bytes = self.words.max(4) * 4;
        let Some(pool) = &self.pool else {
            return;
        };
        let mut p = pool.lock().unwrap_or_else(|e| e.into_inner());
        if bytes > p.limit {
            return;
        }
        // Evict the least recently returned allocations first.
        while p.bytes + bytes > p.limit {
            let (n, _) = p.idle.remove(0);
            p.bytes -= n.max(4) * 4;
        }
        p.idle.push((self.words, self.value.take().unwrap()));
        p.bytes += bytes;
    }
}
