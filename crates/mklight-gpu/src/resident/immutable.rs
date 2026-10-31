//! Bounded immutable per-operation state (aim seeds, transform plans,
//! twiddles, pupils and stencils). Active jobs hold their own Arc leases.
use super::{SystemCacheKey, bad};
use mklight_core::MkResult;
use std::{any::Any, collections::VecDeque, sync::Arc};
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperationCacheKey {
    pub system: SystemCacheKey,
    pub operation: String,
    pub plan_version: u32,
}
type Value = Arc<dyn Any + Send + Sync>;
pub struct ImmutableCache {
    entries: VecDeque<(OperationCacheKey, usize, Value)>,
    bytes: usize,
    limit: usize,
}
impl ImmutableCache {
    pub fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            limit,
        }
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn get<T: Any + Send + Sync>(
        &mut self,
        key: &OperationCacheKey,
    ) -> MkResult<Option<Arc<T>>> {
        let Some(i) = self.entries.iter().position(|(k, _, _)| k == key) else {
            return Ok(None);
        };
        let value = self.entries[i]
            .2
            .clone()
            .downcast::<T>()
            .map_err(|_| bad("operation cache type collision"))?;
        let e = self.entries.remove(i).unwrap();
        self.entries.push_back(e);
        Ok(Some(value))
    }
    pub fn insert<T: Any + Send + Sync>(
        &mut self,
        key: OperationCacheKey,
        bytes: usize,
        value: Arc<T>,
    ) {
        if bytes > self.limit {
            return;
        }
        if let Some(i) = self.entries.iter().position(|(k, _, _)| k == &key) {
            self.bytes -= self.entries.remove(i).unwrap().1;
        }
        while self.bytes + bytes > self.limit || self.entries.len() >= 64 {
            self.bytes -= self.entries.pop_front().unwrap().1;
        }
        self.bytes += bytes;
        self.entries.push_back((key, bytes, value));
    }
}
