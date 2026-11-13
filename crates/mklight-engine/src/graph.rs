//! The engine's analysis dependency graph: shared, content-addressed
//! intermediates under the analysis requests (`docs/ARCHITECTURE.md`,
//! "Analysis graph").
//!
//! Every node is identified by a content key (`mklight_core::memo`):
//! the digest of what it is computed from — snapshot content, material
//! library, configuration, settings that matter, precision and algorithm
//! version. Consumers of one snapshot (and of other snapshots with equal
//! content: undo, configurations a solve leaves unchanged) share nodes; an
//! edit changes the content, so exactly the nodes that depend on it miss
//! and nothing stale is ever returned.
//!
//! | node | key | used by |
//! |---|---|---|
//! | `tracer` | snapshot, library | geometric, extended and radiometry adapters (first order + prepared system fields) |
//! | `field_tracer` | `tracer` key, field selection | geometric analyses with axis fields added |
//! | `configured` | snapshot, library, configuration | wave adapter (configuration, solves, first order, prepared fields; host seeds of the Metal path) |
//! | `pupil_lattice`, `resampled_pupil` | `mklight_wave::shared` keys, scoped by library | CPU f64 wave analyses |
//! | `result` | snapshot, library, analysis, normalised settings, policy; backend selection for explicit requests | repeated requests (refresh, undo) |
//!
//! Material resolution stays in the session's resolved-system cache.
//!
//! **Memory.** One byte budget for all nodes (`EngineConfig::
//! analysis_graph_bytes`; 0 disables the graph). Entries are retained by
//! rebuild cost per byte with recency aging (GreedyDual-Size): an expensive
//! traced pupil outlives cheap recent ones. A single node larger than a
//! quarter of the budget is returned but not stored.
//!
//! **Concurrency.** Builds of distinct nodes run in parallel. A batch
//! claims all its missing keys at once; a caller outside the rayon pool
//! waits for keys another caller is building (queued six-graph refreshes
//! trace each pupil once), while rayon workers build such keys themselves,
//! since blocking a worker could deadlock work stealing. Node kinds have
//! levels (`result` > tracers > `pupil_lattice` > others): a build may
//! wait only for nodes of a strictly lower level, and a batch publishes its
//! own claims before it waits, so waits cannot form cycles.
//!
//! **Rebuild (fresh requests).** A lookup in rebuild mode
//! (`AnalysisRequest::fresh`) never reads a stored node nor waits for
//! another caller's build: it builds the node itself and its result
//! replaces the stored one, so later ordinary lookups get the new value.
//! It claims the key like a miss when nobody is building it (ordinary
//! callers then wait for it), and otherwise builds an unclaimed duplicate
//! that is published over whatever is stored.
//!
//! **Cancellation.** Only successful builds are published. A failed,
//! cancelled or panicking build releases its claim and wakes the waiters,
//! which then build the node themselves.

use mklight_core::platform::Instant;
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use mklight_core::memo::{Built, KeyHasher, Memo, NodeKey, NodeValue};
use mklight_core::{JobContext, MkError, MkResult};
use rayon::prelude::*;
use serde::Serialize;

/// Activity of one node kind.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct KindStats {
    /// Served from the graph (including after waiting for another build).
    pub hits: u64,
    /// Of the hits, those that waited for a concurrent build.
    pub waits: u64,
    /// Built here (misses).
    pub builds: u64,
    /// Builds that failed or were cancelled (nothing published).
    pub failed_builds: u64,
    /// Built although another caller was building it (inside the rayon
    /// pool, where waiting is unsafe).
    pub duplicate_builds: u64,
    /// Of the builds, those made in rebuild mode (fresh requests), which
    /// read nothing stored.
    pub rebuilds: u64,
    /// Larger than a quarter of the budget: returned, not stored.
    pub rejected: u64,
    pub evictions: u64,
    pub entries: usize,
    pub bytes: usize,
    /// Build time spent, and build time the hits avoided (seconds).
    pub build_seconds: f64,
    pub saved_seconds: f64,
}

/// Bytes of a value that must not be stored (it is returned only).
pub(crate) const UNSTORED: usize = usize::MAX;

/// Snapshot of the graph's state (`Engine::analysis_graph_stats`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GraphStats {
    pub budget_bytes: usize,
    pub bytes: usize,
    pub entries: usize,
    pub kinds: BTreeMap<&'static str, KindStats>,
}

struct Entry {
    kind: &'static str,
    value: NodeValue,
    bytes: usize,
    cost: f64,
    priority: f64,
}

#[derive(Default)]
struct State {
    entries: HashMap<NodeKey, Entry>,
    /// Keys being built, with the claim that builds them.
    building: HashMap<NodeKey, u64>,
    next_claim: u64,
    bytes: usize,
    /// GreedyDual-Size inflation value.
    clock: f64,
    stats: BTreeMap<&'static str, KindStats>,
}

impl State {
    fn stats(&mut self, kind: &'static str) -> &mut KindStats {
        self.stats.entry(kind).or_default()
    }

    fn priority(&self, cost: f64, bytes: usize) -> f64 {
        self.clock + cost / bytes.max(1) as f64
    }

    fn hit(&mut self, key: &NodeKey, waited: bool) -> Option<NodeValue> {
        let clock = self.clock;
        let e = self.entries.get_mut(key)?;
        e.priority = clock + e.cost / e.bytes.max(1) as f64;
        let (kind, cost, value) = (e.kind, e.cost, e.value.clone());
        let s = self.stats(kind);
        s.hits += 1;
        s.waits += waited as u64;
        s.saved_seconds += cost;
        Some(value)
    }
}

/// The analysis graph (module docs). Shared by every session of an engine.
pub struct AnalysisGraph {
    state: Mutex<State>,
    changed: Condvar,
    budget: usize,
}

/// Level of a node kind (module docs, concurrency).
fn level(kind: &str) -> u8 {
    match kind {
        "result" => 3,
        "configured" | "tracer" | "field_tracer" => 2,
        "pupil_lattice" => 1,
        _ => 0,
    }
}

thread_local! {
    /// Lowest level of the builds running on this thread (`u8::MAX`:
    /// none). Nested requests may wait only for lower levels.
    static BUILDING: std::cell::Cell<u8> = const { std::cell::Cell::new(u8::MAX) };
}

/// Whether a request for `kind` on this thread may block on another
/// caller's build.
fn may_wait(kind: &str) -> bool {
    rayon::current_thread_index().is_none() && level(kind) < BUILDING.with(|d| d.get())
}

/// Marks a build of a kind running on this thread (`BUILDING`), restored
/// also on unwinding.
struct Depth(u8);

impl Depth {
    fn enter(kind: &str) -> Self {
        let previous = BUILDING.with(|d| d.replace(d.get().min(level(kind))));
        Depth(previous)
    }
}

impl Drop for Depth {
    fn drop(&mut self) {
        BUILDING.with(|d| d.set(self.0));
    }
}

/// Releases an unpublished claim (failed, cancelled or panicking build,
/// or a build never started because another one panicked).
struct Claim<'a> {
    graph: &'a AnalysisGraph,
    key: NodeKey,
    id: u64,
    kind: &'static str,
    done: bool,
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let mut st = self.graph.lock();
        if st.building.get(&self.key) == Some(&self.id) {
            st.building.remove(&self.key);
        }
        st.stats(self.kind).failed_builds += 1;
        drop(st);
        self.graph.changed.notify_all();
    }
}

impl AnalysisGraph {
    pub fn new(budget_bytes: usize) -> Self {
        AnalysisGraph {
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
            budget: budget_bytes,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn stats(&self) -> GraphStats {
        let st = self.lock();
        let mut kinds = st.stats.clone();
        for k in kinds.values_mut() {
            k.entries = 0;
            k.bytes = 0;
        }
        for e in st.entries.values() {
            let k = kinds.entry(e.kind).or_default();
            k.entries += 1;
            k.bytes += e.bytes;
        }
        GraphStats {
            budget_bytes: self.budget,
            bytes: st.bytes,
            entries: st.entries.len(),
            kinds,
        }
    }

    /// Whether `key` is stored (no hit is counted).
    pub fn contains(&self, key: &NodeKey) -> bool {
        self.lock().entries.contains_key(key)
    }

    /// Drop every stored node (statistics are kept).
    pub fn clear(&self) {
        let mut st = self.lock();
        st.entries.clear();
        st.bytes = 0;
    }

    /// Store a built node, evicting the cheapest-to-rebuild entries per
    /// byte until it fits.
    fn publish(&self, st: &mut State, kind: &'static str, key: NodeKey, b: &Built, cost: f64) {
        let s = st.stats(kind);
        s.builds += 1;
        s.build_seconds += cost;
        // The new value replaces the stored one, also when it is too large
        // to be stored itself (a rebuild leaves nothing stale behind).
        if let Some(old) = st.entries.remove(&key) {
            st.bytes -= old.bytes;
        }
        if b.bytes > self.budget / 4 {
            st.stats(kind).rejected += 1;
            return;
        }
        while st.bytes + b.bytes > self.budget {
            let Some((&victim, _)) = st
                .entries
                .iter()
                .min_by(|a, b| a.1.priority.total_cmp(&b.1.priority))
            else {
                break;
            };
            let e = st.entries.remove(&victim).expect("victim");
            st.bytes -= e.bytes;
            st.clock = st.clock.max(e.priority);
            st.stats(e.kind).evictions += 1;
        }
        let priority = st.priority(cost, b.bytes);
        st.bytes += b.bytes;
        st.entries.insert(
            key,
            Entry {
                kind,
                value: b.value.clone(),
                bytes: b.bytes,
                cost,
                priority,
            },
        );
    }

    /// Build `key` under `claim` (`None`: an unclaimed duplicate) and
    /// publish it on success.
    fn build(
        &self,
        kind: &'static str,
        key: NodeKey,
        claim: Option<&mut Claim>,
        rebuild: bool,
        build: &dyn Fn() -> MkResult<Built>,
    ) -> MkResult<NodeValue> {
        let start = Instant::now();
        let b = {
            let _depth = Depth::enter(kind);
            build()?
        };
        let cost = start.elapsed().as_secs_f64();
        self.finish(kind, key, claim, rebuild, &b, cost);
        Ok(b.value)
    }

    /// Publish a successful build and release its claim. An unclaimed
    /// duplicate is published only if nothing is stored, unless it is a
    /// rebuild, which replaces the stored value.
    fn finish(
        &self,
        kind: &'static str,
        key: NodeKey,
        claim: Option<&mut Claim>,
        rebuild: bool,
        b: &Built,
        cost: f64,
    ) {
        let mut st = self.lock();
        st.stats(kind).rebuilds += rebuild as u64;
        match claim {
            Some(g) => {
                if st.building.get(&key) == Some(&g.id) {
                    st.building.remove(&key);
                }
                g.done = true;
                self.publish(&mut st, kind, key, b, cost);
            }
            None => {
                st.stats(kind).duplicate_builds += 1;
                if rebuild || !st.entries.contains_key(&key) {
                    self.publish(&mut st, kind, key, b, cost);
                }
            }
        }
        drop(st);
        self.changed.notify_all();
    }

    /// `Memo::nodes` with the keys already scoped; `rebuild`: module docs.
    fn batch(
        &self,
        kind: &'static str,
        keys: &[NodeKey],
        ctx: &JobContext,
        rebuild: bool,
        build: &(dyn Fn(usize) -> MkResult<Built> + Sync),
    ) -> Vec<MkResult<NodeValue>> {
        // A rebuild never waits: it builds what others are building too.
        let may_wait = !rebuild && may_wait(kind);
        let mut out: Vec<Option<MkResult<NodeValue>>> = (0..keys.len()).map(|_| None).collect();
        let mut pending: Vec<usize> = (0..keys.len()).collect();
        let mut waited = false;
        while !pending.is_empty() {
            // Claim every missing key of this round at once.
            let mut mine: Vec<(usize, Claim)> = Vec::new();
            let mut theirs = Vec::new();
            {
                let mut st = self.lock();
                for &i in &pending {
                    if let Some(v) = (!rebuild).then(|| st.hit(&keys[i], waited)).flatten() {
                        out[i] = Some(Ok(v));
                    } else if st.building.contains_key(&keys[i]) {
                        theirs.push(i);
                    } else {
                        let id = st.next_claim;
                        st.next_claim += 1;
                        st.building.insert(keys[i], id);
                        let claim = Claim {
                            graph: self,
                            key: keys[i],
                            id,
                            kind,
                            done: false,
                        };
                        mine.push((i, claim));
                    }
                }
            }
            let run = |(i, claim): &mut (usize, Claim)| {
                let i = *i;
                (
                    i,
                    self.build(kind, keys[i], Some(claim), rebuild, &|| build(i)),
                )
            };
            let built: Vec<(usize, MkResult<NodeValue>)> = if mine.len() > 1 {
                mine.par_iter_mut().map(run).collect()
            } else {
                mine.iter_mut().map(run).collect()
            };
            drop(mine);
            for (i, r) in built {
                out[i] = Some(r);
            }
            if theirs.is_empty() {
                break;
            }
            if !may_wait {
                let dup: Vec<(usize, MkResult<NodeValue>)> = theirs
                    .par_iter()
                    .map(|&i| (i, self.build(kind, keys[i], None, rebuild, &|| build(i))))
                    .collect();
                for (i, r) in dup {
                    out[i] = Some(r);
                }
                break;
            }
            // Wait until none of `theirs` is being built any more, then
            // take the published ones and claim the rest.
            let mut st = self.lock();
            while theirs.iter().any(|i| st.building.contains_key(&keys[*i])) {
                if let Err(e) = ctx.check() {
                    drop(st);
                    for &i in &theirs {
                        out[i] = Some(Err(e.clone()));
                    }
                    theirs.clear();
                    break;
                }
                st = self
                    .changed
                    .wait_timeout(st, Duration::from_millis(20))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            waited = true;
            pending = theirs;
        }
        out.into_iter()
            .map(|r| r.unwrap_or_else(|| Err(MkError::Numerical("graph node missing".into()))))
            .collect()
    }

    /// One node of `kind`, typed (engine-side nodes), built on this
    /// thread with the protocol of [`Self::batch`]. `build` returns the
    /// value and its bytes; [`UNSTORED`] returns the value without storing
    /// it. `rebuild` builds without reading or waiting (module docs).
    pub(crate) fn single<T, E>(
        &self,
        kind: &'static str,
        key: NodeKey,
        ctx: &JobContext,
        rebuild: bool,
        build: impl FnOnce() -> Result<(T, usize), E>,
    ) -> Result<Arc<T>, E>
    where
        T: Any + Send + Sync,
        E: From<MkError>,
    {
        let may_wait = may_wait(kind);
        let mut waited = false;
        let mut st = self.lock();
        if !rebuild {
            loop {
                if let Some(v) = st.hit(&key, waited) {
                    return Ok(mklight_core::memo::downcast::<T>(v));
                }
                if !(may_wait && st.building.contains_key(&key)) {
                    break;
                }
                while st.building.contains_key(&key) {
                    if let Err(e) = ctx.check() {
                        return Err(e.into());
                    }
                    st = self
                        .changed
                        .wait_timeout(st, Duration::from_millis(20))
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                waited = true;
            }
        }
        // Claim it, unless another caller is building it (a duplicate).
        let mut claim = (!st.building.contains_key(&key)).then(|| {
            let id = st.next_claim;
            st.next_claim += 1;
            st.building.insert(key, id);
            Claim {
                graph: self,
                key,
                id,
                kind,
                done: false,
            }
        });
        drop(st);
        let start = Instant::now();
        let (value, bytes) = {
            let _depth = Depth::enter(kind);
            build()?
        };
        let value = Arc::new(value);
        let b = Built {
            value: value.clone(),
            bytes,
        };
        let cost = start.elapsed().as_secs_f64();
        self.finish(kind, key, claim.as_mut(), rebuild, &b, cost);
        Ok(value)
    }

    /// One node of `kind`, typed (engine-side nodes); `rebuild`: module
    /// docs.
    pub(crate) fn node<T: Any + Send + Sync>(
        &self,
        kind: &'static str,
        key: NodeKey,
        ctx: &JobContext,
        rebuild: bool,
        bytes: impl FnOnce(&T) -> usize,
        build: impl FnOnce() -> MkResult<T>,
    ) -> MkResult<Arc<T>> {
        self.single(kind, key, ctx, rebuild, || {
            let v = build()?;
            let b = bytes(&v);
            Ok((v, b))
        })
    }

    /// A memo for solvers whose keys do not see the material library:
    /// every key is scoped by `scope`. `rebuild`: module docs.
    pub fn scoped(self: &Arc<Self>, scope: NodeKey, rebuild: bool) -> ScopedMemo {
        ScopedMemo {
            graph: self.clone(),
            scope,
            rebuild,
        }
    }
}

/// [`AnalysisGraph::scoped`].
pub struct ScopedMemo {
    graph: Arc<AnalysisGraph>,
    scope: NodeKey,
    rebuild: bool,
}

impl Memo for ScopedMemo {
    fn nodes(
        &self,
        kind: &'static str,
        keys: &[NodeKey],
        ctx: &JobContext,
        build: &(dyn Fn(usize) -> MkResult<Built> + Sync),
    ) -> Vec<MkResult<NodeValue>> {
        let scoped: Vec<NodeKey> = keys
            .iter()
            .map(|k| {
                KeyHasher::new("mklight-engine/scope/1")
                    .key(&self.scope)
                    .str(kind)
                    .key(k)
                    .finish()
            })
            .collect();
        self.graph.batch(kind, &scoped, ctx, self.rebuild, build)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn key(i: u64) -> NodeKey {
        KeyHasher::new("test").u64(i).finish()
    }

    fn get(g: &AnalysisGraph, k: NodeKey, bytes: usize, builds: &AtomicUsize) -> Arc<u64> {
        g.node(
            "t",
            k,
            &JobContext::new(),
            false,
            |_| bytes,
            || {
                builds.fetch_add(1, Ordering::SeqCst);
                Ok(7u64)
            },
        )
        .unwrap()
    }

    #[test]
    fn hits_misses_and_budget() {
        let g = AnalysisGraph::new(1000);
        let n = AtomicUsize::new(0);
        assert_eq!(*get(&g, key(1), 100, &n), 7);
        assert_eq!(*get(&g, key(1), 100, &n), 7);
        assert_eq!(n.load(Ordering::SeqCst), 1);
        // Oversized: returned, never stored.
        get(&g, key(2), 400, &n);
        get(&g, key(2), 400, &n);
        assert_eq!(n.load(Ordering::SeqCst), 3);
        let s = g.stats();
        assert_eq!((s.kinds["t"].hits, s.kinds["t"].rejected), (1, 2));
        // Fill past the budget: total stays bounded.
        for i in 10..30 {
            get(&g, key(i), 200, &n);
        }
        let s = g.stats();
        assert!(s.bytes <= 1000 && s.kinds["t"].evictions > 0, "{s:?}");
    }

    #[test]
    fn rebuilds_replace_the_stored_node() {
        let g = AnalysisGraph::new(1 << 20);
        let ctx = JobContext::new();
        let v = |rebuild, x: u64| *g.node("t", key(1), &ctx, rebuild, |_| 8, || Ok(x)).unwrap();
        assert_eq!(v(false, 1), 1);
        assert_eq!(v(false, 2), 1, "served from the graph");
        assert_eq!(v(true, 3), 3, "a rebuild reads nothing stored");
        assert_eq!(v(false, 4), 3, "and replaces the stored value");
        // Batches (solver memos) too.
        let b = |rebuild, x: u64| {
            let r = g.batch("b", &[key(2)], &ctx, rebuild, &|_| Ok(Built::new(x, 8)));
            *r.into_iter()
                .next()
                .unwrap()
                .unwrap()
                .downcast::<u64>()
                .unwrap()
        };
        assert_eq!((b(false, 1), b(true, 2), b(false, 3)), (1, 2, 2));
        let s = g.stats();
        assert_eq!((s.kinds["t"].builds, s.kinds["t"].rebuilds), (2, 1));
        assert_eq!((s.kinds["b"].builds, s.kinds["b"].rebuilds), (2, 1));
        assert_eq!(s.entries, 2);
        assert!(g.lock().building.is_empty());
    }

    #[test]
    fn expensive_entries_outlive_cheap_recent_ones() {
        let g = AnalysisGraph::new(1000);
        let slow = g
            .batch("t", &[key(1)], &JobContext::new(), false, &|_| {
                std::thread::sleep(Duration::from_millis(30));
                Ok(Built::new(1u64, 200))
            })
            .pop()
            .unwrap();
        assert!(slow.is_ok());
        let n = AtomicUsize::new(0);
        for i in 2..20 {
            get(&g, key(i), 200, &n);
        }
        let before = n.load(Ordering::SeqCst);
        // Still cached although it is the least recently used.
        let again = g
            .batch("t", &[key(1)], &JobContext::new(), false, &|_| {
                Ok(Built::new(2u64, 200))
            })
            .pop()
            .unwrap()
            .unwrap();
        assert_eq!(*again.downcast::<u64>().unwrap(), 1);
        assert_eq!(n.load(Ordering::SeqCst), before);
    }

    #[test]
    fn failed_and_cancelled_builds_publish_nothing() {
        let g = AnalysisGraph::new(1 << 20);
        let ctx = JobContext::new();
        let r = g.node(
            "t",
            key(1),
            &ctx,
            false,
            |_: &u64| 8,
            || Err(MkError::Cancelled),
        );
        assert!(matches!(r, Err(MkError::Cancelled)));
        let r = g.node::<u64>(
            "t",
            key(1),
            &ctx,
            false,
            |_| 8,
            || Err(MkError::Trace("x".into())),
        );
        assert!(r.is_err());
        // A panicking build releases its claim too.
        let p = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.node::<u64>("t", key(1), &ctx, false, |_| 8, || panic!("boom"))
        }));
        assert!(p.is_err());
        let n = AtomicUsize::new(0);
        assert_eq!(*get(&g, key(1), 8, &n), 7);
        assert_eq!(n.load(Ordering::SeqCst), 1);
        let s = g.stats();
        assert_eq!(s.kinds["t"].failed_builds, 3);
        assert!(g.lock().building.is_empty());
    }

    #[test]
    fn concurrent_callers_build_once() {
        let g = Arc::new(AnalysisGraph::new(1 << 20));
        let n = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (g, n) = (g.clone(), n.clone());
                std::thread::spawn(move || {
                    let keys: Vec<NodeKey> = (0..6).map(key).collect();
                    g.batch("t", &keys, &JobContext::new(), false, &|i| {
                        n.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(20));
                        Ok(Built::new(i as u64, 8))
                    })
                    .into_iter()
                    .map(|r| *r.unwrap().downcast::<u64>().unwrap())
                    .collect::<Vec<_>>()
                })
            })
            .collect();
        for t in threads {
            assert_eq!(t.join().unwrap(), vec![0, 1, 2, 3, 4, 5]);
        }
        assert_eq!(n.load(Ordering::SeqCst), 6);
        assert!(g.stats().kinds["t"].waits > 0);
    }

    #[test]
    fn waiters_rebuild_after_a_cancelled_builder() {
        let g = Arc::new(AnalysisGraph::new(1 << 20));
        let started = Arc::new(std::sync::Barrier::new(2));
        let cancelled = JobContext::new();
        let builder = {
            let (g, started, ctx) = (g.clone(), started.clone(), cancelled.clone());
            std::thread::spawn(move || {
                g.node::<u64>(
                    "t",
                    key(1),
                    &ctx,
                    false,
                    |_| 8,
                    || {
                        started.wait();
                        std::thread::sleep(Duration::from_millis(50));
                        Err(MkError::Cancelled)
                    },
                )
            })
        };
        started.wait();
        let v = g
            .node::<u64>("t", key(1), &JobContext::new(), false, |_| 8, || Ok(5))
            .unwrap();
        assert_eq!(*v, 5);
        assert!(builder.join().unwrap().is_err());
        // A waiting caller whose own job is cancelled stops waiting.
        let blocker = {
            let g = g.clone();
            std::thread::spawn(move || {
                g.node::<u64>(
                    "t",
                    key(2),
                    &JobContext::new(),
                    false,
                    |_| 8,
                    || {
                        std::thread::sleep(Duration::from_millis(200));
                        Ok(1)
                    },
                )
            })
        };
        std::thread::sleep(Duration::from_millis(30));
        let ctx = JobContext::new();
        ctx.cancel.cancel();
        let r = g.node::<u64>("t", key(2), &ctx, false, |_| 8, || Ok(2));
        assert!(matches!(r, Err(MkError::Cancelled)));
        assert_eq!(*blocker.join().unwrap().unwrap(), 1);
    }

    #[test]
    fn rayon_workers_do_not_wait() {
        let g = Arc::new(AnalysisGraph::new(1 << 20));
        let g2 = g.clone();
        // Hold key 1 in flight from a plain thread, then request it from
        // inside the pool: the worker builds a duplicate instead of waiting.
        let gate = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let (gate2, release2) = (gate.clone(), release.clone());
        let holder = std::thread::spawn(move || {
            g2.node::<u64>(
                "t",
                key(1),
                &JobContext::new(),
                false,
                |_| 8,
                || {
                    gate2.wait();
                    release2.wait();
                    Ok(1)
                },
            )
        });
        gate.wait();
        let v = rayon::scope(|_| {
            g.node::<u64>("t", key(1), &JobContext::new(), false, |_| 8, || Ok(1))
        })
        .unwrap();
        assert_eq!(*v, 1);
        release.wait();
        holder.join().unwrap().unwrap();
        assert_eq!(g.stats().kinds["t"].duplicate_builds, 1);
    }
}
