//! Plan data kept on the device between device operations (GPU.md §11.1,
//! "Residency").
//!
//! Plan buffers live on the host (`backend` module docs). Without residency
//! every device operation uploaded its inputs and read its outputs back,
//! 10–15 ms per operation. Now a device operation leaves its outputs in
//! device buffers (*mirrors*, one per plan buffer) and the next device
//! operation reads them there:
//!
//! * [`Mirrors::load`] gives an input view as a batch resource: the mirror
//!   when the device copy is current (a device-side slice when the view does
//!   not start at a bindable offset), else an upload of the host words.
//! * [`Mirrors::store`] copies a batch resource into an output's mirror on
//!   the device; [`Mirrors::output`] hands out the mirror itself for a kernel
//!   to write. Either marks the host copy stale.
//! * Before an operation runs its CPU-f32 specification, its stale inputs
//!   are read back ([`Mirrors::sync_reads`]); what it writes on the host
//!   drops the mirror. When the plan ends, only its final outputs are read
//!   back ([`Mirrors::finish`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use mklight_core::MkResult;
use mklight_gpu::JobStats;
use mklight_gpu::resident::cpu::Execution;
use mklight_gpu::resident::{BufferId, PlanView, ResidentOp, ResidentPlan, accesses};

use crate::device::*;

/// Word counts of the last `READ_BACK_LOG` buffers read back from the device
/// by any plan of this process, in order (diagnostics: tests check what is
/// not read back).
pub const READ_BACK_LOG: usize = 4096;
pub static READ_BACKS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Words one readback batch carries at most (several small buffers share a
/// batch; a device stages a batch's reads together).
const READ_BATCH_WORDS: usize = 16 << 20;

struct Mirror {
    key: BufferKey,
    words: usize,
    /// The host buffer holds the same data (else only the device does).
    host: bool,
}

/// The device copies of one plan's buffers.
pub struct Mirrors {
    gpu: Arc<dyn Gpu>,
    map: HashMap<BufferId, Mirror>,
    /// Outputs a device operation stored on the device in this operation.
    stored: HashSet<BufferId>,
    /// Bytes not uploaded or read back thanks to residency (diagnostics).
    pub saved_bytes: u64,
}

impl Mirrors {
    pub fn new(gpu: Arc<dyn Gpu>) -> Self {
        Self {
            gpu,
            map: HashMap::new(),
            stored: HashSet::new(),
            saved_bytes: 0,
        }
    }

    /// Whether the device holds `id`'s current contents.
    pub fn on_device(&self, id: BufferId) -> bool {
        self.map.contains_key(&id)
    }

    /// Input `v` (validated, contiguous) as a resource of `batch`.
    pub fn load<'a>(&mut self, batch: &mut Batch<'a>, x: &'a Execution, v: &PlanView) -> usize {
        let w = v.desc.element.words();
        let (off, len) = (v.offset * w, v.desc.len().unwrap_or(0) * w);
        match self.map.get(&v.buffer) {
            Some(m) => {
                self.saved_bytes += (len * 4) as u64;
                if off == 0 && len == m.words {
                    batch.persistent(m.key)
                } else {
                    batch.slice(m.key, off, len)
                }
            }
            None => batch.upload(x.view(v)),
        }
    }

    /// The mirror of `id` as a read-only resource of `batch` when the device
    /// copy is current (for kernels that pack several plan buffers).
    pub fn resource(&mut self, batch: &mut Batch<'_>, id: BufferId) -> Option<usize> {
        self.map.get(&id).map(|m| batch.persistent(m.key))
    }

    /// The mirror of output `id` (allocated on first use), for a kernel to
    /// write whole or for [`Mirrors::store`]; the host copy becomes stale.
    pub fn output(&mut self, x: &Execution, id: BufferId) -> MkResult<BufferKey> {
        self.stored.insert(id);
        if let Some(m) = self.map.get_mut(&id) {
            m.host = false;
            return Ok(m.key);
        }
        let words = x.words(id).len();
        let key = self.gpu.allocate(words)?;
        self.map.insert(
            id,
            Mirror {
                key,
                words,
                host: false,
            },
        );
        Ok(key)
    }

    /// Copy `len` words of `batch` resource `from` (at `from_offset`) into
    /// output `id` at `to_offset`, on the device, after the passes.
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &mut self,
        batch: &mut Batch<'_>,
        x: &Execution,
        from: usize,
        from_offset: usize,
        id: BufferId,
        to_offset: usize,
        len: usize,
    ) -> MkResult<()> {
        let key = self.output(x, id)?;
        self.saved_bytes += (len * 4) as u64;
        batch.copy(from, from_offset, key, to_offset, len);
        Ok(())
    }

    /// Make `id` current on the host (a small explicit readback for words
    /// the host inspects inside an operation).
    pub fn read_back(
        &mut self,
        id: BufferId,
        x: &mut Execution,
        stats: &mut JobStats,
    ) -> MkResult<()> {
        self.read_back_all(vec![id], x, stats)
    }

    /// [`Mirrors::read_back`] of every buffer of `ids`, in as few batches as
    /// [`READ_BATCH_WORDS`] allows.
    pub fn read_back_all(
        &mut self,
        mut ids: Vec<BufferId>,
        x: &mut Execution,
        stats: &mut JobStats,
    ) -> MkResult<()> {
        ids.sort_by_key(|b| b.0);
        ids.dedup();
        ids.retain(|id| self.map.get(id).is_some_and(|m| !m.host));
        let (mut group, mut words) = (Vec::new(), 0);
        for id in ids {
            let w = self.map[&id].words;
            if !group.is_empty() && words + w > READ_BATCH_WORDS {
                self.read_group(std::mem::take(&mut group), x, stats)?;
                words = 0;
            }
            group.push(id);
            words += w;
        }
        if group.is_empty() {
            return Ok(());
        }
        self.read_group(group, x, stats)
    }

    /// One batch reading back the distinct stale mirrors `ids`.
    fn read_group(
        &mut self,
        ids: Vec<BufferId>,
        x: &mut Execution,
        stats: &mut JobStats,
    ) -> MkResult<()> {
        // Only device operations' outputs have mirrors: plan-owned buffers.
        let (gpu, map) = (&self.gpu, &self.map);
        let b = x.with_outputs(&ids, |_, words| {
            let mut batch = Batch::default();
            for (id, w) in ids.iter().zip(words.iter_mut()) {
                let m = &map[id];
                let r = batch.persistent(m.key);
                batch.read(r, 0, &mut w[..m.words]);
            }
            gpu.run(batch)
        })?;
        stats.device_seconds += b.device_seconds;
        stats.command_buffers += 1;
        let mut log = READ_BACKS.lock().ok();
        for id in &ids {
            let m = self.map.get_mut(id).unwrap();
            if let Some(log) = &mut log {
                if log.len() >= READ_BACK_LOG {
                    log.drain(..READ_BACK_LOG / 2);
                }
                log.push(m.words);
            }
            m.host = true;
        }
        Ok(())
    }

    /// Before `op` runs its specification: its inputs current on the host.
    pub fn sync_reads(
        &mut self,
        op: &ResidentOp,
        x: &mut Execution,
        stats: &mut JobStats,
    ) -> MkResult<()> {
        if let ResidentOp::TraceUpload(_) = op {
            return Ok(());
        }
        let ids = accesses(op).reads.iter().map(|v| v.buffer).collect();
        self.read_back_all(ids, x, stats)
    }

    /// After operation `op` ran (on the device or as its specification):
    /// outputs it wrote on the host lose their mirrors.
    pub fn after(&mut self, op: &ResidentOp) {
        for id in accesses(op).writes {
            if !self.stored.contains(&id)
                && let Some(m) = self.map.remove(&id)
            {
                self.gpu.release(m.key);
            }
        }
        self.stored.clear();
    }

    /// At the end of the plan: its final outputs current on the host.
    pub fn finish(
        &mut self,
        plan: &ResidentPlan,
        x: &mut Execution,
        stats: &mut JobStats,
    ) -> MkResult<()> {
        self.read_back_all(plan.outputs.clone(), x, stats)
    }
}

impl Drop for Mirrors {
    fn drop(&mut self) {
        for (_, m) in self.map.drain() {
            self.gpu.release(m.key);
        }
    }
}
