//! The device seam: what the backend asks of a WebGPU device.
//!
//! The backend never touches wgpu objects. It describes each unit of device
//! work as a [`Batch`] (buffers to upload or allocate, compute passes, and
//! the ranges to read back into host slices) and hands it to a [`Gpu`],
//! which blocks until the batch has completed:
//!
//! * natively, [`crate::wgpu_device::WgpuDevice`] records and submits the
//!   batch on the calling thread (wgpu's native objects are `Send + Sync`);
//! * in the browser, wgpu objects are not `Send` and buffers map
//!   asynchronously, while the engine runs synchronously on rayon worker
//!   threads. There [`crate::proxy::GpuProxy`] forwards the batch to the one
//!   thread that owns the `GPUDevice` and blocks the calling thread until
//!   that thread's event loop has completed it (GPU.md §11).
//!
//! A batch borrows its uploads and readback destinations; the caller blocks
//! for the whole call, so on the web the GPU thread reads and writes them in
//! the shared WebAssembly memory without a copy.
//!
//! Binding convention of every kernel (group 0): binding 0 is the
//! environment block (`mkl_env`, the opaque zero of `common.wgsl`), binding 1
//! the pass's uniform parameters when it has some, and bindings 2… the
//! storage buffers in [`Pass::storage`] order.

use mklight_core::MkResult;
use serde::{Deserialize, Serialize};

/// A compiled entry point: `module` names a WGSL source of
/// [`crate::kernels`], `entry` the `@compute` function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Kernel {
    pub module: &'static str,
    pub entry: &'static str,
}

impl Kernel {
    pub const fn new(module: &'static str, entry: &'static str) -> Self {
        Self { module, entry }
    }
}

/// How a pass binds a storage buffer; it must match the kernel's
/// `var<storage, read>` / `var<storage, read_write>` declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    ReadWrite,
}

/// A device buffer that outlives batches (uploaded systems), read-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferKey(pub u64);

/// A buffer a batch uses, by index in [`Batch::resources`].
pub enum Resource<'a> {
    /// A persistent buffer.
    Persistent(BufferKey),
    /// Words uploaded for this batch.
    Upload(&'a [u32]),
    /// Zero-initialised device storage of this many words.
    Scratch(usize),
}

/// A storage binding: a whole resource, or the word range `offset..offset+len`
/// (`offset` a multiple of the device's storage offset alignment).
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub resource: usize,
    pub access: Access,
    pub range: Option<(usize, usize)>,
}

impl Binding {
    pub fn read(resource: usize) -> Self {
        Self {
            resource,
            access: Access::Read,
            range: None,
        }
    }
    pub fn write(resource: usize) -> Self {
        Self {
            resource,
            access: Access::ReadWrite,
            range: None,
        }
    }
}

/// One compute dispatch.
pub struct Pass {
    pub kernel: Kernel,
    /// Uniform parameter words (binding 1), a multiple of 4 words.
    pub uniform: Option<Vec<u32>>,
    pub storage: Vec<Binding>,
    pub workgroups: [u32; 3],
}

/// Words `offset..offset + into.len()` of a resource, copied into `into`
/// after the batch completed.
pub struct Read<'a> {
    pub resource: usize,
    pub offset: usize,
    pub into: &'a mut [u32],
}

/// Device work submitted and awaited as one unit.
#[derive(Default)]
pub struct Batch<'a> {
    pub resources: Vec<Resource<'a>>,
    pub passes: Vec<Pass>,
    pub reads: Vec<Read<'a>>,
}

impl<'a> Batch<'a> {
    pub fn upload(&mut self, words: &'a [u32]) -> usize {
        self.resources.push(Resource::Upload(words));
        self.resources.len() - 1
    }
    pub fn scratch(&mut self, words: usize) -> usize {
        self.resources.push(Resource::Scratch(words));
        self.resources.len() - 1
    }
    pub fn persistent(&mut self, key: BufferKey) -> usize {
        self.resources.push(Resource::Persistent(key));
        self.resources.len() - 1
    }
    pub fn read(&mut self, resource: usize, offset: usize, into: &'a mut [u32]) {
        self.reads.push(Read {
            resource,
            offset,
            into,
        });
    }
}

/// What a completed batch cost.
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchStats {
    /// Submission to completion of the device work (s): the device time,
    /// including queueing behind other submissions.
    pub device_seconds: f64,
    pub uploaded_bytes: u64,
    pub readback_bytes: u64,
    pub passes: u32,
}

/// The adapter and the limits the device was created with.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdapterSummary {
    /// Adapter name ("Apple M2 Pro").
    pub name: String,
    pub vendor: String,
    /// The API wgpu runs on ("Metal", "Vulkan", "BrowserWebGpu").
    pub backend: String,
    pub driver: String,
    /// Whether the adapter shares memory with the CPU (an integrated GPU).
    pub integrated: bool,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub min_storage_buffer_offset_alignment: u32,
}

/// A WebGPU device as the backend uses it (module docs).
pub trait Gpu: Send + Sync {
    fn summary(&self) -> &AdapterSummary;
    /// Upload words to a persistent read-only buffer.
    fn persist(&self, words: &[u32]) -> MkResult<BufferKey>;
    /// Free a persistent buffer (later batches must not name it).
    fn release(&self, key: BufferKey);
    /// Create the pipelines of `kernels` ahead of their first use.
    fn prepare(&self, kernels: &[(Kernel, PassShape)]) -> MkResult<()>;
    /// Run `batch` and block until it completed and its reads are written.
    fn run(&self, batch: Batch<'_>) -> MkResult<BatchStats>;
}

/// The binding shape of a pass, which with the kernel identifies a pipeline.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PassShape {
    pub uniform: bool,
    pub storage: Vec<Access>,
}

impl Pass {
    pub fn shape(&self) -> PassShape {
        PassShape {
            uniform: self.uniform.is_some(),
            storage: self.storage.iter().map(|b| b.access).collect(),
        }
    }
}

/// Words of a uniform block from 4-byte values, padded to 16 bytes.
pub fn uniform_words(words: &[u32]) -> Vec<u32> {
    let mut v = words.to_vec();
    v.resize(words.len().div_ceil(4).max(1) * 4, 0);
    v
}
