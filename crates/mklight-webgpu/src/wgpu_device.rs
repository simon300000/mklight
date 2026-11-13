//! [`Gpu`] on a wgpu device. The same code records batches natively (Metal,
//! Vulkan, D3D12) and on the browser's WebGPU; only waiting for completion
//! differs: natively [`WgpuDevice::run`] polls the device until the batch's
//! readbacks are mapped, on the web the GPU thread awaits the mapping
//! ([`WgpuDevice::run_async`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mklight_core::{MkError, MkResult};

use crate::device::*;
use crate::kernels;

/// Limits the backend is designed for: the WebGPU defaults every browser
/// grants without asking (`maxStorageBufferBindingSize` 128 MiB,
/// `maxBufferSize` 256 MiB, 8 storage buffers per stage, 256 invocations
/// per workgroup, 16 KiB of workgroup storage), so a native run exercises the
/// chunking a browser needs.
pub fn design_limits() -> wgpu::Limits {
    wgpu::Limits::default()
}

pub struct WgpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    summary: AdapterSummary,
    env: wgpu::Buffer,
    modules: Mutex<HashMap<&'static str, Arc<wgpu::ShaderModule>>>,
    pipelines: Mutex<HashMap<(Kernel, PassShape), Arc<Pipeline>>>,
    persistent: Mutex<HashMap<u64, Arc<wgpu::Buffer>>>,
    next_key: AtomicU64,
}

struct Pipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// Recorded and submitted work whose readbacks are being mapped.
pub struct Submitted<'a> {
    staging: Vec<(wgpu::Buffer, &'a mut [u32])>,
    stats: BatchStats,
    started: std::time::Instant,
}

fn backend_error(what: &str, e: impl std::fmt::Display) -> MkError {
    MkError::Backend(format!("webgpu {what}: {e}"))
}

impl WgpuDevice {
    /// Open the default high-performance adapter natively with the design
    /// limits. `Ok(None)` when there is no adapter.
    pub fn open_native() -> MkResult<Option<Self>> {
        pollster::block_on(Self::open(wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        )))
    }

    /// Open the default adapter of `instance` (the browser's `navigator.gpu`
    /// on the web).
    pub async fn open(instance: wgpu::Instance) -> MkResult<Option<Self>> {
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
        {
            Ok(a) => a,
            Err(_) => return Ok(None),
        };
        let info = adapter.get_info();
        let limits = design_limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mklight"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| backend_error("device", e))?;
        let summary = AdapterSummary {
            name: info.name.clone(),
            vendor: format!("{:#x}", info.vendor),
            backend: format!("{:?}", info.backend),
            driver: format!("{} {}", info.driver, info.driver_info)
                .trim()
                .to_string(),
            integrated: info.device_type == wgpu::DeviceType::IntegratedGpu,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
        };
        let env = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mkl_env"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-initialised: mkl_env.x is the opaque zero.
        Ok(Some(Self {
            device,
            queue,
            summary,
            env,
            modules: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            persistent: Mutex::new(HashMap::new()),
            next_key: AtomicU64::new(1),
        }))
    }

    fn module(&self, name: &'static str) -> MkResult<Arc<wgpu::ShaderModule>> {
        if let Some(m) = self.modules.lock().unwrap().get(name) {
            return Ok(m.clone());
        }
        let src = kernels::source(name)
            .ok_or_else(|| MkError::Backend(format!("webgpu: no kernel module {name}")))?;
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(backend_error(&format!("module {name}"), e));
        }
        let m = Arc::new(m);
        self.modules.lock().unwrap().insert(name, m.clone());
        Ok(m)
    }

    fn pipeline(&self, kernel: Kernel, shape: &PassShape) -> MkResult<Arc<Pipeline>> {
        let key = (kernel, shape.clone());
        if let Some(p) = self.pipelines.lock().unwrap().get(&key) {
            return Ok(p.clone());
        }
        let module = self.module(kernel.module)?;
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        if shape.uniform {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                ..entries[0]
            });
        }
        for (i, a) in shape.storage.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: *a == Access::Read,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(kernel.entry),
                entries: &entries,
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(kernel.entry),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(kernel.entry),
                // Every kernel writes its workgroup memory before reading
                // it. naga's zero-initialisation is one invocation storing
                // the whole array per group (a 2048² FFT ran 100× slower);
                // browsers initialise regardless.
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(backend_error(
                &format!("pipeline {}::{}", kernel.module, kernel.entry),
                e,
            ));
        }
        let p = Arc::new(Pipeline { layout, pipeline });
        self.pipelines.lock().unwrap().insert(key, p.clone());
        Ok(p)
    }

    fn buffer(
        &self,
        words: usize,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> MkResult<wgpu::Buffer> {
        let size = (words.max(4) * 4) as u64;
        if size > self.summary.max_buffer_size {
            return Err(MkError::Unsupported(format!(
                "webgpu: {label} buffer of {size} bytes exceeds maxBufferSize"
            )));
        }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        }))
    }

    /// Record and submit `batch`; its readbacks complete after mapping.
    pub fn submit<'a>(&self, batch: Batch<'a>) -> MkResult<Submitted<'a>> {
        let started = std::time::Instant::now();
        let mut stats = BatchStats::default();
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut buffers: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(batch.resources.len());
        let mut sizes = Vec::with_capacity(batch.resources.len());
        for r in &batch.resources {
            let (b, words) = match r {
                Resource::Persistent(k) => {
                    let b = self
                        .persistent
                        .lock()
                        .unwrap()
                        .get(&k.0)
                        .cloned()
                        .ok_or_else(|| MkError::Backend("webgpu: released buffer".into()))?;
                    let words = (b.size() / 4) as usize;
                    (b, words)
                }
                Resource::Upload(w) => {
                    let b = self.buffer(w.len(), storage, "upload")?;
                    if !w.is_empty() {
                        self.queue.write_buffer(&b, 0, bytemuck::cast_slice(w));
                    }
                    stats.uploaded_bytes += (w.len() * 4) as u64;
                    (Arc::new(b), w.len().max(4))
                }
                Resource::Scratch(n) => {
                    (Arc::new(self.buffer(*n, storage, "scratch")?), (*n).max(4))
                }
            };
            buffers.push(b);
            sizes.push(words);
        }
        // Every pass's uniform block at a 256-byte aligned offset of one buffer.
        let align = 256usize;
        let mut uniform_bytes = Vec::<u8>::new();
        let mut offsets = Vec::with_capacity(batch.passes.len());
        for p in &batch.passes {
            if let Some(u) = &p.uniform {
                let off = uniform_bytes.len().div_ceil(align) * align;
                uniform_bytes.resize(off, 0);
                uniform_bytes.extend_from_slice(bytemuck::cast_slice(u));
                offsets.push(Some((off as u64, (u.len() * 4) as u64)));
            } else {
                offsets.push(None);
            }
        }
        let uniforms = if uniform_bytes.is_empty() {
            None
        } else {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uniforms"),
                size: uniform_bytes.len().div_ceil(16) as u64 * 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            uniform_bytes.resize(uniform_bytes.len().div_ceil(16) * 16, 0);
            self.queue.write_buffer(&b, 0, &uniform_bytes);
            Some(b)
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (p, off) in batch.passes.iter().zip(&offsets) {
            let pipe = self.pipeline(p.kernel, &p.shape())?;
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: self.env.as_entire_binding(),
            }];
            if let Some((o, n)) = off {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniforms.as_ref().unwrap(),
                        offset: *o,
                        size: wgpu::BufferSize::new(*n),
                    }),
                });
            }
            for (i, b) in p.storage.iter().enumerate() {
                let (offset, len) = b.range.unwrap_or((0, sizes[b.resource]));
                let bytes = (len.max(1) * 4) as u64;
                if bytes > self.summary.max_storage_buffer_binding_size {
                    return Err(MkError::Unsupported(format!(
                        "webgpu: {}::{} binding of {bytes} bytes exceeds maxStorageBufferBindingSize",
                        p.kernel.module, p.kernel.entry
                    )));
                }
                entries.push(wgpu::BindGroupEntry {
                    binding: 2 + i as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers[b.resource],
                        offset: (offset * 4) as u64,
                        size: wgpu::BufferSize::new(bytes),
                    }),
                });
            }
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(p.kernel.entry),
                layout: &pipe.layout,
                entries: &entries,
            });
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(p.kernel.entry),
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipe.pipeline);
            cp.set_bind_group(0, &group, &[]);
            let [x, y, z] = p.workgroups;
            cp.dispatch_workgroups(x, y, z);
            stats.passes += 1;
        }
        let mut staging = Vec::with_capacity(batch.reads.len());
        for r in batch.reads {
            if r.into.is_empty() {
                continue;
            }
            let bytes = (r.into.len() * 4) as u64;
            let s = self.buffer(
                r.into.len(),
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                "readback",
            )?;
            encoder.copy_buffer_to_buffer(
                &buffers[r.resource],
                (r.offset * 4) as u64,
                &s,
                0,
                bytes,
            );
            stats.readback_bytes += bytes;
            staging.push((s, r.into));
        }
        self.queue.submit([encoder.finish()]);
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(backend_error("batch", e));
        }
        for (s, _) in &staging {
            s.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        Ok(Submitted {
            staging,
            stats,
            started,
        })
    }

    /// Copy the mapped readbacks out (after the device has completed).
    pub fn finish(&self, s: Submitted<'_>) -> MkResult<BatchStats> {
        let mut stats = s.stats;
        for (b, into) in s.staging {
            {
                let range = b
                    .slice(..)
                    .get_mapped_range()
                    .map_err(|e| backend_error("readback", e))?;
                let n = into.len();
                into.copy_from_slice(&bytemuck::cast_slice::<u8, u32>(&range)[..n]);
            }
            b.unmap();
        }
        stats.device_seconds = s.started.elapsed().as_secs_f64();
        Ok(stats)
    }

    /// Block (natively) until every submission completed.
    fn wait(&self) -> MkResult<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map(|_| ())
            .map_err(|e| backend_error("poll", e))
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
}

impl Gpu for WgpuDevice {
    fn summary(&self) -> &AdapterSummary {
        &self.summary
    }
    fn persist(&self, words: &[u32]) -> MkResult<BufferKey> {
        let b = self.buffer(
            words.len(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "persistent",
        )?;
        self.queue.write_buffer(&b, 0, bytemuck::cast_slice(words));
        let k = self.next_key.fetch_add(1, Ordering::Relaxed);
        self.persistent.lock().unwrap().insert(k, Arc::new(b));
        Ok(BufferKey(k))
    }
    fn release(&self, key: BufferKey) {
        self.persistent.lock().unwrap().remove(&key.0);
    }
    fn prepare(&self, kernels: &[(Kernel, PassShape)]) -> MkResult<()> {
        for (k, s) in kernels {
            self.pipeline(*k, s)?;
        }
        Ok(())
    }
    fn run(&self, batch: Batch<'_>) -> MkResult<BatchStats> {
        let s = self.submit(batch)?;
        self.wait()?;
        self.finish(s)
    }
}
