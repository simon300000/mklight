//! The GPU thread: one thread owns the WebGPU device; every other thread
//! reaches it through a [`GpuProxy`] (GPU.md §11, "Browser integration").
//!
//! In the browser, wgpu's objects are not `Send` (they are JavaScript
//! objects of one worker) and buffers map asynchronously, while the engine
//! is synchronous and runs on rayon threads (Web Workers sharing one
//! WebAssembly memory). So:
//!
//! * the GPU worker opens the device and runs an event loop
//!   ([`crate::web::run_gpu_thread`]): it pops requests from the shared
//!   [`Channel`], awaits each batch (`mapAsync` included) and sleeps in
//!   `Atomics.waitAsync` on the channel's doorbell word while idle;
//! * an engine thread calling [`Gpu::run`] on the proxy pushes a request
//!   that points at its batch (uploads and readback destinations stay where
//!   they are: the memory is shared), rings the doorbell
//!   (`Atomics.notify`) and blocks on the request's completion
//!   (`Atomics.wait` under `std`'s condition variable) until the GPU thread
//!   has written the readbacks.
//!
//! Natively the same protocol runs over a plain thread ([`spawn_native`]),
//! which the tests use to check it against direct calls.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use mklight_core::{MkError, MkResult};

use crate::device::*;

/// A request's command. Pointers refer to the caller's stack and heap, valid
/// until the caller is released (it blocks for the whole call).
enum Command {
    Run(*mut Option<Batch<'static>>),
    Persist(*const [u32]),
    Allocate(usize),
    Release(BufferKey),
    Prepare(*const [(Kernel, PassShape)]),
}

enum Outcome {
    Stats(MkResult<BatchStats>),
    Key(MkResult<BufferKey>),
    Unit(MkResult<()>),
}

#[derive(Default)]
struct Reply {
    slot: Mutex<Option<Outcome>>,
    done: Condvar,
}

pub struct Request {
    command: Command,
    reply: Arc<Reply>,
}

// SAFETY: the pointers in `command` are only dereferenced by the GPU thread
// while the requesting thread blocks in `GpuProxy::call` (which keeps the
// pointees alive and unaliased) and before the reply is published.
unsafe impl Send for Request {}

impl Request {
    fn finish(self, outcome: Outcome) {
        *self.reply.slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome);
        self.reply.done.notify_all();
    }

    /// Execute on a device the calling thread may block on (native).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn serve_blocking(self, device: &crate::WgpuDevice) {
        // SAFETY: see `unsafe impl Send for Request`.
        let outcome = unsafe {
            match self.command {
                Command::Run(b) => Outcome::Stats(Gpu::run(device, (*b).take().unwrap())),
                Command::Persist(w) => Outcome::Key(device.persist_words(&*w)),
                Command::Allocate(n) => Outcome::Key(device.allocate_words(n)),
                Command::Release(k) => {
                    device.release_key(k);
                    Outcome::Unit(Ok(()))
                }
                Command::Prepare(k) => Outcome::Unit(Gpu::prepare(device, &*k)),
            }
        };
        self.finish(outcome);
    }

    /// Execute on the GPU thread's event loop (browser).
    pub async fn serve_async(self, device: &crate::WgpuDevice) {
        // SAFETY: see `unsafe impl Send for Request`.
        let outcome = unsafe {
            match self.command {
                Command::Run(b) => {
                    let batch = (*b).take().unwrap();
                    Outcome::Stats(device.run_async(batch).await)
                }
                Command::Persist(w) => Outcome::Key(device.persist_words(&*w)),
                Command::Allocate(n) => Outcome::Key(device.allocate_words(n)),
                Command::Release(k) => {
                    device.release_key(k);
                    Outcome::Unit(Ok(()))
                }
                Command::Prepare(k) => Outcome::Unit(device.prepare_async(&*k).await),
            }
        };
        self.finish(outcome);
    }
}

/// The request queue between engine threads and the GPU thread.
#[derive(Default)]
pub struct Channel {
    queue: Mutex<VecDeque<Request>>,
    /// Bumped and notified for every request: the word the GPU thread waits
    /// on (`Atomics.waitAsync` in the browser).
    doorbell: AtomicI32,
    /// Wakes a native server thread.
    pending: Condvar,
    closed: std::sync::atomic::AtomicBool,
}

impl Channel {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn push(&self, r: Request) -> MkResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(MkError::Backend(
                "webgpu: the GPU thread has stopped".into(),
            ));
        }
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(r);
        self.doorbell.fetch_add(1, Ordering::SeqCst);
        #[cfg(target_arch = "wasm32")]
        crate::web::notify(self.doorbell.as_ptr());
        self.pending.notify_all();
        Ok(())
    }

    /// The next request, if any.
    pub fn pop(&self) -> Option<Request> {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
    }

    /// The doorbell word (its address is what the GPU thread waits on).
    pub fn doorbell(&self) -> &AtomicI32 {
        &self.doorbell
    }

    /// Stop accepting requests and fail the queued ones.
    pub fn close(&self, reason: &str) {
        self.closed.store(true, Ordering::Release);
        while let Some(r) = self.pop() {
            let e = || MkError::Backend(format!("webgpu: {reason}"));
            let o = match r.command {
                Command::Run(_) => Outcome::Stats(Err(e())),
                Command::Persist(_) | Command::Allocate(_) => Outcome::Key(Err(e())),
                _ => Outcome::Unit(Err(e())),
            };
            r.finish(o);
        }
        self.pending.notify_all();
    }

    /// Block until a request is queued or the channel is closed (native).
    #[cfg(not(target_arch = "wasm32"))]
    fn wait_native(&self) -> Option<Request> {
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(r) = q.pop_front() {
                return Some(r);
            }
            if self.closed.load(Ordering::Acquire) {
                return None;
            }
            q = self.pending.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// [`Gpu`] for threads that do not own the device.
pub struct GpuProxy {
    channel: Arc<Channel>,
    summary: AdapterSummary,
}

impl GpuProxy {
    pub fn new(channel: Arc<Channel>, summary: AdapterSummary) -> Self {
        Self { channel, summary }
    }

    fn call(&self, command: Command) -> MkResult<Outcome> {
        let reply = Arc::new(Reply::default());
        self.channel.push(Request {
            command,
            reply: reply.clone(),
        })?;
        let mut slot = reply.slot.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(o) = slot.take() {
                return Ok(o);
            }
            slot = reply.done.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
    }
}

fn protocol() -> MkError {
    MkError::Backend("webgpu: GPU thread protocol".into())
}

impl Gpu for GpuProxy {
    fn summary(&self) -> &AdapterSummary {
        &self.summary
    }
    fn persist(&self, words: &[u32]) -> MkResult<BufferKey> {
        match self.call(Command::Persist(words as *const [u32]))? {
            Outcome::Key(k) => k,
            _ => Err(protocol()),
        }
    }
    fn release(&self, key: BufferKey) {
        let _ = self.call(Command::Release(key));
    }
    fn allocate(&self, words: usize) -> MkResult<BufferKey> {
        match self.call(Command::Allocate(words))? {
            Outcome::Key(k) => k,
            _ => Err(protocol()),
        }
    }
    fn prepare(&self, kernels: &[(Kernel, PassShape)]) -> MkResult<()> {
        match self.call(Command::Prepare(kernels as *const _))? {
            Outcome::Unit(r) => r,
            _ => Err(protocol()),
        }
    }
    fn run(&self, batch: Batch<'_>) -> MkResult<BatchStats> {
        let mut slot = Some(batch);
        // SAFETY: lifetime erasure only; this thread blocks in `call` until
        // the GPU thread has taken and dropped the batch.
        let ptr = (&mut slot as *mut Option<Batch<'_>>).cast::<Option<Batch<'static>>>();
        match self.call(Command::Run(ptr))? {
            Outcome::Stats(s) => s,
            _ => Err(protocol()),
        }
    }
}

/// Serve `channel` from a new native thread that owns `device` (the
/// browser protocol without a browser; tests and diagnostics). Dropping the
/// returned proxy does not stop the thread; `Channel::close` does.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_native(device: crate::WgpuDevice) -> MkResult<(GpuProxy, Arc<Channel>)> {
    let channel = Channel::new();
    let summary = device.summary().clone();
    let ch = channel.clone();
    std::thread::Builder::new()
        .name("mklight-webgpu-device".into())
        .spawn(move || {
            while let Some(r) = ch.wait_native() {
                r.serve_blocking(&device);
            }
        })
        .map_err(|e| MkError::Backend(format!("webgpu: GPU thread: {e}")))?;
    Ok((GpuProxy::new(channel.clone(), summary), channel))
}
