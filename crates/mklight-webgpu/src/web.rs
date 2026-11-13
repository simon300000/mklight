//! The browser side of the GPU thread (GPU.md §11, "Browser integration").
//!
//! The web build runs the engine on rayon worker threads
//! (`wasm-bindgen-rayon`) and dedicates one more worker to the GPU: the
//! web host starts it and calls [`run_gpu_thread`] there, which opens
//! `navigator.gpu`'s adapter with the design limits, publishes the shared
//! [`Channel`] and serves it forever. Engine threads call [`connect`] (it
//! blocks until the GPU thread has opened the device or reported why it
//! cannot) and give the proxy to `WebGpuBackend::with_gpu`; the engine
//! registers that as `gpu:webgpu` (`GpuProvider::webgpu_with`).
//!
//! The GPU worker must never block: it sleeps in `Atomics.waitAsync` on the
//! channel's doorbell and awaits each batch's `mapAsync`. Engine threads
//! block in `Atomics.wait` (through `std`'s condition variables), which
//! Web Workers allow; the page's main thread must not call into the engine
//! synchronously.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};

use mklight_core::{MkError, MkResult};
use wasm_bindgen::JsCast;

use crate::WgpuDevice;
use crate::device::AdapterSummary;
use crate::proxy::{Channel, GpuProxy};

type Published = Result<(Arc<Channel>, AdapterSummary), String>;

/// What the GPU thread published (statics live in the shared memory).
static PUBLISHED: Mutex<Option<Published>> = Mutex::new(None);
static PUBLISHED_CHANGED: Condvar = Condvar::new();

fn publish(p: Published) {
    *PUBLISHED.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
    PUBLISHED_CHANGED.notify_all();
}

/// The GPU worker's entry point: open the device and serve the engine's
/// threads until the page goes away. Resolves only if the device cannot be
/// opened (the reason is published to [`connect`]) or is lost.
pub async fn run_gpu_thread() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let device = match WgpuDevice::open(instance).await {
        Ok(Some(d)) => d,
        Ok(None) => return publish(Err("this browser offers no WebGPU adapter".into())),
        Err(e) => return publish(Err(format!("the WebGPU device could not be opened: {e}"))),
    };
    let channel = Channel::new();
    publish(Ok((channel.clone(), device.adapter_summary().clone())));
    loop {
        let seen = channel.doorbell().load(Ordering::SeqCst);
        while let Some(r) = channel.pop() {
            r.serve_async(&device).await;
        }
        if channel.doorbell().load(Ordering::SeqCst) == seen {
            wait_async(channel.doorbell().as_ptr(), seen).await;
        }
    }
}

fn shared_view() -> js_sys::Int32Array {
    let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
    js_sys::Int32Array::new(&memory.buffer())
}

/// `Atomics.notify` on an i32 of the shared memory (wakes the GPU thread's
/// `waitAsync`).
pub(crate) fn notify(word: *mut i32) {
    let _ = js_sys::Atomics::notify(&shared_view(), word as u32 / 4);
}

/// `Atomics.waitAsync` on an i32 of the shared memory: resolves once the
/// word no longer holds `value` (or was notified).
async fn wait_async(word: *mut i32, value: i32) {
    let view = shared_view();
    let index = word as u32 / 4;
    let Ok(r) = js_sys::Atomics::wait_async(&view, index, value) else {
        return;
    };
    let is_async = js_sys::Reflect::get(&r, &"async".into())
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if is_async && let Ok(p) = js_sys::Reflect::get(&r, &"value".into()) {
        let _ = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(p)).await;
    }
}

/// The GPU thread's proxy, once it has opened the device (blocks the
/// calling worker up to `timeout`). Never call it on the page's main
/// thread.
pub fn connect(timeout: std::time::Duration) -> MkResult<GpuProxy> {
    let start = crate::clock::Instant::now();
    let mut p = PUBLISHED.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        match &*p {
            Some(Ok((channel, summary))) => {
                return Ok(GpuProxy::new(channel.clone(), summary.clone()));
            }
            Some(Err(reason)) => return Err(MkError::Unsupported(reason.clone())),
            None => {}
        }
        let left = timeout.saturating_sub(start.elapsed());
        if left.is_zero() {
            return Err(MkError::Unsupported(
                "the WebGPU thread did not start (no GPU worker)".into(),
            ));
        }
        p = PUBLISHED_CHANGED
            .wait_timeout(p, left)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}
