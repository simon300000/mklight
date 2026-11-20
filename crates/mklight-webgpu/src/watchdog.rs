//! The pipeline-creation watchdog.
//!
//! A browser compiles a WGSL pipeline into the platform's shader language
//! and hands that to the driver's compiler, which can take far longer than
//! the WGSL suggests: on some drivers the trace kernel's compilation never
//! finishes. Work submitted with such a pipeline would wait forever, and
//! with it every later request to the device (the GPU thread serves them in
//! order).
//!
//! So [`crate::WgpuDevice`] makes each pipeline ready before any batch uses
//! it: in the browser it asks for the pipeline with
//! `createComputePipelineAsync` and races that against a timer
//! ([`Watchdog::timeout`]). A pipeline that is not ready in time marks the
//! device as failed ([`Health`]): every later request fails at once with the
//! reason, and the engine treats the backend as unavailable
//! (`ComputeBackend::unavailable`), so `auto` routes to the CPU and an
//! explicit request is refused. No work is ever submitted with a pipeline
//! that did not compile.
//!
//! Natively wgpu creates pipelines synchronously and the drivers finish, so
//! the race resolves at once and creation is as before; the race still runs
//! there so that tests can simulate a compilation that never finishes
//! ([`Watchdog::stalled`]).

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};
use std::time::Duration;

/// How long a pipeline may take to become ready.
///
/// 45 s: where compilation works, every pipeline (the trace kernel, the
/// largest, included) is ready within a few seconds even with cold caches,
/// so this leaves an order of magnitude for slow machines; and an analysis
/// that meets a compiler that never finishes ends within the minute with
/// the reason instead of appearing hung. A pipeline that does finish later
/// than this is lost to WebGPU until the failure expires (the web host
/// remembers it), which is the cheaper mistake. Each pipeline gets its own
/// budget, but after the first failure the device fails every request at
/// once, so a run waits at most this long.
pub const PIPELINE_TIMEOUT: Duration = Duration::from_secs(45);

/// The watchdog's settings for one device.
#[derive(Clone, Debug, PartialEq)]
pub struct Watchdog {
    /// How long a pipeline may take to become ready.
    pub timeout: Duration,
    /// Testing only: kernel modules (`"trace"`, …) whose pipelines never
    /// become ready, as on a driver whose compiler does not finish.
    pub stalled: Vec<String>,
}

impl Default for Watchdog {
    fn default() -> Self {
        Self {
            timeout: PIPELINE_TIMEOUT,
            stalled: Vec::new(),
        }
    }
}

impl Watchdog {
    pub(crate) fn stalls(&self, module: &str) -> bool {
        self.stalled.iter().any(|m| m == module)
    }
}

/// Whether a device can still run work: shared by the GPU thread, which
/// records a pipeline failure, and the engine threads, which read it.
#[derive(Debug, Default)]
pub struct Health(Mutex<Option<String>>);

impl Health {
    /// Why the device can no longer run work, once known.
    pub fn failure(&self) -> Option<String> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Record why the device can no longer run work (the first reason
    /// stays).
    pub fn fail(&self, reason: impl Into<String>) {
        let mut f = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if f.is_none() {
            *f = Some(reason.into());
        }
    }
}

/// A pipeline's creation as a future: `Ok` once the pipeline is ready,
/// `Err` with the browser's message when its creation failed.
pub(crate) type Probe<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + 'a>>;

/// A probe that never completes (a simulated stalled compilation).
pub(crate) fn stalled<'a>() -> Probe<'a> {
    Box::pin(std::future::pending())
}

/// `probe`'s outcome, or `None` when it did not complete within `timeout`.
/// The timer starts only if the probe is not complete when first polled.
pub(crate) async fn race(probe: Probe<'_>, timeout: Duration) -> Option<Result<(), String>> {
    Race {
        probe,
        timeout,
        timer: None,
    }
    .await
}

struct Race<'a> {
    probe: Probe<'a>,
    timeout: Duration,
    timer: Option<Timer>,
}

impl Future for Race<'_> {
    type Output = Option<Result<(), String>>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Poll::Ready(r) = self.probe.as_mut().poll(cx) {
            return Poll::Ready(Some(r));
        }
        let timeout = self.timeout;
        let timer = self.timer.get_or_insert_with(|| Timer::new(timeout));
        match Pin::new(timer).poll(cx) {
            Poll::Ready(()) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A one-shot timer: a sleeping thread natively.
#[cfg(not(target_arch = "wasm32"))]
struct Timer(std::sync::Arc<Mutex<(bool, Option<std::task::Waker>)>>);

#[cfg(not(target_arch = "wasm32"))]
impl Timer {
    fn new(d: Duration) -> Self {
        let state = std::sync::Arc::new(Mutex::new((false, None::<std::task::Waker>)));
        let s = state.clone();
        let spawned = std::thread::Builder::new()
            .name("mklight-webgpu-watchdog".into())
            .spawn(move || {
                std::thread::sleep(d);
                let mut g = s.lock().unwrap_or_else(|e| e.into_inner());
                g.0 = true;
                if let Some(w) = g.1.take() {
                    w.wake();
                }
            });
        if spawned.is_err() {
            // No thread for the timer: expire at once rather than never.
            state.lock().unwrap_or_else(|e| e.into_inner()).0 = true;
        }
        Self(state)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Future for Timer {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut g = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if g.0 {
            Poll::Ready(())
        } else {
            g.1 = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// A one-shot timer: `setTimeout` in the browser.
#[cfg(target_arch = "wasm32")]
struct Timer(wasm_bindgen_futures::JsFuture);

#[cfg(target_arch = "wasm32")]
impl Timer {
    fn new(d: Duration) -> Self {
        let ms = d.as_millis().min(i32::MAX as u128) as f64;
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let global = js_sys::global();
            let set_timeout = js_sys::Reflect::get(&global, &"setTimeout".into())
                .ok()
                .and_then(|f| wasm_bindgen::JsCast::dyn_into::<js_sys::Function>(f).ok());
            match set_timeout {
                Some(f) => {
                    let _ = f.call2(&global, &resolve, &ms.into());
                }
                // No timers: expire at once rather than never.
                None => {
                    let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
                }
            }
        });
        Self(wasm_bindgen_futures::JsFuture::from(promise))
    }
}

#[cfg(target_arch = "wasm32")]
impl Future for Timer {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        Pin::new(&mut self.0).poll(cx).map(|_| ())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn a_ready_probe_wins_without_a_timer() {
        let r = pollster::block_on(race(Box::pin(async { Ok(()) }), Duration::ZERO));
        assert_eq!(r, Some(Ok(())));
        let r = pollster::block_on(race(Box::pin(async { Err("bad".into()) }), Duration::ZERO));
        assert_eq!(r, Some(Err("bad".into())));
    }

    #[test]
    fn a_stalled_probe_times_out() {
        let t = std::time::Instant::now();
        let r = pollster::block_on(race(stalled(), Duration::from_millis(50)));
        assert_eq!(r, None);
        assert!(t.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn health_keeps_the_first_failure() {
        let h = Health::default();
        assert_eq!(h.failure(), None);
        h.fail("first");
        h.fail("second");
        assert_eq!(h.failure().as_deref(), Some("first"));
    }
}
