// The GPU worker (docs/GPU.md §11): owns the WebGPU device for the whole
// engine. Started by the engine worker with the module and the shared
// memory like a host worker, it runs `gpu_thread`, an async loop that serves
// the engine's threads through a request channel in the shared memory and
// sleeps in `Atomics.waitAsync`; it never blocks.

/// <reference lib="webworker" />

import init, * as wasm from '../pkg/mklight_wasm.js'

declare const self: DedicatedWorkerGlobalScope

self.onmessage = async (ev: MessageEvent<{ module: WebAssembly.Module; memory: WebAssembly.Memory; stack: number }>) => {
  self.onmessage = null
  const { module, memory, stack } = ev.data
  await init({ module_or_path: module, memory, thread_stack_size: stack })
  self.postMessage('running')
  // Resolves only when the browser has no usable adapter; the engine learns
  // the reason when it opens gpu:webgpu.
  await wasm.gpu_thread()
}
