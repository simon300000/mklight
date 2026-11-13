// A host worker for the engine's own threads (scheduler workers, progress
// and event threads; mklight_core::platform). The engine worker starts it
// with the module and the shared memory; it then runs engine threads
// forever (`thread_host` never returns).

/// <reference lib="webworker" />

import init, * as wasm from '../pkg/mklight_wasm.js'

declare const self: DedicatedWorkerGlobalScope

self.onmessage = async (ev: MessageEvent<{ module: WebAssembly.Module; memory: WebAssembly.Memory; stack: number }>) => {
  self.onmessage = null
  const { module, memory, stack } = ev.data
  await init({ module_or_path: module, memory, thread_stack_size: stack })
  self.postMessage('running')
  wasm.thread_host()
}
