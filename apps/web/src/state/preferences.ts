// App preferences that outlive a document (the macOS app's user defaults):
// where analyses compute. Stored per browser; reading never fails.

import { create } from 'zustand'
import type { BackendRequest, EngineInfo } from '@/engine/types'

export type BackendPreference = 'auto' | 'cpu_reference' | 'gpu'

export const BACKEND_CHOICES: { value: BackendPreference; title: string; help: string }[] = [
  { value: 'auto', title: 'Automatic', help: 'The GPU where the engine has a path and it pays off, else the CPU.' },
  { value: 'cpu_reference', title: 'CPU reference (f64)', help: 'The double-precision reference path for every analysis.' },
  { value: 'gpu', title: 'GPU', help: 'The GPU (f32) where the engine has a path; others fall back to the CPU.' },
]

const KEY = 'mklight.backend'

function load(): BackendPreference {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'cpu_reference' || v === 'gpu' ? v : 'auto'
  } catch {
    return 'auto'
  }
}

interface PreferencesState {
  backend: BackendPreference
  setBackend(b: BackendPreference): void
}

export const usePreferences = create<PreferencesState>()((set) => ({
  backend: load(),
  setBackend(backend) {
    try {
      localStorage.setItem(KEY, backend)
    } catch {
      // Private windows may refuse storage; the choice still holds for the session.
    }
    set({ backend })
  },
}))

/** The engine's `backend` request for a preference: `gpu` names the first
 *  available GPU backend the engine lists (none: automatic). */
export function backendRequest(pref: BackendPreference, info: EngineInfo | null | undefined): BackendRequest {
  if (pref === 'cpu_reference') return { kind: 'cpu_reference' }
  if (pref === 'gpu') {
    const gpu = (info?.backends ?? [])
      .map((b) => (typeof b === 'string' ? { id: b, available: true } : b))
      .find((b) => b.available !== false && /gpu|webgpu|metal|wgpu/i.test(b.id))
    if (gpu) return { kind: 'gpu', name: gpu.id }
  }
  return { kind: 'auto' }
}
