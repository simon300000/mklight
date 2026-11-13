// The contract between the analysis pane and the dedicated result renderers.
// Each renderer group exports a `Record<analysis, Renderer>`; `results/index.tsx`
// merges them, and anything without a renderer (or whose result does not have
// the expected shape) falls back to the generic data view.
import type { JSX } from 'react'
import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import type { System } from '@/model/system'

export interface KeyNumber {
  label: string
  value: string
  help?: string
}

export interface RendererProps {
  analysis: string
  result: AnalysisResult
  system: System
  /** The pane's effective settings (defaults merged with its own). */
  settings: JsonObject
  /** Merges a patch into the pane's own settings (view controls such as the
   *  image simulation's source and field of view); absent in read-only use. */
  onSettingsChange?: (patch: JsonObject) => void
  /** The pane's view-only state (`views[].ui`, never sent to the engine). */
  ui?: JsonObject
  /** Merges a patch into that state; a null value removes its key. */
  onUiChange?: (patch: JsonObject) => void
}

export interface Renderer {
  /** False when the result does not have this renderer's shape (the generic view is used). */
  accepts?: (json: JsonObject) => boolean
  View: (p: RendererProps) => JSX.Element
  keyNumbers?: (result: AnalysisResult, system?: System) => KeyNumber[]
}

export type Renderers = Record<string, Renderer>

export type { Json }
