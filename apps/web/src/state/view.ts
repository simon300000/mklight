// Window state (docs/APP.md "State model"): selection, inspector page,
// visibility of the panels, the layout's view options, transient messages.
// It never enters the design; its layout part is saved in `views`.

import { create } from 'zustand'
import type { LayoutMode } from '@/views/layout/camera'

export type InspectorPage = 'surface' | 'system' | 'optimize' | 'glass' | 'tolerance' | 'solves'
export type RayColoring = 'field' | 'wavelength'

export interface Toast {
  id: number
  text: string
  tone?: 'info' | 'warning' | 'error'
  /** A button in the toast (it runs on a user action, so it may open windows). */
  action?: { label: string; run: () => void }
}

/** An autosaved document with unsaved changes, offered in the welcome window. */
export interface Recovery {
  name: string
  time: number
}

interface ViewState {
  selection: number
  inspectorPage: InspectorPage
  inspectorVisible: boolean
  sidebarVisible: boolean
  /** Whether the lens data and the layout are open in one of the design's
   *  windows; they follow the panel arrangement, and setting them shows or
   *  hides the panel. */
  lensDataVisible: boolean
  layoutVisible: boolean
  focusedPane: string | null
  welcomeOpen: boolean
  issuesOpen: boolean
  sidebarQuery: string
  /** The layout's view: Y–Z or X–Z section, or 3-D (saved as `ui.mode`). */
  layoutMode: LayoutMode
  rayColoring: RayColoring
  /** Vertical stretch of the section views (1 = true proportions). */
  stretch: number
  /** The stretch was chosen (in the menu, or saved with the project): the
   *  first fit keeps it instead of choosing one for a slim design. */
  stretchChosen: boolean
  /** Counts document openings: the first fit after one may pick a stretch
   *  and frames a slim design on its lens. */
  layoutEpoch: number
  /** Beam footprints in the 3-D view (runs `footprint`). */
  showFootprints: boolean
  /** Incremented to ask the layout to fit; `fitLens` frames the lens only. */
  fitRequest: number
  fitLens: boolean
  recovery: Recovery | null
  toasts: Toast[]
  select(i: number): void
  setPage(p: InspectorPage): void
  set(p: Partial<Omit<ViewState, 'set' | 'select' | 'setPage' | 'toast' | 'dismiss' | 'fit'>>): void
  /** Fits the layout (Fit, ⌘0); `lens`: the lens only. */
  fit(lens?: boolean): void
  toast(text: string, tone?: Toast['tone'], action?: Toast['action']): void
  dismiss(id: number): void
}

let toastId = 0

export const useView = create<ViewState>()((set, get) => ({
  selection: 1,
  inspectorPage: 'surface',
  inspectorVisible: true,
  sidebarVisible: true,
  lensDataVisible: true,
  layoutVisible: true,
  focusedPane: null,
  welcomeOpen: false,
  issuesOpen: false,
  sidebarQuery: '',
  layoutMode: 'yz',
  rayColoring: 'field',
  stretch: 1,
  stretchChosen: false,
  layoutEpoch: 0,
  showFootprints: false,
  fitRequest: 0,
  fitLens: false,
  recovery: null,
  toasts: [],
  select: (i) => set({ selection: i }),
  setPage: (p) => set({ inspectorPage: p, inspectorVisible: true }),
  set: (p) => set(p),
  fit: (lens = false) => set({ fitLens: lens, fitRequest: get().fitRequest + 1 }),
  toast(text, tone = 'info', action) {
    const id = ++toastId
    set({ toasts: [...get().toasts, { id, text, tone, action }] })
    setTimeout(() => get().dismiss(id), action ? 12000 : tone === 'error' ? 6000 : 3500)
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
}))
