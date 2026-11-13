// Window state (docs/APP.md "State model"): selection, inspector page,
// visibility of the panels, transient messages. It never enters the design.

import { create } from 'zustand'

export type InspectorPage = 'surface' | 'system' | 'optimize' | 'glass' | 'tolerance' | 'solves'

export interface Toast {
  id: number
  text: string
  tone?: 'info' | 'warning' | 'error'
}

interface ViewState {
  selection: number
  inspectorPage: InspectorPage
  inspectorVisible: boolean
  sidebarVisible: boolean
  lensDataVisible: boolean
  layoutVisible: boolean
  focusedPane: string | null
  welcomeOpen: boolean
  issuesOpen: boolean
  sidebarQuery: string
  /** Lens data height as a share of the content column. */
  lensShare: number
  /** Layout width as a share of the lower row. */
  layoutShare: number
  toasts: Toast[]
  select(i: number): void
  setPage(p: InspectorPage): void
  set(p: Partial<Omit<ViewState, 'set' | 'select' | 'setPage' | 'toast' | 'dismiss'>>): void
  toast(text: string, tone?: Toast['tone']): void
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
  lensShare: 0.36,
  layoutShare: 0.4,
  toasts: [],
  select: (i) => set({ selection: i }),
  setPage: (p) => set({ inspectorPage: p, inspectorVisible: true }),
  set: (p) => set(p),
  toast(text, tone = 'info') {
    const id = ++toastId
    set({ toasts: [...get().toasts, { id, text, tone }] })
    setTimeout(() => get().dismiss(id), tone === 'error' ? 6000 : 3500)
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
}))
