import { create } from 'zustand'

/** Whether the command palette (⌘K) is open. */
export const usePalette = create<{ open: boolean; set(open: boolean): void }>()((set) => ({
  open: false,
  set: (open) => set({ open }),
}))
