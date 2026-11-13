// The import report of a `.zmx` / `.seq` file (docs/APP.md "Lens prescription
// files"): warnings grouped as Problems, Glasses, Surfaces, Not Imported and
// Assumed or Approximated.

import { create } from 'zustand'
import type { ImportWarning } from '@/engine/types'

interface ImportReportState {
  fileName: string | null
  warnings: ImportWarning[]
  open: boolean
  /** The imported file, to import it again after Load Glass Catalog…. */
  source: { data: ArrayBuffer; format: 'zmx' | 'seq' } | null
  show(fileName: string, warnings: ImportWarning[], source?: { data: ArrayBuffer; format: 'zmx' | 'seq' }): void
  close(): void
}

export const useImportReport = create<ImportReportState>()((set) => ({
  fileName: null,
  warnings: [],
  open: false,
  source: null,
  show: (fileName, warnings, source) => set({ fileName, warnings, open: true, source: source ?? null }),
  close: () => set({ open: false }),
}))

export const IMPORT_GROUPS = ['Problems', 'Glasses', 'Surfaces', 'Not Imported', 'Assumed or Approximated'] as const

export function importGroup(w: ImportWarning): (typeof IMPORT_GROUPS)[number] {
  switch (w.kind ?? w.category) {
    case 'unknown_glass':
      return 'Glasses'
    case 'unsupported_surface':
    case 'non_sequential':
      return 'Surfaces'
    case 'unsupported_record':
    case 'solve_dropped':
      return 'Not Imported'
    case 'approximated':
    case 'defaulted':
      return 'Assumed or Approximated'
    default:
      return 'Problems'
  }
}
