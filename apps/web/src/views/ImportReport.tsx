import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { IMPORT_GROUPS, importGroup, useImportReport } from '@/state/importReport'
import { loadGlassCatalog } from '@/state/commands'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { useWorkspace } from '@/state/workspace'
import type { System } from '@/model/system'

/** Load Glass Catalog…, then import the lens file again with it (one undoable step). */
async function loadCatalogAndImportAgain() {
  const r = useImportReport.getState()
  const name = await loadGlassCatalog()
  if (!name || !r.source) return
  const engine = useWorkspace.getState().engine
  if (!engine) return
  try {
    const imported = await engine.importPrescription(r.source.data.slice(0), r.source.format)
    const system = imported.system as System
    if (!system.name) system.name = (r.fileName ?? '').replace(/\.(zmx|seq)$/i, '')
    const doc = useDocument.getState()
    doc.replace({ ...doc.project, system }, 'Import Lens File Again')
    if (imported.warnings.length) r.show(r.fileName ?? '', imported.warnings, r.source)
    else {
      r.close()
      useView.getState().toast(`Imported ${r.fileName} again: everything mapped.`)
    }
  } catch (e) {
    useView.getState().toast(`Import: ${e instanceof Error ? e.message : String(e)}`, 'error')
  }
}

/** The import report sheet (`ImportReportSheet`): the warnings grouped, each
 *  with its surface and line, and Copy Report. A clean import shows a note. */
export function ImportReport() {
  const r = useImportReport()
  const groups = IMPORT_GROUPS.map((g) => ({ g, items: r.warnings.filter((w) => importGroup(w) === g) })).filter((x) => x.items.length)
  const text = groups.map(({ g, items }) => [g, ...items.map((w) => `  ${w.surface != null ? `S${w.surface} ` : ''}${w.line != null ? `line ${w.line}: ` : ''}${w.message}`)].join('\n')).join('\n\n')
  return (
    <Dialog open={r.open} onOpenChange={(o) => !o && r.close()}>
      <DialogContent className="w-[min(560px,92vw)] p-5">
        <DialogTitle className="text-[15px] font-semibold">Imported {r.fileName}</DialogTitle>
        <DialogDescription className="mb-3 text-[12px] text-secondary">
          {r.warnings.length === 0
            ? 'Everything in the file was imported. Save it as an MKLight project to keep it.'
            : 'Some of the file did not map exactly. Saving asks for a new .mklight file; the original is never written.'}
        </DialogDescription>
        <div className="max-h-[50vh] space-y-3 overflow-y-auto">
          {groups.map(({ g, items }) => (
            <section key={g}>
              <h4 className="mb-1 text-[12px] font-semibold">{g}</h4>
              <ul className="space-y-1 text-[12px]">
                {items.map((w, k) => (
                  <li key={k} className="select-text">
                    {w.surface != null && <span className="mr-1 text-secondary">Surface {w.surface}</span>}
                    {w.line != null && <span className="mr-1 text-secondary">line {w.line}</span>}
                    {w.message}
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
        <div className="mt-4 flex justify-end gap-2">
          {r.warnings.length > 0 && <Button onClick={() => void navigator.clipboard?.writeText(text)}>Copy Report</Button>}
          {r.source && r.warnings.some((w) => importGroup(w) === 'Glasses') && (
            <Button onClick={() => void loadCatalogAndImportAgain()} title="Loads a glass catalog and imports the file again with it">
              Load Glass Catalog…
            </Button>
          )}
          <Button variant="prominent" onClick={r.close}>
            Done
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
