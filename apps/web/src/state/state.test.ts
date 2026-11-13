import { beforeEach, describe, expect, it } from 'vitest'
import { createMockEngine } from '@/engine-mock'
import { projectFromSystem, SAMPLES } from '@/model/project'
import { clone, curvatureOf } from '@/model/system'
import { useDocument, undoName, redoName, isDirty } from './document'
import { useWorkspace, isStale, LAYOUT } from './workspace'
import { eventShortcut } from './commands'

const dg = () => clone(SAMPLES.find((s) => s.id === 'double_gauss')!.system)

describe('document undo', () => {
  beforeEach(() => useDocument.getState().open(projectFromSystem(dg()), null))

  it('records named edits and restores whole design states', () => {
    const doc = useDocument.getState()
    const before = curvatureOf(doc.project.system.surfaces[1].shape)
    doc.edit('Change Radius', (s) => {
      s.surfaces[1].shape = { ...s.surfaces[1].shape, curvature: 0.01 }
    })
    expect(undoName(useDocument.getState())).toBe('Change Radius')
    expect(isDirty(useDocument.getState())).toBe(true)
    useDocument.getState().undo()
    expect(curvatureOf(useDocument.getState().project.system.surfaces[1].shape)).toBe(before)
    expect(redoName(useDocument.getState())).toBe('Change Radius')
    useDocument.getState().redo()
    expect(curvatureOf(useDocument.getState().project.system.surfaces[1].shape)).toBe(0.01)
  })

  it('does not record an edit that changes nothing', () => {
    const changed = useDocument.getState().edit('Rename', (s) => void (s.name = s.name))
    expect(changed).toBe(false)
    expect(useDocument.getState().past).toHaveLength(0)
  })
})

describe('workspace', () => {
  it('runs panes against snapshots and marks results stale when the design changes', async () => {
    const ws = useWorkspace.getState()
    await ws.setEngine(createMockEngine({ latencyMs: 0 }))
    const sys = dg()
    ws.designChanged(sys)
    await new Promise((r) => setTimeout(r, 400))
    expect(useWorkspace.getState().snapshot).not.toBeNull()
    const key = useWorkspace.getState().addPane('spot_diagram')
    await new Promise((r) => setTimeout(r, 200))
    const run = useWorkspace.getState().runs[key]
    expect(run.status).toBe('done')
    expect(useWorkspace.getState().runs[LAYOUT]?.status).toBe('done')
    const edited = clone(sys)
    edited.surfaces[1].thickness = 3
    useWorkspace.getState().updatePane(key, { auto: false })
    useWorkspace.getState().designChanged(edited)
    await new Promise((r) => setTimeout(r, 400))
    const s = useWorkspace.getState()
    expect(isStale(s, s.runs[key], 'spot_diagram', {})).toBe(true)
  })

  it('reports structural errors as issues', async () => {
    const sys = dg()
    sys.stop = 99
    useWorkspace.getState().designChanged(sys)
    await new Promise((r) => setTimeout(r, 400))
    expect(useWorkspace.getState().modelIssues.length).toBeGreaterThan(0)
  })
})

describe('shortcuts', () => {
  it('reads the physical key with modifiers', () => {
    const e = new KeyboardEvent('keydown', { key: 'Z', code: 'KeyZ', ctrlKey: true, metaKey: true, shiftKey: true })
    const sc = eventShortcut(e)
    expect(sc).toMatch(/shift\+mod\+z$/)
  })
})
