import { FolderOpen, Plus } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { isSurfaceTypeSample, SAMPLES, type Sample } from '@/model/project'
import { newDocument, openFromPicker, openSample } from '@/state/files'
import { useView } from '@/state/view'
import { useWorkspace } from '@/state/workspace'
import { cssVar, seriesColor, useDark } from '@/lib/theme'
import { readScene, type Scene } from './LayoutView'

const thumbCache = new Map<string, Promise<Scene | null>>()

/** The sample's layout, computed by the engine on a snapshot of its own (`SampleThumbnails`). */
function sampleScene(sample: Sample): Promise<Scene | null> {
  let p = thumbCache.get(sample.id)
  if (!p) {
    const engine = useWorkspace.getState().engine
    if (!engine) return Promise.resolve(null)
    p = engine
      .openSnapshot(sample.system)
      .then((snap) => engine.submit({ analysis: 'layout', settings: { rays_per_field: 3, include_3d: false } }, snap).result)
      .then(readScene)
      .catch(() => null)
    thumbCache.set(sample.id, p)
  }
  return p
}

function Thumbnail({ sample }: { sample: Sample }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  const engine = useWorkspace((s) => s.engine)
  const [scene, setScene] = useState<Scene | null>(null)
  useEffect(() => {
    let live = true
    if (engine) void sampleScene(sample).then((sc) => live && setScene(sc))
    return () => {
      live = false
    }
  }, [sample, engine])
  useEffect(() => {
    const c = ref.current
    const g = c?.getContext('2d')
    if (!c || !g || !scene) return
    const w = 200, h = 96, dpr = devicePixelRatio || 1
    c.width = w * dpr
    c.height = h * dpr
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, h)
    const { xyz, offsets, kinds, flags, fields } = scene
    // Slim, long designs are framed on their lens.
    let z0 = Infinity, z1 = -Infinity, hm = 0
    for (let p = 0; p < kinds.length; p++) {
      if (!(flags[p] & 1) || (kinds[p] !== 0 && kinds[p] !== 7)) continue
      for (let i = offsets[p]; i < offsets[p + 1]; i++) {
        z0 = Math.min(z0, xyz[3 * i + 2]); z1 = Math.max(z1, xyz[3 * i + 2]); hm = Math.max(hm, Math.abs(xyz[3 * i + 1]))
      }
    }
    if (!Number.isFinite(z0)) return
    const scale = Math.min((w - 24) / Math.max(z1 - z0, 1e-6), (h - 12) / Math.max(2 * hm, 1e-6))
    const X = (z: number) => w / 2 + (z - (z0 + z1) / 2) * scale
    const Y = (y: number) => h / 2 - y * scale
    const path = (p: number) => {
      g.beginPath()
      for (let i = offsets[p]; i < offsets[p + 1]; i++) {
        const x = X(xyz[3 * i + 2]), y = Y(xyz[3 * i + 1])
        if (i === offsets[p]) g.moveTo(x, y)
        else g.lineTo(x, y)
      }
    }
    g.save()
    g.beginPath()
    g.rect(0, 0, w, h)
    g.clip()
    for (const k of [7, 5, 0, 4]) {
      for (let p = 0; p < kinds.length; p++) {
        if (kinds[p] !== k || !(flags[p] & 1)) continue
        path(p)
        if (k === 7) {
          g.fillStyle = dark ? 'rgba(120,170,255,0.16)' : 'rgba(80,140,230,0.13)'
          g.fill()
        } else {
          g.strokeStyle = k === 5 ? seriesColor(Math.max(fields[p], 0), dark) : cssVar('--label')
          g.globalAlpha = k === 5 ? 0.55 : 1
          g.lineWidth = k === 5 ? 0.6 : 0.9
          g.stroke()
          g.globalAlpha = 1
        }
      }
    }
    g.restore()
  }, [scene, dark])
  return <canvas ref={ref} style={{ width: 200, height: 96 }} aria-hidden />
}

function SampleCard({ sample }: { sample: Sample }) {
  return (
    <button
      type="button"
      data-testid={`sample-${sample.id}`}
      onClick={() => openSample(sample.id)}
      className="flex cursor-default flex-col items-center gap-1 rounded-xl border border-separator bg-content p-2 text-center hover:border-accent hover:bg-accent-soft"
    >
      <Thumbnail sample={sample} />
      <span className="line-clamp-2 text-[12px] font-medium">{sample.system.name}</span>
    </button>
  )
}

/** The welcome window: New Design (prominent) and Open, and the engine's
 *  sample designs as cards; samples with extended surface types are listed
 *  under "Surface Types". */
export function Welcome() {
  const open = useView((s) => s.welcomeOpen)
  const set = useView((s) => s.set)
  const basic = SAMPLES.filter((s) => !isSurfaceTypeSample(s))
  const extended = SAMPLES.filter(isSurfaceTypeSample)
  return (
    <Dialog open={open} onOpenChange={(o) => set({ welcomeOpen: o })}>
      <DialogContent className="flex w-[min(940px,94vw)] flex-col p-0">
        <div className="flex items-end gap-4 border-b border-separator px-6 pb-4 pt-6">
          <div className="flex-1">
            <DialogTitle className="text-[22px] font-semibold tracking-tight">MKLight</DialogTitle>
            <DialogDescription className="text-secondary">Optical design. Start a new design, open a project or a lens file, or begin from a sample.</DialogDescription>
          </div>
          <Button variant="prominent" onClick={newDocument}>
            <Plus /> New Design
          </Button>
          <Button onClick={() => void openFromPicker()}>
            <FolderOpen /> Open…
          </Button>
        </div>
        <div className="max-h-[64vh] overflow-y-auto px-6 py-4">
          <h3 className="mb-2 text-[12px] font-semibold text-secondary">Samples</h3>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(212px,1fr))] gap-3">
            {basic.map((s) => (
              <SampleCard key={s.id} sample={s} />
            ))}
          </div>
          <h3 className="mb-2 mt-5 text-[12px] font-semibold text-secondary">Surface Types</h3>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(212px,1fr))] gap-3">
            {extended.map((s) => (
              <SampleCard key={s.id} sample={s} />
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
