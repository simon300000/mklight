import { FolderOpen, Plus } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { isSurfaceTypeSample, SAMPLES, type Sample } from '@/model/project'
import { curvatureOf, materialText } from '@/model/system'
import { nf } from '@/model/format'
import { newDocument, openFromPicker, openSample } from '@/state/files'
import { useView } from '@/state/view'
import { cssVar, useDark } from '@/lib/theme'

/** A small sketch of the sample's elements from its prescription (no optics:
 *  vertices from thicknesses, spherical profiles from curvatures, display only). */
function Thumbnail({ sample }: { sample: Sample }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  useEffect(() => {
    const c = ref.current
    const g = c?.getContext('2d')
    if (!c || !g) return
    const w = 200, h = 96, dpr = devicePixelRatio || 1
    c.width = w * dpr
    c.height = h * dpr
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    const s = sample.system.surfaces
    const zs: number[] = []
    let z = 0
    for (let i = 1; i < s.length; i++) {
      zs[i] = z
      const t = nf(s[i].thickness)
      z += Number.isFinite(t) ? Math.abs(t) : 0
    }
    // Frame the lens: the elements, not the long back focus.
    const glass = s.map((x, i) => (i > 0 && i < s.length - 1 && (materialText(x) !== '' || materialText(s[i - 1]) !== '') ? i : -1)).filter((i) => i > 0)
    const idx = glass.length ? glass : s.map((_, i) => i).slice(1, -1)
    const z0 = Math.min(...idx.map((i) => zs[i])), z1 = Math.max(...idx.map((i) => zs[i]))
    const span = Math.max(z1 - z0, 1)
    const ap = nf((sample.system.aperture as { value?: number }).value) || 10
    const semi = Math.max(ap / 2, span * 0.25, 1)
    const scale = Math.min((w - 40) / span, (h - 16) / (2 * semi * 1.15))
    const X = (zz: number) => w / 2 + (zz - (z0 + z1) / 2) * scale
    const Y = (y: number) => h / 2 - y * scale
    g.clearRect(0, 0, w, h)
    g.strokeStyle = cssVar('--label')
    g.lineWidth = 1
    for (const i of idx) {
      const c0 = curvatureOf(s[i].shape) ?? 0
      g.beginPath()
      for (let k = 0; k <= 24; k++) {
        const y = -semi + (2 * semi * k) / 24
        const q = 1 - c0 * c0 * y * y
        const sg = q > 0 ? (c0 * y * y) / (1 + Math.sqrt(q)) : 0
        if (k) g.lineTo(X(zs[i] + sg), Y(y))
        else g.moveTo(X(zs[i] + sg), Y(y))
      }
      g.stroke()
    }
    g.strokeStyle = cssVar('--secondary')
    g.setLineDash([3, 3])
    g.beginPath()
    g.moveTo(4, h / 2)
    g.lineTo(w - 4, h / 2)
    g.stroke()
  }, [sample, dark])
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
