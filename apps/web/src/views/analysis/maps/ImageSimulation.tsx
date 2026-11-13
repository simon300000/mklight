// The image simulation viewer: the simulated sensor image beside its
// source (side by side, split or blinking) with exposure, zoom and pan; the
// source menu (built-in patterns or Choose Image…, also by dropping a file),
// the field of view, the patch centre (axis, design field points or a small
// field diagram) and the resolution; a caption naming the patch and PSF.
import { Minus, Plus, Table } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select } from '@/components/ui/select'
import { nf } from '@/model/format'
import { nums, obj } from '@/views/results/common'
import type { KeyNumber, RendererProps } from '../registry'
import { SegmentedOrMenu } from './Controls'
import { lengthText } from './fieldMap'
import {
  EXTENTS,
  FULL_EXTENT,
  PATTERNS,
  SIZES,
  centerFromRun,
  centerJSON,
  centerTitle,
  clippedFraction,
  coordinates,
  distortionPercent,
  extentFromRun,
  extentTitle,
  fieldPoints,
  maxDistortion,
  minRelativeIllumination,
  outsideField,
  patternImage,
  readImageSim,
  readNodes,
  readSource,
  resampleLinear,
  resampleSrgb,
  snap,
  sourceJSON,
  targetSize,
  toRgba,
  type FieldPointChoice,
  type ImageSimData,
  type LinearImage,
} from './imageSim'

type Compare = 'side' | 'split' | 'blink'

/** Decoded user images by the source object they produced (to resample on a
 *  size change without decoding again); the samples live in the settings. */
const originals = new WeakMap<object, { name: string; pixels: ImageData }>()

/** Decodes an image file to sRGB pixels, at most 2048 px on a side. */
async function decodeFile(file: File): Promise<ImageData | null> {
  try {
    const bmp = await createImageBitmap(file)
    const s = Math.min(1, 2048 / Math.max(bmp.width, bmp.height))
    const w = Math.max(1, Math.round(bmp.width * s)), h = Math.max(1, Math.round(bmp.height * s))
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    const ctx = c.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(bmp, 0, 0, w, h)
    bmp.close?.()
    return ctx.getImageData(0, 0, w, h)
  } catch {
    return null
  }
}

/** The field's aspect (width / height in mm) of a result. */
function fieldAspect(j: JsonObject): number | null {
  const s = nums(j.size), p = nums(j.pixel_size_mm)
  if (s.length !== 2 || p.length !== 2) return null
  const a = (s[0] * p[0]) / (s[1] * p[1])
  return Number.isFinite(a) && a > 0 ? a : null
}

/** A linear image drawn into a pixel-sized canvas, scaled by CSS with sharp pixels. */
function ImageCanvas({ img, gain, className, style }: { img: LinearImage | null; gain: number; className?: string; style?: React.CSSProperties }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || !img) return
    c.width = img.width
    c.height = img.height
    let ctx: CanvasRenderingContext2D | null = null
    try {
      ctx = c.getContext('2d')
    } catch {
      ctx = null
    }
    if (!ctx) return
    const id = ctx.createImageData(img.width, img.height)
    id.data.set(toRgba(img, gain))
    ctx.putImageData(id, 0, 0)
  }, [img, gain])
  if (!img) return <div className="flex h-full w-full items-center justify-center text-[11px] text-white/60">No preview</div>
  return <canvas ref={ref} className={className} style={{ imageRendering: 'pixelated', ...style }} />
}

function Tag({ children }: { children: ReactNode }) {
  return <span className="pointer-events-none absolute top-1 left-1 rounded bg-black/55 px-1.5 py-[1px] text-[10px] font-semibold text-white">{children}</span>
}

/** One framed image, fitted at the image's aspect, zoomed about the centre and panned. */
function Frame(props: { img: LinearImage | null; gain: number; aspect: number; zoom: number; pan: [number, number]; title?: string; children?: ReactNode; split?: number; under?: LinearImage | null; underGain?: number }) {
  const [box, setBox] = useState<HTMLDivElement | null>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  useEffect(() => {
    if (!box) return
    const read = () => {
      const r = box.getBoundingClientRect()
      setSize({ w: r.width, h: r.height })
    }
    read()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(read)
    ro.observe(box)
    return () => ro.disconnect()
  }, [box])
  const w = Math.min(size.w, size.h * props.aspect), h = w / Math.max(props.aspect, 1e-6)
  const tf = `translate(${props.pan[0]}px, ${props.pan[1]}px) scale(${props.zoom})`
  return (
    <div ref={setBox} className="relative flex h-full min-h-0 w-full min-w-0 items-center justify-center">
      <div className="relative overflow-hidden bg-black shadow-[0_0_0_0.5px_var(--separator)]" style={{ width: w || '100%', height: h || '100%' }}>
        {props.under !== undefined && <ImageCanvas img={props.under} gain={props.underGain ?? 1} className="absolute inset-0 h-full w-full" style={{ transform: tf }} />}
        <div className="absolute inset-0" style={props.split !== undefined ? { clipPath: `inset(0 ${(1 - props.split) * 100}% 0 0)` } : undefined}>
          <ImageCanvas img={props.img} gain={props.gain} className="absolute inset-0 h-full w-full" style={{ transform: tf }} />
        </div>
        {props.title && <Tag>{props.title}</Tag>}
        {props.children}
      </div>
    </div>
  )
}

/** A small diagram of the normalised field: the disk, the design's field
 *  points and the patch; a click or drag moves the centre, snapping to the
 *  axis and the field points. */
function FieldCenterPicker(props: { center: [number, number]; extent: number; points: FieldPointChoice[]; onSet: (c: [number, number]) => void }) {
  const side = 200, span = 1.15, k = side / 2 / span
  const [drag, setDrag] = useState<[number, number] | null>(null)
  const c = drag ?? props.center
  const at = (e: React.PointerEvent): [number, number] => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect()
    const f: [number, number] = [(e.clientX - r.left - side / 2) / k, (side / 2 - (e.clientY - r.top)) / k]
    return snap(f, props.points.map((p) => p.field), 0.06)
  }
  const X = (v: number) => side / 2 + v * k, Y = (v: number) => side / 2 - v * k
  const e = props.extent
  return (
    <div className="flex w-[200px] flex-col gap-1.5">
      <h3 className="text-[13px] font-semibold">Patch Center</h3>
      <svg
        width={side}
        height={side}
        role="img"
        aria-label={`Field diagram, patch center ${centerTitle(c, props.points)}`}
        className="touch-none"
        onPointerDown={(ev) => {
          ;(ev.currentTarget as SVGElement).setPointerCapture?.(ev.pointerId)
          setDrag(at(ev))
        }}
        onPointerMove={(ev) => drag && setDrag(at(ev))}
        onPointerUp={(ev) => {
          setDrag(null)
          props.onSet(at(ev))
        }}
      >
        <circle cx={side / 2} cy={side / 2} r={k} fill="var(--selected)" stroke="var(--secondary)" strokeOpacity={0.6} />
        <path d={`M0 ${side / 2}H${side}M${side / 2} 0V${side}`} stroke="var(--secondary)" strokeOpacity={0.35} strokeWidth={0.5} strokeDasharray="3 3" />
        {props.points.map((p, i) => (
          <circle key={i} cx={X(p.field[0])} cy={Y(p.field[1])} r={3} fill="var(--secondary)" />
        ))}
        <rect x={X(c[0] - e)} y={Y(c[1] + e)} width={Math.max(2 * e * k, 3)} height={Math.max(2 * e * k, 3)} fill="var(--accent-soft)" stroke="var(--accent)" strokeWidth={1.5} />
        <circle cx={X(c[0])} cy={Y(c[1])} r={2.5} fill="var(--accent)" />
      </svg>
      <span className="num text-[12px]">{centerTitle(c, props.points)}</span>
      {outsideField(c, e) && <span className="text-[11px] text-warning">Part of the patch lies outside the design's field.</span>}
      <span className="text-[11px] text-secondary">Click to center the patch; it snaps to the axis and the field points.</span>
    </div>
  )
}

function FieldMappingTable({ data }: { data: ImageSimData }) {
  const f4 = (v: number) => (Number.isFinite(v) ? v.toFixed(4).replace('-', '−') : '—')
  return (
    <div className="flex flex-col gap-1">
      <h3 className="text-[13px] font-semibold">Field Mapping</h3>
      <p className="text-[11px] text-secondary">PSF grid nodes; the image is mapped bilinearly between them.</p>
      <div className="max-h-[320px] overflow-auto">
        <table className="num text-[11px] select-text">
          <thead className="text-secondary">
            <tr>
              <th className="pr-3 text-left font-medium">Field (Hx, Hy)</th>
              <th className="px-2 text-right font-medium">Ideal (mm)</th>
              <th className="px-2 text-right font-medium">Real (mm)</th>
              <th className="px-2 text-right font-medium">Distortion</th>
              <th className="px-2 text-right font-medium">Rel. illum.</th>
            </tr>
          </thead>
          <tbody>
            {data.nodes.map((n, k) => {
              const d = distortionPercent(n)
              return (
                <tr key={k}>
                  <td className="pr-3">{coordinates(n.field)}</td>
                  <td className="px-2 text-right">
                    {f4(n.ideal[0])}, {f4(n.ideal[1])}
                  </td>
                  <td className="px-2 text-right">
                    {f4(n.real[0])}, {f4(n.real[1])}
                  </td>
                  <td className="px-2 text-right">{Number.isFinite(d) ? `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(3)} %` : '—'}</td>
                  <td className="px-2 text-right">{Number.isFinite(n.ri) ? n.ri.toFixed(3) : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function ImageSimulationView(p: RendererProps) {
  const data = useMemo(() => readImageSim(p.result), [p.result])
  const runSettings = (p.result.runInfo.settings ?? {}) as JsonObject
  const source = useMemo(() => readSource(runSettings.source ?? p.settings.source), [runSettings.source, p.settings.source])
  const sourceImage = useMemo<LinearImage | null>(() => {
    if (!data) return null
    if (source.kind === 'pattern') return patternImage(source.pattern, data.image.width, data.image.height, 4)
    if (source.kind === 'buffer') return source.image
    return null
  }, [data, source])
  const [compare, setCompare] = useState<Compare>('side')
  const [exposure, setExposure] = useState(0)
  const [zoom, setZoomState] = useState(1)
  const [pan, setPan] = useState<[number, number]>([0, 0])
  const [split, setSplit] = useState(0.5)
  const [blinkSource, setBlinkSource] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const dragFrom = useRef<{ x: number; y: number; pan: [number, number] } | null>(null)

  useEffect(() => {
    if (compare !== 'blink') return
    const t = setInterval(() => setBlinkSource((b) => !b), 700)
    return () => clearInterval(t)
  }, [compare])

  if (!data) return <div className="p-4 text-secondary">No simulated image in this result.</div>

  const gain = Math.pow(2, exposure)
  const clipped = clippedFraction(data.image, gain)
  const aspect = (() => {
    const [px, py] = data.pixelSize
    const w = data.image.width * (px > 0 ? px : 1), h = data.image.height * (py > 0 ? py : 1)
    return w / Math.max(h, 1e-12)
  })()
  const setZoom = (z: number) => {
    const n = Math.min(Math.max(z, 1), 32)
    setZoomState(n)
    if (n === 1) setPan([0, 0])
  }

  const optics = obj(p.settings.optics) ?? {}
  const extent = nf(optics.field_extent) || FULL_EXTENT
  const fc = nums(optics.field_center)
  const center: [number, number] = fc.length === 2 ? [fc[0], fc[1]] : [0, 0]
  const size = Math.floor(nums(p.settings.size)[0] || 64)
  const points = fieldPoints(p.system)
  const edit = p.onSettingsChange
  const setOptics = (patch: JsonObject) => edit?.({ optics: { ...optics, ...patch } })

  const runExtent = extentFromRun(data)
  const widthMm = data.pixelSize[0] * data.image.width
  const mmPerExtent = runExtent && Number.isFinite(widthMm) && widthMm > 0 ? widthMm / 2 / runExtent : null
  const extentLabel = (e: number) => {
    let t = extentTitle(e)
    if (mmPerExtent) {
      const w = 2 * mmPerExtent * e
      t += ` · ${lengthText(w)} wide, pixel ${lengthText(w / size)}`
    }
    return t
  }

  const importFile = async (file: File) => {
    setBusy(true)
    const px = await decodeFile(file)
    setBusy(false)
    if (!px) return setError(`${file.name} could not be read`)
    setError(null)
    const [w, h] = targetSize(p.settings.size)
    const lin = resampleSrgb(px, w, h, fieldAspect(p.result.json))
    const src = sourceJSON(lin)
    originals.set(src, { name: file.name, pixels: px })
    edit?.({ source: src })
  }

  const setSize = (n: number) => {
    const patch: JsonObject = { size: [n, n] }
    // An imported image follows the resolution.
    const cur = obj(p.settings.source)
    if (cur?.kind === 'buffer') {
      const [w, h] = targetSize([n, n])
      const orig = originals.get(cur)
      const img = orig ? resampleSrgb(orig.pixels, w, h, fieldAspect(p.result.json)) : readSource(cur).kind === 'buffer' ? resampleLinear((readSource(cur) as { image: LinearImage }).image, w, h) : null
      if (img) {
        const src = sourceJSON(img)
        if (orig) originals.set(src, orig)
        patch.source = src
      }
    }
    edit?.(patch)
  }

  const curSource = obj(p.settings.source)
  const sourceValue = curSource?.kind === 'buffer' ? '__image' : curSource?.kind === 'path' ? '__path' : String(curSource?.pattern ?? 'grid')
  const importedName = curSource ? originals.get(curSource)?.name : undefined
  const sourceTitle = source.kind === 'pattern' ? `${PATTERNS.find((x) => x.id === source.pattern)?.title ?? source.pattern} pattern` : source.kind === 'buffer' ? (importedName ?? 'Imported image') : source.path.split('/').pop()

  const runCenter = centerFromRun(data)
  const psfAuto = (nf(optics.psf_size) || 0) === 0

  const panHandlers = {
    onPointerDown: (e: React.PointerEvent) => {
      if (compare === 'split' && e.altKey) return
      dragFrom.current = { x: e.clientX, y: e.clientY, pan }
      ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
    },
    onPointerMove: (e: React.PointerEvent) => {
      const d = dragFrom.current
      if (!d) return
      if (compare === 'split' && zoom === 1) {
        const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
        setSplit(Math.min(1, Math.max(0, (e.clientX - r.left) / Math.max(r.width, 1))))
        return
      }
      if (zoom > 1) setPan([d.pan[0] + e.clientX - d.x, d.pan[1] + e.clientY - d.y])
    },
    onPointerUp: () => (dragFrom.current = null),
    onWheel: (e: React.WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      setZoom(zoom * (e.deltaY < 0 ? 1.25 : 0.8))
    },
    onDragOver: (e: React.DragEvent) => edit && e.preventDefault(),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault()
      const f = Array.from(e.dataTransfer.files).find((x) => x.type.startsWith('image/'))
      if (f && edit) void importFile(f)
    },
  }

  const sim = compare === 'blink' && blinkSource ? sourceImage : data.image
  return (
    <div className="absolute inset-0 flex flex-col gap-1.5 p-2.5 text-[12px]">
      <div className="relative z-10 flex flex-wrap items-center gap-x-3 gap-y-1">
        {edit && (
          <>
            <Select
              aria-label="Source"
              value={sourceValue}
              onChange={(e) => {
                const v = e.target.value
                if (v === '__choose') fileInput.current?.click()
                else if (!v.startsWith('__')) edit({ source: { kind: 'pattern', pattern: v } })
              }}
            >
              {PATTERNS.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.title}
                </option>
              ))}
              {sourceValue === '__image' && <option value="__image">{importedName ?? 'Imported image'}</option>}
              {sourceValue === '__path' && <option value="__path">{sourceTitle}</option>}
              <option value="__choose">Choose Image…</option>
            </Select>
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              className="hidden"
              aria-hidden
              onChange={(e) => {
                const f = e.target.files?.[0]
                e.target.value = ''
                if (f) void importFile(f)
              }}
            />
            <Select aria-label="Field of view" value={String(EXTENTS.find((e) => Math.abs(e - extent) < 1e-9) ?? extent)} onChange={(e) => setOptics({ field_extent: Number(e.target.value) })} title="The simulated part of the field: a square around the patch center. A smaller patch has finer pixels, so the blur shows.">
              {!EXTENTS.some((e) => Math.abs(e - extent) < 1e-9) && <option value={String(extent)}>View {extentTitle(extent)}</option>}
              {EXTENTS.map((e) => (
                <option key={e} value={String(e)}>
                  View {extentLabel(e)}
                </option>
              ))}
            </Select>
            <span className="inline-flex items-center gap-1">
              <Select
                aria-label="Patch center"
                value={String(points.findIndex((q) => Math.hypot(q.field[0] - center[0], q.field[1] - center[1]) < 1e-4))}
                onChange={(e) => {
                  const k = Number(e.target.value)
                  if (k >= 0) setOptics({ field_center: centerJSON(points[k].field) })
                }}
              >
                {!points.some((q) => Math.hypot(q.field[0] - center[0], q.field[1] - center[1]) < 1e-4) && <option value="-1">At {centerTitle(center, points)}</option>}
                {points.map((q, k) => (
                  <option key={k} value={k}>
                    {q.title === 'On axis' ? q.title : `At ${q.title}`}
                  </option>
                ))}
              </Select>
              <Popover>
                <PopoverTrigger asChild>
                  <Button size="sm" variant="ghost">
                    Pick…
                  </Button>
                </PopoverTrigger>
                <PopoverContent>
                  <FieldCenterPicker center={center} extent={extent} points={points} onSet={(c) => setOptics({ field_center: centerJSON(c) })} />
                </PopoverContent>
              </Popover>
            </span>
            <Select aria-label="Resolution" value={String(size)} onChange={(e) => setSize(Number(e.target.value))} title="Output pixels per side">
              {!SIZES.includes(size) && <option value={String(size)}>{size} px</option>}
              {SIZES.map((n) => (
                <option key={n} value={n}>
                  {n} × {n} px
                </option>
              ))}
            </Select>
          </>
        )}
        <SegmentedOrMenu
          label="Compare"
          value={compare}
          onChange={setCompare}
          options={[
            { value: 'side', title: 'Side by Side' },
            { value: 'split', title: 'Split' },
            { value: 'blink', title: 'Blink' },
          ]}
        />
        <label className="inline-flex items-center gap-1.5 text-secondary">
          Exposure
          <input type="range" min={-4} max={4} step={0.5} value={exposure} onChange={(e) => setExposure(Number(e.target.value))} aria-label="Exposure" className="w-[90px] accent-[var(--accent)]" />
          <span className="num w-[48px] text-label">
            {exposure >= 0 ? '+' : '−'}
            {Math.abs(exposure).toFixed(1)} EV
          </span>
        </label>
        <span className="inline-flex items-center gap-0.5">
          <Button size="iconSm" variant="ghost" aria-label="Zoom out" disabled={zoom <= 1} onClick={() => setZoom(zoom / 2)}>
            <Minus />
          </Button>
          <span className="num min-w-[40px] text-center">{Math.round(zoom * 100)}%</span>
          <Button size="iconSm" variant="ghost" aria-label="Zoom in" disabled={zoom >= 32} onClick={() => setZoom(zoom * 2)}>
            <Plus />
          </Button>
          <Button size="sm" variant="ghost" disabled={zoom === 1 && pan[0] === 0 && pan[1] === 0} onClick={() => setZoom(1)}>
            Fit
          </Button>
        </span>
        <Popover>
          <PopoverTrigger asChild>
            <Button size="sm" variant="ghost" disabled={!data.nodes.length}>
              <Table /> Field Mapping
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto">
            <FieldMappingTable data={data} />
          </PopoverContent>
        </Popover>
      </div>
      <div className="relative min-h-0 flex-1 cursor-grab touch-none overflow-hidden select-none" {...panHandlers} aria-label="Images; drag to pan, drop an image file to use it as the source">
        {compare === 'side' ? (
          <div className="flex h-full gap-2 @max-[480px]:flex-col">
            <Frame img={sourceImage} gain={1} aspect={aspect} zoom={zoom} pan={pan} title={`Source · ${sourceTitle}`} />
            <Frame img={data.image} gain={gain} aspect={aspect} zoom={zoom} pan={pan} title="Simulated" />
          </div>
        ) : compare === 'split' ? (
          <Frame img={sourceImage} gain={1} aspect={aspect} zoom={zoom} pan={pan} split={split} under={data.image} underGain={gain}>
            <span className="pointer-events-none absolute top-0 bottom-0 w-px bg-white/80 shadow-[0_0_0_1px_rgb(0_0_0/0.4)]" style={{ left: `${split * 100}%` }} />
            <Tag>Source | Simulated</Tag>
          </Frame>
        ) : (
          <Frame img={sim} gain={blinkSource ? 1 : gain} aspect={aspect} zoom={zoom} pan={pan} title={blinkSource ? 'Source' : 'Simulated'} />
        )}
        {busy && <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-white">Reading image…</div>}
      </div>
      <div className="num flex flex-wrap gap-x-4 gap-y-0.5 text-[11px] text-secondary">
        <span>
          {data.image.width} × {data.image.height} px{data.image.channels === 3 ? ', RGB' : ', monochrome'}
        </span>
        {Number.isFinite(data.pixelSize[0]) && <span>Pixel {lengthText(data.pixelSize[0])}{Math.abs(data.pixelSize[0] - data.pixelSize[1]) > 1e-9 * data.pixelSize[0] ? ` × ${lengthText(data.pixelSize[1])}` : ''}</span>}
        {runCenter && <span title={patchHelp(data, runCenter)}>At {centerTitle(runCenter, points)}</span>}
        {data.psfSize !== null && data.psfSpacing !== null && (
          <span title={`The PSF window and sample spacing this run used${psfAuto ? ', chosen by the engine from the blur and diffraction' : ''}.`}>
            PSF {data.psfSize} × {lengthText(data.psfSpacing)}
          </span>
        )}
        {clipped > 0.001 && <span className="text-warning">{(clipped * 100).toFixed(1)}% clipped</span>}
        {error && <span className="text-warning">{error}</span>}
        {source.kind === 'pattern' && (
          <span title="The engine evaluates the built-in pattern analytically at each pixel's exact source point, averaged over the pixel; the source shown here is drawn with the same definition.">Pattern exact</span>
        )}
      </div>
    </div>
  )
}

function patchHelp(d: ImageSimData, c: [number, number]): string {
  let t = `Patch centered at normalized field ${coordinates(c)}`
  let best: (typeof d.nodes)[number] | null = null, bd = Infinity
  for (const n of d.nodes) {
    const x = Math.hypot(n.field[0] - c[0], n.field[1] - c[1])
    if (x < bd) ((bd = x), (best = n))
  }
  if (best && Number.isFinite(best.real[0])) t += `, image point (${best.real[0].toFixed(3)}, ${best.real[1].toFixed(3)}) mm`
  return t + '.'
}

export function imageSimulationKeyNumbers(r: AnalysisResult): KeyNumber[] {
  const j = r.json
  const nodes = readNodes(j)
  const out: KeyNumber[] = []
  const px = nums(j.pixel_size_mm)[0]
  if (Number.isFinite(px)) out.push({ label: 'Pixel', value: lengthText(px), help: 'Output pixel pitch on the sensor' })
  const m = maxDistortion(nodes)
  if (m !== null) out.push({ label: 'Max distortion', value: `${m >= 0 ? '+' : '−'}${Math.abs(m).toFixed(3)} %`, help: 'Largest distortion of the field mapping nodes' })
  const ri = minRelativeIllumination(nodes)
  if (ri !== null) out.push({ label: 'Min RI', value: ri.toFixed(3), help: 'Lowest relative illumination of the field mapping nodes' })
  return out
}
