// The design's and a candidate's layouts side by side (LayoutComparison of
// GlassSubstitutionPanel.swift): small 2-D sections from the engine's `layout`.

import { useEffect, useRef, useState } from 'react'
import type { ModelSnapshot } from '@/engine/types'
import { useWorkspace } from '@/state/workspace'
import { cssVar, seriesColor, useDark } from '@/lib/theme'
import { readScene, type Scene } from '../../LayoutView'

const cache = new Map<string, Promise<Scene | null>>()

function sceneOf(snap: ModelSnapshot): Promise<Scene | null> {
  let p = cache.get(snap.hash)
  if (!p) {
    const engine = useWorkspace.getState().engine
    if (!engine) return Promise.resolve(null)
    try {
      p = engine
        .submit({ analysis: 'layout', settings: { section: 'yz', rays_per_field: 3, include_3d: false } }, snap)
        .result.then(readScene)
        .catch(() => null)
    } catch {
      p = Promise.resolve(null)
    }
    if (cache.size > 40) cache.clear()
    cache.set(snap.hash, p)
  }
  return p
}

function Thumb({ title, snap }: { title: string; snap: ModelSnapshot | null }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  const [scene, setScene] = useState<Scene | null>(null)
  useEffect(() => {
    let live = true
    setScene(null)
    if (snap) void sceneOf(snap).then((s) => live && setScene(s))
    return () => {
      live = false
    }
  }, [snap])
  useEffect(() => {
    const c = ref.current
    let g: CanvasRenderingContext2D | null = null
    try {
      g = c?.getContext('2d') ?? null
    } catch {
      g = null
    }
    if (!c || !g) return
    const w = c.clientWidth || 160, h = 88, dpr = devicePixelRatio || 1
    c.width = w * dpr
    c.height = h * dpr
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, h)
    if (!scene) return
    const { xyz, offsets, kinds, flags, fields } = scene
    let z0 = Infinity, z1 = -Infinity, hm = 0
    for (let p = 0; p < kinds.length; p++) {
      if (!(flags[p] & 1) || (kinds[p] !== 0 && kinds[p] !== 7)) continue
      for (let i = offsets[p]; i < offsets[p + 1]; i++) {
        z0 = Math.min(z0, xyz[3 * i + 2])
        z1 = Math.max(z1, xyz[3 * i + 2])
        hm = Math.max(hm, Math.abs(xyz[3 * i + 1]))
      }
    }
    if (!Number.isFinite(z0)) return
    const scale = Math.min((w - 16) / Math.max(z1 - z0, 1e-6), (h - 10) / Math.max(2 * hm, 1e-6))
    const X = (z: number) => w / 2 + (z - (z0 + z1) / 2) * scale
    const Y = (y: number) => h / 2 - y * scale
    for (const k of [7, 5, 0, 4]) {
      for (let p = 0; p < kinds.length; p++) {
        if (kinds[p] !== k || !(flags[p] & 1)) continue
        g.beginPath()
        for (let i = offsets[p]; i < offsets[p + 1]; i++) {
          const x = X(xyz[3 * i + 2]), y = Y(xyz[3 * i + 1])
          if (i === offsets[p]) g.moveTo(x, y)
          else g.lineTo(x, y)
        }
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
  }, [scene, dark])
  return (
    <figure className="min-w-0 flex-1" aria-label={`${title} layout`}>
      <figcaption className="mb-0.5 text-[11px] text-secondary">{title}</figcaption>
      <div className="rounded-md bg-content">
        <canvas ref={ref} className="block w-full" style={{ height: 88 }} aria-hidden />
      </div>
    </figure>
  )
}

/** Before: the design's snapshot; Candidate: the previewed one. */
export function LayoutCompare() {
  const snap = useWorkspace((s) => s.snapshot)
  const preview = useWorkspace((s) => s.preview)
  return (
    <div className="flex gap-2" data-testid="layout-compare">
      <Thumb title="Before" snap={snap} />
      <Thumb title="Candidate" snap={preview?.snapshot ?? null} />
    </div>
  )
}
