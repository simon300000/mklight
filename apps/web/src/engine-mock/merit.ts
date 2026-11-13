// MOCK ENGINE — a merit function on the mock optics and a damped Gauss-Newton
// (Levenberg–Marquardt) minimiser. The design jobs (optimize, glass
// substitution, tolerance) run on it: the numbers come from the mock's real-ray
// trace, so edits change them, but they are not the engine's.
import type { Json, JsonObject } from '../engine/types'
import { MockOptics } from './optics'
import { parseSystem, type Obj } from './model'
import { pupilSamples, firstOrderEntries } from './analyses'
import { asPath, getParamValue, setParamValue, type Path } from './params'

export interface Operand {
  kind: string
  target: number
  weight: number
  comparison: 'equal' | 'less_than' | 'greater_than'
  role: 'residual' | 'constraint'
  field: JsonObject
  wavelength: JsonObject
  config: number
  enabled: boolean
  label: string
  raw: Obj
}

export interface OperandRow {
  index: number
  value: number
  residual: number | null
  contribution: number | null
  error: string | null
}

export interface Eval {
  rows: number[]
  /** Sum of the residual-role operands' w·r². */
  merit: number
  /** Normalised constraint violation. */
  violation: number
  operands: OperandRow[]
  ok: boolean
}

const CONSTRAINT_WEIGHT = 1e5
/** Operand kinds read from the first-order table. */
const FIRST_ORDER_KEYS: Record<string, string> = {
  efl: 'efl',
  bfl: 'bfl',
  total_track: 'total_track',
  track: 'total_track',
  working_f_number: 'working_f_number',
  f_number: 'image_f_number',
  image_f_number: 'image_f_number',
  paraxial_image_height: 'paraxial_image_height',
  magnification: 'magnification',
  petzval_radius: 'petzval_radius',
}

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d)

export function readOperand(raw: Obj, forceConstraint = false): Operand {
  const kind = typeof raw.kind === 'string' ? raw.kind : String(raw.kind?.type ?? 'rms_spot')
  return {
    kind,
    target: num(raw.target, 0),
    weight: num(raw.weight, 1),
    comparison: raw.comparison === 'less_than' || raw.comparison === 'greater_than' ? raw.comparison : 'equal',
    role: forceConstraint || raw.role === 'constraint' ? 'constraint' : 'residual',
    field: (raw.field && typeof raw.field === 'object' ? raw.field : { field: 'all' }) as JsonObject,
    wavelength: (raw.wavelength && typeof raw.wavelength === 'object' ? raw.wavelength : { wavelength: 'primary' }) as JsonObject,
    config: num(raw.config, 0),
    enabled: raw.enabled !== false,
    label: typeof raw.label === 'string' ? raw.label : '',
    raw,
  }
}

function fieldIndices(sel: JsonObject, count: number): number[] {
  const t = String(sel.field ?? sel.type ?? 'all')
  if (t === 'index' && typeof sel.index === 'number') return sel.index >= 0 && sel.index < count ? [sel.index] : []
  if (t === 'list' && Array.isArray(sel.indices)) return (sel.indices as number[]).filter((i) => i >= 0 && i < count)
  return Array.from({ length: count }, (_, i) => i)
}

export interface MeritSetup {
  operands: Operand[]
  spec: JsonObject
  rings: number
}

/** The merit of a design job: explicit spec, or the default RMS spot per field, plus extra operands, constraints and the EFL hold. */
export function buildMerit(s: Obj, system: Obj): MeritSetup {
  const nF = (system.fields?.fields ?? []).length || 1
  const spec = s.merit && typeof s.merit === 'object' ? (s.merit as Obj) : null
  let ops: Operand[]
  if (spec && Array.isArray(spec.operands)) ops = (spec.operands as Obj[]).map((o) => readOperand(o))
  else {
    const fw: number[] = (system.fields?.fields ?? []).map((f: Obj) => num(f.weight, 1))
    const sum = fw.reduce((a, b) => a + b, 0) || 1
    ops = Array.from({ length: nF }, (_, i) =>
      readOperand({
        kind: { type: 'rms_spot' },
        target: 0,
        weight: (fw[i] ?? 1) / sum,
        comparison: 'equal',
        role: 'residual',
        field: { field: 'index', index: i },
        wavelength: { wavelength: 'primary' },
        config: 0,
        enabled: true,
        label: `RMS spot, field ${i + 1}`,
      }),
    )
  }
  for (const o of Array.isArray(s.operands) ? (s.operands as Obj[]) : []) ops.push(readOperand(o))
  for (const o of Array.isArray(s.constraints) ? (s.constraints as Obj[]) : []) ops.push(readOperand(o, true))
  if (s.hold_efl !== false && !ops.some((o) => o.kind === 'efl')) {
    const efl = new MockOptics(parseSystem(system as JsonObject)).efl(parseSystem(system as JsonObject).primary)
    if (efl !== null && Number.isFinite(efl))
      ops.push(
        readOperand({
          kind: { type: 'efl' },
          target: efl,
          weight: 1,
          comparison: 'equal',
          role: 'constraint',
          field: { field: 'all' },
          wavelength: { wavelength: 'primary' },
          config: 0,
          enabled: true,
          label: 'hold EFL',
        }),
      )
  }
  const sampling = ((s.default_merit as Obj | undefined)?.sampling ?? {}) as Obj
  return {
    operands: ops,
    spec: {
      schema_version: 1,
      name: typeof spec?.name === 'string' ? spec.name : 'default',
      on_failure: (spec?.on_failure as Json) ?? { mode: 'penalty', value: 1000 },
      operands: ops.map(operandJson),
    },
    rings: Math.min(Math.max(num(sampling.rings, 2), 1), 2),
  }
}

export function operandJson(o: Operand): Json {
  return {
    ...(o.raw as JsonObject),
    kind: { type: o.kind },
    target: o.target,
    weight: o.weight,
    comparison: o.comparison,
    role: o.role,
    field: o.field,
    wavelength: o.wavelength,
    config: o.config,
    enabled: o.enabled,
    label: o.label,
  }
}

interface SpotRows {
  dx: number[]
  dy: number[]
  failed: number
  n: number
  rms: number
}

/** Evaluates the merit of one system. */
export function evaluateMerit(system: Obj, setup: MeritSetup): Eval {
  let o: MockOptics
  try {
    o = new MockOptics(parseSystem(system as JsonObject))
  } catch {
    return { rows: [1], merit: 1e6, violation: 0, operands: [], ok: false }
  }
  const pupil = pupilSamples('hexapolar', setup.rings, 1)
  const cache = new Map<string, SpotRows>()
  const spot = (f: number, w: number): SpotRows => {
    const key = `${f}:${w}`
    const hit = cache.get(key)
    if (hit) return hit
    const fp = o.fieldPoint(f)
    const xs: number[] = []
    const ys: number[] = []
    let failed = 0
    for (const [px, py] of pupil) {
      const r = o.trace(o.makeRay(fp, px, py), w).atImage
      if (r) {
        xs.push(r.p[0])
        ys.push(r.p[1])
      } else failed++
    }
    const m = xs.length
    const cx = xs.reduce((a, b) => a + b, 0) / (m || 1)
    const cy = ys.reduce((a, b) => a + b, 0) / (m || 1)
    const dx = xs.map((x) => x - cx)
    const dy = ys.map((y) => y - cy)
    const rms = m ? Math.sqrt(dx.reduce((a, d, i) => a + d * d + dy[i] * dy[i], 0) / m) : 1
    const r: SpotRows = { dx, dy, failed, n: pupil.length, rms }
    cache.set(key, r)
    return r
  }
  const wl = (sel: JsonObject): number => {
    const t = String(sel.wavelength ?? sel.type ?? 'primary')
    return t === 'index' && typeof sel.index === 'number' && sel.index < o.wavelengthsUm.length ? sel.index : o.primary
  }
  let firstOrder: Map<string, number> | null = null

  const rows: number[] = []
  const operands: OperandRow[] = []
  let merit = 0
  let violation = 0
  setup.operands.forEach((op, index) => {
    if (!op.enabled) {
      operands.push({ index, value: NaN, residual: null, contribution: null, error: null })
      return
    }
    let value = NaN
    let error: string | null = null
    const constraint = op.role === 'constraint'
    const nFields = o.sys.fields.length
    const w = wl(op.wavelength)
    const fkey = FIRST_ORDER_KEYS[op.kind]
    let spotRows: number[] | null = null
    if (fkey) {
      if (!firstOrder) {
        firstOrder = new Map()
        for (const e of firstOrderEntries(o, w)) firstOrder.set(e.key, e.value)
      }
      const v = firstOrder.get(fkey)
      if (v === undefined || !Number.isFinite(v)) error = `${op.kind} is undefined`
      else value = v
    } else {
      const sel = fieldIndices(op.field, nFields)
      if (sel.length === 0) error = 'no field selected'
      else {
        const spots = sel.map((f) => spot(f, w))
        value = Math.sqrt(spots.reduce((a, sp) => a + sp.rms * sp.rms, 0) / spots.length)
        if (!constraint && op.comparison === 'equal' && op.target === 0 && op.weight > 0) {
          spotRows = []
          const k = Math.sqrt(op.weight / sel.length)
          for (const sp of spots) {
            const c = k / Math.sqrt(sp.n)
            for (let i = 0; i < sp.dx.length; i++) spotRows.push(c * sp.dx[i], c * sp.dy[i])
            for (let i = 0; i < sp.failed; i++) spotRows.push(c)
          }
        }
      }
    }
    if (error) {
      rows.push(Math.sqrt(op.weight) * 1)
      if (!constraint) merit += op.weight
      operands.push({ index, value: NaN, residual: null, contribution: null, error })
      return
    }
    const d = value - op.target
    const residual = op.comparison === 'less_than' ? Math.max(d, 0) : op.comparison === 'greater_than' ? Math.min(d, 0) : d
    if (constraint) {
      const scale = Math.abs(op.target) > 1e-9 ? Math.abs(op.target) : 1
      violation += Math.abs(residual) / scale
      rows.push(Math.sqrt(CONSTRAINT_WEIGHT) * (residual / scale))
      operands.push({ index, value, residual: null, contribution: null, error: null })
    } else {
      const c = op.weight * residual * residual
      merit += c
      if (spotRows) for (const r of spotRows) rows.push(r)
      else rows.push(Math.sqrt(op.weight) * residual)
      operands.push({ index, value, residual, contribution: c, error: null })
    }
  })
  return { rows, merit, violation, operands, ok: true }
}

export function operandReport(setup: MeritSetup, start: Eval, now: Eval): JsonObject[] {
  const total = now.operands.reduce((a, r) => a + (r.contribution ?? 0), 0)
  const fin = (x: number | null): Json => (x === null || !Number.isFinite(x) ? null : x)
  return setup.operands.map((op, i) => {
    const r = now.operands[i]
    return {
      ...(operandJson(op) as JsonObject),
      index: i,
      kind: op.kind,
      label: op.label,
      start: fin(start.operands[i]?.value ?? null),
      value: fin(r?.value ?? null),
      residual: fin(r?.residual ?? null),
      contribution: fin(r?.contribution ?? null),
      contribution_fraction: r?.contribution != null && total > 0 ? r.contribution / total : null,
      error: r?.error ?? null,
    }
  })
}

// --- Variables and the working copy ----------------------------------------

export interface Variable {
  path: Path
  start: number
  lower: number | null
  upper: number | null
  fd: number
  label: string
}

const STEP_SCALE: Record<string, number> = { curvature: 0.01, thickness: 1, conic: 0.1, asphere: 1e-6 }

export function readVariables(list: unknown, system: Obj, defaults: Path[], labeller: (p: Path) => string): Variable[] {
  const entries: Obj[] = Array.isArray(list) ? (list as Obj[]) : []
  const specs: Obj[] = entries.length ? entries : defaults.map((p) => ({ param: p }))
  const out: Variable[] = []
  for (const e of specs) {
    const p = asPath(e.param)
    if (!p) continue
    const v = getParamValue(system, p)
    if (v === null || !Number.isFinite(v)) continue
    const lower = typeof e.lower === 'number' ? e.lower : null
    const upper = typeof e.upper === 'number' ? e.upper : null
    const scale = STEP_SCALE[p.param] ?? 1
    out.push({
      path: p,
      start: v,
      lower,
      upper,
      fd: typeof e.fd_step === 'number' && e.fd_step > 0 ? e.fd_step : 1e-4 * Math.max(Math.abs(v), scale),
      label: typeof e.label === 'string' && e.label ? e.label : labeller(p),
    })
  }
  return out
}

/** Implicit bounds of the default merit: gaps keep at least 0.05 mm. */
export function boundsOf(v: Variable): [number, number] {
  const lo = v.lower ?? (v.path.param === 'thickness' ? 0.05 : -Infinity)
  const hi = v.upper ?? Infinity
  return [lo, hi]
}

export function applyValues(work: Obj, vars: Variable[], x: number[]): void {
  vars.forEach((v, i) => setParamValue(work, v.path, x[i]))
}

// --- Levenberg–Marquardt -----------------------------------------------------

function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length
  const M = A.map((r, i) => [...r, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    if (Math.abs(M[p][c]) < 1e-300) return null
    ;[M[c], M[p]] = [M[p], M[c]]
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c]
      if (f !== 0) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  const x = new Array<number>(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n]
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]
    x[r] = s / M[r][r]
  }
  return x.every(Number.isFinite) ? x : null
}

export interface LmStep {
  accepted: boolean
  converged: boolean
}

/**
 * One LM minimiser of Σ rows² (objective = merit + penalised constraints)
 * over the variables of a working copy of the system.
 */
export class Lm {
  x: number[]
  eval: Eval
  evaluations = 0
  lambda = 1e-3
  private obj: number
  private smallSteps = 0
  readonly bounds: [number, number][]

  constructor(
    readonly work: Obj,
    readonly vars: Variable[],
    readonly setup: MeritSetup,
    x0?: number[],
  ) {
    this.x = (x0 ?? vars.map((v) => v.start)).slice()
    this.bounds = vars.map(boundsOf)
    this.x = this.x.map((v, i) => clamp(v, this.bounds[i]))
    this.eval = this.evaluate(this.x)
    this.obj = objective(this.eval)
  }

  evaluate(x: number[]): Eval {
    applyValues(this.work, this.vars, x)
    this.evaluations++
    return evaluateMerit(this.work, this.setup)
  }

  /** The working copy holds the best point after this. */
  step(): LmStep {
    const n = this.vars.length
    const r0 = this.eval.rows
    if (n === 0) return { accepted: false, converged: true }
    const m = r0.length
    const J: number[][] = []
    for (let i = 0; i < n; i++) {
      const h = this.vars[i].fd
      const xp = this.x.slice()
      xp[i] = clamp(xp[i] + h, this.bounds[i])
      let dh = xp[i] - this.x[i]
      if (Math.abs(dh) < h * 0.5) {
        xp[i] = clamp(this.x[i] - h, this.bounds[i])
        dh = xp[i] - this.x[i]
      }
      const ev = this.evaluate(xp)
      const col = new Array<number>(m)
      const len = Math.min(m, ev.rows.length)
      for (let k = 0; k < m; k++) col[k] = k < len && dh !== 0 ? (ev.rows[k] - r0[k]) / dh : 0
      J.push(col)
    }
    const g = J.map((c) => c.reduce((a, v, k) => a + v * r0[k], 0))
    const H = J.map((ci) => J.map((cj) => ci.reduce((a, v, k) => a + v * cj[k], 0)))
    let accepted = false
    for (let attempt = 0; attempt < 5 && !accepted; attempt++) {
      const A = H.map((row, i) => row.map((v, j) => (i === j ? v + this.lambda * Math.max(H[i][i], 1e-30) : v)))
      const d = solveLinear(A, g.map((v) => -v))
      if (!d) {
        this.lambda *= 10
        continue
      }
      const xn = this.x.map((v, i) => clamp(v + d[i], this.bounds[i]))
      const ev = this.evaluate(xn)
      const o = objective(ev)
      if (ev.ok && o < this.obj) {
        const rel = (this.obj - o) / Math.max(this.obj, 1e-300)
        this.smallSteps = rel < 1e-5 ? this.smallSteps + 1 : 0
        this.x = xn
        this.eval = ev
        this.obj = o
        this.lambda = Math.max(this.lambda / 4, 1e-9)
        accepted = true
      } else this.lambda = Math.min(this.lambda * 6, 1e8)
    }
    applyValues(this.work, this.vars, this.x)
    return { accepted, converged: !accepted || this.smallSteps >= 2 }
  }
}

const objective = (e: Eval) => e.rows.reduce((a, r) => a + r * r, 0)
const clamp = (v: number, [lo, hi]: [number, number]) => Math.min(Math.max(v, lo), hi)
