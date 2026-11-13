// MOCK ENGINE — lens prescription import (.zmx, .seq) and AGF glass catalogs.
// A minimal real parser: it maps what the mock engine can hold and reports the
// rest as warnings (docs/IMPORT.md). Output is the engine's OpticalSystem JSON.
import { EngineError, type CatalogInfo, type ImportWarning, type JsonObject, type OpticalSystem, type PrescriptionImport } from '../engine/types'
import { BUILTIN_CATALOG, CATALOG, type Obj } from './model'

// --- Text decoding -----------------------------------------------------------

export function decodeText(data: ArrayBuffer): { text: string; encoding: string } {
  const b = new Uint8Array(data)
  if (b.length >= 2 && b[0] === 0xff && b[1] === 0xfe) return { text: new TextDecoder('utf-16le').decode(b.subarray(2)), encoding: 'utf-16le' }
  if (b.length >= 2 && b[0] === 0xfe && b[1] === 0xff) return { text: new TextDecoder('utf-16be').decode(b.subarray(2)), encoding: 'utf-16be' }
  if (b.length >= 3 && b[0] === 0xef && b[1] === 0xbb && b[2] === 0xbf) return { text: new TextDecoder('utf-8').decode(b.subarray(3)), encoding: 'utf-8' }
  // UTF-16 without a BOM: every other byte is zero.
  if (b.length >= 4 && b[1] === 0 && b[3] === 0) return { text: new TextDecoder('utf-16le').decode(b), encoding: 'utf-16le' }
  try {
    return { text: new TextDecoder('utf-8', { fatal: true }).decode(b), encoding: 'utf-8' }
  } catch {
    return { text: new TextDecoder('latin1').decode(b), encoding: 'latin-1' }
  }
}

// --- Shared building blocks ---------------------------------------------------

const LENGTH_SCALE: Record<string, number> = { MM: 1, CM: 10, M: 1000, IN: 25.4 }

function surface(id: number, label: string, shape: JsonObject, thickness: number | 'inf', material: JsonObject, interaction = 'refract'): Obj {
  return {
    id,
    label,
    shape,
    thickness,
    material,
    interaction,
    semi_diameter: { mode: 'auto' },
    aperture: null,
    placement: null,
    coating: null,
    scatter: null,
    diffraction_efficiency: { type: 'ideal' },
    edge: { mechanical_semi_diameter: null, edge_scatter: null },
  }
}

function makeShape(curvature: number, conic: number, coefficients: number[]): JsonObject {
  while (coefficients.length && coefficients[coefficients.length - 1] === 0) coefficients.pop()
  if (coefficients.length) return { type: 'even_asphere', curvature, conic, coefficients }
  if (curvature === 0 && conic === 0) return { type: 'plane' }
  return { type: 'conic', curvature, conic }
}

type Warn = (kind: string, line: number | null, surface: number | null, message: string) => void

/** A material from a glass name (and optional model n_d, V_d). */
function resolveGlass(name: string, nd: number | null, vd: number | null, line: number, surfaceIndex: number, warn: Warn): { material: JsonObject; mirror: boolean } {
  const raw = name.trim()
  const up = raw.toUpperCase()
  if (up === '' || up === 'AIR' || up === 'VACUUM') return { material: { type: 'air' }, mirror: false }
  if (up === 'MIRROR' || up === 'REFL') return { material: { type: 'air' }, mirror: true }
  if (up === '___BLANK') return { material: { type: 'model', nd: nd ?? 1.5, vd: vd ?? 50, dpgf: 0 }, mirror: false }
  const candidates = [up, up.replace(/_[A-Z]+$/, '')]
  for (const c of candidates) if (CATALOG[c]) return { material: { type: 'catalog', catalog: null, name: c }, mirror: false }
  // A six-digit glass code or "n:v".
  const code = /^(\d{3})(\d{3})$/.exec(up)
  if (code) return { material: { type: 'model', nd: 1 + Number(code[1]) / 1000, vd: Number(code[2]) / 10, dpgf: 0 }, mirror: false }
  const nv = /^([\d.]+):([\d.]+)$/.exec(up)
  if (nv) return { material: { type: 'model', nd: Number(nv[1]), vd: Number(nv[2]), dpgf: 0 }, mirror: false }
  if (nd !== null && vd !== null && nd > 1) {
    warn('unknown_glass', line, surfaceIndex, `glass '${raw}' is in no loaded catalog: a model glass n_d ${nd}, V_d ${vd} is used`)
    return { material: { type: 'model', nd, vd, dpgf: 0 }, mirror: false }
  }
  warn('unknown_glass', line, surfaceIndex, `glass '${raw}' is in no loaded catalog; it resolves once a catalog containing it is loaded`)
  return { material: { type: 'catalog', catalog: null, name: raw }, mirror: false }
}

interface Draft {
  name: string
  notes: string
  unit: number
  aperture: { type: string; value?: number } | null
  fieldKind: string
  fieldX: number[]
  fieldY: number[]
  fieldW: number[]
  nFields: number | null
  wavelengths: { um: number; weight: number }[]
  primary: number | null
  temperature: number
  pressure: number
  rayAiming: string
}

function finish(d: Draft, surfaces: Obj[], stop: number | null, solves: Obj[], warn: Warn): OpticalSystem {
  if (surfaces.length === 0) throw new EngineError('failed', 'no surfaces found: not a lens prescription')
  // Object first and image last.
  if (surfaces[surfaces.length - 1].thickness !== 0 || surfaces.length < 3) {
    const last = surfaces[surfaces.length - 1]
    if (surfaces.length < 3 || last.label !== 'IMA') {
      warn('defaulted', null, null, 'no image surface: a plane image surface was appended')
      surfaces.push(surface(0, 'IMA', { type: 'plane' }, 0, { type: 'air' }))
    }
  }
  surfaces.forEach((s, i) => {
    s.id = i
    if (!s.label) s.label = i === 0 ? 'OBJ' : i === surfaces.length - 1 ? 'IMA' : ''
  })
  for (const sv of solves) if (sv.target.surface === -1) sv.target.surface = surfaces[surfaces.length - 2].id
  let st = stop
  if (st === null || st < 1 || st > surfaces.length - 2) {
    warn('defaulted', null, null, 'no stop surface: surface 1 is the stop')
    st = 1
  }
  let aperture = d.aperture
  if (!aperture) {
    warn('defaulted', null, null, 'no system aperture: an entrance pupil diameter of 10 mm is used')
    aperture = { type: 'entrance_pupil_diameter', value: 10 }
  }
  let wl = d.wavelengths
  if (!wl.length) {
    warn('defaulted', null, null, 'no wavelengths: the d line (0.5875618 µm) is used')
    wl = [{ um: 0.5875618, weight: 1 }]
  }
  const nF = Math.max(d.nFields ?? 0, 0) || Math.max(d.fieldY.length, d.fieldX.length, 1)
  const fields = Array.from({ length: nF }, (_, i) => ({
    x: d.fieldX[i] ?? 0,
    y: d.fieldY[i] ?? 0,
    weight: d.fieldW[i] ?? 1,
    vignetting: { vdx: 0, vdy: 0, vcx: 0, vcy: 0 },
  }))
  if (d.fieldY.length === 0 && d.fieldX.length === 0) warn('defaulted', null, null, 'no fields: a single on-axis field is used')
  const primary = d.primary !== null && d.primary >= 0 && d.primary < wl.length ? d.primary : Math.floor((wl.length - 1) / 2)
  const system: Obj = {
    name: d.name,
    notes: d.notes,
    aperture,
    fields: { kind: d.fieldKind, fields },
    wavelengths: { wavelengths: wl, primary },
    surfaces,
    stop: st,
    ray_aiming: { mode: d.rayAiming },
    environment: { temperature_c: d.temperature, pressure_atm: d.pressure },
    coatings: {},
    mechanics: [],
    sensor: null,
    configurations: [],
  }
  if (solves.length) system.solves = solves
  if (stop !== null && st === stop && aperture.type === 'float_by_stop_size') system.aperture = { type: 'float_by_stop_size' }
  return system as OpticalSystem
}

const newDraft = (): Draft => ({
  name: '',
  notes: '',
  unit: 1,
  aperture: null,
  fieldKind: 'object_angle',
  fieldX: [],
  fieldY: [],
  fieldW: [],
  nFields: null,
  wavelengths: [],
  primary: null,
  temperature: 20,
  pressure: 1,
  rayAiming: 'off',
})

const numbers = (parts: string[]): number[] => parts.map(Number).filter((x) => Number.isFinite(x))

// --- ZMX --------------------------------------------------------------------------

/** Records that only affect drawing, bookkeeping or defaults the mock needs not map. */
const ZMX_IGNORED = new Set([
  'VERS', 'MODE', 'PFIL', 'GFAC', 'PUSH', 'SDMA', 'ROPD', 'PICB', 'POLS', 'GLRS', 'HIDE', 'SLAB', 'MIRR', 'VDXN', 'VDYN', 'VCXN', 'VCYN',
  'VDX', 'VDY', 'VCX', 'VCY', 'PWAV', 'NSCD', 'NSTR', 'BLNK', 'FWGN', 'COFN', 'TOL', 'TOFF', 'PZUP', 'SQAP', 'WWGN', 'WWGT', 'ZVAN',
  'RSCE', 'CONF', 'ENVD', 'PRES', 'UDAF', 'OPDX', 'POLB', 'ANAR', 'GCAT', 'FTYP', 'NOTE', 'NAME', 'UNIT', 'ENPD', 'FNUM', 'FLOA',
])

export function parseZmx(text: string): PrescriptionImport {
  const warnings: ImportWarning[] = []
  const warn: Warn = (kind, line, surf, message) => warnings.push({ kind, line, surface: surf, message })
  const d = newDraft()
  const surfaces: Obj[] = []
  const solves: Obj[] = []
  let stop: number | null = null
  let primaryN: number | null = null
  const counts = new Map<string, number>()
  interface Cur {
    index: number
    label: string
    type: string
    curv: number
    disz: number | 'inf'
    conic: number
    parm: number[]
    glass: { name: string; nd: number | null; vd: number | null; line: number } | null
    diam: number | null
    floating: boolean
    line: number
  }
  let cur: Cur | null = null
  const flush = () => {
    if (!cur) return
    const c = cur
    const f = d.unit
    const idx = surfaces.length
    let shape: JsonObject
    const typ = c.type.toUpperCase()
    const curvature = c.curv / f
    if (typ === 'EVENASPH') {
      const coef: number[] = []
      for (let k = 1; k < 8; k++) coef.push((c.parm[k] ?? 0) * f ** (1 - 2 * (k + 1)))
      if ((c.parm[0] ?? 0) !== 0) warn('unsupported_surface', c.line, idx, 'the r² term (PARM 1) of the even asphere is dropped')
      shape = makeShape(curvature, c.conic, coef)
      if (shape.type === 'plane') shape = { type: 'conic', curvature: 0, conic: 0 }
    } else if (typ === 'PARAXIAL') {
      shape = { type: 'paraxial', focal_length: (c.parm[0] ?? 0) * f }
    } else {
      if (typ !== 'STANDARD') warn(typ === 'NONSEQCO' ? 'non_sequential' : 'unsupported_surface', c.line, idx, `surface type ${typ}: the base ${c.conic !== 0 ? 'conic' : 'sphere'} is kept`)
      shape = makeShape(curvature, c.conic, [])
    }
    let material: JsonObject = { type: 'air' }
    let mirror = false
    if (c.glass) {
      const g = resolveGlass(c.glass.name, c.glass.nd, c.glass.vd, c.glass.line, idx, warn)
      material = g.material
      mirror = g.mirror
    }
    const s = surface(idx, c.label, shape, c.disz === 'inf' ? 'inf' : c.disz * f, material, mirror ? 'reflect' : 'refract')
    if (c.diam !== null && c.floating) s.semi_diameter = { mode: 'fixed', value: c.diam * f }
    surfaces.push(s)
    cur = null
  }
  const lines = text.replace(/\r/g, '').split('\n')
  lines.forEach((rawLine, li) => {
    const lineNo = li + 1
    const line = rawLine.trim()
    if (!line) return
    const parts = line.split(/\s+/)
    const key = parts[0].toUpperCase()
    const rest = line.slice(parts[0].length).trim()
    if (key === 'SURF') {
      flush()
      cur = { index: Number(parts[1]), label: '', type: 'STANDARD', curv: 0, disz: 0, conic: 0, parm: [], glass: null, diam: null, floating: false, line: lineNo }
      return
    }
    if (cur) {
      switch (key) {
        case 'TYPE':
          cur.type = parts[1] ?? 'STANDARD'
          return
        case 'CURV':
          cur.curv = Number(parts[1]) || 0
          if (Number(parts[3]) > 0) counts.set('pickup', (counts.get('pickup') ?? 0) + 1)
          return
        case 'DISZ':
          cur.disz = /^inf/i.test(parts[1] ?? '') || Number(parts[1]) >= 1e10 ? 'inf' : Number(parts[1]) || 0
          return
        case 'CONI':
          cur.conic = Number(parts[1]) || 0
          return
        case 'GLAS':
          cur.glass = { name: parts[1] ?? '', nd: parts[4] !== undefined ? Number(parts[4]) : null, vd: parts[5] !== undefined ? Number(parts[5]) : null, line: lineNo }
          return
        case 'STOP':
          stop = surfaces.length
          return
        case 'DIAM':
          cur.diam = Number(parts[1]) || null
          return
        case 'FLAP':
          cur.floating = true
          return
        case 'COMM':
          cur.label = rest
          return
        case 'PARM': {
          const k = Number(parts[1])
          if (Number.isInteger(k) && k >= 1 && k <= 20) {
            cur.parm[k - 1] = Number(parts[2]) || 0
            return
          }
          warn('malformed', lineNo, surfaces.length, `PARM record not understood: ${line}`)
          return
        }
        case 'MIRR':
        case 'HIDE':
        case 'SLAB':
        case 'PZUP':
        case 'SQAP':
          return
        case 'CLAP':
        case 'ELAP':
          warn('approximated', lineNo, surfaces.length, `${key} aperture: the mock engine computes the clear aperture`)
          return
        case 'XDAT':
          warn('unsupported_record', lineNo, surfaces.length, `${key} extra data is dropped`)
          return
        default:
          break
      }
    }
    switch (key) {
      case 'NAME':
        d.name = rest
        return
      case 'NOTE':
        d.notes = rest.replace(/^\d+\s+/, '')
        return
      case 'UNIT':
        d.unit = LENGTH_SCALE[(parts[1] ?? 'MM').toUpperCase()] ?? 1
        return
      case 'ENPD':
        d.aperture = { type: 'entrance_pupil_diameter', value: (Number(parts[1]) || 10) * d.unit }
        return
      case 'FNUM':
        d.aperture = { type: 'image_space_f_number', value: Number(parts[1]) || 4 }
        return
      case 'OBNA':
        d.aperture = { type: 'object_space_na', value: Number(parts[1]) || 0.1 }
        return
      case 'FLOA':
        d.aperture = { type: 'float_by_stop_size' }
        return
      case 'FTYP': {
        const t = Number(parts[1])
        d.fieldKind = t === 1 ? 'object_height' : t === 2 ? 'paraxial_image_height' : t === 3 ? 'real_image_height' : 'object_angle'
        if (t === 4) warn('approximated', lineNo, null, 'theodolite fields are read as object angles')
        const n = Number(parts[3])
        if (Number.isInteger(n) && n > 0) d.nFields = n
        return
      }
      case 'XFLN':
        d.fieldX = numbers(parts.slice(1))
        return
      case 'YFLN':
        d.fieldY = numbers(parts.slice(1))
        return
      case 'FLDS': {
        // FLDS n x1 y1 x2 y2 …: a compact field list.
        const v = numbers(parts.slice(1))
        const n = v[0]
        for (let i = 0; i < n; i++) {
          d.fieldX[i] = v[1 + 2 * i] ?? 0
          d.fieldY[i] = v[2 + 2 * i] ?? 0
        }
        d.nFields = n
        return
      }
      case 'FWGT':
      case 'FWGN':
        d.fieldW = numbers(parts.slice(1))
        return
      case 'WAVM': {
        const um = Number(parts[2])
        if (Number.isFinite(um) && um > 0) d.wavelengths[Number(parts[1]) - 1] = { um, weight: Number(parts[3]) || 1 }
        return
      }
      case 'PWAV':
        primaryN = Number(parts[1])
        return
      case 'RAIM': {
        const m = Number(parts[2])
        d.rayAiming = m === 2 || m === 1 ? 'real' : 'off'
        if (m === 1) warn('approximated', lineNo, null, 'paraxial ray aiming is read as real ray aiming')
        return
      }
      case 'ENVD':
        d.temperature = Number(parts[1]) || 20
        d.pressure = Number(parts[2]) || 1
        return
      default:
        break
    }
    if (ZMX_IGNORED.has(key)) return
    if (/^(MNUM|MOFF|TOL|CONF)/.test(key)) {
      counts.set(key, (counts.get(key) ?? 0) + 1)
      return
    }
    warn('unsupported_record', lineNo, null, `record ${key} is not mapped and was dropped`)
  })
  flush()
  d.wavelengths = d.wavelengths.filter(Boolean)
  const nW = d.wavelengths.length
  d.primary = primaryN !== null && primaryN >= 1 && primaryN <= nW ? primaryN - 1 : null
  for (const [k, n] of counts) warn('unsupported_record', null, null, `${n} ${k} record${n > 1 ? 's' : ''} dropped (multi-configuration and tolerance data are not imported)`)
  const system = finish(d, surfaces, stop, solves, warn)
  return { format: 'zmx', encoding: '', system, warnings }
}

// --- SEQ (CODE V) -----------------------------------------------------------------

const SEQ_ASPH = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J']
const SEQ_IGNORED = new Set(['LEN', 'NEW', 'RDM', 'CUM', 'GO', 'END', 'OUT', 'ASP', 'WTF', 'WTW', 'REF', 'XAN', 'YAN', 'XOB', 'YOB', 'XIM', 'YIM', 'XRI', 'YRI', 'SO', 'S', 'SI'])

export function parseSeq(text: string): PrescriptionImport {
  const warnings: ImportWarning[] = []
  const warn: Warn = (kind, line, surf, message) => warnings.push({ kind, line, surface: surf, message })
  const d = newDraft()
  const surfaces: Obj[] = []
  const solves: Obj[] = []
  let stop: number | null = null
  let modeCurvature = false
  const asph: number[][] = []
  const conics: number[] = []
  const curvs: number[] = []
  const flagged = new Map<string, number>()
  const flagLine = new Map<string, number>()
  const lines = text.replace(/\r/g, '').split('\n')
  let last = -1
  let hasObject = false
  const target = (q: string | undefined): number => {
    if (q === undefined) return last
    const m = /^S(\d+)$/i.exec(q)
    if (m) return Number(m[1])
    if (/^SO$/i.test(q)) return 0
    if (/^SI$/i.test(q)) return surfaces.length - 1
    return last
  }
  const ensure = (i: number) => {
    while (asph.length <= i) {
      asph.push([])
      conics.push(0)
    }
  }
  lines.forEach((rawLine, li) => {
    const lineNo = li + 1
    const noComment = rawLine.split('!')[0]
    for (const cmdRaw of noComment.split(';')) {
      const cmd = cmdRaw.trim()
      if (!cmd) continue
      const m = /^('[^']*'|"[^"]*"|\S+)\s*(.*)$/.exec(cmd)
      if (!m) continue
      const head = m[1].toUpperCase()
      const args = m[2].match(/'[^']*'|"[^"]*"|\S+/g) ?? []
      if (head === 'SO' || head === 'S' || head === 'SI') {
        const kind = head
        const a = args.slice(0)
        const rNum = Number(a[0])
        const tNum = Number(a[1])
        if (!Number.isFinite(rNum) && a.length) {
          warn('malformed', lineNo, surfaces.length, `surface line not understood: ${cmd}`)
          continue
        }
        const idx = surfaces.length
        const radiusRaw = Number.isFinite(rNum) ? rNum : 0
        const curvature = modeCurvature ? radiusRaw * (1 / d.unit) : radiusRaw === 0 ? 0 : 1 / (radiusRaw * d.unit)
        const thick = kind === 'SI' ? 0 : Number.isFinite(tNum) ? tNum : 0
        const thickness: number | 'inf' = kind === 'SO' && thick >= 1e10 ? 'inf' : thick * d.unit
        let glassName = a[2] ?? ''
        glassName = glassName.replace(/^['"]|['"]$/g, '')
        const g = resolveGlass(glassName, null, null, lineNo, idx, warn)
        if (kind === 'SO') hasObject = true
        else if (!hasObject) {
          surfaces.push(surface(0, 'OBJ', { type: 'plane' }, 'inf', { type: 'air' }))
          asph.push([])
          conics.push(0)
          curvs.push(0)
          hasObject = true
        }
        const index = surfaces.length
        ensure(index)
        surfaces.push(surface(index, '', { type: 'plane' }, thickness, g.material, g.mirror ? 'reflect' : 'refract'))
        curvs[index] = curvature
        last = index
        if (kind === 'SO') surfaces[index].thickness = thickness
        continue
      }
      switch (head) {
        case 'TIT':
        case 'TITLE':
          d.name = (args[0] ?? '').replace(/^['"]|['"]$/g, '')
          break
        case 'DIM':
          d.unit = args[0]?.toUpperCase() === 'I' ? 25.4 : args[0]?.toUpperCase() === 'C' ? 10 : 1
          break
        case 'RDM':
          modeCurvature = false
          break
        case 'CUM':
          modeCurvature = true
          break
        case 'EPD':
          d.aperture = { type: 'entrance_pupil_diameter', value: Number(args[0]) * d.unit }
          break
        case 'FNO':
          d.aperture = { type: 'image_space_f_number', value: Number(args[0]) }
          break
        case 'NAO':
          d.aperture = { type: 'object_space_na', value: Number(args[0]) }
          break
        case 'NA':
          d.aperture = { type: 'paraxial_working_f_number', value: 1 / (2 * Number(args[0])) }
          warn('approximated', lineNo, null, 'image-space NA is read as the paraxial working F/# 1/(2 NA)')
          break
        case 'WL':
          d.wavelengths = numbers(args).map((nm, i) => ({ um: nm / 1000, weight: d.wavelengths[i]?.weight ?? 1 }))
          break
        case 'WTW':
          numbers(args).forEach((w, i) => {
            if (d.wavelengths[i]) d.wavelengths[i].weight = w
          })
          break
        case 'REF':
          d.primary = Number(args[0]) - 1
          break
        case 'YAN':
        case 'XAN':
        case 'YOB':
        case 'XOB':
        case 'YIM':
        case 'XIM':
        case 'YRI':
        case 'XRI': {
          const kind = head[1] === 'A' ? 'object_angle' : head[1] === 'O' ? 'object_height' : head[1] === 'I' ? 'paraxial_image_height' : 'real_image_height'
          d.fieldKind = kind
          if (head[0] === 'Y') d.fieldY = numbers(args)
          else d.fieldX = numbers(args)
          break
        }
        case 'WTF':
          d.fieldW = numbers(args)
          break
        case 'STO':
          stop = last
          break
        case 'REFL':
          if (surfaces[last]) surfaces[last].interaction = 'reflect'
          break
        case 'K': {
          const t = last
          ensure(t)
          conics[t] = Number(args[0]) || 0
          break
        }
        case 'RDY':
        case 'CUY': {
          const t = target(args.length > 1 ? args[0] : undefined)
          const v = Number(args[args.length - 1])
          if (surfaces[t]) curvs[t] = head === 'CUY' ? v / d.unit : v === 0 ? 0 : 1 / (v * d.unit)
          break
        }
        case 'THI': {
          const t = target(args.length > 1 ? args[0] : undefined)
          if (surfaces[t]) surfaces[t].thickness = Number(args[args.length - 1]) * d.unit
          break
        }
        case 'GLA': {
          const t = target(args.length > 1 ? args[0] : undefined)
          if (surfaces[t]) {
            const g = resolveGlass(args[args.length - 1] ?? '', null, null, lineNo, t, warn)
            surfaces[t].material = g.material
          }
          break
        }
        case 'PIM':
          solves.push({ target: { param: 'thickness', surface: -1 }, kind: { type: 'marginal_ray_height', height: 0 } })
          break
        case 'CCY':
        case 'THC':
        case 'CCX':
        case 'RMD':
        case 'CIR':
          if (head === 'CIR' || head === 'RMD') {
            warn('approximated', lineNo, last, `${head}: the mock engine computes clear apertures`)
          } else {
            flagged.set(head, (flagged.get(head) ?? 0) + 1)
            if (!flagLine.has(head)) flagLine.set(head, lineNo)
          }
          break
        default: {
          const ai = SEQ_ASPH.indexOf(head)
          if (ai >= 0 && args.length) {
            ensure(last)
            asph[last][ai] = (Number(args[0]) || 0) * d.unit ** -(3 + 2 * ai)
            break
          }
          if (SEQ_IGNORED.has(head)) break
          warn('unsupported_record', lineNo, null, `command ${head} is not mapped and was dropped`)
        }
      }
    }
  })
  for (const [k, n] of flagged) warn('solve_dropped', flagLine.get(k) ?? null, null, `${n} variable flag${n > 1 ? 's' : ''} (${k}) dropped`)
  // Shapes.
  surfaces.forEach((s, i) => {
    s.shape = makeShape(curvs[i] ?? 0, conics[i] ?? 0, [...(asph[i] ?? [])].map((v) => v ?? 0))
    if (s.shape.type === 'plane' && (conics[i] ?? 0) !== 0) s.shape = { type: 'conic', curvature: 0, conic: conics[i] }
  })
  const system = finish(d, surfaces, stop, solves, warn)
  return { format: 'seq', encoding: '', system, warnings }
}

export function importPrescription(data: ArrayBuffer, format: 'zmx' | 'seq'): PrescriptionImport {
  const { text, encoding } = decodeText(data)
  if (!text.trim()) throw new EngineError('failed', 'the file is empty')
  const r = format === 'zmx' ? parseZmx(text) : parseSeq(text)
  const sys = r.system as Obj
  for (const w of structural(sys)) r.warnings.push(w)
  return { ...r, encoding }
}

/** Cheap validation warnings (CONVENTIONS §13 subset). */
function structural(sys: Obj): ImportWarning[] {
  const out: ImportWarning[] = []
  const n = sys.surfaces.length
  if (n < 3) out.push({ kind: 'validation', line: null, surface: null, message: 'a system needs an object, at least one surface and an image' })
  return out
}

// --- AGF catalogs -----------------------------------------------------------------

export function parseAgf(fileName: string, text: string): { name: string; glasses: Record<string, { nd: number; vd: number; status: string }> } {
  const base = fileName.split(/[\\/]/).pop() ?? fileName
  const name = base.replace(/\.[^.]*$/, '').toUpperCase() || 'USER'
  const glasses: Record<string, { nd: number; vd: number; status: string }> = {}
  for (const raw of text.replace(/\r/g, '').split('\n')) {
    const line = raw.trim()
    if (!/^NM\s/i.test(line)) continue
    const p = line.split(/\s+/)
    const gname = p[1]?.replace(/^['"]|['"]$/g, '')
    const nd = Number(p[4])
    const vd = Number(p[5])
    if (!gname || !(nd > 1) || !Number.isFinite(vd)) continue
    const status = ['standard', 'preferred', 'obsolete', 'special', 'melt'][Number(p[7])] ?? 'standard'
    glasses[gname.toUpperCase()] = { nd, vd, status }
  }
  return { name, glasses }
}

export function loadAgf(fileName: string, text: string): CatalogInfo {
  const { name, glasses } = parseAgf(fileName, text)
  const names = Object.keys(glasses)
  if (!names.length) throw new EngineError('failed', 'no glasses found: not an AGF catalog')
  for (const [g, v] of Object.entries(glasses)) CATALOG[g] = { nd: v.nd, vd: v.vd, catalog: name, status: v.status }
  return { name, source: 'user', glass_count: names.length }
}

export { BUILTIN_CATALOG }
