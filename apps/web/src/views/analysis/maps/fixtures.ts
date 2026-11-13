// Hand-made results shaped by docs/RESULTS.md for the map views (shared by
// maps.test.tsx and results.test.tsx).
import type { AnalysisResult, JsonObject, TypedArray } from '@/engine/types'
import type { System } from '@/model/system'

export const system = {
  name: 't',
  notes: '',
  aperture: { type: 'entrance_pupil_diameter', value: 10 },
  fields: { kind: 'object_angle', fields: [{ x: 0, y: 0, weight: 1 }, { x: 0, y: 10, weight: 1 }, { x: 0, y: 14, weight: 1 }] },
  wavelengths: { wavelengths: [{ um: 0.5876, weight: 1 }], primary: 0 },
  surfaces: [],
  stop: 1,
} as unknown as System

function result(analysis: string, json: JsonObject, buffers: Record<string, TypedArray>, settings: JsonObject = {}, shapes?: Record<string, number[]>): AnalysisResult {
  return { json, buffers, shapes, runInfo: { snapshot_hash: 'h', analysis, settings, backend: 'cpu_reference', warnings: [] } }
}

// MARK: Fixtures shaped by docs/RESULTS.md §8

function fieldMap(analysis: string, columns: { name: string; unit: string }[], f: (hx: number, hy: number, c: number) => number, nx = 5, ny = 5, settings: JsonObject = {}) {
  const fc = new Float64Array(nx * ny * 2), img = new Float64Array(nx * ny * 2), vals = new Float64Array(nx * ny * columns.length)
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i
      const hx = nx > 1 ? -1 + (2 * i) / (nx - 1) : 0, hy = ny > 1 ? -1 + (2 * j) / (ny - 1) : 0
      fc.set([hx, hy], 2 * k)
      const out = Math.hypot(hx, hy) > 1 + 1e-9 && analysis === 'relative_illumination_map'
      img.set(out ? [NaN, NaN] : [hx * 12, hy * 12], 2 * k)
      columns.forEach((_, c) => (vals[k * columns.length + c] = out ? NaN : f(hx, hy, c)))
    }
  return result(
    analysis,
    { columns: columns.map((c) => ({ ...c, definition: `${c.name} definition` })), field_coordinates: 'field_coordinates', image_mm: 'image_mm', values: 'values', shape: [nx, ny], wavelength: 0 },
    { field_coordinates: fc, image_mm: img, values: vals },
    settings,
  )
}

export const pupilAberration = fieldMap('pupil_aberration', [{ name: 'stop_error_x', unit: 'mm' }, { name: 'stop_error_y', unit: 'mm' }, { name: 'variation_x', unit: '1' }, { name: 'variation_y', unit: '1' }], (x, y, c) => [0.01 * x, 0.012 * y, 0.02 * x * x, -0.01 * y * y][c])
export const riMap = fieldMap('relative_illumination_map', [{ name: 'relative_illumination', unit: '1' }, { name: 'vignetting_fraction', unit: '1' }], (x, y, c) => (c === 0 ? 1 - 0.13 * (x * x + y * y) : 0.05 * (x * x + y * y)), 9, 9)
export const cra = fieldMap('chief_ray_angle', [{ name: 'incidence', unit: 'deg' }, { name: 'limit', unit: 'deg' }, { name: 'margin', unit: 'deg' }, { name: 'angle_x', unit: 'deg' }], (x, y, c) => {
  const r = Math.hypot(x, y) * 12
  return [r * 2, 20, 20 - r * 2, x * 20][c]
}, 5, 5, { sensor_limit: [[0, 25], [12, 20]] })
export const craSweep = fieldMap('chief_ray_angle', [{ name: 'incidence', unit: 'deg' }, { name: 'margin', unit: 'deg' }], (_x, y, c) => (c === 0 ? Math.abs(y) * 24 : 20 - Math.abs(y) * 24), 1, 11, { sensor_limit: [[0, 25], [12, 18]] })

const N = 37
const pupil = new Float64Array(2 * N), inten = new Float64Array(N), diat = new Float64Array(N), ret = new Float64Array(N), reached = new Uint8Array(N)
for (let k = 0; k < N; k++) {
  const a = (k / N) * 2 * Math.PI, r = Math.sqrt(k / N)
  pupil.set([r * Math.cos(a), r * Math.sin(a)], 2 * k)
  inten[k] = 0.9 - 0.1 * r
  diat[k] = 0.01 * r
  ret[k] = 0.02 * r * r
  reached[k] = k === 5 ? 0 : 1
}
export const polarization = result(
  'polarization_pupil_map',
  { field: { hx: 0, hy: 1, vignetting: 0 }, grid: 7, max_diattenuation: 0.01, max_retardance: 0.02, mean_intensity: 0.86, rms_retardance: 'nan', sample_count: N, undefined_retarder: 0, wavelength: 0, wavelength_um: 0.5876, pupil: 'pupil' },
  { pupil, intensity: inten, diattenuation: diat, retardance: ret, reached },
)

export const ghostPaths = result('ghost_paths', {
  ghosts: [
    { label: 'R4·R2', path: [4, 2], relative_irradiance: 2.1e-4, focus_from_sensor: -0.4, spot_diameter: 0.3, reflectance_product: 1.6e-3, transmittance: 0.9, vignetting: 1, image_height: 0.1, near_focus: true },
    { label: 'R6·R1', path: [6, 1], relative_irradiance: 'nan', focus_from_sensor: 'inf', spot_diameter: 12, reflectance_product: 1e-3, transmittance: 0.9, vignetting: 0.5, image_height: 2, near_focus: false },
  ],
  truncated: true,
  signal_diameter: 0.01,
  wavelength: 0,
}, {})

const map16 = (f: (i: number, j: number) => number, n = 16) => Float64Array.from({ length: n * n }, (_, k) => f(k % n, Math.floor(k / n)))
export const ghostIrr = result(
  'ghost_irradiance',
  {
    ghosts: [{ label: 'R4·R2', flux: 1e-4, flux_ratio: 1e-4, peak_ratio: null, map_index: 0, fields: [] }],
    map: { bins: [16, 16], half_size: [3, 2] },
    primary: { label: 'primary', flux: 1, fields: [] },
    total_flux_ratio: 1e-4,
    total_peak_ratio: null,
    truncated: false,
    coverage: {},
  },
  { primary_map: map16((i, j) => Math.exp(-((i - 8) ** 2 + (j - 8) ** 2))), total_ghost_map: map16((i, j) => 1e-4 * Math.exp(-((i - 5) ** 2 + (j - 9) ** 2) / 20)), path_maps: map16(() => 1e-5) },
  {},
  { primary_map: [16, 16], total_ghost_map: [16, 16], path_maps: [1, 16, 16] },
)

export const stray = result(
  'stray_light',
  {
    classes: 30,
    detector: { surface: 9, nx: 16, ny: 16, half_x: 4, half_y: 4, accept: 'front' },
    emitted: 1,
    map_classes: ['ghost', 'scatter'],
    stray_flux: 0.002,
    stray_fraction: 0.002,
    stray_std_error: null,
    totals: [
      { class: 'primary', flux: 0.9, included: false, hits: 100, std_error: 0.01 },
      { class: 'ghost', flux: 0.001, included: true, hits: 3, std_error: null },
    ],
    top_signatures: [],
    warnings: ['few hits'],
  },
  { total_irradiance: map16((i) => i * 1e-6), class_irradiance: Float64Array.from({ length: 2 * 256 }, (_, k) => k * 1e-8) },
)

const W = 24, H = 16
const simImage = Float64Array.from({ length: W * H }, (_, k) => (k % 7) / 6)
export const imageSim = result(
  'image_simulation',
  {
    channels: 1,
    field_grid: 2,
    field_mapping: [
      { field: [-0.1, 0.6], ideal_mm: [-1, 6], real_mm: [-1.01, 6.05], relative_illumination: 0.93, pupil_sampling: [] },
      { field: [0.1, 0.6], ideal_mm: [1, 6], real_mm: [1.01, 6.05], relative_illumination: 0.93, pupil_sampling: [] },
      { field: [-0.1, 0.8], ideal_mm: [-1, 8], real_mm: [-1.01, 8.1], relative_illumination: 0.88, pupil_sampling: [] },
      { field: [0.1, 0.8], ideal_mm: [1, 8], real_mm: [1.01, 8.1], relative_illumination: 'nan', pupil_sampling: [] },
    ],
    image: 'image',
    origin_mm: [-1, 6],
    pixel_size_mm: [2 / W, 2 / H],
    psf_size: 505,
    psf_spacing_mm: 0.00039,
    size: [W, H],
  },
  { image: simImage },
  { source: { kind: 'pattern', pattern: 'siemens_star' }, size: [W, H], optics: { field_center: [0, 0.7], field_extent: 0.1 } },
)

export const FIXTURES: [string, AnalysisResult][] = [
  ['pupil_aberration', pupilAberration],
  ['relative_illumination_map', riMap],
  ['chief_ray_angle', cra],
  ['chief_ray_angle', craSweep],
  ['polarization_pupil_map', polarization],
  ['ghost_paths', ghostPaths],
  ['ghost_irradiance', ghostIrr],
  ['stray_light', stray],
  ['image_simulation', imageSim],
]
