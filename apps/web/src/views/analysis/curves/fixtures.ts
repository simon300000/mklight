// Hand-made results shaped by docs/RESULTS.md for the curve and table views
// (shared by curves.test.tsx and results.test.tsx).
import type { AnalysisResult, JsonObject, TypedArray } from '@/engine/types'
import type { System } from '@/model/system'

export const system = {
  name: 'Test',
  notes: '',
  aperture: { type: 'entrance_pupil_diameter', value: 10 },
  fields: { kind: 'object_angle', fields: [{ x: 0, y: 0, weight: 1 }, { x: 0, y: 10, weight: 1 }] },
  wavelengths: { wavelengths: [{ um: 0.4861, weight: 1 }, { um: 0.5876, weight: 1 }, { um: 0.6563, weight: 1 }], primary: 1 },
  surfaces: [],
  stop: 1,
} as unknown as System

const f64 = (...v: number[]) => new Float64Array(v)
const lin = (n: number, a: number, b: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))

export function res(analysis: string, json: JsonObject, buffers: Record<string, TypedArray> = {}): AnalysisResult {
  return { json, buffers, runInfo: { snapshot_hash: 'h', analysis, settings: {}, backend: 'cpu_reference', warnings: [] } }
}

export const FIXTURES: Record<string, AnalysisResult> = {
  first_order: res('first_order', {
    entries: [
      { key: 'efl', label: 'Effective focal length', value: 50, unit: 'mm' },
      { key: 'working_f_number', label: 'Working F/#', value: 'inf', unit: '' },
      { key: 'bfl', label: 'Back focal length', value: 40.123, unit: 'mm' },
    ],
    wavelength_um: 0.5876,
    paraxial_xy: {
      primary_wavelength: 1,
      symmetry: 'meridional',
      squeeze_definition: 'Signed EFL_y / EFL_x',
      warnings: ['Nominal vertices'],
      wavelengths: [
        { wavelength: 0, wavelength_um: 0.4861, efl_x: 49, efl_y: 51, bfl_x: 39, bfl_y: 41, squeeze: 1.04, doe_power: [] },
        { wavelength: 1, wavelength_um: 0.5876, efl_x: 50, efl_y: 52, bfl_x: 40, bfl_y: 42, squeeze: 1.04, doe_power: [{ surface: 3, power: 0.001 }] },
      ],
    },
  }),
  system_summary: res('system_summary', {
    name: 'Doublet',
    aperture: { type: 'entrance_pupil_diameter', value: 10 },
    fields: { kind: 'object_angle', fields: [{ x: 0, y: 0 }] },
    environment: { temperature_c: 20, pressure_atm: 1 },
    first_to_image_distance: 55.2,
    object_at_infinity: true,
    stop: 1,
    symmetry: 'rotational',
    primary_wavelength: 1,
    wavelengths_um: [0.4861, 0.5876, 0.6563],
    warnings: [],
    surfaces: [
      { index: 0, label: 'Object', type: 'plane', radius: 'inf', thickness: 'inf', material: null, nd_after: 1, semi_diameter: { mode: 'auto' } },
      { index: 1, label: 'L1', type: 'sphere', radius: 61.2, thickness: 4, material: 'N-BK7', nd_after: 1.5168, vd_after: 64.17, is_stop: true, semi_diameter: { mode: 'fixed', value: 6 } },
      { index: 2, label: 'Image', type: 'plane', radius: 'inf', thickness: 0, material: null, nd_after: 1, vd_after: null, semi_diameter: { mode: 'traced', value: 2.1 } },
    ],
  }),
  validate: res('validate', {
    ok: false,
    errors: 1,
    warnings: 1,
    issues: [
      { code: 'w', message: 'Wavelength outside glass data', severity: 'warning', surface: 2 },
      { code: 'e', message: 'Material not found', severity: 'error', surface: null },
    ],
  }),
  seidel: res('seidel', {
    status: 'applicable',
    petzval_radius: -120.5,
    surfaces: [
      { surface: 1, s1: 0.01, s2: -0.002, s3: 0.003, s4: 0.004, s5: 'nan', cl: 0.001, ct: -0.0004 },
      { surface: 2, s1: -0.008, s2: 0.001, s3: -0.001, s4: -0.002, s5: 0.0001, cl: -0.0008, ct: 0.0002 },
    ],
    sums: { s1: 0.002, s2: -0.001, s3: 0.002, s4: 0.002, s5: 0.0001, cl: 0.0002, ct: -0.0002 },
    wavefront_waves: { w040: 0.12, w131: -0.03 },
  }),
  chromatic_focal_shift: res('chromatic_focal_shift', {
    wavelengths_um: lin(5, 0.45, 0.65),
    focal_shift: [0.02, 0.005, 0, 0.004, 'nan'],
    focal_shift_x: [0.02, 0.005, 0, 0.004, 0.01],
    primary_wavelength_um: 0.55,
    range: 0.02,
  }),
  longitudinal_aberration: res('longitudinal_aberration', {
    pupil: lin(5, 0, 1),
    primary_paraxial_focus: 0,
    curves: [
      { wavelength: 0, wavelength_um: 0.4861, axis: 'y', aberration: [0.01, 0.008, 0.004, -0.002, -0.01], focus: [], paraxial_focus: 0.01 },
      { wavelength: 1, wavelength_um: 0.5876, axis: 'y', aberration: [0, -0.001, -0.004, -0.008, 'nan'], focus: [], paraxial_focus: 0 },
    ],
  }),
  lateral_colour: res('lateral_colour', {
    airy_radius: 0.003,
    field_unit: '°',
    field_values: [0, 5, 10],
    normalized_field: [0, 0.5, 1],
    lateral_colour: [0, 0.001, -0.002],
    curves: [
      { wavelength: 0, wavelength_um: 0.4861, offset: [0, 0.001, 0.002] },
      { wavelength: 2, wavelength_um: 0.6563, offset: [0, -0.0005, -0.001] },
    ],
  }),
  relative_illumination: res('relative_illumination', {
    field_unit: '°',
    field_values: [0, 5, 10],
    relative_illumination: [1, 0.95, 0.82],
    vignetting_fraction: [0, 0.02, 0.1],
    cos4_object: [1, 0.97, 0.88],
  }),
  footprint: res(
    'footprint',
    {
      surface: 2,
      semi_diameter: 6,
      max_radius: 5.2,
      sets: [
        { field: 0, wavelength: 0, points: [0, 3], clipped: [0, 1], launched: 4, lost_before: 0 },
        { field: 0, wavelength: 1, points: [3, 2], clipped: [1, 0], launched: 2, lost_before: 0 },
      ],
    },
    { points: f64(0, 0, 1, 1, -2, 3, 4, 'x' as never, 0.5, -0.5), clipped: f64(6.1, 0) },
  ),
  grid_distortion: res(
    'grid_distortion',
    { grid_size: 2, max_abs_distortion_percent: 1.2346, wavelength: 1 },
    { paraxial: f64(-1, -1, 1, -1, -1, 1, 1, 1), real: f64(-1.01, -1.01, 1.01, -1.01, NaN, NaN, 1.01, 1.01), field: f64(-1, -1, 1, -1, -1, 1, 1, 1), distortion_percent: f64(1, 1, 1, 1) },
  ),
  ray_grid: res(
    'ray_grid',
    { arrived: 7, launched: 8, pupil_count: 2, fields: [0, 1], wavelengths: [0, 1], wavelengths_um: [0.4861, 0.5876], field_values: [[0, 0], [0, 10]] },
    {
      points: f64(...Array.from({ length: 16 }, (_, i) => (i === 5 ? NaN : i * 0.001))),
      chief: f64(0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1),
      pupil: f64(0, 0, 0, 1),
    },
  ),
  best_focus: res('best_focus', {
    defocus: -0.012,
    merit: 0.004,
    merit_at_zero: 0.006,
    mean_square_defocus: -0.011,
    converged: true,
    evaluations: 23,
    through_focus_defocus: lin(5, -0.05, 0.05),
    through_focus_merit: [0.01, 0.006, 0.004, 0.007, 0.012],
  }),
  encircled_energy: res('encircled_energy', {
    radius: lin(4, 0, 0.01),
    wavelengths: [0, 1, 2],
    fields: [
      { field: 0, field_value: [0, 0], encircled: [0, 0.5, 0.9, 1], ensquared: [0, 0.55, 0.92, 1], r50: 0.003, r80: 0.006, r90: 0.007, half_width80: 0.005, rays_arrived: 100 },
      { field: 1, field_value: [0, 10], encircled: [0, 0.3, 0.7, 1], ensquared: [0, 0.35, 0.75, 1], r50: 0.004, r80: 'nan', r90: 0.009, half_width80: 0.007, rays_arrived: 100 },
    ],
  }),
  diffraction_encircled_energy: res(
    'diffraction_encircled_energy',
    {
      radius_mm: 'radius_mm',
      fields: [{ field: 0, field_value: [0, 0], encircled: 'encircled_0', ensquared: 'ensquared_0', diffraction_limit: 'diffraction_limit_0', r50_mm: 0.002, r80_mm: 0.004, r90_mm: 0.006, grid_size: 64, spacing_mm: 0.001 }],
    },
    { radius_mm: f64(0, 0.002, 0.004, 0.008), encircled_0: f64(0, 0.5, 0.8, 0.95), ensquared_0: f64(0, 0.55, 0.85, 0.97), diffraction_limit_0: f64(0, 0.6, 0.85, 0.97) },
  ),
  geometric_mtf: res('geometric_mtf', {
    frequencies: lin(4, 0, 60),
    cutoff_frequency: 340,
    diffraction_limit: null,
    field_values: [[0, 0], [0, 10]],
    curves: [
      { field: 0, orientation: 'tangential', modulation: [1, 0.8, 0.6, 0.4] },
      { field: 0, orientation: 'sagittal', modulation: [1, 0.82, 0.62, 'nan'] },
      { field: 1, orientation: 'tangential', modulation: [1, 0.7, 0.5, 0.3] },
      { field: 1, orientation: 'sagittal', modulation: [1, 0.75, 0.55, 0.35] },
    ],
  }),
  mtf_vs_field: res(
    'mtf_vs_field',
    { normalized_field: 'normalized_field', field_values: 'field_values', frequencies: 'frequencies', curves: [{ frequency: 10, tangential: 'tangential_0', sagittal: 'sagittal_0' }, { frequency: 30, tangential: 'tangential_1', sagittal: 'sagittal_1' }], pupil_sampling: [] },
    { normalized_field: f64(0, 0.5, 1), frequencies: f64(10, 30), tangential_0: f64(0.9, 0.85, 0.7), sagittal_0: f64(0.9, 0.88, NaN), tangential_1: f64(0.6, 0.5, 0.3), sagittal_1: f64(0.6, 0.55, 0.4) },
  ),
  through_focus_mtf: res(
    'through_focus_mtf',
    {
      focus_mm: 'focus_mm',
      frequencies: 'frequencies',
      fields: [
        { field: 0, field_value: [0, 0], curves: [{ frequency: 20, tangential: 't00', sagittal: 's00' }] },
        { field: 1, field_value: [0, 10], curves: [{ frequency: 20, tangential: 't10', sagittal: 's10' }] },
      ],
    },
    { focus_mm: f64(-0.02, 0, 0.02), frequencies: f64(20), t00: f64(0.5, 0.8, 0.6), s00: f64(0.5, 0.79, 0.6), t10: f64(0.4, 0.7, 0.5), s10: f64(0.4, 0.6, 0.5) },
  ),
  line_spread: res(
    'line_spread',
    { fields: [{ field: 0, field_value: [0, 0], positions_mm: 'positions_mm_0', x: 'x_0', y: 'y_0', pupil_sampling: [] }] },
    { positions_mm_0: f64(-0.01, 0, 0.01), x_0: f64(1, 100, 1), y_0: f64(2, 90, 2) },
  ),
  system_transmission: res('system_transmission', {
    fields: [{ hx: 0, hy: 0 }, { hx: 0, hy: 1 }],
    wavelengths: [0, 1],
    polychromatic: [0.92, 0.85],
    points: [
      { field: 0, wavelength: 0, wavelength_um: 0.4861, transmission: 0.9, mean_unvignetted: 0.9, unvignetted_fraction: 1, samples: 10, failed: 0 },
      { field: 0, wavelength: 1, wavelength_um: 0.5876, transmission: 0.93, mean_unvignetted: 0.93, unvignetted_fraction: 1, samples: 10, failed: 0 },
      { field: 1, wavelength: 0, wavelength_um: 0.4861, transmission: 0.84, mean_unvignetted: 0.9, unvignetted_fraction: 0.93, samples: 10, failed: 0 },
      { field: 1, wavelength: 1, wavelength_um: 0.5876, transmission: 0.86, mean_unvignetted: 0.92, unvignetted_fraction: 0.93, samples: 10, failed: 0 },
    ],
  }),
}
