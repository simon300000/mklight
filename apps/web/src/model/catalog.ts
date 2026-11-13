// Analysis titles and categories, as the macOS app shows them
// (apps/macos/Sources/MKLightUI/ViewModels/AnalysisCatalog.swift).

import type { AnalysisDescriptor } from '@/engine/types'

const TITLES: Record<string, string> = {
  first_order: 'First-Order Data', system_summary: 'System Data', validate: 'Validation',
  spot_diagram: 'Spot Diagram', ray_fan: 'Ray Fan', opd_fan: 'OPD Fan',
  field_curvature_distortion: 'Field Curvature & Distortion', grid_distortion: 'Grid Distortion',
  seidel: 'Seidel Coefficients', longitudinal_aberration: 'Longitudinal Aberration',
  chromatic_focal_shift: 'Chromatic Focal Shift', lateral_colour: 'Lateral Color',
  relative_illumination: 'Relative Illumination', footprint: 'Footprint', encircled_energy: 'Encircled Energy',
  geometric_mtf: 'Geometric MTF', best_focus: 'Best Focus', ray_grid: 'Ray Grid', layout: 'Layout',
  wavefront_map: 'Wavefront Map', psf: 'PSF', mtf: 'MTF', mtf_vs_field: 'MTF vs Field',
  through_focus_mtf: 'Through-Focus MTF', diffraction_encircled_energy: 'Diffraction Encircled Energy',
  line_spread: 'Line Spread', ghost_paths: 'Ghost Paths', ghost_irradiance: 'Ghost Irradiance',
  stray_light: 'Stray Light', polarization_pupil_map: 'Polarization Pupil Map',
  system_transmission: 'System Transmission', glass_info: 'Glass Data', solves: 'Solves',
  optimize: 'Optimize', tolerance: 'Tolerance', rng_stream: 'Random Stream',
  pupil_aberration: 'Pupil Aberration', relative_illumination_map: 'Relative Illumination Map',
  chief_ray_angle: 'Chief Ray Angle', glass_map: 'Glass Map', image_simulation: 'Image Simulation',
  glass_substitution: 'Glass Substitution',
}

const SHORT: Record<string, string> = {
  first_order: 'First Order', system_summary: 'System', spot_diagram: 'Spot',
  field_curvature_distortion: 'Field Curv. & Dist.', grid_distortion: 'Grid Dist.', seidel: 'Seidel',
  longitudinal_aberration: 'Long. Aberration', chromatic_focal_shift: 'Focal Shift',
  relative_illumination: 'Rel. Illumination', encircled_energy: 'Encircled En.', geometric_mtf: 'Geom. MTF',
  wavefront_map: 'Wavefront', through_focus_mtf: 'Thru-Focus MTF', diffraction_encircled_energy: 'Diffr. Enc. En.',
  ghost_irradiance: 'Ghost Irrad.', polarization_pupil_map: 'Pol. Pupil Map', system_transmission: 'Transmission',
  pupil_aberration: 'Pupil Aberr.', relative_illumination_map: 'RI Map', chief_ray_angle: 'CRA',
  image_simulation: 'Image Sim.', glass_substitution: 'Glass Subst.',
}

const CATEGORY_TITLES: Record<string, string> = {
  model: 'System', materials: 'Materials', geometric: 'Geometric', wave: 'Wave Optics',
  radiometry: 'Radiometry', polarization: 'Polarization', ghost: 'Ghosts', stray: 'Stray Light',
  optimization: 'Design', diagnostic: 'Diagnostics',
}
const CATEGORY_ORDER = ['geometric', 'wave', 'model', 'radiometry', 'polarization', 'ghost', 'stray', 'materials']

/** Analyses that are design jobs or internal; not offered as panes. */
const NOT_PANES = new Set(['convert_spec', 'optimize', 'tolerance', 'glass_substitution', 'merit_evaluation', 'rng_stream', 'layout'])

const SMALL = new Set(['a', 'an', 'and', 'at', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'vs'])
function titleCase(s: string) {
  return s
    .split(' ')
    .map((w, k) => (k > 0 && SMALL.has(w.toLowerCase()) ? w.toLowerCase() : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(' ')
}

export function analysisTitle(name: string, d?: AnalysisDescriptor): string {
  return TITLES[name] ?? (d ? titleCase(d.title) : titleCase(name.replace(/_/g, ' ')))
}
export const analysisShortTitle = (name: string) => SHORT[name] ?? analysisTitle(name)
export const categoryTitle = (c: string) => CATEGORY_TITLES[c] ?? c.charAt(0).toUpperCase() + c.slice(1)

export function groupedAnalyses(list: AnalysisDescriptor[]): { category: string; items: AnalysisDescriptor[] }[] {
  const groups = new Map<string, AnalysisDescriptor[]>()
  for (const d of list) {
    if (NOT_PANES.has(d.name) || d.category === 'diagnostic' || d.category === 'optimization') continue
    groups.set(d.category, [...(groups.get(d.category) ?? []), d])
  }
  const order = [...CATEGORY_ORDER, ...[...groups.keys()].filter((c) => !CATEGORY_ORDER.includes(c)).sort()]
  return order
    .filter((c) => groups.has(c))
    .map((c) => ({ category: c, items: groups.get(c)!.sort((a, b) => analysisTitle(a.name).localeCompare(analysisTitle(b.name))) }))
}
