import {
  Activity,
  ArrowDownRight,
  ArrowLeftRight,
  BarChart3,
  Circle,
  CircleDashed,
  CircleDot,
  Droplet,
  Grid3x3,
  Image,
  Lightbulb,
  LineChart,
  ListChecks,
  Rainbow,
  ScatterChart,
  Sigma,
  Sparkle,
  Sun,
  Table,
  TrendingDown,
  Waves,
  type LucideIcon,
} from 'lucide-react'

const ICONS: Record<string, LucideIcon> = {
  first_order: ListChecks, system_summary: Table, validate: ListChecks, spot_diagram: CircleDot,
  ray_fan: LineChart, opd_fan: Activity, field_curvature_distortion: TrendingDown, grid_distortion: Grid3x3,
  seidel: BarChart3, longitudinal_aberration: ArrowLeftRight, chromatic_focal_shift: Rainbow, lateral_colour: Rainbow,
  relative_illumination: Sun, footprint: Circle, encircled_energy: Circle, geometric_mtf: TrendingDown,
  best_focus: CircleDashed, ray_grid: Grid3x3, wavefront_map: Waves, psf: Sparkle, mtf: TrendingDown,
  mtf_vs_field: LineChart, through_focus_mtf: LineChart, diffraction_encircled_energy: Circle, line_spread: Activity,
  ghost_paths: ArrowLeftRight, ghost_irradiance: Sparkle, stray_light: Lightbulb, polarization_pupil_map: Grid3x3,
  system_transmission: ArrowDownRight, glass_info: Droplet, solves: Sigma, pupil_aberration: CircleDashed,
  relative_illumination_map: Sun, chief_ray_angle: ArrowDownRight, glass_map: ScatterChart, image_simulation: Image,
}

export function AnalysisIcon({ name, className }: { name: string; className?: string }) {
  const I = ICONS[name] ?? ScatterChart
  return <I className={className} aria-hidden />
}
