import type { Renderers } from '../registry'
import { acceptsFieldMap } from './fieldMap'
import { ChiefRayAngleView, PupilAberrationView, RelativeIlluminationMapView, fieldMapKeyNumbers } from './FieldMaps'
import { ImageSimulationView, imageSimulationKeyNumbers } from './ImageSimulation'
import {
  GhostIrradianceView,
  GhostPathsView,
  PolarizationView,
  StrayLightView,
  ghostIrradianceKeyNumbers,
  ghostPathsKeyNumbers,
  polarizationKeyNumbers,
  strayLightKeyNumbers,
} from './Radiometry'

/** Field maps, polarisation, ghosts, stray light and image simulation. */
export const MAP_RENDERERS: Renderers = {
  pupil_aberration: { accepts: acceptsFieldMap, View: PupilAberrationView, keyNumbers: fieldMapKeyNumbers('pupil_aberration') },
  relative_illumination_map: { accepts: acceptsFieldMap, View: RelativeIlluminationMapView, keyNumbers: fieldMapKeyNumbers('relative_illumination_map') },
  chief_ray_angle: { accepts: acceptsFieldMap, View: ChiefRayAngleView, keyNumbers: fieldMapKeyNumbers('chief_ray_angle') },
  polarization_pupil_map: { accepts: (j) => typeof j.pupil === 'string' || 'mean_intensity' in j, View: PolarizationView, keyNumbers: polarizationKeyNumbers },
  ghost_paths: { accepts: (j) => Array.isArray(j.ghosts), View: GhostPathsView, keyNumbers: ghostPathsKeyNumbers },
  ghost_irradiance: { accepts: (j) => Array.isArray(j.ghosts) || typeof j.map === 'object', View: GhostIrradianceView, keyNumbers: ghostIrradianceKeyNumbers },
  stray_light: { accepts: (j) => Array.isArray(j.totals) || typeof j.detector === 'object', View: StrayLightView, keyNumbers: strayLightKeyNumbers },
  image_simulation: { accepts: (j) => typeof j.image === 'string' || Array.isArray(j.size), View: ImageSimulationView, keyNumbers: imageSimulationKeyNumbers },
}
