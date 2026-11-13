import type { Renderers } from '../registry'
import { GEOMETRIC_RENDERERS } from './geometric'
import { TABLE_RENDERERS } from './tables'
import { WAVE_RENDERERS } from './wave'

/** Tables and curve plots (first order … system transmission). */
export const CURVE_RENDERERS: Renderers = { ...TABLE_RENDERERS, ...GEOMETRIC_RENDERERS, ...WAVE_RENDERERS }
