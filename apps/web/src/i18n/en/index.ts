// The English UI catalog: the source language and the fallback. One file per
// area of the app; each has a zh-Hans twin with the same keys.
import type { Localized } from '../types'
import { analysis } from './analysis'
import { app } from './app'
import { common } from './common'
import { glass } from './glass'
import { inspector } from './inspector'
import { lens } from './lens'
import { maps } from './maps'
import { optimize } from './optimize'
import { tolerance } from './tolerance'

export const en = { common, app, lens, inspector, analysis, maps, glass, optimize, tolerance }

export type Catalog = typeof en
/** What another language must provide: every key of `en`, nothing else. */
export type Translation = Localized<Catalog>
