// 简体中文. Loaded on demand (a separate chunk), so English users never
// download it. Terms follow locales/zh-Hans/GLOSSARY.md.
import type { Translation } from '../en'
import { analysis } from './analysis'
import { app } from './app'
import { common } from './common'
import { glass } from './glass'
import { inspector } from './inspector'
import { lens } from './lens'
import { maps } from './maps'
import { optimize } from './optimize'
import { tolerance } from './tolerance'

export const zhHans: Translation = { common, app, lens, inspector, analysis, maps, glass, optimize, tolerance }
