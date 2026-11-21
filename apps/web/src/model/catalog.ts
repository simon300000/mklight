// Analysis titles and categories, as the macOS app shows them
// (apps/macos/Sources/MKLightUI/ViewModels/AnalysisCatalog.swift). The titles
// are in the `analysis` catalog (titles, shortTitles, categories), by engine name.

import type { AnalysisDescriptor } from '@/engine/types'
import { currentLanguage, engineText, tMaybe } from '@/i18n'

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
  const own = tMaybe(`analysis.titles.${name}`)
  if (own !== undefined) return own
  if (!d) return titleCase(name.replace(/_/g, ' '))
  const translated = engineText(d.title)
  return translated !== d.title ? translated : titleCase(d.title)
}
export const analysisShortTitle = (name: string) => tMaybe(`analysis.shortTitles.${name}`) ?? analysisTitle(name)
export const categoryTitle = (c: string) => tMaybe(`analysis.categories.${c}`) ?? engineText(c.charAt(0).toUpperCase() + c.slice(1))

export function groupedAnalyses(list: AnalysisDescriptor[]): { category: string; items: AnalysisDescriptor[] }[] {
  const groups = new Map<string, AnalysisDescriptor[]>()
  for (const d of list) {
    if (NOT_PANES.has(d.name) || d.category === 'diagnostic' || d.category === 'optimization') continue
    groups.set(d.category, [...(groups.get(d.category) ?? []), d])
  }
  const order = [...CATEGORY_ORDER, ...[...groups.keys()].filter((c) => !CATEGORY_ORDER.includes(c)).sort()]
  return order
    .filter((c) => groups.has(c))
    .map((c) => ({ category: c, items: groups.get(c)!.sort((a, b) => analysisTitle(a.name).localeCompare(analysisTitle(b.name), currentLanguage())) }))
}
