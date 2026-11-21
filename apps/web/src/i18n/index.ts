// UI languages (docs/WEB.md "Languages"): English, the source, and 简体中文.
//
//   t('app.menu.file')                     a string in the current language
//   t('common.errors', { count: 3 })       `{name}` placeholders; `count` picks a plural form
//   const t = useT()                       in components: re-renders on a language change
//   engineText(s), engineEnum(v), engineIssue(…)  text the engine writes (./engine.ts)
//
// A language's catalog (and its engine catalog) loads on demand; switching is
// live. The choice ("System", or a language) is kept per browser.

import { createElement, Fragment, useMemo, type ReactNode } from 'react'
import { create } from 'zustand'
import { setEngineCatalog, loadEngineCatalog, type EngineCatalog } from './engine'
import { en, type Catalog, type Translation } from './en'
import { fill } from './fill'
import type { Params, Paths, Plural } from './types'

export { engineEnum, engineIssue, engineText, parseStructuralIssue } from './engine'
export type { Params, Plural } from './types'

export type Lang = 'en' | 'zh-Hans'
export type LanguageChoice = 'system' | Lang
export type Key = Paths<Catalog>

/** The languages, each named in itself (the picker shows them so). */
export const LANGUAGES: { value: Lang; name: string }[] = [
  { value: 'en', name: 'English' },
  { value: 'zh-Hans', name: '简体中文' },
]

// MARK: Choosing the language

/** The UI language for the browser's preferred languages: the first that is
 *  English or Simplified Chinese (zh, zh-CN, zh-SG, zh-Hans…). Traditional
 *  Chinese (zh-TW, zh-HK, zh-MO, zh-Hant…) stays English for now. */
export function systemLanguage(langs: readonly string[] = typeof navigator !== 'undefined' ? navigator.languages ?? [navigator.language] : []): Lang {
  for (const tag of langs) {
    const parts = tag.toLowerCase().split(/[-_]/)
    if (parts[0] === 'en') return 'en'
    if (parts[0] !== 'zh') continue
    if (parts.includes('hans')) return 'zh-Hans'
    if (parts.includes('hant') || parts.some((p) => p === 'tw' || p === 'hk' || p === 'mo')) return 'en'
    return 'zh-Hans'
  }
  return 'en'
}

const STORAGE = 'mklight.language'

function storedChoice(): LanguageChoice {
  try {
    const v = localStorage.getItem(STORAGE)
    return v === 'en' || v === 'zh-Hans' ? v : 'system'
  } catch {
    return 'system'
  }
}

function storeChoice(c: LanguageChoice) {
  try {
    if (c === 'system') localStorage.removeItem(STORAGE)
    else localStorage.setItem(STORAGE, c)
  } catch {
    // Private windows may refuse storage; the choice holds for the session.
  }
}

export const resolveChoice = (c: LanguageChoice): Lang => (c === 'system' ? systemLanguage() : c)

// MARK: Catalogs

interface Loaded {
  ui: Translation
  engine: EngineCatalog | null
}

const loaded: Partial<Record<Lang, Loaded>> = { en: { ui: en, engine: null } }
let current: Translation = en
let currentLang: Lang = 'en'

async function load(lang: Lang): Promise<Loaded> {
  const have = loaded[lang]
  if (have) return have
  const [{ zhHans }, engine] = await Promise.all([import('./zh-Hans'), loadEngineCatalog(lang)])
  return (loaded[lang] = { ui: zhHans, engine })
}

function activate(lang: Lang) {
  const l = loaded[lang]!
  current = l.ui
  currentLang = lang
  setEngineCatalog(l.engine)
  if (typeof document !== 'undefined') document.documentElement.lang = lang
}

interface I18nState {
  /** What the user picked: the system's language or a fixed one. */
  choice: LanguageChoice
  /** The language shown (its catalog is loaded). */
  lang: Lang
  setChoice(c: LanguageChoice): Promise<void>
}

let request = 0

/** Shows `lang` once its catalog is loaded; a later request wins. */
async function show(lang: Lang) {
  const mine = ++request
  try {
    await load(lang)
  } catch {
    return // offline and not cached: stay in the language shown
  }
  if (mine !== request) return
  activate(lang)
  useI18n.setState({ lang })
}

export const useI18n = create<I18nState>()((set) => ({
  choice: 'system',
  lang: 'en',
  async setChoice(choice) {
    storeChoice(choice)
    set({ choice })
    await show(resolveChoice(choice))
  },
}))

/** Before the first render: the stored choice, its catalog loaded, so the app
 *  opens in its language without showing English first. */
export async function initI18n(): Promise<void> {
  const choice = storedChoice()
  useI18n.setState({ choice })
  await show(resolveChoice(choice))
  if (typeof window !== 'undefined') {
    window.addEventListener('languagechange', () => {
      if (useI18n.getState().choice === 'system') void show(systemLanguage())
    })
  }
}

// MARK: Lookup

function find(root: unknown, key: string): string | Plural | undefined {
  let v: unknown = root
  for (const part of key.split('.')) {
    if (v == null || typeof v !== 'object') return undefined
    v = (v as Record<string, unknown>)[part]
  }
  return typeof v === 'string' || (v != null && typeof v === 'object' && 'other' in v) ? (v as string | Plural) : undefined
}

const pluralRules = new Map<Lang, Intl.PluralRules>()
function pluralForm(p: Plural, count: unknown): string {
  if (typeof count !== 'number') return p.other
  let rules = pluralRules.get(currentLang)
  if (!rules) pluralRules.set(currentLang, (rules = new Intl.PluralRules(currentLang)))
  return rules.select(count) === 'one' ? p.one : p.other
}

function render(v: string | Plural, params?: Params): string {
  return fill(typeof v === 'string' ? v : pluralForm(v, params?.count), params)
}

/** A UI string in the current language (English where it has none). */
export function t(key: Key, params?: Params): string {
  const v = find(current, key) ?? find(en, key)
  return v == null ? key : render(v, params)
}

/** A string by a key built at run time (`analysis.titles.${name}`), or
 *  undefined when the catalog has no such key: for tables keyed by engine names. */
export function tMaybe(key: string, params?: Params): string | undefined {
  const v = find(current, key) ?? find(en, key)
  return v == null ? undefined : render(v, params)
}

/** The current language (outside React; components use `useLang()`). */
export const currentLanguage = (): Lang => currentLang

/** The current language; the component re-renders when it changes. */
export const useLang = (): Lang => useI18n((s) => s.lang)

/** `t` for a component: it re-renders on a language change, and the
 *  function's identity changes with the language (for memo and effect deps). */
export function useT(): typeof t {
  const lang = useLang()
  return useMemo(() => ((key, params) => t(key, params)) as typeof t, [lang])
}

/** A string with elements in it: `{name}` placeholders found in `parts` become
 *  those nodes (a key, a link), the rest are filled from `params`. */
export function tRich(key: Key, parts: Record<string, ReactNode>, params?: Params): ReactNode {
  const pieces = t(key, params).split(/\{(\w+)\}/)
  return createElement(Fragment, null, ...pieces.map((p, k) => (k % 2 ? (p in parts ? parts[p] : `{${p}}`) : p)))
}
