import { afterEach, describe, expect, it } from 'vitest'
import { en } from './en'
import { zhHans } from './zh-Hans'
import { engineEnum, engineIssue, engineText, parseStructuralIssue, setEngineCatalog } from './engine'
import { currentLanguage, systemLanguage, t, tMaybe, useI18n } from '.'

type Tree = { [k: string]: string | Tree }

/** Every leaf of a catalog by its dotted key (a plural is one leaf per form). */
function leaves(tree: Tree, prefix = ''): Map<string, string> {
  const out = new Map<string, string>()
  for (const [k, v] of Object.entries(tree)) {
    if (typeof v === 'string') out.set(prefix + k, v)
    else for (const [kk, vv] of leaves(v, `${prefix}${k}.`)) out.set(kk, vv)
  }
  return out
}

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()

afterEach(async () => {
  await useI18n.getState().setChoice('system')
  localStorage.clear()
})

describe('system language', () => {
  it('maps Simplified Chinese tags to zh-Hans and keeps Traditional in English', () => {
    for (const tag of ['zh', 'zh-CN', 'zh-SG', 'zh-Hans', 'zh-Hans-CN', 'zh_CN']) expect(systemLanguage([tag])).toBe('zh-Hans')
    for (const tag of ['zh-TW', 'zh-HK', 'zh-MO', 'zh-Hant', 'zh-Hant-TW', 'en-US', 'de-DE']) expect(systemLanguage([tag])).toBe('en')
    expect(systemLanguage(['de-DE', 'zh-CN', 'en'])).toBe('zh-Hans')
    expect(systemLanguage(['en-GB', 'zh-CN'])).toBe('en')
    expect(systemLanguage([])).toBe('en')
  })
})

describe('t', () => {
  it('fills placeholders and picks English plural forms', () => {
    expect(t('common.surfaceN', { n: 3 })).toBe('Surface 3')
    expect(t('common.errors', { count: 1 })).toBe('1 error')
    expect(t('common.errors', { count: 2 })).toBe('2 errors')
    expect(tMaybe('common.nothing')).toBeUndefined()
  })

  it('switches live to Chinese, persists the choice, and back', async () => {
    await useI18n.getState().setChoice('zh-Hans')
    expect(currentLanguage()).toBe('zh-Hans')
    expect(useI18n.getState().lang).toBe('zh-Hans')
    expect(document.documentElement.lang).toBe('zh-Hans')
    expect(localStorage.getItem('mklight.language')).toBe('zh-Hans')
    expect(t('common.surfaceN', { n: 3 })).toBe('第 3 面')
    expect(t('common.errors', { count: 1 })).toBe('1 个错误')
    await useI18n.getState().setChoice('en')
    expect(t('common.cancel')).toBe('Cancel')
    expect(document.documentElement.lang).toBe('en')
  })
})

describe('catalogs', () => {
  const source = leaves(en as unknown as Tree)
  const zh = leaves(zhHans as unknown as Tree)

  it('zh-Hans has every English key and the same placeholders', () => {
    const bad: string[] = []
    for (const [k, v] of source) {
      // A plural may be one string in Chinese: `a.b.one` / `a.b.other` → `a.b`.
      const z = zh.get(k) ?? zh.get(k.replace(/\.(one|other)$/, ''))
      if (z == null) bad.push(`missing ${k}`)
      else if (placeholders(z).join() !== placeholders(v).join() && !(k.endsWith('.one') && placeholders(v).length === 0))
        bad.push(`${k}: {${placeholders(v)}} vs {${placeholders(z)}}`)
    }
    expect(bad).toEqual([])
  })

  it('no Chinese string is empty (joiners aside)', () => {
    expect([...zh].filter(([k, v]) => !v.trim() && source.get(k)?.trim()).map(([k]) => k)).toEqual([])
  })
})

describe('engine text', () => {
  afterEach(() => setEngineCatalog(null))

  it('looks up exact English, enums and issue codes, and falls back to English', () => {
    setEngineCatalog({ strings: { 'Spot Diagram': '点列图' }, enums: { hexapolar: '六角极坐标' }, diagnostics: { thick_neg: '第 {surface} 面的厚度为负', bad: '{path} 无效' } })
    expect(engineText('Spot Diagram')).toBe('点列图')
    expect(engineText('Spot diagram')).toBe('Spot diagram')
    expect(engineText(null)).toBeUndefined()
    expect(engineEnum('hexapolar')).toBe('六角极坐标')
    expect(engineEnum('grid')).toBeUndefined()
    expect(engineIssue('thick_neg', { surface: 4 }, 'Negative thickness -1 mm')).toEqual({ text: '第 4 面的厚度为负', detail: 'Negative thickness -1 mm' })
    // A template naming a value the issue lacks, or an unknown code: the message.
    expect(engineIssue('thick_neg', { surface: null }, 'Negative thickness')).toEqual({ text: 'Negative thickness' })
    expect(engineIssue('other', { surface: 4 }, 'Odd')).toEqual({ text: 'Odd' })
    const s = parseStructuralIssue('invalid model: [bad] surfaces[3].radius: must be finite')
    expect(s).toEqual({ code: 'bad', path: 'surfaces[3].radius', message: 'must be finite' })
    expect(engineIssue(s!.code, { path: s!.path }, s!.message).text).toBe('surfaces[3].radius 无效')
  })
})
