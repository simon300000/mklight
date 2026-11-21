// The UI text the three apps share (locales/<id>/ui.json) must read the same in each app's
// catalog: the macOS String Catalog, these web modules and the Windows .resw (which must also
// stay valid, with the en-US keys and placeholders). Runs on every machine that builds the web
// app. Fix with `node scripts/i18n-ui.mjs extract|apply` (docs/I18N.md, "Shared UI text").
import { describe, expect, it } from 'vitest'
import { locales, problems } from '../../../../scripts/i18n-ui.mjs'

describe('shared UI text', () => {
  it('has a ui.json for Simplified Chinese', () => {
    expect(locales()).toContain('zh-Hans')
  })

  for (const locale of locales())
    it(`every app translates a shared English string as ${locale}/ui.json does`, () => {
      expect(problems(locale)).toEqual([])
    })
})
