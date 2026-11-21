#!/usr/bin/env node
// The UI strings the three apps share, translated once (docs/I18N.md, "Shared UI text").
//
//   node scripts/i18n-ui.mjs extract [locale]   report shared strings and conflicts, add agreed ones to ui.json
//   node scripts/i18n-ui.mjs apply   [locale]   write ui.json's translations into each app's catalog
//   node scripts/i18n-ui.mjs check   [locale]   what the web test checks: exit 1 on a difference
//
// locale defaults to zh-Hans. No dependencies: Node only. Each app keeps its native catalog; this
// script edits the translated text in place (the xcstrings JSON as Xcode writes it, the .resw XML
// <value> text, the web modules' string literals) and nothing else, so `apply` twice changes nothing.
//
// A string is shared when its English, with placeholders normalised to {0}, {1}… in order of first
// appearance, is exactly the same in two or more apps. Left out: web plurals ({one, other}), strings
// without a letter outside placeholders, macOS strings marked not to translate, and strings listed in
// ui.json's "context" (same English, different meaning).
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const MAC = 'apps/macos/Sources/MKLightUI/Resources/Localizable.xcstrings'
const WEB = 'apps/web/src/i18n'
const WIN = 'apps/windows/src/MKLight.App/Strings'
const WEB_LOCALE = (l) => l
const WIN_LOCALE = (l) => (l === 'en' ? 'en-US' : l)
export const APPS = ['macos', 'web', 'windows']

// ---------------------------------------------------------------- placeholders

// A printf conversion as Foundation reads it: %@, %lld, %.1f, %2$@ (and %% as a literal %).
const MAC_SPEC = /%(?:(\d+)\$)?([-+#0]*\d*(?:\.\d+)?(?:hh|h|ll|l|q|L|z|t|j)?[@dDiuUxXoOfFeEgGaAcCsSp])|%%/g
const WEB_SPEC = /\{(\w+)\}/g
const WIN_SPEC = /\{(\d+)((?:,-?\d+)?(?::[^}]*)?)\}/g

/** English → {norm, args}: `args[i]` is the app's own placeholder for {i}. */
function parseSource(app, s) {
  const args = [] // { id, token }
  const index = new Map()
  let seq = 0
  const take = (id, token) => {
    if (!index.has(id)) {
      index.set(id, args.length)
      args.push({ id, token })
    }
    return `{${index.get(id)}}`
  }
  let norm
  if (app === 'macos')
    norm = s.replace(MAC_SPEC, (m, pos, spec) => (m === '%%' ? '%' : take(pos ? +pos : ++seq, spec)))
  else if (app === 'web') norm = s.replace(WEB_SPEC, (m, name) => take(name, m))
  else norm = s.replace(WIN_SPEC, (m, n) => take(+n, m))
  return { norm, args, index }
}

/** A translation → the normalised form, by the English's placeholders; null when it names one
 *  the English lacks. */
function normalise(app, src, s) {
  let seq = 0
  let bad = false
  const at = (id) => {
    if (!src.index.has(id)) bad = true
    return `{${src.index.get(id)}}`
  }
  let out
  if (app === 'macos') out = s.replace(MAC_SPEC, (m, pos) => (m === '%%' ? '%' : at(pos ? +pos : ++seq)))
  else if (app === 'web') out = s.replace(WEB_SPEC, (m, name) => at(name))
  else out = s.replace(WIN_SPEC, (m, n) => at(+n))
  return bad ? null : out
}

/** The normalised form → the app's text, with the English's own placeholders. */
function render(app, entry, norm) {
  const { src, en } = entry
  const order = [...norm.matchAll(/\{(\d+)\}/g)].map((m) => +m[1])
  if (app === 'macos') {
    const escape = /%%/.test(en) || /%%/.test(entry.zh ?? '')
    const inOrder = order.every((n, i) => n === i) && src.args.every((a, i) => a.id === i + 1)
    const positional = !inOrder || (entry.zh != null && /%\d+\$/.test(entry.zh))
    let out = ''
    let last = 0
    for (const m of norm.matchAll(/\{(\d+)\}/g)) {
      out += (escape ? norm.slice(last, m.index).replace(/%/g, '%%') : norm.slice(last, m.index))
      const a = src.args[+m[1]]
      out += positional ? `%${a.id}$${a.token}` : `%${a.token}`
      last = m.index + m[0].length
    }
    return out + (escape ? norm.slice(last).replace(/%/g, '%%') : norm.slice(last))
  }
  return norm.replace(/\{(\d+)\}/g, (m, n) => src.args[+n].token)
}

const hasLetter = (norm) => /[A-Za-z]/.test(norm.replace(/\{\d+\}/g, ''))

// ---------------------------------------------------------------- macOS: String Catalog

/** A JSON string literal at `i`: [value, end]. */
function jsonString(text, i) {
  let j = i + 1
  while (text[j] !== '"') j += text[j] === '\\' ? 2 : 1
  return [JSON.parse(text.slice(i, j + 1)), j + 1]
}

/** Where each zh value sits in the file text: key → [start, end] of its JSON string. Walks the
 *  file once, tracking the path of object keys. */
function macSpans(text, locale) {
  const spans = new Map()
  const path = []
  let pendingKey = null
  let i = 0
  while (i < text.length) {
    const c = text[i]
    if (c === '"') {
      const [v, end] = jsonString(text, i)
      let k = end
      while (/\s/.test(text[k])) k++
      if (text[k] === ':') {
        pendingKey = v
        i = k + 1
        continue
      }
      if (path.length === 6 && path[1] === 'strings' && path[3] === 'localizations' && path[4] === locale && path[5] === 'stringUnit' && pendingKey === 'value')
        spans.set(path[2], [i, end])
      pendingKey = null
      i = end
      continue
    }
    if (c === '{') path.push(pendingKey ?? '')
    else if (c === '}') path.pop()
    else if (c === '[') path.push(pendingKey ?? '')
    else if (c === ']') path.pop()
    if (c === '{' || c === '[' || c === ',') pendingKey = null
    i++
  }
  return spans
}

function readMac(locale) {
  const file = join(ROOT, MAC)
  const text = readFileSync(file, 'utf8')
  const json = JSON.parse(text)
  const out = []
  for (const [key, v] of Object.entries(json.strings)) {
    if (v.shouldTranslate === false || v.extractionState === 'stale') continue
    const l = v.localizations ?? {}
    const en = l.en?.stringUnit?.value ?? key
    const zl = l[locale]
    if (l.en?.variations || zl?.variations) {
      out.push({ app: 'macos', id: key, en, plural: true })
      continue
    }
    out.push({ app: 'macos', id: key, en, zh: zl?.stringUnit?.value })
  }
  return { entries: out, text, file }
}

function writeMac(cat, locale, changes) {
  if (!changes.length) return
  const spans = macSpans(cat.text, locale)
  const edits = changes.map(({ entry, value }) => {
    const span = spans.get(entry.id)
    if (!span) throw new Error(`macOS: no ${locale} stringUnit for ${JSON.stringify(entry.id)}`)
    return [span, JSON.stringify(value)]
  })
  writeFileSync(cat.file, splice(cat.text, edits))
}

// ---------------------------------------------------------------- web: TypeScript modules

/** The string leaves of an `export const x = { … }` module: dotted path → {value, start, end, quote}.
 *  Handles what the catalogs use: identifier or quoted keys, nested objects, '…' and "…" literals,
 *  comments. */
function webLeaves(text) {
  const leaves = new Map()
  let i = text.indexOf('{', text.search(/export const \w+(?::[^=]+)?=/))
  let key = null
  const skip = () => {
    for (;;) {
      while (/\s|,/.test(text[i])) i++
      if (text.startsWith('//', i)) i = text.indexOf('\n', i)
      else if (text.startsWith('/*', i)) i = text.indexOf('*/', i) + 2
      else return
    }
  }
  const literal = () => {
    const q = text[i]
    let j = i + 1
    let v = ''
    while (text[j] !== q) {
      if (text[j] === '\\') {
        const e = text[j + 1]
        if (e === 'u') {
          v += String.fromCharCode(parseInt(text.slice(j + 2, j + 6), 16))
          j += 6
          continue
        }
        v += { n: '\n', t: '\t', r: '\r' }[e] ?? e
        j += 2
      } else v += text[j++]
    }
    const r = { value: v, start: i, end: j + 1, quote: q }
    i = j + 1
    return r
  }
  i++
  const prefix = []
  for (;;) {
    skip()
    if (text[i] === '}') {
      i++
      if (!prefix.length) break
      prefix.pop()
      continue
    }
    // key
    if (text[i] === "'" || text[i] === '"') key = literal().value
    else {
      const m = /^[A-Za-z_$][\w$]*/.exec(text.slice(i))
      if (!m) throw new Error(`web: unexpected ${JSON.stringify(text.slice(i, i + 20))}`)
      key = m[0]
      i += key.length
    }
    skip()
    if (text[i] !== ':') throw new Error(`web: expected ':' after ${key}`)
    i++
    skip()
    if (text[i] === '{') {
      i++
      prefix.push(key)
      continue
    }
    if (text[i] !== "'" && text[i] !== '"') throw new Error(`web: ${[...prefix, key].join('.')} is not a string literal`)
    leaves.set([...prefix, key].join('.'), literal())
  }
  return leaves
}

function readWeb(locale) {
  const out = []
  const files = []
  for (const name of readdirSync(join(ROOT, WEB, 'en')).sort()) {
    if (name === 'index.ts' || !name.endsWith('.ts')) continue
    const area = name.slice(0, -3)
    const enLeaves = webLeaves(readFileSync(join(ROOT, WEB, 'en', name), 'utf8'))
    const zhFile = join(ROOT, WEB, WEB_LOCALE(locale), name)
    const zhText = existsSync(zhFile) ? readFileSync(zhFile, 'utf8') : ''
    const zhLeaves = zhText ? webLeaves(zhText) : new Map()
    files.push({ area, file: zhFile, text: zhText, leaves: zhLeaves })
    const plurals = new Set()
    for (const k of enLeaves.keys()) if (/\.(one|other)$/.test(k)) plurals.add(k.replace(/\.(one|other)$/, ''))
    for (const [k, v] of enLeaves) {
      const id = `${area}.${k}`
      const base = k.replace(/\.(one|other)$/, '')
      if (plurals.has(base) && base !== k) {
        if (k.endsWith('.other')) out.push({ app: 'web', id: `${area}.${base}`, en: v.value, plural: true })
        continue
      }
      out.push({ app: 'web', id, en: v.value, zh: zhLeaves.get(k)?.value, area, key: k })
    }
  }
  return { entries: out, files }
}

function writeWeb(cat, changes) {
  for (const f of cat.files) {
    const edits = []
    for (const { entry, value } of changes) {
      if (entry.area !== f.area) continue
      const lit = f.leaves.get(entry.key)
      if (!lit) throw new Error(`web: no literal for ${entry.id}`)
      const q = lit.quote
      const body = value.replace(/\\/g, '\\\\').replace(new RegExp(q, 'g'), `\\${q}`).replace(/\n/g, '\\n')
      edits.push([[lit.start, lit.end], q + body + q])
    }
    if (edits.length) writeFileSync(f.file, splice(f.text, edits))
  }
}

// ---------------------------------------------------------------- Windows: .resw

const XML_ENT = { lt: '<', gt: '>', amp: '&', quot: '"', apos: "'" }
const xmlDecode = (s) =>
  s.replace(/&(#x[0-9a-fA-F]+|#\d+|\w+);/g, (m, e) =>
    e[0] === '#' ? String.fromCodePoint(e[1] === 'x' ? parseInt(e.slice(2), 16) : +e.slice(1)) : (XML_ENT[e] ?? m),
  )
const xmlEncode = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/** Checks the file is well-formed enough for a .resw: tags balance, every & is an entity, no raw
 *  < in text, and every <data> was read. Returns problems. */
export function reswProblems(text, count) {
  const problems = []
  const body = text.replace(/<\?xml[^>]*\?>/, '').replace(/<!--[\s\S]*?-->/g, '')
  const stack = []
  for (const m of body.matchAll(/<(\/?)([A-Za-z_][\w:.-]*)((?:\s+[\w:.-]+\s*=\s*(?:"[^"<]*"|'[^'<]*'))*)\s*(\/?)>|<|&(?!(?:#x[0-9a-fA-F]+|#\d+|lt|gt|amp|quot|apos);)/g)) {
    if (m[0] === '<') problems.push(`raw '<' at ${m.index}`)
    else if (m[0] === '&') problems.push(`raw '&' at ${m.index}`)
    else if (m[4]) continue
    else if (m[1]) {
      if (stack.pop() !== m[2]) problems.push(`unbalanced </${m[2]}> at ${m.index}`)
    } else stack.push(m[2])
  }
  if (stack.length) problems.push(`unclosed <${stack.join('>, <')}>`)
  const datas = (body.match(/<data\s/g) ?? []).length
  if (datas !== count) problems.push(`${datas} <data> elements, ${count} read`)
  return problems
}

const RESW_DATA = /<data\s+name="([^"]+)"[^>]*>\s*<value>([\s\S]*?)<\/value>/g

function reswValues(text) {
  const m = new Map()
  for (const d of text.matchAll(RESW_DATA)) {
    const start = d.index + d[0].length - '</value>'.length - d[2].length
    m.set(d[1], { value: xmlDecode(d[2]), start, end: start + d[2].length })
  }
  return m
}

function readWin(locale) {
  const enText = readFileSync(join(ROOT, WIN, 'en-US', 'Resources.resw'), 'utf8')
  const file = join(ROOT, WIN, WIN_LOCALE(locale), 'Resources.resw')
  const text = existsSync(file) ? readFileSync(file, 'utf8') : ''
  const en = reswValues(enText)
  const zh = reswValues(text)
  const entries = [...en].map(([name, v]) => ({ app: 'windows', id: name, en: v.value, zh: zh.get(name)?.value }))
  return { entries, file, text, values: zh, enText, enValues: en }
}

function writeWin(cat, changes) {
  if (!changes.length) return
  const edits = changes.map(({ entry, value }) => {
    const v = cat.values.get(entry.id)
    if (!v) throw new Error(`windows: no value for ${entry.id}`)
    return [[v.start, v.end], xmlEncode(value)]
  })
  writeFileSync(cat.file, splice(cat.text, edits))
}

// ---------------------------------------------------------------- shared

function splice(text, edits) {
  edits.sort((a, b) => b[0][0] - a[0][0])
  for (const [[s, e], v] of edits) text = text.slice(0, s) + v + text.slice(e)
  return text
}

/** The locales that have a ui.json. */
export function locales() {
  const dir = join(ROOT, 'locales')
  return readdirSync(dir).filter((l) => existsSync(join(dir, l, 'ui.json'))).sort()
}

export function uiPath(locale = 'zh-Hans') {
  return join(ROOT, 'locales', locale, 'ui.json')
}

export function readUi(locale = 'zh-Hans') {
  const p = uiPath(locale)
  if (!existsSync(p)) return { locale, strings: {}, context: {} }
  const ui = JSON.parse(readFileSync(p, 'utf8'))
  return { ...ui, strings: ui.strings ?? {}, context: ui.context ?? {} }
}

function writeUi(locale, ui) {
  const sorted = (o) => Object.fromEntries(Object.entries(o).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
  const out = { locale, about: ui.about, strings: sorted(ui.strings), context: sorted(ui.context) }
  if (!out.about) delete out.about
  writeFileSync(uiPath(locale), JSON.stringify(out, null, 2) + '\n')
}

/** Reads the three catalogs and groups their strings by normalised English. */
export function collect(locale = 'zh-Hans') {
  const cats = { macos: readMac(locale), web: readWeb(locale), windows: readWin(locale) }
  const groups = new Map() // norm → entries
  const skipped = { plural: [], noLetters: 0, badPlaceholder: [] }
  for (const app of APPS)
    for (const e of cats[app].entries) {
      if (e.plural) {
        skipped.plural.push(`${app}: ${e.id}`)
        continue
      }
      e.src = parseSource(app, e.en)
      if (!hasLetter(e.src.norm)) {
        skipped.noLetters++
        continue
      }
      if (e.zh != null) {
        e.zhNorm = normalise(app, e.src, e.zh)
        if (e.zhNorm == null) skipped.badPlaceholder.push(`${app}: ${e.id}`)
      }
      if (!groups.has(e.src.norm)) groups.set(e.src.norm, [])
      groups.get(e.src.norm).push(e)
    }
  const shared = new Map([...groups].filter(([, es]) => new Set(es.map((e) => e.app)).size >= 2))
  return { cats, groups, shared, skipped }
}

/** Every way the catalogs and ui.json disagree; empty when they agree. */
export function problems(locale = 'zh-Hans') {
  const ui = readUi(locale)
  const { cats, shared, skipped } = collect(locale)
  const out = []
  for (const [norm, es] of shared) {
    if (norm in ui.context) continue
    const want = ui.strings[norm]
    if (want == null) {
      out.push(`missing from ui.json: ${JSON.stringify(norm)} (${[...new Set(es.map((e) => e.app))].join(', ')})`)
      continue
    }
    for (const e of es)
      if (e.zhNorm !== want) out.push(`${e.app} ${e.id}: ${JSON.stringify(e.zh ?? null)}, ui.json ${JSON.stringify(want)}`)
  }
  for (const k of Object.keys(ui.strings)) {
    if (!shared.has(k)) out.push(`ui.json has ${JSON.stringify(k)}, which is not shared`)
    else if (k in ui.context) out.push(`ui.json has ${JSON.stringify(k)} in both strings and context`)
    else if (!ui.strings[k].trim()) out.push(`ui.json: empty translation for ${JSON.stringify(k)}`)
    else {
      const ph = (s) => [...s.matchAll(/\{(\d+)\}/g)].map((m) => +m[1])
      const need = new Set(ph(k))
      const got = new Set(ph(ui.strings[k]))
      if ([...got].some((n) => !need.has(n)) || [...need].some((n) => !got.has(n)))
        out.push(`ui.json: placeholders of ${JSON.stringify(k)} differ`)
    }
  }
  for (const k of Object.keys(ui.context)) if (!shared.has(k)) out.push(`ui.json context has ${JSON.stringify(k)}, which is not shared`)
  for (const s of skipped.badPlaceholder) out.push(`placeholder not in the English: ${s}`)
  // Windows: the file must stay valid, with the en-US keys and placeholders.
  const win = cats.windows
  if (win.text) {
    for (const p of reswProblems(win.text, win.values.size)) out.push(`windows ${locale}: ${p}`)
    for (const p of reswProblems(win.enText, win.enValues.size)) out.push(`windows en-US: ${p}`)
    const enKeys = [...win.enValues.keys()].join('\n')
    if ([...win.values.keys()].sort().join('\n') !== [...win.enValues.keys()].sort().join('\n') && enKeys)
      out.push(`windows ${locale}: keys differ from en-US`)
    const ph = (s) => [...s.matchAll(WIN_SPEC)].map((m) => +m[1]).sort().filter((n, i, a) => a.indexOf(n) === i).join()
    for (const [k, v] of win.values) {
      const en = win.enValues.get(k)
      if (en && ph(en.value) !== ph(v.value)) out.push(`windows ${k}: placeholders {${ph(en.value)}} vs {${ph(v.value)}}`)
    }
  }
  return out
}

function extract(locale) {
  const ui = readUi(locale)
  const { shared, skipped } = collect(locale)
  const conflicts = []
  let added = 0
  for (const [norm, es] of shared) {
    if (norm in ui.context || norm in ui.strings) continue
    const zhs = [...new Set(es.map((e) => e.zhNorm).filter((z) => z != null))]
    if (zhs.length === 1 && es.every((e) => e.zhNorm != null)) {
      ui.strings[norm] = zhs[0]
      added++
    } else conflicts.push([norm, es])
  }
  writeUi(locale, ui)
  console.log(`${shared.size} shared strings; ${Object.keys(ui.strings).length} in ui.json (${added} added), ${Object.keys(ui.context).length} by context`)
  console.log(`skipped: ${skipped.plural.length} plurals, ${skipped.noLetters} without letters, ${skipped.badPlaceholder.length} with a placeholder the English lacks`)
  for (const s of skipped.plural) console.log(`  plural  ${s}`)
  for (const s of skipped.badPlaceholder) console.log(`  placeholder  ${s}`)
  if (conflicts.length) {
    console.log(`\n${conflicts.length} conflicts (same English, different ${locale}); choose one in ui.json "strings", or list the English in "context" with a reason:`)
    for (const [norm, es] of conflicts) {
      console.log(`\n  ${JSON.stringify(norm)}`)
      for (const e of es) console.log(`    ${e.app.padEnd(8)} ${JSON.stringify(e.zhNorm ?? e.zh ?? null)}  ${e.id}`)
    }
  }
  return conflicts.length
}

function apply(locale) {
  const ui = readUi(locale)
  const { cats, shared } = collect(locale)
  const changes = { macos: [], web: [], windows: [] }
  const unwritten = []
  for (const [norm, es] of shared) {
    const want = ui.strings[norm]
    if (want == null || norm in ui.context) continue
    for (const e of es) {
      if (e.zhNorm === want) continue
      // Only an existing translation is replaced: a missing one has no place to go yet (a
      // String Catalog entry, a web module key or a .resw <data>), so add the entry first.
      if (e.zh == null) {
        unwritten.push(`${e.app} ${e.id}`)
        continue
      }
      changes[e.app].push({ entry: e, value: render(e.app, e, want) })
    }
  }
  writeMac(cats.macos, locale, changes.macos)
  writeWeb(cats.web, changes.web)
  writeWin(cats.windows, changes.windows)
  if (unwritten.length) console.log(`not written, no ${locale} entry to replace: ${unwritten.join(', ')}`)
  for (const app of APPS) {
    console.log(`${app}: ${changes[app].length} changed`)
    for (const { entry, value } of changes[app]) console.log(`  ${entry.id}: ${JSON.stringify(entry.zh ?? null)} → ${JSON.stringify(value)}`)
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const [cmd, locale = 'zh-Hans'] = process.argv.slice(2)
  if (cmd === 'extract') process.exit(extract(locale) ? 1 : 0)
  else if (cmd === 'apply') apply(locale)
  else if (cmd === 'check') {
    const p = problems(locale)
    for (const s of p) console.log(s)
    console.log(p.length ? `${p.length} problems` : 'shared UI text agrees')
    process.exit(p.length ? 1 : 0)
  } else {
    console.error('usage: node scripts/i18n-ui.mjs extract|apply|check [locale]')
    process.exit(2)
  }
}
