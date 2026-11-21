import type { Params } from './types'

/** Replaces `{name}` with `params.name`. An unknown name stays as written,
 *  unless `strict`, when the result is null. */
export function fill(template: string, params: Params | undefined, strict: true): string | null
export function fill(template: string, params: Params | undefined, strict?: false): string
export function fill(template: string, params: Params | undefined, strict = false): string | null {
  if (!params || !template.includes('{')) return template
  let missing = false
  const out = template.replace(/\{(\w+)\}/g, (m, name: string) => {
    const v = params[name]
    if (v === undefined) {
      missing = true
      return m
    }
    return String(v)
  })
  return strict && missing ? null : out
}
