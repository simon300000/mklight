import { describe, expect, it } from 'vitest'
import { catalogText } from './catalogs'

const sample = 'CC test\r\nNM A 2 517642 1.5168 64.17\r\n'

function utf16(text: string, littleEndian: boolean): Uint8Array {
  const out = new Uint8Array(2 + text.length * 2)
  out.set(littleEndian ? [0xff, 0xfe] : [0xfe, 0xff])
  const view = new DataView(out.buffer)
  for (let i = 0; i < text.length; i++) view.setUint16(2 + 2 * i, text.charCodeAt(i), littleEndian)
  return out
}

describe('catalogText', () => {
  it('reads UTF-8', () => {
    expect(catalogText(new TextEncoder().encode(sample))).toBe(sample)
  })
  it('reads UTF-16 by its byte order mark and drops the mark', () => {
    expect(catalogText(utf16(sample, true))).toBe(sample)
    expect(catalogText(utf16(sample, false).buffer as ArrayBuffer)).toBe(sample)
  })
})
