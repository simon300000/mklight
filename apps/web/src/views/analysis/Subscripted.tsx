// Labels with real subscripts (`Subscripted`): `S_III` and `C_L` set the part
// after `_` (to the end of its word) as a subscript.
import { Fragment } from 'react'

export interface Piece {
  text: string
  sub: boolean
}

export function subscriptPieces(label: string): Piece[] {
  const out: Piece[] = []
  const add = (text: string, sub: boolean) => {
    if (!text) return
    const last = out[out.length - 1]
    if (last && last.sub === sub) last.text += text
    else out.push({ text, sub })
  }
  label.split(' ').forEach((word, k) => {
    if (k > 0) add(' ', false)
    const u = word.indexOf('_')
    if (u > 0) {
      add(word.slice(0, u), false)
      add(word.slice(u + 1), true)
    } else add(word, false)
  })
  return out
}

export function Subscripted({ text, className }: { text: string; className?: string }) {
  return (
    <span className={className}>
      {subscriptPieces(text).map((p, k) => (p.sub ? <sub key={k} className="text-[0.8em]">{p.text}</sub> : <Fragment key={k}>{p.text}</Fragment>))}
    </span>
  )
}
