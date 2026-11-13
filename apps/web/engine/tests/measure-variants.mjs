// Times and result digests of several engine builds in Chromium (docs/WEB.md
// "SIMD"). Build each variant and its harness first, e.g.
//   scripts/build-wasm.sh --no-simd --libm-fma && (cd apps/web/engine && npx vite build --outDir ../dist-A)
//   scripts/build-wasm.sh && (cd apps/web/engine && npx vite build --outDir ../dist-D)
// then, in apps/web/engine (after `npm test` has written tests/.reference):
//   node tests/measure-variants.mjs A D
// Prints per variant: [1-thread ms, all-threads ms] per analysis and a
// SHA-256 prefix of each result (equal digests = bitwise-identical results).
import { chromium } from 'playwright'
import { readFileSync } from 'node:fs'
import { spawn } from 'node:child_process'
const variants = process.argv.slice(2)
const text = readFileSync('tests/.reference/cooke.mklight', 'utf8')
const timed = [
  ['spot250', { analysis: 'spot_diagram', settings: { rings: 250, include_points: false } }],
  ['geometric_mtf', { analysis: 'geometric_mtf', settings: {} }],
  ['psf512', { analysis: 'psf', settings: { grid_size: 512 } }],
  ['through_focus_mtf', { analysis: 'through_focus_mtf', settings: {} }],
]
const digested = [
  ['spot', { analysis: 'spot_diagram', settings: {} }],
  ['spot250pts', { analysis: 'spot_diagram', settings: { rings: 120 } }],
  ['ray_fan', { analysis: 'ray_fan', settings: {} }],
  ['mtf', { analysis: 'mtf', settings: {} }],
  ['psf', { analysis: 'psf', settings: {} }],
  ['psf512', { analysis: 'psf', settings: { grid_size: 512 } }],
  ['geometric_mtf', { analysis: 'geometric_mtf', settings: {} }],
  ['through_focus_mtf', { analysis: 'through_focus_mtf', settings: {} }],
  ['opd_fan', { analysis: 'opd_fan', settings: {} }],
  ['wavefront_map', { analysis: 'wavefront_map', settings: {} }],
]
const b = await chromium.launch()
const out = {}
let port = 4800
for (const v of variants) {
  port++
  const srv = spawn('node', ['tests/static-server.mjs', `dist-${v}`, String(port), '--isolate'], { stdio: 'ignore' })
  await new Promise(r => setTimeout(r, 500))
  const p = await b.newPage()
  await p.goto(`http://localhost:${port}/`)
  await p.waitForFunction(() => 'mk' in window)
  out[v] = await p.evaluate(async ([t, timed, digested]) => {
    const one = (await window.mk.create(1)).index
    const all = (await window.mk.create()).index
    await window.mk.openProject(one, t); await window.mk.openProject(all, t)
    const res = { time: {}, digest: {} }
    for (const [k, q] of digested) res.digest[k] = await window.mk.digest(all, { ...q, backend: 'cpu_reference' })
    for (const [k, q] of timed) {
      const req = { ...q, backend: 'cpu_reference' }
      await window.mk.time(all, req, 1)
      const t1 = Math.min(...await window.mk.time(one, req, 3))
      const tn = Math.min(...await window.mk.time(all, req, 3))
      res.time[k] = [Math.round(t1), Math.round(tn)]
    }
    return res
  }, [text, timed, digested])
  console.log(v, JSON.stringify(out[v]))
  await p.close(); srv.kill()
}
await b.close()
