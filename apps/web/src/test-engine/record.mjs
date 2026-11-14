// Records src/test-engine/recordings.json from the real engine, through the CLI.
//
//   cargo build --release -p mklight-cli     (once, and after engine changes)
//   pnpm record:engine                        (from apps/web)
//
// For each entry of RUNS it runs `mklight run <project> <analysis> --json
// --buffers --no-timing` on the double Gauss sample and keeps the result JSON,
// the buffers, the run info and the envelope's other keys. Settings are kept
// small so the file stays small. Re-record after a change of a result format.
import { execFileSync, spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '../../../..')
const cli = process.env.MKLIGHT_CLI ?? join(root, 'target/release/mklight')

/** What the replay engine can answer: analysis, exact settings (null: defaults). */
const RUNS = [
  ['system_summary', null],
  ['validate', null],
  ['layout', { section: 'yz', rays_per_field: 7, include_3d: false }],
  ['first_order', null],
  ['seidel', null],
  ['spot_diagram', { through_focus: { min: null, max: null, steps: 3 } }],
  ['ray_fan', null],
  ['opd_fan', null],
  ['mtf', null],
  ['psf', { grid_size: 32, wavelengths: { type: 'primary' } }],
  ['field_curvature_distortion', null],
  ['wavefront_map', { pupil: { grid: 32 }, fields: { type: 'all' } }],
  ['glass_map', null],
  ['glass_info', { name: 'N-BK7' }],
  ['optimize', { max_iterations: 5 }],
]

const dir = mkdtempSync(join(tmpdir(), 'mklight-record-'))
const project = join(dir, 'double_gauss.mklight')
const run = (args, opts = {}) => execFileSync(cli, args, { encoding: 'utf8', maxBuffer: 1 << 28, ...opts })
run(['new', project, '--sample', 'double_gauss'])

const descriptors = JSON.parse(run(['analyses', '--json', '--no-timing'])).analyses

const runs = []
for (const [analysis, settings] of RUNS) {
  const args = ['run', project, analysis, '--json', '--buffers', '--no-timing', '--progress']
  if (settings) args.push('--settings', JSON.stringify(settings))
  const r = spawnSync(cli, args, { encoding: 'utf8', maxBuffer: 1 << 28 })
  if (r.status !== 0) throw new Error(`${analysis}: exit ${r.status}\n${r.stderr}`)
  const env = JSON.parse(r.stdout)
  const progress = r.stderr
    .split('\n')
    .filter((l) => l.startsWith('{'))
    .map((l) => JSON.parse(l))
    .map((p) => ({ progress: p.progress, stage: p.stage }))
  const { result, buffers = [], run_info, ...envelope } = env
  runs.push({
    analysis,
    settings,
    json: result,
    buffers: buffers.map((b) => ({ name: b.name, dtype: b.dtype, shape: b.shape, data: b.data })),
    run_info,
    envelope,
    progress: thin(progress, 8),
  })
}

// The engine's answer for a system whose stop surface does not exist.
const bad = JSON.parse(readFileSync(project, 'utf8'))
bad.system.stop = 99
writeFileSync(project, JSON.stringify(bad))
const v = spawnSync(cli, ['validate', project, '--json', '--no-timing'], { encoding: 'utf8' })
const issues = JSON.parse(v.stdout).issues.filter((i) => i.severity === 'error').map((i) => i.message)
const invalidStop = { message: issues[0], issues }

// Numbers keep 7 significant digits (a fixture, not a reference): the file stays small.
const round = (_k, x) => (typeof x === 'number' && Number.isFinite(x) && !Number.isInteger(x) ? Number(x.toPrecision(7)) : x)
writeFileSync(join(here, 'recordings.json'), JSON.stringify({ descriptors, invalidStop, runs }, round))
console.log(`recorded ${runs.length} runs`)

function thin(list, n) {
  if (list.length <= n) return list
  return Array.from({ length: n }, (_, i) => list[Math.round((i * (list.length - 1)) / (n - 1))])
}
