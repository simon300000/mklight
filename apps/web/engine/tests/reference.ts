// Global setup: native reference results from the same engine crates
// (the `mklight` CLI on this machine), for the web results to match.
// Writes tests/.reference/: cooke.mklight and <analysis>.json envelopes
// with inline buffer data (CPU f64 reference backend).
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, rmSync } from 'node:fs'
import { join, resolve } from 'node:path'

export const REF_DIR = resolve(import.meta.dirname, '.reference')
export const ROOT = resolve(import.meta.dirname, '../../../..')

/** Analyses compared against the native engine, with their settings. */
export const COMPARED: Record<string, object> = {
  spot_diagram: {},
  mtf: {},
  psf: {},
  ray_fan: {},
}

export default function setup() {
  rmSync(REF_DIR, { recursive: true, force: true })
  mkdirSync(REF_DIR, { recursive: true })
  execFileSync('cargo', ['build', '-p', 'mklight-cli', '--release'], { cwd: ROOT, stdio: 'inherit' })
  const cli = join(ROOT, 'target/release/mklight')
  if (!existsSync(cli)) throw new Error(`no CLI at ${cli}`)
  const project = join(REF_DIR, 'cooke.mklight')
  execFileSync(cli, ['new', project, '--sample', 'cooke_triplet', '--force'], { stdio: 'inherit' })
  for (const [analysis, settings] of Object.entries(COMPARED)) {
    execFileSync(
      cli,
      ['run', project, analysis, '--backend', 'cpu', '--json', '--buffers', '--settings', JSON.stringify(settings),
        '--out', join(REF_DIR, `${analysis}.json`)],
      { stdio: ['ignore', 'ignore', 'inherit'] },
    )
  }
}
