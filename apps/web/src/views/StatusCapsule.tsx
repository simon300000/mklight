import { AlertTriangle, CheckCircle2, Loader2, XOctagon } from 'lucide-react'
import type { JsonObject } from '@/engine/types'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { fmt, nf, sig } from '@/model/format'
import { Segmented } from '@/components/ui/segmented'
import type { System } from '@/model/system'
import { BACKEND_CHOICES, usePreferences } from '@/state/preferences'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { allIssues, SUMMARY, useWorkspace } from '@/state/workspace'
import { cn } from '@/lib/utils'
import { JOB_VERBS, useJobs, type JobKind } from '@/state/jobs'

/** One entry per running job ("Optimizing · 479"); a click shows its panel. */
function JobEntries() {
  const jobs = useJobs((s) => s.jobs)
  const setPage = useView((s) => s.setPage)
  const running = (Object.keys(jobs) as JobKind[]).filter((k) => jobs[k].phase === 'running')
  if (!running.length) return null
  return (
    <>
      {running.map((k) => {
        const d = jobs[k].detail as Record<string, unknown> | null
        const n = d && (d.iteration ?? d.evaluations ?? d.trials ?? d.combinations)
        return (
          <button
            key={k}
            type="button"
            data-testid={`status-job-${k}`}
            onClick={() => setPage(k)}
            className="flex h-full shrink-0 cursor-default items-center gap-1.5 border-r border-separator px-3 hover:bg-hover first:rounded-l-full"
          >
            <Loader2 className="size-3 animate-spin text-accent" />
            <span className="num">
              {JOB_VERBS[k]}
              {n != null ? ` · ${String(n)}` : ` · ${Math.round(jobs[k].progress * 100)}%`}
            </span>
          </button>
        )
      })}
    </>
  )
}
import { IssuesList } from './Issues'

function entry(json: JsonObject | undefined, key: string): number | null {
  const e = (json?.entries as JsonObject[] | undefined)?.find((x) => x.key === key)
  return e ? nf(e.value) : null
}

export interface XYRow {
  wavelength: number
  wavelengthUm: number
  eflX: number
  eflY: number
  bflX: number
  bflY: number
  squeeze: number
  doePower: { surface: number; power: number }[]
}

/** The engine's first order of both sections (`paraxial_xy` of `first_order`,
 *  docs/RESULTS.md) and whether the design lacks rotational symmetry (the
 *  engine's `symmetry`; else toroids, biconics and freeforms decide). */
export function anamorphicSummary(json: JsonObject | undefined | null, system: System | null | undefined) {
  const xy = (json?.paraxial_xy as JsonObject | undefined) ?? null
  const symmetry = ((xy?.symmetry ?? json?.symmetry) as string | undefined) ?? null
  const rows: XYRow[] = ((xy?.wavelengths as JsonObject[] | undefined) ?? []).flatMap((w, k) => {
    const eflY = nf(w.efl_y)
    if (Number.isNaN(eflY)) return []
    const eflX = w.efl_x == null ? NaN : nf(w.efl_x)
    return [
      {
        wavelength: typeof w.wavelength === 'number' ? w.wavelength : k,
        wavelengthUm: nf(w.wavelength_um),
        eflX,
        eflY,
        bflX: w.bfl_x == null ? NaN : nf(w.bfl_x),
        bflY: w.bfl_y == null ? NaN : nf(w.bfl_y),
        squeeze: w.squeeze != null ? nf(w.squeeze) : eflX !== 0 ? eflY / eflX : NaN,
        doePower: ((w.doe_power as JsonObject[] | undefined) ?? []).map((d) => ({ surface: Number(d.surface), power: nf(d.power) })),
      },
    ]
  })
  const primaryIndex = typeof xy?.primary_wavelength === 'number' ? xy.primary_wavelength : null
  const primary = rows.find((r) => r.wavelength === primaryIndex) ?? rows[0] ?? null
  const nonRotational = ['toroid', 'biconic', 'xy_polynomial', 'zernike_sag']
  const anamorphic = symmetry ? symmetry !== 'rotational' : !!system?.surfaces?.some((s) => nonRotational.includes(s.shape.type))
  return {
    anamorphic,
    symmetry,
    primary,
    rows,
    squeezeDefinition: (xy?.squeeze_definition as string | undefined) ?? null,
    warnings: ((xy?.warnings as string[] | undefined) ?? []).filter((w) => typeof w === 'string'),
  }
}

/** `1.5000×`, signed. */
export const squeezeText = (v: number, digits = 4) => (Number.isFinite(v) ? `${v.toFixed(digits).replace('-', '−')}×` : sig(v))

/** The toolbar's status capsule (`DesignStatus`): activity, EFL, F/#, total
 *  track and the issues button; its popover has the first-order table and the
 *  engine's state. */
export function StatusCapsule() {
  const run = useWorkspace((s) => s.runs[SUMMARY])
  const runs = useWorkspace((s) => s.runs)
  const engine = useWorkspace((s) => s.engine)
  const engineError = useWorkspace((s) => s.engineError)
  const modelIssues = useWorkspace((s) => s.modelIssues)
  const validationIssues = useWorkspace((s) => s.validationIssues)
  const pending = useWorkspace((s) => s.snapshotPending)
  const issuesOpen = useView((s) => s.issuesOpen)
  const setView = useView((s) => s.set)
  const issues = allIssues({ modelIssues, validationIssues })
  const errors = issues.filter((i) => i.severity === 'error').length
  const busy = pending || Object.values(runs).some((r) => r.status === 'running')
  const json = run?.result?.json
  const system = useDocument((s) => s.project.system)
  const efl = entry(json, 'efl')
  const fno = entry(json, 'working_f_number') ?? entry(json, 'image_f_number')
  const tt = entry(json, 'total_track')
  const xy = anamorphicSummary(json, system)

  return (
    <div className="flex h-7 max-w-full items-center rounded-full border border-control-border bg-control/60 text-[12px] shadow-[0_0.5px_1px_rgb(0_0_0/0.06)]">
      <JobEntries />
      <Popover>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="flex h-full min-w-0 cursor-default items-center gap-2 rounded-l-full pl-3 pr-2 hover:bg-hover"
            aria-label="Design status"
            data-testid="status-capsule"
          >
            {engineError ? (
              <span className="text-danger">Engine unavailable</span>
            ) : !engine ? (
              <span className="flex items-center gap-1.5 text-secondary">
                <Loader2 className="size-3 animate-spin" /> Starting engine
              </span>
            ) : (
              <>
                {busy && <Loader2 className="size-3 shrink-0 animate-spin text-secondary" aria-label="Computing" />}
                {engine.info.isMock && (
                  <span className="rounded bg-warning/15 px-1 text-[10px] font-semibold uppercase tracking-wide text-warning" title="Mock engine: synthetic results">
                    Mock
                  </span>
                )}
                {errors > 0 ? (
                  <span className="text-secondary">Design has errors</span>
                ) : (
                  <span className={cn('num truncate', run?.status === 'running' && 'opacity-70')}>
                    {xy.anamorphic && xy.primary ? (
                      <>
                        <span className="text-secondary">EFL y</span> {sig(xy.primary.eflY, 5)} <span className="text-secondary">· Squeeze</span>{' '}
                        {squeezeText(xy.primary.squeeze, 2)}
                      </>
                    ) : (
                      <>
                        <span className="text-secondary">{xy.anamorphic ? 'EFL y' : 'EFL'}</span> {sig(efl, 5)}
                      </>
                    )}{' '}
                    <span className="text-secondary">·</span> <span className="text-secondary">F/</span>
                    {sig(fno, 3)} <span className="text-secondary">· TT</span> {sig(tt, 5)}
                  </span>
                )}
              </>
            )}
          </button>
        </PopoverTrigger>
        <PopoverContent className="w-[340px]">
          <StatusDetails json={json} />
        </PopoverContent>
      </Popover>
      <Popover open={issuesOpen} onOpenChange={(o) => setView({ issuesOpen: o })}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="flex h-full cursor-default items-center gap-1 rounded-r-full border-l border-separator pl-2 pr-3 hover:bg-hover"
            aria-label={issues.length ? `${issues.length} issues` : 'No issues'}
            data-testid="issues-button"
          >
            {issues.length === 0 ? (
              <CheckCircle2 className="size-3.5 text-secondary" />
            ) : (
              <>
                {errors > 0 ? <XOctagon className="size-3.5 text-danger" /> : <AlertTriangle className="size-3.5 text-warning" />}
                <span className="num">{issues.length}</span>
              </>
            )}
          </button>
        </PopoverTrigger>
        <PopoverContent className="w-[380px]" align="end">
          <IssuesList />
        </PopoverContent>
      </Popover>
    </div>
  )
}

const Y_SECTION = new Set(['efl', 'bfl', 'image_f_number', 'working_f_number', 'paraxial_image_height'])

/** The x and y first order at the primary wavelength (anamorphic designs). */
function AnamorphicTable({ json }: { json: JsonObject | undefined }) {
  const system = useDocument((s) => s.project.system)
  const a = anamorphicSummary(json, system)
  if (!a.anamorphic) return null
  const p = a.primary
  if (!p) {
    return <p className="mt-1.5 text-[11px] text-secondary">This design is not rotationally symmetric. The engine’s first-order data covers the y section.</p>
  }
  return (
    <table className="mt-1.5 w-full text-[12px]" data-testid="anamorphic-table">
      <thead>
        <tr className="text-[11px] text-secondary">
          <th />
          <th className="text-right font-normal">x (mm)</th>
          <th className="text-right font-normal">y (mm)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="py-[1px] pr-2 text-secondary">EFL</td>
          <td className="num text-right">{fmt(p.eflX, 4)}</td>
          <td className="num text-right">{fmt(p.eflY, 4)}</td>
        </tr>
        <tr>
          <td className="py-[1px] pr-2 text-secondary">BFL</td>
          <td className="num text-right">{fmt(p.bflX, 4)}</td>
          <td className="num text-right">{fmt(p.bflY, 4)}</td>
        </tr>
        <tr title={`Squeeze: ${a.squeezeDefinition ?? 'signed EFL y / EFL x'}`}>
          <td className="py-[1px] pr-2 text-secondary">Squeeze</td>
          <td colSpan={2} className="num text-right">
            {squeezeText(p.squeeze)}
          </td>
        </tr>
        {a.symmetry && (
          <tr>
            <td className="py-[1px] pr-2 text-secondary">Symmetry</td>
            <td colSpan={2} className="text-right">
              {a.symmetry === 'meridional' ? 'Meridional plane' : a.symmetry === 'none' ? 'None' : 'Rotational'}
            </td>
          </tr>
        )}
      </tbody>
    </table>
  )
}

/** Where analyses compute: Automatic, CPU reference (f64) or GPU. */
export function BackendPicker() {
  const backend = usePreferences((s) => s.backend)
  const setBackend = usePreferences((s) => s.setBackend)
  return (
    <div className="mt-2 flex items-center justify-between gap-2 text-[12px]" title="Where analyses compute. Automatic uses the GPU where the engine has a path and it pays off.">
      <span className="text-secondary">Compute on</span>
      <Segmented
        label="Compute on"
        value={backend}
        onChange={setBackend}
        options={BACKEND_CHOICES.map((c) => ({ value: c.value, title: c.value === 'cpu_reference' ? 'CPU (f64)' : c.title, help: c.help }))}
      />
    </div>
  )
}

function StatusDetails({ json }: { json: JsonObject | undefined }) {
  const system = useDocument((s) => s.project.system)
  const anamorphic = anamorphicSummary(json, system).anamorphic
  const engine = useWorkspace((s) => s.engine)
  const engineError = useWorkspace((s) => s.engineError)
  const snapshot = useWorkspace((s) => s.snapshot)
  const entries = (json?.entries as JsonObject[] | undefined) ?? []
  const info = engine?.info
  const rows = ['efl', 'bfl', 'image_f_number', 'working_f_number', 'image_na', 'entrance_pupil_diameter', 'exit_pupil_diameter', 'total_track', 'max_field', 'paraxial_image_height']
  return (
    <div className="space-y-3">
      <section>
        <h3 className="mb-1 text-[13px] font-semibold">First Order</h3>
        {entries.length === 0 ? (
          <p className="text-secondary">No first-order data yet.</p>
        ) : (
          <table className="w-full text-[12px]">
            <tbody>
              {rows
                .map((k) => entries.find((e) => e.key === k))
                .filter(Boolean)
                .map((e) => (
                  <tr key={String(e!.key)}>
                    <td className="py-[1px] pr-2 text-secondary">
                      {String(e!.label)}
                      {anamorphic && Y_SECTION.has(String(e!.key)) ? ' (y)' : ''}
                    </td>
                    <td className="num text-right">{fmt(nf(e!.value), 4)}</td>
                    <td className="w-8 pl-1 text-secondary">{String(e!.unit ?? '')}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
        <AnamorphicTable json={json} />
      </section>
      <section>
        <h3 className="mb-1 text-[13px] font-semibold">Engine</h3>
        {engineError ? (
          <p className="text-danger">{engineError}</p>
        ) : info ? (
          <>
            {info.isMock && (
              <p className="mb-1.5 flex gap-1.5 text-[12px] text-warning">
                <AlertTriangle className="mt-[1px] size-3.5 shrink-0" />
                Mock engine: results are synthetic. The WASM engine is not part of this build.
              </p>
            )}
            <EngineInfoTable />
            <div className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 text-[12px]">
              <span className="text-secondary">Snapshot</span>
              <span className="num font-mono text-[11px]">{snapshot ? snapshot.hash.slice(0, 12) : '–'}</span>
            </div>
            <BackendPicker />
          </>
        ) : (
          <p className="text-secondary">Starting…</p>
        )}
      </section>
    </div>
  )
}

/** Engine readiness, threads, cross-origin isolation and WebGPU. */
export function EngineInfoTable() {
  const info = useWorkspace((s) => s.engine?.info)
  if (!info) return null
  const isolated = typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : info.isolated
  const yes = (b: boolean) => (b ? 'Yes' : 'No')
  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-[1px] text-[12px]" data-testid="engine-info">
      <span className="text-secondary">Engine</span>
      <span>
        {info.name} <span className="num text-secondary">{info.version}</span>
      </span>
      <span className="text-secondary">Status</span>
      <span className="text-success">Ready</span>
      <span className="text-secondary">Threads</span>
      <span className="num">{info.threads}</span>
      <span className="text-secondary">Cross-origin isolated</span>
      <span title="Threads need SharedArrayBuffer, which needs cross-origin isolation (COOP/COEP headers).">{yes(isolated)}</span>
      <span className="text-secondary">WebGPU</span>
      <span>{yes(info.webgpu)}</span>
      <span className="text-secondary">Backends</span>
      <span>
        {info.backends
          .filter((b) => typeof b === 'string' || b.available !== false)
          .map((b) => (typeof b === 'string' ? b : (b.title ?? b.id)))
          .join(', ') || '–'}
      </span>
    </div>
  )
}
