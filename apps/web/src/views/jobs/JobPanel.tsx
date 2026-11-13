// The shared pieces of every design job panel (apps/macos/Sources/MKLightUI/Views/Jobs/JobPanel.swift):
// `JobPanelFrame` (header, scrolling body, glass action bar), `JobActions`
// (start; running with progress, Cancel and Stop; review with Discard and Apply
// to Design; failed with Back and Try Again; done), `JobFailure` and
// `AppliedNotice`. In a job panel Return is the primary action.

import { AlertTriangle, CheckCircle2, Gauge, Loader2, Play, RotateCw, Sigma, Square, Target, Wand2 } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { JOB_TITLES, useJobs, type JobKind, type JobState } from '@/state/jobs'
import { useView } from '@/state/view'
import { cssVar, useDark } from '@/lib/theme'
import { cn, isMac } from '@/lib/utils'

export const JOB_ICONS: Record<JobKind, React.ReactNode> = {
  optimize: <Wand2 />,
  glass: <Gauge />,
  tolerance: <Target />,
  solves: <Sigma />,
}

/** Header, a scrolling body, and the glass action bar at the bottom. Return
 *  triggers the element marked `data-primary` in the bar; from a number or
 *  text field it commits the field first, then runs the action. */
export function JobPanelFrame({
  kind,
  subtitle,
  children,
  actions,
}: {
  kind: JobKind
  subtitle: string
  children: React.ReactNode
  actions: React.ReactNode
}) {
  const bar = useRef<HTMLDivElement>(null)
  return (
    <div
      className="flex h-full min-h-0 flex-col"
      data-testid={`job-${kind}`}
      onKeyDown={(e) => {
        const t = e.target as HTMLElement
        if (e.key !== 'Enter' || e.defaultPrevented || e.nativeEvent.isComposing || e.metaKey || e.ctrlKey || e.altKey) return
        if (t.tagName === 'TEXTAREA' || t.tagName === 'BUTTON' || t.tagName === 'SELECT' || t.tagName === 'A') return
        // Only the panel's own fields: a popover's search (portalled) keeps its Return.
        if (!e.currentTarget.contains(t) || t.closest('[role="combobox"],[role="menu"],[role="dialog"]')) return
        const primary = () => bar.current?.querySelector<HTMLButtonElement>('[data-primary]:not(:disabled)')
        if (t.tagName === 'INPUT') {
          const type = (t as HTMLInputElement).type
          if (type !== 'text' && type !== 'number' && type !== '') return
          // The field commits on Return (blur); the primary action runs once the
          // commit has rendered, so it sees the new value (macOS default action).
          e.preventDefault()
          ;(t as HTMLInputElement).blur()
          setTimeout(() => primary()?.click(), 0)
          return
        }
        const p = primary()
        if (p) {
          e.preventDefault()
          p.click()
        }
      }}
    >
      <div className="flex shrink-0 items-center gap-3 border-b border-separator px-4 py-3">
        <span className="flex size-8 items-center justify-center text-accent [&_svg]:size-6">{JOB_ICONS[kind]}</span>
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold">{JOB_TITLES[kind]}</h2>
          <p className="truncate text-[11px] text-secondary">{subtitle}</p>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">{children}</div>
      <div ref={bar} className="flex shrink-0 items-center gap-2 border-t border-separator px-3 py-3" data-testid="job-actions">
        {actions}
      </div>
    </div>
  )
}

function GlassAction({
  title,
  icon,
  prominent,
  destructive,
  disabled,
  help,
  onClick,
}: {
  title: string
  icon?: React.ReactNode
  prominent?: boolean
  destructive?: boolean
  disabled?: boolean
  help?: string
  onClick: () => void
}) {
  return (
    <Button
      variant={prominent ? 'prominent' : 'default'}
      className={cn('h-8 rounded-full px-4 transition-all', prominent && 'flex-1 font-semibold', destructive && 'text-danger')}
      disabled={disabled}
      title={help}
      data-primary={prominent ? '' : undefined}
      onClick={onClick}
    >
      {icon}
      {title}
    </Button>
  )
}

/** A capsule with the run's progress and its stage or live detail. */
export function GlassProgress({ job, detail }: { job: JobState; detail?: string }) {
  const pct = job.progress >= 0.01 ? `${Math.round(job.progress * 100)}%` : ''
  return (
    <div className="glass-popover flex h-8 min-w-0 flex-1 items-center gap-2 rounded-full px-3 text-[12px]" role="status" data-testid="job-progress">
      <Loader2 className="size-3.5 shrink-0 animate-spin text-secondary" />
      <span className="num min-w-0 flex-1 truncate">{detail || job.stage || 'Running'}</span>
      {pct && <span className="num shrink-0 text-secondary">{pct}</span>}
    </div>
  )
}

/** The action bar's states, built the same way by every job. */
export const JobActions = {
  start(title: string, onClick: () => void, opts?: { disabled?: boolean; help?: string }) {
    return <GlassAction title={title} icon={<Play />} prominent disabled={opts?.disabled} help={opts?.help} onClick={onClick} />
  },
  running(kind: JobKind, job: JobState, opts?: { detail?: string; canStop?: boolean; stopHelp?: string }) {
    const jobs = useJobs.getState()
    return (
      <>
        <GlassProgress job={job} detail={opts?.detail} />
        <GlassAction title="Cancel" destructive help="End the run and discard it" onClick={() => jobs.cancel(kind)} />
        <GlassAction
          title="Stop"
          icon={<Square />}
          disabled={!opts?.canStop || job.stopped}
          help={opts?.stopHelp ?? 'Stop now and keep the best result so far'}
          onClick={() => jobs.stop(kind)}
        />
      </>
    )
  },
  review(kind: JobKind, apply: () => void, opts?: { applyDisabled?: boolean; discard?: boolean }) {
    return (
      <>
        {opts?.discard !== false && (
          <GlassAction title="Discard" help="Drop this result; the design stays as it is" onClick={() => useJobs.getState().discard(kind)} />
        )}
        <GlassAction title="Apply to Design" prominent disabled={opts?.applyDisabled} help="Apply the result as one step you can undo" onClick={apply} />
      </>
    )
  },
  failed(kind: JobKind, opts?: { back?: boolean; retryDisabled?: boolean }) {
    const jobs = useJobs.getState()
    return (
      <>
        {opts?.back !== false && <GlassAction title="Back" help="Back to the job's settings" onClick={() => jobs.back(kind)} />}
        <GlassAction
          title="Try Again"
          icon={<RotateCw />}
          prominent
          disabled={opts?.retryDisabled}
          help="Run the job again with the same settings"
          onClick={() => jobs.retry(kind)}
        />
      </>
    )
  },
  done(kind: JobKind, help = 'Back to the job’s settings') {
    return <GlassAction title="Done" prominent help={help} onClick={() => useJobs.getState().done(kind)} />
  },
}

export function JobFailure({ title, reason }: { title: string; reason: string }) {
  return (
    <div className="flex flex-col items-center gap-2 px-4 py-10 text-center" role="alert" data-testid="job-failure">
      <AlertTriangle className="size-8 text-secondary" />
      <h3 className="text-[15px] font-semibold">{title}</h3>
      <p className="select-text whitespace-pre-wrap text-[12px] text-secondary">{reason}</p>
    </div>
  )
}

export function AppliedNotice() {
  return (
    <p className="mb-3 flex items-center gap-1.5 text-[12px] text-secondary" data-testid="applied-notice">
      <CheckCircle2 className="size-4 text-success" />
      Applied to the design. Undo with {isMac ? '⌘Z' : 'Ctrl+Z'}.
    </p>
  )
}

export function BigNumber({ label, value, testId }: { label: string; value: string; testId?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-secondary">{label}</div>
      <div className="num truncate text-[17px] font-medium" data-testid={testId}>
        {value}
      </div>
    </div>
  )
}

/** A small line of a series (log scale optional), for live merit and the like. */
export function Sparkline({ values, log, height = 36 }: { values: number[]; log?: boolean; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  useEffect(() => {
    const c = ref.current
    const g = c?.getContext('2d')
    if (!c || !g) return
    const w = c.clientWidth || 200
    const dpr = devicePixelRatio || 1
    c.width = w * dpr
    c.height = height * dpr
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, height)
    const v = values.filter((x) => Number.isFinite(x) && (!log || x > 0)).map((x) => (log ? Math.log10(x) : x))
    if (v.length < 2) return
    const lo = Math.min(...v), hi = Math.max(...v)
    const span = hi - lo || 1
    g.strokeStyle = cssVar('--accent')
    g.lineWidth = 1.5
    g.beginPath()
    v.forEach((y, i) => {
      const X = (i / (v.length - 1)) * (w - 2) + 1
      const Y = height - 2 - ((y - lo) / span) * (height - 4)
      if (i) g.lineTo(X, Y)
      else g.moveTo(X, Y)
    })
    g.stroke()
  }, [values, log, height, dark])
  return <canvas ref={ref} className="w-full" style={{ height }} role="img" aria-label="Progress of the merit" />
}

/** Opens the inspector on a job's page (menu commands, status capsule). */
export function showJob(kind: JobKind) {
  useView.getState().setPage(kind)
}
