// The design's variables with their bounds (apps/macos/Sources/MKLightUI/Views/Jobs/MeritEditor.swift,
// `VariablesBoundsList`): min and max inline (empty is unbounded) in the row's
// unit, pin (equal bounds: kept but held), remove; out-of-bounds values,
// automatic semi-diameters, solved and vanished parameters are flagged.
// Curvatures are bounded as a radius window in mm by default (model/bounds.ts).

import { CircleHelp, Pin, PinOff, XCircle } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import {
  RADIUS_HELP,
  boundsActionName,
  boundsOf,
  counterpart,
  curvatureBounds,
  defaultBounds,
  isCurvature,
  isPinned,
  outOfBounds,
  passesThroughFlat,
  radiusWindow,
  type Bounds,
  type CurvatureBoundsUnit,
} from '@/model/bounds'
import { sig } from '@/model/format'
import { getParam, paramTitle, paramUnit, sameParam, solveFor, surfaceIndexOfId, variablesOf, type VariableSpec } from '@/model/params'
import { radiusFromCurvature, type System } from '@/model/system'
import { useDocument } from '@/state/document'
import { editVariables, useOptimize } from '@/state/optimize'
import { cn } from '@/lib/utils'
import { NumberInput } from '../SettingsForm'

export function VariablesBoundsList() {
  const project = useDocument((s) => s.project)
  const unit = useOptimize((s) => s.curvatureUnit)
  const setUnit = useOptimize((s) => s.setCurvatureUnit)
  const vars = variablesOf(project)
  const system = project.system
  return (
    <div data-testid="variables-list">
      {vars.some((v) => isCurvature(v.param)) && (
        <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="text-[11px] text-secondary">Curvatures as</span>
          <Segmented<CurvatureBoundsUnit>
            label="Bound curvatures as"
            value={unit}
            onChange={setUnit}
            options={[
              { value: 'radius', title: <span className="whitespace-nowrap">Radius mm</span>, help: 'Enter and read the bounds of radii as a radius window in mm' },
              { value: 'curvature', title: <span className="whitespace-nowrap">Curvature 1/mm</span>, help: 'Enter and read the bounds as the curvature c = 1/R the optimizer uses' },
            ]}
          />
          <span className="ml-auto text-secondary" title={RADIUS_HELP} aria-label="About radius bounds">
            <CircleHelp className="size-3.5" />
          </span>
        </div>
      )}
      {vars.map((v, k) => (
        <VariableRow key={JSON.stringify(v.param)} system={system} variable={v} unit={unit} first={k === 0} />
      ))}
    </div>
  )
}

function VariableRow({ system, variable, unit: shownUnit, first }: { system: System; variable: VariableSpec; unit: CurvatureBoundsUnit; first: boolean }) {
  const [rejection, setRejection] = useState<string | null>(null)
  const path = variable.param
  const bounds = boundsOf(variable)
  const value = getParam(system, path)
  const asRadius = isCurvature(path) && shownUnit === 'radius'
  const label = paramTitle(system, path)
  const unit = asRadius ? 'mm' : paramUnit(path)
  const shown = asRadius ? radiusWindow(bounds.lower, bounds.upper) : { min: bounds.lower, max: bounds.upper }
  const pinned = isPinned(bounds)
  const solve = solveFor(system, path)

  const setBounds = (b: Bounds) => {
    if (b.lower === bounds.lower && b.upper === bounds.upper) return
    editVariables(boundsActionName(bounds, b), (list) =>
      list.map((v) => (sameParam(system, v.param, path) ? { ...v, lower: b.lower, upper: b.upper } : v)),
    )
  }

  const setShown = (min: number | null, max: number | null) => {
    if (!asRadius) {
      setRejection(null)
      setBounds({ lower: min, upper: max })
      return
    }
    const r = curvatureBounds(min, max)
    if (r.ok) {
      setRejection(null)
      setBounds(r.bounds)
    } else setRejection(r.error)
  }

  let problem: string | null = rejection
  if (!problem && solve) problem = 'Set by a solve: not varied'
  if (!problem && path.param === 'semi_diameter') {
    const i = typeof path.surface === 'number' ? surfaceIndexOfId(system, path.surface) : -1
    if (i >= 0 && system.surfaces[i].semi_diameter.mode !== 'fixed') problem = 'The semi-diameter is automatic: set a value in the lens data first.'
  }
  if (!problem && value == null) problem = 'This parameter does not exist on the surface any more.'
  if (!problem && bounds.lower != null && bounds.upper != null && bounds.lower > bounds.upper) problem = 'The minimum is above the maximum.'
  if (!problem && value != null && outOfBounds(bounds, value)) {
    const now = asRadius ? `R ${sig(radiusFromCurvature(value), 6)} mm` : `${sig(value, 6)}${unit === '–' ? '' : ' ' + unit}`
    problem = `Now ${now}: outside its bounds`
  }

  let caption = ''
  if (problem) caption = problem
  else if (value != null) {
    if (isCurvature(path)) {
      caption = asRadius
        ? `now R ${sig(radiusFromCurvature(value), 6)}` + (passesThroughFlat(shown) ? ' · through ∞' : '')
        : `now c ${sig(value, 4)}`
    } else caption = `now ${sig(value, 6)}`
  }

  const other = counterpart(bounds, shownUnit)
  let help = other ? `Bounds as ${other}. ` : ''
  if (asRadius) help += RADIUS_HELP
  else if (isCurvature(path)) help += 'Bounds on the curvature c = 1/R (1/mm).'
  else help += unit === '–' ? 'Bounds of a dimensionless parameter.' : `Bounds in ${unit}.`
  const solvedHelp = solve
    ? 'A solve sets this value, so the optimizer leaves the variable out of the run. Remove the variable, or remove the solve to vary it.'
    : null

  return (
    <div className={cn('py-1', !first && 'border-t border-separator')} data-testid="variable-row">
      <div className="flex min-w-0 items-center gap-1" title={solvedHelp ?? problem ?? help}>
        <span className="shrink-0 text-[13px]">{label}</span>
        {pinned && <Pin className="size-3 shrink-0 text-accent" aria-label="pinned" />}
        <span className="num min-w-0 flex-1 truncate pl-1 text-[11px] text-secondary">{problem ? '' : caption}</span>
        <Button
          variant="ghost"
          size="iconSm"
          disabled={value == null}
          aria-label={pinned ? 'Unpin' : 'Pin'}
          title={pinned ? 'Unpin: let the optimizer vary it again' : 'Pin at its value (equal bounds): keep it as a variable but hold it'}
          onClick={() => {
            if (pinned) setBounds(defaultBounds(path, system))
            else if (value != null) setBounds({ lower: value, upper: value })
          }}
        >
          {pinned ? <PinOff /> : <Pin />}
        </Button>
        <Button
          variant="ghost"
          size="iconSm"
          aria-label="Remove variable"
          title="Fix this parameter (remove the variable)"
          onClick={() => editVariables('Remove Variable', (list) => list.filter((v) => !sameParam(system, v.param, path)))}
        >
          <XCircle />
        </Button>
      </div>
      <div className="flex items-center gap-1" title={help}>
        <NumberInput
          ariaLabel={`Minimum of ${label}${unit === '–' ? '' : ' in ' + unit}`}
          value={shown.min}
          nullable
          placeholder="min −∞"
          onCommit={(v) => setShown(v, shown.max)}
        />
        <span className="text-secondary">…</span>
        <NumberInput
          ariaLabel={`Maximum of ${label}${unit === '–' ? '' : ' in ' + unit}`}
          value={shown.max}
          nullable
          placeholder={asRadius ? 'max ∞' : 'max +∞'}
          onCommit={(v) => setShown(shown.min, v)}
        />
        <span className="w-9 shrink-0 truncate text-[11px] text-secondary" title={unit === '–' ? 'Dimensionless' : `The bounds are in ${unit}`}>
          {unit}
        </span>
      </div>
      {problem && (
        <p className="pt-0.5 text-[11px] text-warning" role="status">
          {problem}
        </p>
      )}
    </div>
  )
}
