import { MinusCircle, Plus } from 'lucide-react'
import type { JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Select } from '@/components/ui/select'
import { paramTitle, sameParam, type ParamPath } from '@/model/params'
import {
  SOLVE_KINDS,
  kindFits,
  newSolveKind,
  removeSolve,
  setSolve,
  solveFor,
  solveKindInfo,
  solveProblem,
  solveTargets,
  solvesOf,
  type Solve,
  type SolveKindId,
} from '@/model/solves'
import { imageIndex, type System } from '@/model/system'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { NumberInput } from '../SettingsForm'
import { FormRow, Section } from './Form'

const SOURCE_PARAMS: { param: string; title: string }[] = [
  { param: 'curvature', title: 'Curvature' },
  { param: 'thickness', title: 'Thickness' },
  { param: 'conic', title: 'Conic' },
  { param: 'semi_diameter', title: 'Semi-diameter' },
]

function SurfacePicker({ system, value, onChange, label }: { system: System; value: number; onChange: (id: number) => void; label: string }) {
  return (
    <Select aria-label={label} className="w-full" value={String(value)} onChange={(e) => onChange(Number(e.target.value))}>
      {system.surfaces.map((s, i) => (
        <option key={s.id} value={s.id}>
          {i === 0 ? 'Object' : i === imageIndex(system) ? 'Image' : `Surface ${i}`}
          {s.label ? ` · ${s.label}` : ''}
        </option>
      ))}
    </Select>
  )
}

/** The kind's numbers and references, edited in place ("Change Solve"). */
function SolveEditor({ system, solve, onChange }: { system: System; solve: Solve; onChange: (kind: JsonObject) => void }) {
  const k = solve.kind
  const info = solveKindInfo(String(k.type))
  const num = (key: string, label: string, unit = '', nullable = false) => (
    <FormRow key={key} label={label} unit={unit}>
      <NumberInput
        ariaLabel={`Solve ${label}`}
        value={typeof k[key] === 'number' ? (k[key] as number) : null}
        nullable={nullable}
        onCommit={(v) => onChange({ ...k, [key]: v })}
      />
    </FormRow>
  )
  switch (k.type) {
    case 'pickup': {
      const src = (k.source as ParamPath) ?? { param: 'thickness', surface: system.surfaces[1]?.id ?? 0 }
      return (
        <>
          <FormRow label="Source">
            <Select
              aria-label="Pickup parameter"
              className="w-full"
              value={String(src.param)}
              onChange={(e) => onChange({ ...k, source: { param: e.target.value, surface: (src.surface as number) ?? system.surfaces[1].id } })}
            >
              {SOURCE_PARAMS.map((p) => (
                <option key={p.param} value={p.param}>
                  {p.title}
                </option>
              ))}
              {!SOURCE_PARAMS.some((p) => p.param === src.param) && <option value={String(src.param)}>{paramTitle(system, src)}</option>}
            </Select>
          </FormRow>
          {typeof src.surface === 'number' && (
            <FormRow label="of">
              <SurfacePicker system={system} label="Pickup surface" value={src.surface} onChange={(id) => onChange({ ...k, source: { ...src, surface: id } })} />
            </FormRow>
          )}
          {num('scale', 'Scale')}
          {num('offset', 'Offset')}
        </>
      )
    }
    case 'position':
      return (
        <>
          <FormRow label="From">
            <SurfacePicker system={system} label="From surface" value={Number(k.from)} onChange={(id) => onChange({ ...k, from: id })} />
          </FormRow>
          <FormRow label="To">
            <SurfacePicker system={system} label="To surface" value={Number(k.to)} onChange={(id) => onChange({ ...k, to: id })} />
          </FormRow>
          {num('distance', 'Distance', 'mm')}
        </>
      )
    case 'center_of_curvature':
      return (
        <FormRow label="On vertex of">
          <SurfacePicker system={system} label="Centre of curvature surface" value={Number(k.surface)} onChange={(id) => onChange({ ...k, surface: id })} />
        </FormRow>
      )
    case 'edge_thickness':
      return (
        <>
          {num('value', 'Edge thickness', 'mm')}
          {num('radius', 'At radius', 'mm', true)}
        </>
      )
    default:
      return info?.value ? num(info.value.key, info.value.label, info.value.unit) : null
  }
}

/** Solves on this surface's parameters (docs/SOLVES.md): each with its rule
 *  and numbers, a kind switch and Remove; Add Solve offers every kind that
 *  fits each parameter. Apply Solves (⌃⌘L) previews and applies them. */
export function SolvesSection({ index }: { index: number }) {
  const system = useDocument((s) => s.project.system)
  const edit = useDocument((s) => s.edit)
  const surface = system.surfaces[index]
  const targets = solveTargets(system, index)
  const solves = solvesOf(system).filter((s) => s.target.surface === surface?.id)
  if (!surface || (!solves.length && !targets.length)) return null

  const change = (target: ParamPath, kind: JsonObject) => {
    const why = solveProblem({ target, kind })
    if (why) return useView.getState().toast(why, 'warning')
    edit('Change Solve', (s) => setSolve(s, target, kind))
  }
  const add = (target: ParamPath, id: SolveKindId) =>
    edit('Add Solve', (s) => {
      setSolve(s, target, newSolveKind(id, s, target))
    })

  return (
    <Section
      title="Solves"
      testId="solves-section"
      footer={solves.length ? 'Solved values follow on Apply Solves (⌃⌘L); the optimizer applies them first.' : undefined}
      accessory={
        targets.length > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm">
                <Plus /> Add Solve
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {targets.map((t) => (
                <DropdownMenuSub key={t.title}>
                  <DropdownMenuSubTrigger>
                    {t.title}
                    {solveFor(system, t.path) && <span className="text-[11px] opacity-60">· solved</span>}
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    {t.kinds.map((id) => {
                      const info = solveKindInfo(id)!
                      return (
                        <DropdownMenuItem key={id} onSelect={() => add(t.path, id)}>
                          {info.title}
                        </DropdownMenuItem>
                      )
                    })}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        )
      }
    >
      {solves.length === 0 ? (
        <p className="py-1 text-[12px] text-secondary">No solves on this surface.</p>
      ) : (
        solves.map((s) => {
          const fitting = SOLVE_KINDS.filter((k) => kindFits(k.id, s.target))
          return (
            <div key={JSON.stringify(s.target)} className="border-b border-separator py-1 last:border-0" data-testid="solve-row">
              <div className="flex items-center gap-2">
                <span className="w-[4.5rem] shrink-0 truncate text-[12px] font-medium" title={paramTitle(system, s.target)}>
                  {targets.find((t) => sameParam(system, t.path, s.target))?.title ?? paramTitle(system, s.target)}
                </span>
                <Select
                  aria-label="Solve kind"
                  className="min-w-0 flex-1"
                  value={String(s.kind.type)}
                  onChange={(e) => change(s.target, newSolveKind(e.target.value as SolveKindId, system, s.target))}
                >
                  {fitting.map((k) => (
                    <option key={k.id} value={k.id}>
                      {k.title}
                    </option>
                  ))}
                </Select>
                <button
                  type="button"
                  aria-label="Remove Solve"
                  title="Remove Solve"
                  className="text-secondary hover:text-label"
                  onClick={() => edit('Remove Solve', (sys) => removeSolve(sys, s.target))}
                >
                  <MinusCircle className="size-3.5" />
                </button>
              </div>
              <SolveEditor system={system} solve={s} onChange={(k) => change(s.target, k)} />
            </div>
          )
        })
      )}
    </Section>
  )
}
