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
import { useT } from '@/i18n'
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

const SOURCE_PARAMS: { param: string; key: 'curvature' | 'thickness' | 'conic' | 'semiDiameter' }[] = [
  { param: 'curvature', key: 'curvature' },
  { param: 'thickness', key: 'thickness' },
  { param: 'conic', key: 'conic' },
  { param: 'semi_diameter', key: 'semiDiameter' },
]

function SurfacePicker({ system, value, onChange, label }: { system: System; value: number; onChange: (id: number) => void; label: string }) {
  const t = useT()
  return (
    <Select aria-label={label} className="w-full" value={String(value)} onChange={(e) => onChange(Number(e.target.value))}>
      {system.surfaces.map((s, i) => (
        <option key={s.id} value={s.id}>
          {i === 0 ? t('inspector.solves.object') : i === imageIndex(system) ? t('inspector.solves.image') : t('common.surfaceN', { n: i })}
          {s.label ? ` · ${s.label}` : ''}
        </option>
      ))}
    </Select>
  )
}

/** The kind's numbers and references, edited in place ("Change Solve"). */
function SolveEditor({ system, solve, onChange }: { system: System; solve: Solve; onChange: (kind: JsonObject) => void }) {
  const t = useT()
  const k = solve.kind
  const info = solveKindInfo(String(k.type))
  const num = (key: string, label: string, unit = '', nullable = false) => (
    <FormRow key={key} label={label} unit={unit}>
      <NumberInput
        ariaLabel={t('inspector.solves.valueLabel', { label })}
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
          <FormRow label={t('inspector.solves.source')}>
            <Select
              aria-label={t('inspector.solves.pickupParameter')}
              className="w-full"
              value={String(src.param)}
              onChange={(e) => onChange({ ...k, source: { param: e.target.value, surface: (src.surface as number) ?? system.surfaces[1].id } })}
            >
              {SOURCE_PARAMS.map((p) => (
                <option key={p.param} value={p.param}>
                  {t(`inspector.solves.sources.${p.key}`)}
                </option>
              ))}
              {!SOURCE_PARAMS.some((p) => p.param === src.param) && <option value={String(src.param)}>{paramTitle(system, src)}</option>}
            </Select>
          </FormRow>
          {typeof src.surface === 'number' && (
            <FormRow label={t('inspector.solves.of')}>
              <SurfacePicker system={system} label={t('inspector.solves.pickupSurface')} value={src.surface} onChange={(id) => onChange({ ...k, source: { ...src, surface: id } })} />
            </FormRow>
          )}
          {num('scale', t('inspector.solves.scale'))}
          {num('offset', t('inspector.solves.offset'))}
        </>
      )
    }
    case 'position':
      return (
        <>
          <FormRow label={t('inspector.solves.from')}>
            <SurfacePicker system={system} label={t('inspector.solves.fromSurface')} value={Number(k.from)} onChange={(id) => onChange({ ...k, from: id })} />
          </FormRow>
          <FormRow label={t('inspector.solves.to')}>
            <SurfacePicker system={system} label={t('inspector.solves.toSurface')} value={Number(k.to)} onChange={(id) => onChange({ ...k, to: id })} />
          </FormRow>
          {num('distance', t('inspector.solves.value.distance'), 'mm')}
        </>
      )
    case 'center_of_curvature':
      return (
        <FormRow label={t('inspector.solves.onVertexOf')}>
          <SurfacePicker system={system} label={t('inspector.solves.centerSurface')} value={Number(k.surface)} onChange={(id) => onChange({ ...k, surface: id })} />
        </FormRow>
      )
    case 'edge_thickness':
      return (
        <>
          {num('value', t('inspector.solves.value.edgeThickness'), 'mm')}
          {num('radius', t('inspector.solves.atRadius'), 'mm', true)}
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
  const t = useT()
  const system = useDocument((s) => s.project.system)
  const edit = useDocument((s) => s.edit)
  const surface = system.surfaces[index]
  const targets = solveTargets(system, index)
  const solves = solvesOf(system).filter((s) => s.target.surface === surface?.id)
  if (!surface || (!solves.length && !targets.length)) return null

  const change = (target: ParamPath, kind: JsonObject) => {
    const why = solveProblem({ target, kind })
    if (why) return useView.getState().toast(why, 'warning')
    edit(t('inspector.solves.undoChange'), (s) => setSolve(s, target, kind))
  }
  const add = (target: ParamPath, id: SolveKindId) =>
    edit(t('inspector.solves.undoAdd'), (s) => {
      setSolve(s, target, newSolveKind(id, s, target))
    })

  return (
    <Section
      title={t('inspector.solves.title')}
      testId="solves-section"
      footer={solves.length ? t('inspector.solves.footer') : undefined}
      accessory={
        targets.length > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm">
                <Plus /> {t('inspector.solves.add')}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {targets.map((tg) => (
                <DropdownMenuSub key={tg.title}>
                  <DropdownMenuSubTrigger>
                    {tg.title}
                    {solveFor(system, tg.path) && <span className="text-[11px] opacity-60">{t('inspector.solves.solvedMark')}</span>}
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    {tg.kinds.map((id) => {
                      const info = solveKindInfo(id)!
                      return (
                        <DropdownMenuItem key={id} onSelect={() => add(tg.path, id)}>
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
        <p className="py-1 text-[12px] text-secondary">{t('inspector.solves.none')}</p>
      ) : (
        solves.map((s) => {
          const fitting = SOLVE_KINDS.filter((k) => kindFits(k.id, s.target))
          return (
            <div key={JSON.stringify(s.target)} className="border-b border-separator py-1 last:border-0" data-testid="solve-row">
              <div className="flex items-center gap-2">
                <span className="w-[4.5rem] shrink-0 truncate text-[12px] font-medium" title={paramTitle(system, s.target)}>
                  {targets.find((tg) => sameParam(system, tg.path, s.target))?.title ?? paramTitle(system, s.target)}
                </span>
                <Select
                  aria-label={t('inspector.solves.kind')}
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
                  aria-label={t('inspector.solves.remove')}
                  title={t('inspector.solves.remove')}
                  className="text-secondary hover:text-label"
                  onClick={() => edit(t('inspector.solves.remove'), (sys) => removeSolve(sys, s.target))}
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
