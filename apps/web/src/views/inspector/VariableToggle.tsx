import { isVariable, solveFor, variableBoundsKind, variableOf, type ParamPath } from '@/model/params'
import { solveKindTitle, solveLetter } from '@/model/solves'
import { toggleParamVariable } from '@/state/commands'
import { useDocument } from '@/state/document'
import { cn } from '@/lib/utils'

/** The `V` toggle of a value (docs/APP.md "Solves and variables"): accent when
 *  the value is a variable; struck through when a solve sets it (a click
 *  says why); orange for a variable that later got a solve (not varied). */
export function VariableToggle({ path, className }: { path: ParamPath; className?: string }) {
  const project = useDocument((s) => s.project)
  const on = isVariable(project, path)
  const solve = solveFor(project.system, path)
  const v = on ? variableOf(project, path) : null
  const bounds = v ? variableBoundsKind(v) : 'free'
  const help = solve
    ? on
      ? `Set by a solve (${solveKindTitle(solve.kind)}): not varied. Remove the solve or the variable.`
      : `Set by a solve (${solveKindTitle(solve.kind)}), so the optimizer cannot vary it. Remove the solve to make it a variable.`
    : on
      ? `A variable${bounds === 'pinned' ? ', pinned' : bounds === 'bounded' ? ', bounded' : ''}. Click to fix the value (⌃⌘V).`
      : 'Make this value a variable of the optimizer (⌃⌘V).'
  return (
    <button
      type="button"
      aria-label={on ? 'Remove Variable' : 'Make Variable'}
      aria-pressed={on}
      title={help}
      data-testid="variable-toggle"
      onClick={() => toggleParamVariable(path)}
      className={cn(
        'relative flex h-[18px] w-[18px] shrink-0 cursor-default items-center justify-center rounded-[5px] text-[10px] font-bold',
        on && !solve && 'bg-accent text-on-accent',
        on && solve && 'bg-warning text-white',
        !on && !solve && 'text-tertiary hover:bg-hover hover:text-secondary',
        !on && solve && 'text-tertiary line-through decoration-[1.5px]',
        className,
      )}
    >
      V
      {on && bounds === 'bounded' && <span className="absolute bottom-[2px] left-1/2 h-[1.5px] w-[7px] -translate-x-1/2 rounded bg-current" />}
      {on && bounds === 'pinned' && <span className="absolute bottom-[1.5px] left-1/2 size-[3px] -translate-x-1/2 rounded-full bg-current" />}
    </button>
  )
}

/** The letter of the solve that sets a value (`M`, `C`, `F`, `P` …). */
export function SolveBadge({ path }: { path: ParamPath }) {
  const system = useDocument((s) => s.project.system)
  const solve = solveFor(system, path)
  if (!solve) return null
  return (
    <span
      className="flex h-[18px] min-w-[18px] shrink-0 items-center justify-center rounded-[5px] bg-[var(--solve-soft,rgb(142_92_247/0.16))] px-[3px] text-[10px] font-bold text-[var(--solve,#7a4fe0)] dark:text-[#b694ff]"
      title={`Set by a solve: ${solveKindTitle(solve.kind)}`}
      aria-label={`Solve: ${solveKindTitle(solve.kind)}`}
    >
      {solveLetter(solve.kind)}
    </span>
  )
}
