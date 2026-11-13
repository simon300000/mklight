// Which catalogue glasses are candidates, in plain terms (CandidateForm and
// OverrideButton of apps/macos/Sources/MKLightUI/Views/Jobs/GlassSubstitutionPanel.swift).

import { ChevronDown, SlidersHorizontal } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { CatalogInfo } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ALL_STATUSES, candidatesSummary, parseNames, type CandidateStatus, type GlassCandidates, type GlassSlot } from '@/model/glass'
import { useGlassSubstitution } from '@/state/glassSubstitution'
import { useWorkspace } from '@/state/workspace'
import { cn } from '@/lib/utils'
import { NumberInput } from '../../SettingsForm'
import { FormRow, TextInput } from '../../inspector/Form'

/** The engine's loaded catalogues. */
export function useCatalogs(): CatalogInfo[] {
  const engine = useWorkspace((s) => s.engine)
  const [list, setList] = useState<CatalogInfo[]>([])
  useEffect(() => {
    let live = true
    engine
      ?.catalogs()
      .then((c) => live && setList(c))
      .catch(() => {})
    return () => {
      live = false
    }
  }, [engine])
  return list
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

function MenuButton({ title, label }: { title: string; label: string }) {
  return (
    <DropdownMenuTrigger asChild>
      <Button size="sm" className="max-w-full min-w-0" aria-label={label}>
        <span className="truncate">{title}</span>
        <ChevronDown className="size-3" />
      </Button>
    </DropdownMenuTrigger>
  )
}

export function CandidateForm({ candidates: c, onChange, catalogs }: { candidates: GlassCandidates; onChange: (c: GlassCandidates) => void; catalogs: CatalogInfo[] }) {
  const [limits, setLimits] = useState(false)
  const set = (p: Partial<GlassCandidates>) => onChange({ ...c, ...p })
  const limit = (title: string, key: 'minTransmission' | 'maxRelativeCost' | 'maxClimateClass' | 'maxStainClass', help: string, max?: number) => (
    <FormRow className="grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_2rem]" label={title} help={help}>
      <NumberInput ariaLabel={title} value={c[key]} nullable min={0} max={max} placeholder="none" onCommit={(v) => set({ [key]: v })} />
    </FormRow>
  )
  return (
    <div className="text-[12px]" data-testid="candidate-form">
      <FormRow label="Catalogs">
        <DropdownMenu>
          <MenuButton label="Catalogs" title={c.catalogs.length ? c.catalogs.join(', ') : 'All loaded'} />
          <DropdownMenuContent align="end">
            <DropdownMenuItem onSelect={() => set({ catalogs: [] })}>All loaded</DropdownMenuItem>
            <DropdownMenuSeparator />
            {catalogs.map((k) => (
              <DropdownMenuCheckboxItem
                key={k.name}
                checked={c.catalogs.includes(k.name)}
                onSelect={(e) => e.preventDefault()}
                onCheckedChange={(on) => set({ catalogs: on ? [...c.catalogs, k.name] : c.catalogs.filter((x) => x !== k.name) })}
              >
                {k.name} <span className="opacity-60">({k.glass_count})</span>
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </FormRow>
      <FormRow label="Status">
        <DropdownMenu>
          <MenuButton label="Status" title={ALL_STATUSES.filter((s) => c.statuses.includes(s)).map(cap).join(', ')} />
          <DropdownMenuContent align="end">
            {ALL_STATUSES.map((s) => (
              <DropdownMenuCheckboxItem
                key={s}
                checked={c.statuses.includes(s)}
                onSelect={(e) => e.preventDefault()}
                onCheckedChange={(on) => {
                  if (on) set({ statuses: [...c.statuses, s as CandidateStatus] })
                  else if (c.statuses.length > 1) set({ statuses: c.statuses.filter((x) => x !== s) })
                }}
              >
                {cap(s)}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </FormRow>
      <FormRow label="Choose">
        <Select aria-label="Choose" value={c.rule} onChange={(e) => set({ rule: e.target.value as GlassCandidates['rule'] })}>
          <option value="nearest">The nearest glasses</option>
          <option value="window">Glasses within a window</option>
        </Select>
      </FormRow>
      {c.rule === 'nearest' ? (
        <FormRow label="Nearest" help="The glasses nearest to the current one in n_d and V_d">
          <NumberInput ariaLabel="Nearest glasses" value={c.nearest} integer min={1} max={200} onCommit={(v) => v != null && set({ nearest: v })} />
        </FormRow>
      ) : (
        <>
          <FormRow label="n_d ±">
            <NumberInput ariaLabel="n_d window" value={c.windowNd} min={1e-4} onCommit={(v) => set({ windowNd: v ?? 0.05 })} />
          </FormRow>
          <FormRow label="V_d ±">
            <NumberInput ariaLabel="V_d window" value={c.windowVd} min={0.1} onCommit={(v) => set({ windowVd: v ?? 5 })} />
          </FormRow>
        </>
      )}
      <FormRow label="Always include" help="Glass names or catalog:name, comma-separated">
        <TextInput ariaLabel="Always include" value={c.include.join(', ')} placeholder="N-BK7, SCHOTT:F2" onCommit={(t) => set({ include: parseNames(t) })} />
      </FormRow>
      <FormRow label="Never use" help="Glass names or catalog:name, comma-separated">
        <TextInput ariaLabel="Never use" value={c.exclude.join(', ')} placeholder="names" onCommit={(t) => set({ exclude: parseNames(t) })} />
      </FormRow>
      <button
        type="button"
        className="mt-1 flex cursor-default items-center gap-1 text-[12px] text-secondary hover:text-label"
        aria-expanded={limits}
        onClick={() => setLimits(!limits)}
      >
        <ChevronDown className={cn('size-3 transition-transform', !limits && '-rotate-90')} />
        Limits
      </button>
      {limits && (
        <div className="mt-1">
          {limit('Min. transmission', 'minTransmission', "Internal transmission at the shortest wavelength through the lens's center thickness (0…1)", 1)}
          {limit('Max. relative cost', 'maxRelativeCost', 'Catalog relative cost')}
          {limit('Max. climate class', 'maxClimateClass', 'Climate resistance class (CR)')}
          {limit('Max. stain class', 'maxStainClass', 'Stain resistance class (FR)')}
        </div>
      )}
    </div>
  )
}

/** Candidate settings for one lens, in a popover. */
export function OverrideButton({ slot, catalogs }: { slot: GlassSlot; catalogs: CatalogInfo[] }) {
  const own = useGlassSubstitution((s) => s.overrides[slot.id] ?? null)
  const every = useGlassSubstitution((s) => s.candidates)
  const setOverride = useGlassSubstitution((s) => s.setOverride)
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="iconSm"
          className={cn(own && 'text-accent')}
          title="Candidates for this lens only"
          aria-label={`Candidates for surface ${slot.index}`}
        >
          <SlidersHorizontal />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[340px]" align="end">
        <h4 className="mb-2 text-[13px] font-semibold">
          Candidates for S{slot.index} · {slot.material}
        </h4>
        <label className="mb-2 flex items-center gap-2 text-[12px]">
          <Switch checked={own != null} onCheckedChange={(on) => setOverride(slot.id, on ? { ...every } : null)} aria-label="Own candidates for this lens" />
          Own candidates for this lens
        </label>
        {own ? (
          <CandidateForm candidates={own} onChange={(c) => setOverride(slot.id, c)} catalogs={catalogs} />
        ) : (
          <p className="text-[12px] text-secondary">Uses the candidates for every lens ({candidatesSummary(every)}).</p>
        )}
      </PopoverContent>
    </Popover>
  )
}
