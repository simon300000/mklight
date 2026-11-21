// Which catalogue glasses are candidates, in plain terms (CandidateForm and
// OverrideButton of apps/macos/Sources/MKLightUI/Views/Jobs/GlassSubstitutionPanel.swift).

import { ChevronDown, SlidersHorizontal } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { CatalogInfo } from '@/engine/types'
import { useT } from '@/i18n'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ALL_STATUSES, candidatesSummary, parseNames, statusTitle, type CandidateStatus, type GlassCandidates, type GlassSlot } from '@/model/glass'
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
  const t = useT()
  const [limits, setLimits] = useState(false)
  const set = (p: Partial<GlassCandidates>) => onChange({ ...c, ...p })
  const limit = (title: string, key: 'minTransmission' | 'maxRelativeCost' | 'maxClimateClass' | 'maxStainClass', help: string, max?: number) => (
    <FormRow className="grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_2rem]" label={title} help={help}>
      <NumberInput ariaLabel={title} value={c[key]} nullable min={0} max={max} placeholder={t('common.none').toLowerCase()} onCommit={(v) => set({ [key]: v })} />
    </FormRow>
  )
  return (
    <div className="text-[12px]" data-testid="candidate-form">
      <FormRow label={t('glass.cand.catalogs')}>
        <DropdownMenu>
          <MenuButton label={t('glass.cand.catalogs')} title={c.catalogs.length ? c.catalogs.join(', ') : t('glass.cand.allLoaded')} />
          <DropdownMenuContent align="end">
            <DropdownMenuItem onSelect={() => set({ catalogs: [] })}>{t('glass.cand.allLoaded')}</DropdownMenuItem>
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
      <FormRow label={t('glass.cand.status')}>
        <DropdownMenu>
          <MenuButton label={t('glass.cand.status')} title={ALL_STATUSES.filter((s) => c.statuses.includes(s)).map(statusTitle).join(t('glass.list.sep'))} />
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
                {statusTitle(s)}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </FormRow>
      <FormRow label={t('glass.cand.choose')}>
        <Select aria-label={t('glass.cand.choose')} value={c.rule} onChange={(e) => set({ rule: e.target.value as GlassCandidates['rule'] })}>
          <option value="nearest">{t('glass.cand.ruleNearest')}</option>
          <option value="window">{t('glass.cand.ruleWindow')}</option>
        </Select>
      </FormRow>
      {c.rule === 'nearest' ? (
        <FormRow label={t('glass.cand.nearest')} help={t('glass.cand.nearestHelp')}>
          <NumberInput ariaLabel={t('glass.cand.nearestLabel')} value={c.nearest} integer min={1} max={200} onCommit={(v) => v != null && set({ nearest: v })} />
        </FormRow>
      ) : (
        <>
          <FormRow label="n_d ±">
            <NumberInput ariaLabel={t('glass.cand.ndWindow')} value={c.windowNd} min={1e-4} onCommit={(v) => set({ windowNd: v ?? 0.05 })} />
          </FormRow>
          <FormRow label="V_d ±">
            <NumberInput ariaLabel={t('glass.cand.vdWindow')} value={c.windowVd} min={0.1} onCommit={(v) => set({ windowVd: v ?? 5 })} />
          </FormRow>
        </>
      )}
      <FormRow label={t('glass.cand.include')} help={t('glass.cand.namesHelp')}>
        <TextInput ariaLabel={t('glass.cand.include')} value={c.include.join(', ')} placeholder="N-BK7, SCHOTT:F2" onCommit={(t) => set({ include: parseNames(t) })} />
      </FormRow>
      <FormRow label={t('glass.cand.exclude')} help={t('glass.cand.namesHelp')}>
        <TextInput ariaLabel={t('glass.cand.exclude')} value={c.exclude.join(', ')} placeholder={t('glass.cand.names')} onCommit={(t) => set({ exclude: parseNames(t) })} />
      </FormRow>
      <button
        type="button"
        className="mt-1 flex cursor-default items-center gap-1 text-[12px] text-secondary hover:text-label"
        aria-expanded={limits}
        onClick={() => setLimits(!limits)}
      >
        <ChevronDown className={cn('size-3 transition-transform', !limits && '-rotate-90')} />
        {t('glass.cand.limits')}
      </button>
      {limits && (
        <div className="mt-1">
          {limit(t('glass.cand.minTransmission'), 'minTransmission', t('glass.cand.minTransmissionHelp'), 1)}
          {limit(t('glass.cand.maxCost'), 'maxRelativeCost', t('glass.cand.maxCostHelp'))}
          {limit(t('glass.cand.maxClimate'), 'maxClimateClass', t('glass.cand.maxClimateHelp'))}
          {limit(t('glass.cand.maxStain'), 'maxStainClass', t('glass.cand.maxStainHelp'))}
        </div>
      )}
    </div>
  )
}

/** Candidate settings for one lens, in a popover. */
export function OverrideButton({ slot, catalogs }: { slot: GlassSlot; catalogs: CatalogInfo[] }) {
  const t = useT()
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
          title={t('glass.cand.forLensOnly')}
          aria-label={t('glass.cand.forSurface', { n: slot.index })}
        >
          <SlidersHorizontal />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[340px]" align="end">
        <h4 className="mb-2 text-[13px] font-semibold">
          {t('glass.cand.forSlot', { n: slot.index, material: slot.material })}
        </h4>
        <label className="mb-2 flex items-center gap-2 text-[12px]">
          <Switch checked={own != null} onCheckedChange={(on) => setOverride(slot.id, on ? { ...every } : null)} aria-label={t('glass.cand.own')} />
          {t('glass.cand.own')}
        </label>
        {own ? (
          <CandidateForm candidates={own} onChange={(c) => setOverride(slot.id, c)} catalogs={catalogs} />
        ) : (
          <p className="text-[12px] text-secondary">{t('glass.cand.usesAll', { summary: candidatesSummary(every) })}</p>
        )}
      </PopoverContent>
    </Popover>
  )
}
