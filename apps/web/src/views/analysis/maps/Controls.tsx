// Small controls of the map views: a segmented control that becomes a pop-up
// menu in narrow panes (`SegmentedOrMenu`), and the field-map column picker
// with the selected column's definition beside it (below it when narrow).
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { columnTitleWithUnit, type FieldColumn } from './fieldMap'

export function SegmentedOrMenu<T extends string>(props: {
  label: string
  value: T
  options: { value: T; title: string; help?: string }[]
  onChange: (v: T) => void
  className?: string
}) {
  return (
    <>
      <Segmented
        label={props.label}
        value={props.value}
        options={props.options}
        onChange={props.onChange}
        className={cn('@max-[420px]:hidden', props.className)}
      />
      <Select aria-label={props.label} value={props.value} onChange={(e) => props.onChange(e.target.value as T)} className={cn('hidden @max-[420px]:inline-flex', props.className)}>
        {props.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.title}
          </option>
        ))}
      </Select>
    </>
  )
}

export function ColumnPicker(props: { columns: FieldColumn[]; indices?: number[]; value: number; onChange: (c: number) => void }) {
  const idx = props.indices ?? props.columns.map((_, k) => k)
  const col = props.columns[props.value]
  return (
    <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-0.5 @max-[420px]:flex-col @max-[420px]:items-start">
      <Select aria-label="Quantity" value={String(props.value)} onChange={(e) => props.onChange(Number(e.target.value))} title={col?.definition}>
        {idx.map((k) => (
          <option key={k} value={k} title={props.columns[k].definition}>
            {columnTitleWithUnit(props.columns[k])}
          </option>
        ))}
      </Select>
      {col?.definition && <span className="min-w-0 truncate text-[11px] text-secondary @max-[420px]:line-clamp-3 @max-[420px]:whitespace-normal">{col.definition}</span>}
    </div>
  )
}

/** Label/value rows beside a map. */
export function KeyValues(props: { rows: [string, string][]; className?: string }) {
  return (
    <div className={cn('grid grid-cols-[auto_auto] content-start gap-x-3 gap-y-[1px] text-[11px]', props.className)}>
      {props.rows.map(([k, v], i) => (
        <div key={i} className="contents">
          <span className="truncate text-secondary">{k}</span>
          <span className="num text-right select-text">{v}</span>
        </div>
      ))}
    </div>
  )
}
