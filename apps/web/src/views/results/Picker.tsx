// A small native picker for choosing one of several images.
export function Picker(props: { value: number; onChange: (k: number) => void; options: string[]; label: string }) {
  return (
    <div className="flex shrink-0 items-center gap-2 px-3 pt-2 text-[12px] text-secondary">
      <label className="flex items-center gap-2">
        <span>{props.label}</span>
        <select
          className="rounded-md border border-separator bg-content px-1.5 py-0.5 text-[12px] text-label"
          value={props.value}
          onChange={(e) => props.onChange(Number(e.target.value))}
        >
          {props.options.map((o, k) => (
            <option key={k} value={k}>
              {o}
            </option>
          ))}
        </select>
      </label>
    </div>
  )
}
