import { cn } from '@/lib/utils'

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  className,
  label,
}: {
  value: T
  options: { value: T; title: React.ReactNode; help?: string }[]
  onChange: (v: T) => void
  className?: string
  label?: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className={cn('inline-flex rounded-md bg-selected p-[2px]', className)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          title={o.help}
          onClick={() => onChange(o.value)}
          className={cn(
            'cursor-default rounded-[5px] px-2 py-[1px] text-[12px]',
            o.value === value
              ? 'bg-content text-label shadow-[0_0.5px_2px_rgb(0_0_0/0.18)] dark:bg-white/20 dark:shadow-none'
              : 'text-secondary hover:text-label',
          )}
        >
          {o.title}
        </button>
      ))}
    </div>
  )
}
