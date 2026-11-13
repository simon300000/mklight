import { cn } from '@/lib/utils'

/** The shared header of the lens data and the analysis panes: icon and title,
 *  a caption state line, and small circular buttons (`PanelHeaderTitle`). */
export function PanelHeader({
  icon,
  title,
  caption,
  children,
  focused,
  titleSlot,
  className,
}: {
  icon: React.ReactNode
  title: React.ReactNode
  caption?: React.ReactNode
  children?: React.ReactNode
  focused?: boolean
  titleSlot?: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex h-9 shrink-0 items-center gap-2 border-b border-separator px-2.5', className)}>
      <span className={cn('flex shrink-0 [&_svg]:size-4', focused ? 'text-accent' : 'text-secondary')}>{icon}</span>
      <div className="flex min-w-0 flex-1 items-baseline gap-2">
        {titleSlot ?? <h2 className="truncate text-[13px] font-semibold">{title}</h2>}
        {caption && <span className="min-w-0 truncate text-[11px] text-secondary @max-[260px]:hidden">{caption}</span>}
      </div>
      <div className="flex shrink-0 items-center gap-0.5">{children}</div>
    </div>
  )
}
