import { cn } from '@/lib/utils'
import { usePanelTabbed } from '@/views/panels/context'

/** The shared header of the lens data and the analysis panes: icon and title,
 *  a caption state line, and small circular buttons (`PanelHeaderTitle`).
 *  As one of several tabs (`inTabGroup`, by default from the panel's group)
 *  the tab carries the icon and title: the header keeps the caption, the
 *  buttons and a title slot's other controls (the analysis menu's chevron),
 *  and the title stays for assistive technology only. */
export function PanelHeader({
  icon,
  title,
  caption,
  children,
  focused,
  titleSlot,
  className,
  inTabGroup,
}: {
  icon: React.ReactNode
  title: React.ReactNode
  caption?: React.ReactNode
  children?: React.ReactNode
  focused?: boolean
  titleSlot?: React.ReactNode
  className?: string
  inTabGroup?: boolean
}) {
  const tabbedGroup = usePanelTabbed()
  const tabbed = inTabGroup ?? tabbedGroup
  return (
    <div className={cn('flex h-9 shrink-0 items-center gap-2 border-b border-separator px-2.5', className)} data-tabbed={tabbed || undefined}>
      {!tabbed && <span className={cn('flex shrink-0 [&_svg]:size-4', focused ? 'text-accent' : 'text-secondary')}>{icon}</span>}
      <div className={cn('flex min-w-0 flex-1 items-baseline gap-2', tabbed && 'items-center gap-1 [&_h2]:sr-only')}>
        {titleSlot ?? <h2 className="truncate text-[13px] font-semibold">{title}</h2>}
        {caption && <span className="min-w-0 truncate text-[11px] text-secondary @max-[260px]:hidden">{caption}</span>}
      </div>
      <div className="flex shrink-0 items-center gap-0.5">{children}</div>
    </div>
  )
}
