// Menus for views that may sit in a separate window: the same look as
// components/ui/dropdown-menu, but portalled into the window the view is in.

import { DropdownMenu as M } from 'radix-ui'
import { Check, ChevronRight } from 'lucide-react'
import * as React from 'react'
import { cn } from '@/lib/utils'
import { usePortalContainer } from './context'

export const Menu = M.Root
export const MenuTrigger = M.Trigger
export const MenuSub = M.Sub
export const MenuRadioGroup = M.RadioGroup

const content =
  'glass-popover z-50 min-w-[12rem] overflow-y-auto rounded-lg p-1 text-[13px] text-label max-h-[var(--radix-dropdown-menu-content-available-height)]'
const item =
  'relative flex cursor-default select-none items-center gap-2 rounded-[5px] px-2 py-[3px] outline-none data-[highlighted]:bg-accent data-[highlighted]:text-on-accent data-[disabled]:opacity-40 [&_svg]:size-3.5'

export function MenuContent({ className, sideOffset = 4, ...props }: React.ComponentProps<typeof M.Content>) {
  const container = usePortalContainer()
  return (
    <M.Portal container={container}>
      <M.Content sideOffset={sideOffset} className={cn(content, className)} {...props} />
    </M.Portal>
  )
}

export function MenuItem({ className, ...props }: React.ComponentProps<typeof M.Item>) {
  return <M.Item className={cn(item, className)} {...props} />
}

export function MenuCheckboxItem({ className, children, ...props }: React.ComponentProps<typeof M.CheckboxItem>) {
  return (
    <M.CheckboxItem className={cn(item, 'pl-6', className)} {...props}>
      <span className="absolute left-1.5 flex items-center">
        <M.ItemIndicator>
          <Check />
        </M.ItemIndicator>
      </span>
      {children}
    </M.CheckboxItem>
  )
}

export function MenuRadioItem({ className, children, ...props }: React.ComponentProps<typeof M.RadioItem>) {
  return (
    <M.RadioItem className={cn(item, 'pl-6', className)} {...props}>
      <span className="absolute left-1.5 flex items-center">
        <M.ItemIndicator>
          <Check />
        </M.ItemIndicator>
      </span>
      {children}
    </M.RadioItem>
  )
}

export function MenuLabel({ className, ...props }: React.ComponentProps<typeof M.Label>) {
  return <M.Label className={cn('px-2 pt-1.5 pb-0.5 text-[11px] font-semibold text-secondary', className)} {...props} />
}

export function MenuSeparator({ className, ...props }: React.ComponentProps<typeof M.Separator>) {
  return <M.Separator className={cn('mx-1 my-1 h-px bg-separator', className)} {...props} />
}

export function MenuSubTrigger({ className, children, ...props }: React.ComponentProps<typeof M.SubTrigger>) {
  return (
    <M.SubTrigger className={cn(item, 'data-[state=open]:bg-hover', className)} {...props}>
      {children}
      <ChevronRight className="ml-auto" />
    </M.SubTrigger>
  )
}

export function MenuSubContent({ className, ...props }: React.ComponentProps<typeof M.SubContent>) {
  const container = usePortalContainer()
  return (
    <M.Portal container={container}>
      <M.SubContent className={cn(content, className)} {...props} />
    </M.Portal>
  )
}

/** A menu that opens at a point (a context menu): controlled, anchored to an
 *  invisible trigger placed at the pointer. */
export function PointMenu({ at, onClose, children }: { at: { x: number; y: number } | null; onClose: () => void; children: React.ReactNode }) {
  return (
    <M.Root open={!!at} onOpenChange={(o) => !o && onClose()} modal={false}>
      <M.Trigger asChild>
        <span aria-hidden style={{ position: 'fixed', left: at?.x ?? 0, top: at?.y ?? 0, width: 1, height: 1, pointerEvents: 'none' }} />
      </M.Trigger>
      <MenuContent align="start" sideOffset={0}>
        {children}
      </MenuContent>
    </M.Root>
  )
}
