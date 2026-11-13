import { DropdownMenu as M } from 'radix-ui'
import { Check, ChevronRight } from 'lucide-react'
import * as React from 'react'
import { usePortalContainer } from '@/views/panels/context'
import { cn, shortcutLabel } from '@/lib/utils'

export const DropdownMenu = M.Root
export const DropdownMenuTrigger = M.Trigger
export const DropdownMenuGroup = M.Group
export const DropdownMenuSub = M.Sub
export const DropdownMenuRadioGroup = M.RadioGroup

const menuContent =
  'glass-popover z-50 min-w-[12rem] overflow-y-auto rounded-lg p-1 text-[13px] text-label max-h-[var(--radix-dropdown-menu-content-available-height)]'
const itemBase =
  'relative flex cursor-default select-none items-center gap-2 rounded-[5px] px-2 py-[3px] outline-none data-[highlighted]:bg-accent data-[highlighted]:text-on-accent data-[disabled]:opacity-40 [&_svg]:size-3.5'

export function DropdownMenuContent({ className, sideOffset = 4, ...props }: React.ComponentProps<typeof M.Content>) {
  return (
    <M.Portal container={usePortalContainer()}>
      <M.Content sideOffset={sideOffset} className={cn(menuContent, className)} {...props} />
    </M.Portal>
  )
}

export function DropdownMenuItem({
  className,
  shortcut,
  inset,
  children,
  ...props
}: React.ComponentProps<typeof M.Item> & { shortcut?: string; inset?: boolean }) {
  return (
    <M.Item className={cn(itemBase, inset && 'pl-6', className)} {...props}>
      {children}
      {shortcut && <span className="ml-auto pl-4 text-[12px] opacity-60 num">{shortcutLabel(shortcut)}</span>}
    </M.Item>
  )
}

export function DropdownMenuCheckboxItem({ className, children, ...props }: React.ComponentProps<typeof M.CheckboxItem>) {
  return (
    <M.CheckboxItem className={cn(itemBase, 'pl-6', className)} {...props}>
      <span className="absolute left-1.5 flex items-center">
        <M.ItemIndicator>
          <Check />
        </M.ItemIndicator>
      </span>
      {children}
    </M.CheckboxItem>
  )
}

export function DropdownMenuRadioItem({ className, children, ...props }: React.ComponentProps<typeof M.RadioItem>) {
  return (
    <M.RadioItem className={cn(itemBase, 'pl-6', className)} {...props}>
      <span className="absolute left-1.5 flex items-center">
        <M.ItemIndicator>
          <Check />
        </M.ItemIndicator>
      </span>
      {children}
    </M.RadioItem>
  )
}

export function DropdownMenuLabel({ className, ...props }: React.ComponentProps<typeof M.Label>) {
  return <M.Label className={cn('px-2 pt-1.5 pb-0.5 text-[11px] font-semibold text-secondary', className)} {...props} />
}

export function DropdownMenuSeparator({ className, ...props }: React.ComponentProps<typeof M.Separator>) {
  return <M.Separator className={cn('mx-1 my-1 h-px bg-separator', className)} {...props} />
}

export function DropdownMenuSubTrigger({ className, children, ...props }: React.ComponentProps<typeof M.SubTrigger>) {
  return (
    <M.SubTrigger className={cn(itemBase, 'data-[state=open]:bg-hover', className)} {...props}>
      {children}
      <ChevronRight className="ml-auto" />
    </M.SubTrigger>
  )
}

export function DropdownMenuSubContent({ className, ...props }: React.ComponentProps<typeof M.SubContent>) {
  return (
    <M.Portal container={usePortalContainer()}>
      <M.SubContent className={cn(menuContent, className)} {...props} />
    </M.Portal>
  )
}
