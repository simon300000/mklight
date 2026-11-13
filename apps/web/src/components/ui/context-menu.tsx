import { ContextMenu as M } from 'radix-ui'
import * as React from 'react'
import { usePortalContainer } from '@/views/panels/context'
import { cn, shortcutLabel } from '@/lib/utils'

export const ContextMenu = M.Root
export const ContextMenuTrigger = M.Trigger

export function ContextMenuContent({ className, ...props }: React.ComponentProps<typeof M.Content>) {
  return (
    <M.Portal container={usePortalContainer()}>
      <M.Content className={cn('glass-popover z-50 min-w-[12rem] rounded-lg p-1 text-[13px] text-label', className)} {...props} />
    </M.Portal>
  )
}

export function ContextMenuItem({ className, shortcut, children, ...props }: React.ComponentProps<typeof M.Item> & { shortcut?: string }) {
  return (
    <M.Item
      className={cn(
        'relative flex cursor-default select-none items-center gap-2 rounded-[5px] px-2 py-[3px] outline-none data-[highlighted]:bg-accent data-[highlighted]:text-on-accent data-[disabled]:opacity-40',
        className,
      )}
      {...props}
    >
      {children}
      {shortcut && <span className="ml-auto pl-4 text-[12px] opacity-60">{shortcutLabel(shortcut)}</span>}
    </M.Item>
  )
}

export function ContextMenuSeparator({ className, ...props }: React.ComponentProps<typeof M.Separator>) {
  return <M.Separator className={cn('mx-1 my-1 h-px bg-separator', className)} {...props} />
}
