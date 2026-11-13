import { Tooltip as T } from 'radix-ui'
import * as React from 'react'
import { cn } from '@/lib/utils'

export const TooltipProvider = T.Provider

/** A help tag: shown after a pause, like macOS. */
export function Tip({ label, children, side = 'bottom' }: { label?: React.ReactNode; children: React.ReactElement; side?: 'top' | 'bottom' | 'left' | 'right' }) {
  if (!label) return children
  return (
    <T.Root>
      <T.Trigger asChild>{children}</T.Trigger>
      <T.Portal>
        <T.Content
          side={side}
          sideOffset={6}
          className={cn('glass-popover z-[60] max-w-xs rounded-md px-2 py-1 text-[12px] text-label')}
        >
          {label}
        </T.Content>
      </T.Portal>
    </T.Root>
  )
}
