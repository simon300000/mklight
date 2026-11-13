import { Popover as P } from 'radix-ui'
import * as React from 'react'
import { usePortalContainer } from '@/views/panels/context'
import { cn } from '@/lib/utils'

export const Popover = P.Root
export const PopoverTrigger = P.Trigger
export const PopoverAnchor = P.Anchor
export const PopoverClose = P.Close

export function PopoverContent({ className, align = 'center', sideOffset = 6, ...props }: React.ComponentProps<typeof P.Content>) {
  return (
    <P.Portal container={usePortalContainer()}>
      <P.Content
        align={align}
        sideOffset={sideOffset}
        className={cn('glass-popover z-50 rounded-xl p-3 text-[13px] text-label outline-none', className)}
        {...props}
      />
    </P.Portal>
  )
}
