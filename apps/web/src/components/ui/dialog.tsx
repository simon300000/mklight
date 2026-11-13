import { Dialog as D } from 'radix-ui'
import { X } from 'lucide-react'
import * as React from 'react'
import { usePortalContainer } from '@/views/panels/context'
import { cn } from '@/lib/utils'

export const Dialog = D.Root
export const DialogTrigger = D.Trigger
export const DialogClose = D.Close
export const DialogTitle = D.Title
export const DialogDescription = D.Description

export function DialogContent({ className, children, ...props }: React.ComponentProps<typeof D.Content>) {
  return (
    <D.Portal container={usePortalContainer()}>
      <D.Overlay className="fixed inset-0 z-40 bg-black/20" />
      <D.Content
        className={cn(
          'fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[min(720px,92vw)] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-2xl bg-content text-label shadow-popover outline-none',
          className,
        )}
        {...props}
      >
        {children}
        <D.Close className="absolute right-3 top-3 rounded-full p-1 text-secondary hover:bg-hover" aria-label="Close">
          <X className="size-4" />
        </D.Close>
      </D.Content>
    </D.Portal>
  )
}
