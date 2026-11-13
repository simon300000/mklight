import { ChevronsUpDown } from 'lucide-react'
import * as React from 'react'
import { cn } from '@/lib/utils'

/** A pop-up button: the native select, styled like a macOS pop-up. */
export function Select({ className, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <span className={cn('relative inline-flex min-w-0', className)}>
      <select
        className="h-6 w-full min-w-0 appearance-none rounded-md border border-control-border bg-control pl-2 pr-6 text-[13px] text-label shadow-[0_0.5px_1px_rgb(0_0_0/0.06)] outline-none focus-visible:outline-2 focus-visible:outline-accent"
        {...props}
      >
        {children}
      </select>
      <ChevronsUpDown className="pointer-events-none absolute right-1.5 top-1/2 size-3 -translate-y-1/2 text-secondary" />
    </span>
  )
}
