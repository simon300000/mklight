import { Switch as S } from 'radix-ui'
import * as React from 'react'
import { cn } from '@/lib/utils'

export function Switch({ className, ...props }: React.ComponentProps<typeof S.Root>) {
  return (
    <S.Root
      className={cn(
        'inline-flex h-[18px] w-[30px] shrink-0 cursor-default items-center rounded-full border border-control-border bg-selected transition-colors data-[state=checked]:border-transparent data-[state=checked]:bg-accent',
        className,
      )}
      {...props}
    >
      <S.Thumb className="block size-[14px] translate-x-[1px] rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[13px]" />
    </S.Root>
  )
}
