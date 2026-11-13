import { cva, type VariantProps } from 'class-variance-authority'
import { Slot } from 'radix-ui'
import * as React from 'react'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md text-[13px] font-normal transition-colors disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-4 [&_svg]:shrink-0 cursor-default',
  {
    variants: {
      variant: {
        default: 'bg-control border border-control-border shadow-[0_0.5px_1px_rgb(0_0_0/0.08)] hover:bg-hover active:bg-selected',
        prominent: 'bg-accent text-on-accent hover:brightness-110 active:brightness-95 shadow-sm',
        ghost: 'hover:bg-hover active:bg-selected',
        plain: '',
        destructive: 'text-danger hover:bg-hover',
      },
      size: {
        default: 'h-7 px-3',
        sm: 'h-6 px-2 text-[12px]',
        icon: 'size-7 rounded-full',
        iconSm: 'size-6 rounded-full [&_svg]:size-3.5',
      },
    },
    defaultVariants: { variant: 'default', size: 'default' },
  },
)

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, type = 'button', ...props }, ref) => {
    const Comp = asChild ? Slot.Root : 'button'
    return <Comp ref={ref} type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  },
)
Button.displayName = 'Button'

export { buttonVariants }
