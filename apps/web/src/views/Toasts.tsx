import { AlertTriangle, Info, XOctagon } from 'lucide-react'
import { useView } from '@/state/view'

/** Transient messages float as a glass toast (no modal alerts for routine errors). */
export function Toasts() {
  const toasts = useView((s) => s.toasts)
  const dismiss = useView((s) => s.dismiss)
  return (
    <div className="pointer-events-none fixed inset-x-0 top-16 z-[70] flex flex-col items-center gap-2" aria-live="polite">
      {toasts.map((t) => (
        <button
          type="button"
          key={t.id}
          onClick={() => dismiss(t.id)}
          className="glass-popover pointer-events-auto flex max-w-[560px] cursor-default items-center gap-2 rounded-full px-4 py-1.5 text-[13px]"
          role={t.tone === 'error' ? 'alert' : 'status'}
        >
          {t.tone === 'error' ? (
            <XOctagon className="size-4 shrink-0 text-danger" />
          ) : t.tone === 'warning' ? (
            <AlertTriangle className="size-4 shrink-0 text-warning" />
          ) : (
            <Info className="size-4 shrink-0 text-secondary" />
          )}
          {t.text}
        </button>
      ))}
    </div>
  )
}
