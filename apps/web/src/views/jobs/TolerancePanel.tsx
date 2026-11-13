import { useDocument } from '@/state/document'
import { JobPanelFrame } from './JobPanel'

/** Placeholder: replaced by the full panel. */
export function TolerancePanel() {
  const name = useDocument((s) => s.project.system?.name ?? '')
  return (
    <JobPanelFrame kind="tolerance" subtitle={name} actions={null}>
      <p className="text-secondary">Coming next.</p>
    </JobPanelFrame>
  )
}
