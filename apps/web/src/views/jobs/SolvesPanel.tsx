import { useDocument } from '@/state/document'
import { JobPanelFrame } from './JobPanel'

/** Placeholder: replaced by the full panel. */
export function SolvesPanel() {
  const name = useDocument((s) => s.project.system?.name ?? '')
  return (
    <JobPanelFrame kind="solves" subtitle={name} actions={null}>
      <p className="text-secondary">Coming next.</p>
    </JobPanelFrame>
  )
}
