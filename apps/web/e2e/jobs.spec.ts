import { expect, test } from '@playwright/test'

const mod = process.platform === 'darwin' ? 'Meta' : 'Control'
type Hooks = { mklight: { radius(i: number): number; document: { getState(): { project: { system: unknown } } } } }
const systemJson = (page: import('@playwright/test').Page) =>
  page.evaluate(() => JSON.stringify((window as unknown as Hooks).mklight.document.getState().project.system))

// The Cooke triplet: the optimisation finishes in a few seconds. (Stopping it
// early could leave nothing to apply; Stop is covered by the engine's own tests.)
test('optimize on the engine, apply, undo', async ({ page }) => {
  await page.goto('./?sample=cooke_triplet')
  await expect(page.getByTestId('status-capsule')).toContainText('EFL', { timeout: 60_000 })
  const before = await systemJson(page)

  await page.getByRole('tab', { name: 'Optimize' }).click()
  const panel = page.getByTestId('job-optimize')
  await expect(panel).toBeVisible()
  await panel.getByRole('button', { name: 'Optimize', exact: true }).click()

  // While it runs: a dot on the tab and an entry in the status capsule.
  await expect(page.getByTestId('job-progress').or(panel.getByRole('button', { name: 'Apply to Design' }))).toBeVisible()

  const apply = panel.getByRole('button', { name: 'Apply to Design' })
  await expect(apply).toBeVisible({ timeout: 90_000 })
  await apply.click()
  await expect(panel.getByTestId('applied-notice')).toBeVisible()
  await expect.poll(() => systemJson(page)).not.toBe(before)

  await page.getByRole('button', { name: 'Menu' }).click()
  await expect(page.getByRole('menuitem', { name: 'Undo Apply Optimization' })).toBeVisible()
  await page.keyboard.press('Escape')
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur())
  await page.keyboard.press(`${mod}+z`)
  await expect.poll(() => systemJson(page)).toBe(before)
})
