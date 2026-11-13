import { expect, test, type Page } from '@playwright/test'

const mod = process.platform === 'darwin' ? 'Meta' : 'Control'
type Hooks = {
  mklight: {
    radius(i: number): number
    document: { getState(): { project: { system: unknown } } }
    workspace: { getState(): { engine: { info: { name: string; isMock?: boolean } } | null } }
  }
}
const hooks = (page: Page) => page.evaluate(() => (window as unknown as Hooks).mklight !== undefined)
const systemJson = (page: Page) => page.evaluate(() => JSON.stringify((window as unknown as Hooks).mklight.document.getState().project.system))
const radius = (page: Page, i: number) => page.evaluate((k) => (window as unknown as Hooks).mklight.radius(k), i)

async function editCell(page: Page, col: number, row: number, text: string) {
  const grid = page.getByTestId('lens-data').locator('canvas').first()
  const box = (await grid.boundingBox())!
  const widths = [58, 80, 110, 110, 104, 100, 112, 84]
  const x = box.x + widths.slice(0, col).reduce((a, b) => a + b, 0) + widths[col] / 2
  const y = box.y + 26 + row * 24 + 12
  await page.mouse.dblclick(x, y)
  const editor = page.locator('#portal textarea, #portal input').first()
  await expect(editor).toBeVisible()
  await editor.fill(text)
  await editor.press('Enter')
}

test('the real engine: Cooke triplet, edit a radius, spot and MTF', async ({ page }) => {
  await page.goto('./?engine=wasm&sample=cooke_triplet')
  await expect.poll(() => hooks(page)).toBe(true)
  await expect(page.getByTestId('status-capsule')).toContainText('EFL', { timeout: 60_000 })
  const info = await page.evaluate(() => (window as unknown as Hooks).mklight.workspace.getState().engine?.info)
  expect(info?.name).toBe('MKLight engine')
  expect(info?.isMock).toBeFalsy()

  // Edit a radius in the lens table.
  await editCell(page, 3, 1, '23')
  await expect.poll(() => radius(page, 1)).toBeCloseTo(23, 6)

  // The spot pane (a default pane) and a new MTF pane compute on the engine.
  await expect(page.getByTestId('pane-spot_diagram').getByTestId('key-numbers')).toContainText('RMS')
  await page.getByRole('button', { name: 'Add Analysis' }).click()
  await page.getByRole('menuitem', { name: 'Wave Optics' }).click()
  await page.getByRole('menuitem', { name: 'MTF', exact: true }).click()
  const mtf = page.getByTestId('pane-mtf')
  await expect(mtf).toBeVisible()
  await expect(mtf.getByRole('button', { name: /Run details/ })).toBeVisible({ timeout: 60_000 })
  await expect(mtf.getByRole('button', { name: /Run details/ })).toContainText('CPU')

})

test('the real engine: optimize the Cooke triplet, apply, undo', async ({ page }) => {
  await page.goto('./?engine=wasm&sample=cooke_triplet')
  await expect.poll(() => hooks(page)).toBe(true)
  await expect(page.getByTestId('status-capsule')).toContainText('EFL', { timeout: 60_000 })
  const before = await systemJson(page)
  await page.getByRole('tab', { name: 'Optimize' }).click()
  const panel = page.getByTestId('job-optimize')
  await panel.getByRole('button', { name: 'Optimize', exact: true }).click()
  const apply = panel.getByRole('button', { name: 'Apply to Design' })
  // The run finishes in a few seconds in both engines. (Stopping it early
  // could leave nothing to apply: the merit first improves near iteration 30.
  // Stop itself is covered by the engine's own tests.)
  await expect(apply).toBeVisible({ timeout: 90_000 })
  await apply.click()
  await expect(panel.getByTestId('applied-notice')).toBeVisible()
  await expect.poll(() => systemJson(page)).not.toBe(before)
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur())
  await page.keyboard.press(`${mod}+z`)
  await expect.poll(() => systemJson(page)).toBe(before)
})
