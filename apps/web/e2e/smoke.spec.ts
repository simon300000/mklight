import { expect, test, type Page } from '@playwright/test'

const mod = process.platform === 'darwin' ? 'Meta' : 'Control'

async function openSample(page: Page) {
  await page.goto('./?fresh')
  // The real engine takes a moment to start; the app opens its first document
  // only then, so wait for it before touching the menus.
  await page.waitForFunction(() => (window as unknown as { mklight?: { workspace: { getState(): { engine: unknown } } } }).mklight?.workspace.getState().engine != null, null, { timeout: 60_000 })
  await expect(page.getByTestId('status-capsule')).not.toContainText('Starting engine')
  await page.getByRole('button', { name: 'Menu' }).click()
  await page.getByRole('menuitem', { name: 'Samples…' }).click()
  await page.getByTestId('sample-double_gauss').click()
  await expect(page.getByTestId('document-title')).toHaveText('Double Gauss f50 F/2')
  // The engine answered: the status capsule shows the first-order data.
  await expect(page.getByTestId('status-capsule')).toContainText('EFL')
}

/** Types into the lens table's cell at (column, row) through Glide's editor. */
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

test('open a sample, edit a radius, add a spot pane, undo', async ({ page }) => {
  await openSample(page)
  const radiusBefore = await page.evaluate(() => (window as unknown as { mklight: { radius(i: number): number } }).mklight.radius(1))
  await editCell(page, 3, 1, '40')
  await expect.poll(() => page.evaluate(() => (window as unknown as { mklight: { radius(i: number): number } }).mklight.radius(1))).toBeCloseTo(40, 6)

  // Add a spot diagram pane from the toolbar.
  await page.getByRole('button', { name: 'Add Analysis' }).click()
  await page.getByRole('menuitem', { name: 'Geometric' }).click()
  await page.getByRole('menuitem', { name: 'Spot Diagram' }).click()
  await expect(page.getByTestId('pane-spot_diagram').first()).toBeVisible()
  await expect(page.getByTestId('pane-spot_diagram').last().getByTestId('key-numbers')).toContainText('RMS', { timeout: 10_000 })

  // Undo the radius edit; the name of the action is in the menu.
  await page.getByRole('button', { name: 'Menu' }).click()
  await expect(page.getByRole('menuitem', { name: 'Undo Change Radius' })).toBeVisible()
  await page.keyboard.press('Escape')
  await page.keyboard.press(`${mod}+z`)
  await expect.poll(() => page.evaluate(() => (window as unknown as { mklight: { radius(i: number): number } }).mklight.radius(1))).toBeCloseTo(radiusBefore, 6)
})

test('the inspector edits the system and the status popover shows the engine', async ({ page }) => {
  await openSample(page)
  await page.getByRole('tab', { name: 'System' }).click()
  await expect(page.getByTestId('system-page')).toBeVisible()
  await page.getByRole('button', { name: 'Presets' }).click()
  await page.getByRole('menuitem', { name: 'd only' }).click()
  await expect(page.getByLabel('Wavelength 2')).toHaveCount(0)
  await page.getByTestId('status-capsule').click()
  await expect(page.getByTestId('engine-info')).toContainText('Threads')
  await expect(page.getByTestId('engine-info')).toContainText('Cross-origin isolated')
})
