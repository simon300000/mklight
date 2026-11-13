import { expect, test, type Page } from '@playwright/test'

const mod = process.platform === 'darwin' ? 'Meta' : 'Control'

async function openSample(page: Page) {
  await page.goto('./?engine=mock&sample=double_gauss')
  await expect(page.getByTestId('status-capsule')).toContainText('EFL')
  await expect(page.getByTestId('pane-spot_diagram')).toBeVisible()
}

test('panels: the default arrangement, tabs by dragging a header, 3-D and back', async ({ page }) => {
  await openSample(page)
  const lens = (await page.getByTestId('lens-data').boundingBox())!
  const layout = (await page.getByTestId('layout').boundingBox())!
  const spot = (await page.getByTestId('pane-spot_diagram').boundingBox())!
  // Lens data across the top, the layout and the panes side by side below.
  expect(layout.y).toBeGreaterThan(lens.y + lens.height - 2)
  expect(spot.x).toBeGreaterThan(layout.x + layout.width - 2)

  // Drag the spot diagram by its header onto the ray fan: they become tabs.
  const fan = (await page.getByTestId('pane-ray_fan').boundingBox())!
  await page.mouse.move(spot.x + 12, spot.y + 18)
  await page.mouse.down()
  await page.mouse.move(spot.x + 40, spot.y + 40, { steps: 4 })
  await page.mouse.move(fan.x + fan.width / 2, fan.y + fan.height / 2, { steps: 8 })
  await expect(page.getByTestId('drop-zone')).toContainText('Add as Tab')
  await page.mouse.up()
  await expect(page.getByRole('tab', { name: 'Spot Diagram' })).toBeVisible()
  await page.getByRole('tab', { name: 'Ray Fan' }).click()
  await expect(page.getByTestId('pane-ray_fan')).toBeVisible()

  // ⌘3 shows the 3-D view, ⌘2 the section again.
  await page.keyboard.press(`${mod}+3`)
  await expect(page.getByTestId('layout-3d')).toBeVisible()
  await page.keyboard.press(`${mod}+2`)
  await expect(page.getByTestId('layout-3d')).toBeHidden()
})

test('panels: Move to New Window opens a live separate window', async ({ page }) => {
  await openSample(page)
  await page.getByTestId('layout').getByRole('button', { name: 'More' }).click()
  const popupPromise = page.waitForEvent('popup')
  await page.getByRole('menuitem', { name: 'Move to New Window' }).click()
  const popup = await popupPromise
  await expect(popup.getByTestId('layout')).toBeVisible()
  await expect(page.getByTestId('layout')).toHaveCount(0)
  // Move to Main Window puts it back.
  await popup.getByRole('button', { name: 'Move to Main Window' }).click()
  await expect(page.getByTestId('layout')).toBeVisible()
})
