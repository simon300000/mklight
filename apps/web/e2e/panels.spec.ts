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

  // In a group of several, the tab carries the title: the pane's header keeps
  // its caption and buttons, the title only for assistive technology.
  const fanHeader = page.getByTestId('pane-ray_fan').locator('[data-tabbed]')
  await expect(fanHeader).toHaveCount(1)
  await expect(fanHeader.locator('h2')).toHaveClass(/sr-only|truncate/)
  expect(await fanHeader.locator('h2').evaluate((el) => el.getBoundingClientRect().width)).toBeLessThanOrEqual(1)

  // ⌥⌘3 shows the 3-D view, ⌥⌘Y the section again (⌘digits are the browser's).
  await page.keyboard.press(`Alt+${mod}+3`)
  await expect(page.getByTestId('layout-3d')).toBeVisible()
  await page.keyboard.press(`Alt+${mod}+KeyY`)
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

test('panels: dragging out of the window previews the new window; Escape cancels, a drop opens it', async ({ page }) => {
  await openSample(page)
  const spot = (await page.getByTestId('pane-spot_diagram').boundingBox())!
  await page.mouse.move(spot.x + 12, spot.y + 18)
  await page.mouse.down()
  await page.mouse.move(spot.x + 40, spot.y + 40, { steps: 4 })
  // Out past the window's right edge: the outline of the window to be opened.
  const vw = page.viewportSize()!.width
  await page.mouse.move(vw + 120, spot.y + 60, { steps: 8 })
  await expect(page.getByTestId('new-window-outline')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('new-window-outline')).toHaveCount(0)
  await page.mouse.up()
  await expect(page.getByTestId('pane-spot_diagram')).toBeVisible()

  // Again, and drop: the pane opens in a window of its own.
  await page.mouse.move(spot.x + 12, spot.y + 18)
  await page.mouse.down()
  await page.mouse.move(spot.x + 40, spot.y + 40, { steps: 4 })
  await page.mouse.move(vw + 120, spot.y + 60, { steps: 8 })
  await expect(page.getByTestId('new-window-outline')).toBeVisible()
  const popupPromise = page.waitForEvent('popup')
  await page.mouse.up()
  const popup = await popupPromise
  await expect(popup.getByTestId('pane-spot_diagram')).toBeVisible()
  await expect(page.getByTestId('pane-spot_diagram')).toHaveCount(0)
  // Shortcuts work from the separate window too: ⌃⌘S hides the main window's sidebar.
  await expect(page.getByRole('complementary', { name: 'Sidebar' })).toBeVisible()
  await popup.keyboard.press(`Control+${mod}+KeyS`)
  await expect(page.getByRole('complementary', { name: 'Sidebar' })).toBeHidden()
  await popup.keyboard.press(`Alt+${mod}+3`)
  await expect(page.getByTestId('layout-3d')).toBeVisible()
  await popup.keyboard.press(`Alt+${mod}+KeyE`)
  await expect(page.getByTestId('lens-data')).toHaveCount(0)
})

test('panels: blocked pop-ups dock the separate windows; Open Window restores them at their frames', async ({ page }) => {
  await openSample(page)
  await page.getByTestId('layout').getByRole('button', { name: 'More' }).click()
  const popupPromise = page.waitForEvent('popup')
  await page.getByRole('menuitem', { name: 'Move to New Window' }).click()
  await popupPromise
  // Autosave: the new window's frame settles first, and each change restarts the debounce.
  await page.waitForTimeout(2500)
  const frameNow = () =>
    page.evaluate(() => {
      const m = (window as unknown as { mklight: { arrangement: { getState(): { arr: { windows: { frame: number[] | null }[] } } } } }).mklight
      return m.arrangement.getState().arr.windows[0]?.frame ?? null
    })
  const frame = await frameNow()
  expect(frame).not.toBeNull()
  // The browser now blocks pop-ups that do not follow a click; the project opens again.
  await page.addInitScript(() => {
    const open = window.open.bind(window)
    ;(window as unknown as { __allowPopups: boolean }).__allowPopups = false
    window.open = ((...a: Parameters<typeof window.open>) =>
      (window as unknown as { __allowPopups: boolean }).__allowPopups ? open(...a) : null) as typeof window.open
  })
  await page.goto('./?engine=mock')
  await expect(page.getByTestId('layout')).toBeVisible()
  const toast = page.getByText(/blocked a separate window/)
  await expect(toast).toBeVisible()
  await page.evaluate(() => ((window as unknown as { __allowPopups: boolean }).__allowPopups = true))
  const again = page.waitForEvent('popup')
  await page.getByRole('button', { name: 'Open Window' }).click()
  // The window opens at its saved frame (where the browser honours it).
  const restored = await frameNow()
  const popup = await again
  await expect(popup.getByTestId('layout')).toBeVisible()
  await expect(page.getByTestId('layout')).toHaveCount(0)
  // Saved frames round-trip through AppKit coordinates; allow for rounding.
  restored!.forEach((v, k) => expect(Math.abs(v - frame![k])).toBeLessThan(2))
})

test('panels: a narrow window stacks the panels; the sidebar and inspector float over them', async ({ page }) => {
  await page.setViewportSize({ width: 820, height: 1000 })
  await openSample(page)
  await expect(page.getByRole('complementary', { name: 'Sidebar' })).toBeHidden()
  await expect(page.getByTestId('inspector')).toBeHidden()
  const layout = (await page.getByTestId('layout').boundingBox())!
  const spot = (await page.getByTestId('pane-spot_diagram').boundingBox())!
  expect(spot.y).toBeGreaterThan(layout.y + layout.height - 2)
  expect(spot.width).toBeGreaterThan(700)
  await page.getByRole('button', { name: 'Show Sidebar' }).click()
  await expect(page.getByTestId('sidebar-drawer')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('sidebar-drawer')).toHaveCount(0)
  await page.getByRole('button', { name: 'Inspector', exact: true }).click()
  await expect(page.getByTestId('inspector-sheet')).toBeVisible()
  await page.getByTestId('overlay-scrim').click({ position: { x: 20, y: 500 } })
  await expect(page.getByTestId('inspector-sheet')).toHaveCount(0)
})
