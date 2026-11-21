import { expect, test, type Page } from '@playwright/test'

// Languages (docs/WEB.md "Languages"): the system's language by default, a
// choice in App menu ▸ Language kept per browser, switched live.

const engineReady = (page: Page) =>
  page.waitForFunction(() => (window as unknown as { mklight?: { workspace: { getState(): { engine: unknown } } } }).mklight?.workspace.getState().engine != null, null, { timeout: 60_000 })

/** Catalog keys or unfilled placeholders that leaked into the page. */
async function leaks(page: Page): Promise<string[]> {
  const text = await page.locator('body').innerText()
  const keys = text.match(/\b(?:common|app|lens|inspector|analysis|maps|glass|optimize|tolerance)\.[a-z]\w*(?:\.\w+)*/g) ?? []
  const holes = text.match(/\{[a-zA-Z]\w*\}/g) ?? []
  return [...keys, ...holes]
}

test.describe('a Chinese browser', () => {
  test.use({ locale: 'zh-CN' })

  test('opens in Chinese and shows a sample in Chinese', async ({ page }) => {
    await page.goto('./?fresh&sample=double_gauss')
    await engineReady(page)
    await expect(page.locator('html')).toHaveAttribute('lang', 'zh-Hans')
    await expect(page.getByRole('button', { name: '菜单' })).toBeVisible()
    await expect(page.getByTestId('status-capsule')).toContainText('EFL')

    // An analysis pane in Chinese: the add menu, its title and its key numbers.
    await page.getByRole('button', { name: '添加分析' }).click()
    await page.getByRole('menuitem', { name: '几何' }).click()
    await page.getByRole('menuitem', { name: '点列图' }).click()
    const pane = page.getByTestId('pane-spot_diagram').last()
    await expect(pane).toContainText('点列图')
    await expect(pane.getByTestId('key-numbers')).toContainText('RMS', { timeout: 30_000 })

    // The inspector's System page.
    await page.getByRole('tab', { name: '系统' }).click()
    await expect(page.getByTestId('system-page')).toContainText('波长')
    expect(await leaks(page)).toEqual([])
  })
})

test('Language in the app menu switches live and the choice survives reload', async ({ page }) => {
  await page.goto('./?fresh&sample=double_gauss')
  await engineReady(page)
  await expect(page.locator('html')).toHaveAttribute('lang', 'en')
  await page.getByRole('button', { name: 'Menu' }).click()
  await page.getByTestId('language-menu').click()
  await page.getByRole('menuitemradio', { name: '简体中文' }).click()
  // Live: no reload, the toolbar and the open menu read in Chinese.
  await expect(page.getByRole('button', { name: '菜单' })).toBeVisible()
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-Hans')
  await page.keyboard.press('Escape')
  expect(await leaks(page)).toEqual([])

  // A new visit (a reload; ?fresh skips the welcome window).
  await page.goto('./?fresh&sample=double_gauss')
  await engineReady(page)
  await expect(page.getByRole('button', { name: '菜单' })).toBeVisible()

  // Back to the system's language (English here).
  await page.getByRole('button', { name: '菜单' }).click()
  await page.getByTestId('language-menu').click()
  await page.getByRole('menuitemradio', { name: '跟随系统' }).click()
  await expect(page.getByRole('button', { name: 'Menu' })).toBeVisible()
  expect(await page.evaluate(() => localStorage.getItem('mklight.language'))).toBeNull()
})
