// Types for the parts of i18n-ui.mjs the web app's tests use.
/** Every way the app catalogs and locales/<locale>/ui.json disagree; empty when they agree. */
export function problems(locale?: string): string[]
/** The locales that have a ui.json. */
export function locales(): string[]
export function uiPath(locale?: string): string
