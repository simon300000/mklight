// URL-addressable view state, the web's way: the inspector page (Surface,
// System or a design job), the selected surface and an open dialog live in the
// URL's hash (`#page=optimize&surface=3`, `#dialog=samples`), so reload keeps
// them and Back / Forward step through pages and dialogs. The design itself is
// never in the URL (it lives in the document and its autosave).

import { useView, type InspectorPage } from './view'

const PAGES: InspectorPage[] = ['surface', 'system', 'optimize', 'glass', 'tolerance', 'solves']

interface UrlState {
  page: InspectorPage
  surface: number | null
  dialog: 'samples' | null
}

function read(): UrlState {
  const h = new URLSearchParams(location.hash.replace(/^#/, ''))
  const page = h.get('page') as InspectorPage | null
  const surface = h.get('surface')
  return {
    page: page && PAGES.includes(page) ? page : 'surface',
    surface: surface != null && /^\d+$/.test(surface) ? Number(surface) : null,
    dialog: h.get('dialog') === 'samples' ? 'samples' : null,
  }
}

function hashOf(s: UrlState): string {
  const h = new URLSearchParams()
  if (s.page !== 'surface') h.set('page', s.page)
  if (s.surface != null && s.page === 'surface') h.set('surface', String(s.surface))
  if (s.dialog) h.set('dialog', s.dialog)
  const t = h.toString()
  return t ? `#${t}` : ''
}

function current(): UrlState {
  const v = useView.getState()
  return { page: v.inspectorPage, surface: v.selection, dialog: v.welcomeOpen ? 'samples' : null }
}

let applying = false

function apply(s: UrlState) {
  applying = true
  const v = useView.getState()
  v.set({ inspectorPage: s.page, welcomeOpen: s.dialog === 'samples', ...(s.page !== v.inspectorPage ? { inspectorVisible: true } : {}) })
  if (s.surface != null) v.select(s.surface)
  applying = false
}

/** Restores the view from the URL and keeps the URL following the view. A page
 *  or dialog change is a history entry; a surface change replaces it. */
export function installUrlState() {
  if (location.hash) apply(read())
  window.addEventListener('popstate', () => apply(read()))
  useView.subscribe((s, prev) => {
    if (applying) return
    if (s.inspectorPage === prev.inspectorPage && s.welcomeOpen === prev.welcomeOpen && s.selection === prev.selection) return
    const next = hashOf(current())
    if (next === location.hash || (next === '' && location.hash === '')) return
    const url = location.pathname + location.search + next
    const entry = s.inspectorPage !== prev.inspectorPage || s.welcomeOpen !== prev.welcomeOpen
    if (entry) history.pushState(null, '', url)
    else history.replaceState(null, '', url)
  })
}
