import { createContext, useContext } from 'react'
import { MAIN } from '@/state/arrangement'

/** The design window a view is in: the main window or a separate one. Menus,
 *  popovers and tooltips of a view in a separate window must portal into that
 *  window's document (`container`), not the main window's body. */
export interface PanelWindowContext {
  id: string
  win: Window
  /** Where floating content (menus, popovers) goes in this window; null: the default (main body). */
  portal: HTMLElement | null
}

export const WindowContext = createContext<PanelWindowContext>({ id: MAIN, win: typeof window !== 'undefined' ? window : (null as unknown as Window), portal: null })

export const usePanelWindow = () => useContext(WindowContext)

/** The container for Radix portals of the window a view is in (undefined in the main window). */
export const usePortalContainer = () => useContext(WindowContext).portal ?? undefined
