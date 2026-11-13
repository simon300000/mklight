import { create } from 'zustand'
import { defaultCamera, type Camera } from './camera'

/** The layout's camera. It lives outside the view, so the layout keeps its
 *  framing when it moves to another tab, place or window. */
export const useCamera = create<{
  cam: Camera
  /** The view frames the scene by itself until it is panned or zoomed. */
  fitted: boolean
  /** The last fit framed the lens only. */
  lens: boolean
  /** `layoutEpoch` of the last initial fit. */
  epoch: number
}>()(() => ({ cam: defaultCamera(), fitted: true, lens: false, epoch: -1 }))
