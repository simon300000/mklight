import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'uplot/dist/uPlot.min.css'
import './index.css'
import { App } from './App'
import { initI18n } from './i18n'
import { startSession } from './state/session'

// The language's catalog first, so the app does not open in English and switch.
await initI18n()
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
void startSession()
