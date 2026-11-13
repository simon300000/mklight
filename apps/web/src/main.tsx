import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'uplot/dist/uPlot.min.css'
import './index.css'
import { App } from './App'
import { startSession } from './state/session'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
void startSession()
