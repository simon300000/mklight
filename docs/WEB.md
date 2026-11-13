# MKLight on the web

The web version of MKLight is a static single-page app in `apps/web`: React,
TypeScript and Vite, with the same Rust engine compiled to WebAssembly and run
in a Web Worker. The UI never computes optics; it mirrors the macOS app's
concepts, names and wording (docs/APP.md). SwiftUI stays the standard: features
land in the macOS app first and then here.

<!-- The engine module (apps/web/engine, scripts/build-wasm.sh) is documented in
     its own section, written with that work. -->

## Web UI

### Run and build

Requirements: Node 22 or later and pnpm 10 or later.

```sh
cd apps/web
pnpm install
pnpm dev            # http://localhost:5173, cross-origin isolated
pnpm build          # static dist/, relative paths (base './')
pnpm preview        # serves dist/ with the COOP/COEP headers
```

`pnpm build` writes a static `dist/` with relative asset paths, so it works
from any static host and any subpath (`https://example.org/tools/mklight/`).
There is no server code and no SSR. For the threaded engine the host must
send `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (the dev and preview servers do);
without them the page still works, and the status popover shows
"Cross-origin isolated: No".

**Engine choice.** The UI imports the engine as `@engine`
(`apps/web/engine/src/index.ts`, `createEngine()`). When that module is not in
the checkout, `@engine` resolves to a stub and the app runs on the **mock
engine** (`src/engine-mock`): synthetic but correctly shaped results from a
small rotationally symmetric ray trace, labelled "Mock" in the toolbar and
"backend mock" in every result. Choose explicitly with `VITE_ENGINE=mock|wasm`
at build time or `?engine=mock|wasm` in the URL. Other URL parameters:
`?sample=double_gauss` opens a sample, `?fresh` skips the autosaved document.

**Tests.**

```sh
pnpm test                                   # Vitest: model, state, mock engine, renderers
pnpm exec playwright install chromium webkit  # once
pnpm test:e2e                               # Playwright smoke tests, Chromium and WebKit
pnpm typecheck
```

The Playwright tests build the app with the mock engine, serve it with
`vite preview` on port 4317 and walk the basic workflow: open a sample, edit a
radius in the lens table, add a spot diagram pane, undo.

### Structure

| path | role |
|---|---|
| `src/engine/types.ts` | The engine boundary as the UI codes against it (mirrors the Swift `EngineClient`; JSON keeps the engine's snake_case keys) |
| `src/engine/index.ts` | Chooses the WASM engine or the mock |
| `src/engine-mock/` | The mock engine (`createMockEngine`), real descriptors from `mklight analyses --json` |
| `src/model/` | Pure model code: number format, system edits, surface-type descriptors, the `.mklight` file, samples, analysis titles |
| `src/state/document.ts` | Simulation state: the project and its undo history (named actions) |
| `src/state/workspace.ts` | Snapshots, panes and runs (stale, automatic update, cancel, fresh runs) |
| `src/state/view.ts` | Window state: selection, inspector page, panel visibility, toasts |
| `src/state/files.ts` | New, Samples, Open/Save, Import, IndexedDB autosave |
| `src/state/commands.ts` | Menu commands and keyboard shortcuts |
| `src/views/` | The window: toolbar, sidebar, lens data, layout, analysis panes, inspector |
| `src/views/results/` | Result renderers (spot, fans, MTF, PSF, field curvature, wavefront, tables, generic) |
| `src/components/ui/` | shadcn/ui-style components on Radix and Tailwind |

Document state and view state are separate, as in the Swift app. Every edit
replaces the system with a new value and records the previous project under
the action's name (`Change Radius`, `Insert Surface`, `Apply Wavelength
Preset`, …, the macOS names), so Undo and Redo restore whole design states and
the menu says what they undo. Each design value is debounced (150 ms) and
frozen into an engine snapshot; panes run against snapshots, are stale as soon
as the snapshot or their settings differ, and re-run when their automatic
update is on. Run and Try Again send `fresh: true`.

### Documents

* **New**, **Samples…** (the gallery of `samples.json`, read at build time from
  `apps/macos/Sources/MKLightModel/Resources/samples.json`), **Open…**,
  **Save**, **Save As…**, **Replace with Sample** (undoable).
* **File access.** Where the browser has the File System Access API (Chrome,
  Edge), Open and Save use real file handles and Save writes in place. Elsewhere
  (Safari, Firefox) Open uses a file input and Save downloads the file.
* **Import.** `.zmx` and `.seq` files open through the engine's importer, by
  Open… or by dropping them on the window; the import report groups the
  warnings as in the macOS app. Imported files are never written: Save asks for
  a new `.mklight` file.
* **Autosave.** The current document (with its panes) is kept in IndexedDB and
  restored on the next visit, with its file handle where the browser allows.

### Window

The sidebar (Lens Data, Layout, Issues, then every engine analysis by
category, searchable; a click shows the analysis in the focused pane, ⌥-click
or the context menu opens a new pane), the content (lens data across the top,
the layout and the analysis panes side by side under it, dividers to drag,
double-click for equal shares) and the full-height inspector whose part of the
toolbar holds its page tabs (Surface, System). The toolbar has the app menu
(File, Edit, Lens, View, Analysis), the document title, the status capsule
(EFL, F/#, total track, issues; its popover has the first-order data and the
engine: readiness, threads, cross-origin isolation, WebGPU, backends), Add
Analysis and the inspector toggle.

Keyboard (⌘ on Mac, Ctrl elsewhere): ⌘O open, ⌘S save, ⇧⌘S save as, ⌥⌘N new,
⇧⌘N samples, ⌘Z / ⇧⌘Z undo / redo, ⇧⌘I / ⌥⌘I insert before / after, ⌘⌫ delete
surface, ⇧⌘T make stop, ⌘R run analyses, ⌘. cancel, ⌃⌘I inspector, ⌃⌘S
sidebar, ⌃⌘E issues, ⌥⌘E lens data, ⌥⌘L layout, ⌘1 / ⌥⌘1 Surface / System.
Browsers reserve some keys (⌘N, ⌘T, ⌘W): New is ⌥⌘N on the web, and ⇧⌘T may
reopen a closed tab in some browsers before the page sees it.

### Differences from the macOS app

Not yet on the web: the design jobs (Optimize, Glass Substitution, Tolerance,
Apply Solves), the 3-D layout, separate windows and drag-and-drop panel
arrangement (tabs and splits), variables and solve markers in the table, melt
offsets in the inspector, the run-local spectrum control, glass catalogs,
configurations editing, dedicated views for the field maps, glass map, image
simulation, ghosts and stray light (they show with the generic result view).
