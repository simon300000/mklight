# MKLight on the web

The web version of MKLight is a static single-page app in `apps/web`: React,
TypeScript and Vite, with the same Rust engine compiled to WebAssembly and run
in a Web Worker. The UI never computes optics; it mirrors the macOS app's
concepts, names and wording (docs/APP.md). SwiftUI stays the standard: features
land in the macOS app first and then here.

Two parts: the web UI (`apps/web`, this section first) and the engine module
(`apps/web/engine`, `scripts/build-wasm.sh`; "The engine" below).

## Web UI

### Run and build

Requirements: Node 22 or later and pnpm 10 or later; for the engine the
wasm toolchain of "The engine" below.

```sh
cd apps/web
pnpm install
pnpm build:wasm     # scripts/build-wasm.sh → engine/pkg (once, and after engine changes)
pnpm dev            # http://localhost:5173, cross-origin isolated
pnpm build          # static dist/: the app, the wasm, its workers, the isolation worker
pnpm preview        # serves dist/ with the COOP/COEP headers
```

`pnpm build` writes a static `dist/` with relative paths (`base: './'`), so it
works from any static host and any subpath (`https://example.org/tools/mklight/`):
checked in Chromium and WebKit from a subpath of a plain static server that
sends no special headers. There is no server code and no SSR. The threaded
engine needs a cross-origin isolated page; a host that cannot send
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (GitHub Pages) gets them from
`coi-serviceworker.js`, which the build places next to `index.html` and which
`index.html` loads first. The status popover shows the engine, its threads,
cross-origin isolation and WebGPU.

**The engine.** The UI imports the engine as `@engine`
(`apps/web/engine/src/index.ts`) and shows only what the real engine computes;
there is no stand-in. The wasm package (`engine/pkg`, `pnpm build:wasm`) must
be built. Without it the app still builds (so `pnpm build` does not need the
pinned nightly toolchain), `@engine` points at `src/engine/missing.ts`, and the
app starts in an error state: the status capsule says "Engine unavailable" and
its popover "Engine not built: run npm run build:wasm in apps/web, then
reload", with no results anywhere. `src/engine/adapt.ts` maps the module's
wrappers onto the UI's contract (`src/engine/types.ts`): progress stage text
parsed into live detail as the macOS app does, buffer shapes, the envelope,
backend requests and errors. URL parameters: `?sample=double_gauss` opens a
sample, `?fresh` skips the autosaved document.

**Tests.**

```sh
pnpm test                                     # Vitest: model, state, renderers, arrangement (no engine, no browser)
pnpm exec playwright install chromium webkit  # once
pnpm build:wasm                               # once: the Playwright suite runs on the real engine
pnpm test:e2e                                 # Playwright on the real engine, Chromium and WebKit
pnpm typecheck
pnpm record:engine                            # re-record src/test-engine/recordings.json
```

Vitest needs no engine. The few unit tests that need engine data (workspace
runs, jobs, result renderers) use the replay engine in `src/test-engine/`: an
`EngineClient` that serves results recorded from the real engine
(`recordings.json`, the double Gauss sample, small settings) after a short
latency, and rejects any request it has no recording for. It is imported by
tests only and is not part of the app bundle. `pnpm record:engine`
(`src/test-engine/record.mjs`) records them again with the CLI
(`target/release/mklight run <project> <analysis> --json --buffers --no-timing`;
build it with `cargo build --release -p mklight-cli`); re-record after a change
of the result formats.

`test:e2e` builds the app and serves it on port 4318 (cross-origin isolated):
open a sample, edit a radius in the lens table, add a spot diagram pane, undo;
the System page and the status popover; Optimize → Apply to Design → Undo; the
panel arrangement, 3-D and a separate window; the Cooke triplet with spot and
MTF panes. The tests run one at a time (the engine's thread pool is per page).

### Structure

| path | role |
|---|---|
| `src/engine/types.ts`, `adapt.ts`, `index.ts` | The engine boundary as the UI codes against it, the adapter to the WASM module, the loader (with the "not built" error) |
| `src/test-engine/` | Tests only: the replay engine and its recordings of the real engine |
| `src/model/` | Pure model code: number format, system edits, surface types, param paths, solves, materials, sag, merit, bounds, tolerances, glasses, spectrum, the `.mklight` file, samples |
| `src/state/document.ts` | Simulation state: the project and its undo history (named actions) |
| `src/state/workspace.ts` | Snapshots, preview snapshots, panes and runs (stale, automatic update, cancel, fresh runs) |
| `src/state/jobs.ts` and `optimize.ts`, `tolerance.ts`, `glassSubstitution.ts`, `solves.ts` | The design jobs: the shared run life cycle and each job's own session |
| `src/state/arrangement.ts` | The panel tree of splits and tab groups per window (`PanelArrangement`) |
| `src/state/view.ts`, `preferences.ts` | Window state; the backend preference |
| `src/state/files.ts`, `catalogs.ts` | Documents, import, autosave and recovery; glass catalogs |
| `src/state/commands.ts` | Menu commands and keyboard shortcuts |
| `src/views/` | The window: toolbar, sidebar, lens data, layout (2-D and WebGL2 3-D), panels and separate windows, analysis panes, inspector, job panels |
| `src/views/results/`, `src/views/analysis/`, `src/views/glassmap/` | A view for every analysis, pane settings, run details, the glass map |
| `src/components/ui/` | shadcn/ui-style components on Radix and Tailwind |

Document state and view state are separate, as in the Swift app. Every edit
replaces the system (or the project's variables, merit, tolerances and
catalogs) with a new value and records the previous project under the
action's name (`Change Radius`, `Insert Surface`, `Apply Wavelength Preset`,
`Apply Optimization`, `Apply Solves`, `Apply Glass Substitution`, `Replace
Tolerances`, `Change Trials`, …, the macOS names), so Undo and Redo restore
whole design states and the menu says what they undo. Each design value is
debounced (150 ms) and frozen into an engine snapshot; panes run against
snapshots, are stale as soon as the snapshot or their settings differ, and
re-run when their automatic update is on. Run and Try Again send
`fresh: true`. A preview (an optimisation, substitution or tolerance build)
shows a candidate snapshot in the layout and every pane, labelled "not
applied"; design jobs always run on the design's own snapshot.

### Documents

* **New**, **Samples…** (the gallery of `samples.json`, read at build time from
  `apps/macos/Sources/MKLightModel/Resources/samples.json`, thumbnails drawn
  from the engine's layout), **Open…**, **Save**, **Save As…**, **Replace with
  Sample** (undoable).
* **File access.** Where the browser has the File System Access API (Chrome,
  Edge), Open and Save use real file handles and Save writes in place. Elsewhere
  (Safari, Firefox) Open uses a file input and Save downloads the file.
* **Import.** `.zmx` and `.seq` files open through the engine's importer, by
  Open… or by dropping them on the window; the import report groups the
  warnings as in the macOS app and offers Load Glass Catalog…, which imports
  the file again with it. A clean import shows a short note. Imported files are
  never written: Save asks for a new `.mklight` file.
* **Autosave and recovery.** The current document (with its panes and panel
  arrangement) is kept in IndexedDB. A document with unsaved changes is
  offered for recovery in the welcome window on the next visit.

### Window

Sidebar, content and a full-height inspector, as in the macOS app. The
content is a tree of splits whose leaves are tab groups: panels (the lens
data, the layout, every analysis pane) drag by their header or tab onto drop
zones outlined in the accent colour; dividers drag (double click: equal
shares); View ▸ Panel has Move to New Window, Move to Main Window, Hide/Close
and Reset Panel Layout. The inspector's tabs in its part of the toolbar are
Surface, System and the design jobs (Optimize, Glass Substitution, Tolerance,
Apply Solves), each running job's tab marked with a dot; the status capsule
shows each running job ("Optimizing · 479", click to show it), EFL (EFL y and
the squeeze for anamorphic designs), F/#, total track and the issues; its
popover has the first-order data, the engine and the backend picker.

Keyboard (⌘ on Mac, Ctrl elsewhere; the Mac's ⌃⌘ shortcuts are
Ctrl+Alt+Shift elsewhere): ⌘K command palette, ⌘O open, ⌘S save, ⇧⌘S save
as, ⌥⌘N new, ⌥⌘O samples, ⌘Z / ⇧⌘Z undo / redo, ⇧⌘I / ⌥⌘↩ insert before /
after, ⌘⌫ delete surface, ⌥⌘P make stop, ⌃⌘V toggle variable, ⌃⌘O optimize,
⌃⌘G glass substitution, ⌃⌘T tolerance, ⌃⌘L apply solves, ⌃⌘E issues, ⌘↩ run
analyses, ⌘. cancel, ⌥⌘Y / ⌥⌘X / ⌥⌘3 y–z / x–z / 3-D, ⌥⌘0 fit, ⌥⌘L layout,
⌥⌘E lens data, ⌃⌘I inspector, ⌃⌘S sidebar, ⌥⌘1 / ⌥⌘2 Surface / System. In a
job panel Return commits a field and runs the primary action.

### Web equivalents of macOS features

Where a browser cannot do what the Mac app does, the web app does the nearest
thing:

* **Menu bar** → the app menu in the toolbar (File, Edit, Lens, Design, View,
  Analysis) with the same commands and shortcuts. Browsers keep ⌘N, ⌘T and ⌘W:
  New is ⌥⌘N, and ⇧⌘T may reopen a closed tab in some browsers before the page
  sees it.
* **Separate windows** → same-origin popup windows rendered from the main
  page with React portals, so they share the design live; they open on a user
  action (popup blockers), close with the main window and reopen with the
  project (a blocked popup leaves the panels docked, with Open Windows in a
  toast).
* **Document files** → the File System Access API where available, else an
  input and a download; autosave in IndexedDB instead of autosave in place.
* **Glass catalog paths** → a catalog's text is kept in IndexedDB by name and
  loaded again when a project lists it; one never loaded in this browser
  shows as not loaded, with Locate… to choose the file.
* **Image simulation source files** → the chosen image is kept with the
  pane's settings in the project instead of its path.
* **Metal viewport** → a WebGL2 3-D view and a 2-D canvas for the sections.
* **Liquid Glass** → backdrop-filter translucency on the toolbar and popovers
  only; system font, light and dark from the system.

### Differences from the macOS app

Same features, concepts, engine behaviour and wording (undo names, job
states, Try Again recomputing); web patterns where they read better:

* **Command palette** (⌘K) over every command, analysis, surface and sample,
  beside the toolbar's app menu.
* **URL state**: the inspector page (a design job included), the selected
  surface and the samples dialog are in the URL hash, so reload and
  Back/Forward behave; `?sample=` opens a sample once.
* **Shortcuts** avoid the browser's own: ⌘-digits (tabs), ⌘0 (zoom), ⌘R
  (reload), ⇧⌘T, ⇧⌘N, ⌘N, ⌘T, ⌘W and ⌥⌘I (developer tools) are left alone, so
  Run Analyses is ⌘↩, the layout views are ⌥⌘Y / ⌥⌘X / ⌥⌘3 / ⌥⌘0, the
  inspector pages ⌥⌘1 / ⌥⌘2, Insert Surface After ⌥⌘↩ and Make Stop ⌥⌘P;
  a test keeps every shortcut unique on the Mac and elsewhere.
* **Narrow windows and tablets** (below 900 px): the sidebar is a drawer, the
  inspector a sheet (a scrim or Escape closes them) and the panels stack in a
  scrolling column; panels drag by touch, and each panel's More menu is the
  tap alternative to dragging; charts and maps show values on tap.
* **Standard web dialogs, menus and toasts** (Radix), focusable and labelled
  for assistive technology; the glass map and the operand list take keyboard
  focus.
* **Narrow panes** fold by their width (about 360 / 300 / 260 px), not at
  fixed macOS points; lens column widths are kept per browser session.

### Known gaps

* Undoing Add Glass Catalog starts a new engine session without the catalog
  but does not free the old one until reload.
* Dragging a panel outside the browser: the browser cannot draw outside its
  own windows, so the new-window outline shows only where it overlaps them.
* While popups are blocked, a project's separate windows stay docked, and an
  autosave made then no longer lists them.
* The glass map: Return marks a glass in the substitution panel; the glass's
  data card opens only in the glass map pane.

## The engine

The web version of MKLight runs the **same Rust engine** as the macOS app,
compiled to WebAssembly with real threads, in a Web Worker. No optics logic
exists in TypeScript: the web app talks to the engine through a typed client
(`@engine`, `apps/web/engine/`) that mirrors the Swift `EngineClient` and the
JSON shapes of [FFI.md](FFI.md) and [RESULTS.md](RESULTS.md). The SwiftUI app
stays the standard front end; features land in Swift first, then the web UI,
and the engine needs no porting for them.

This document covers the engine side: the wasm build, threads, isolation,
the worker, the client contract, the WebGPU seam and the tests. The React
app (`apps/web/` outside `engine/`) has its own documentation.

```
 page (main thread, never blocks)            engine worker (wasm "main" thread)
 ┌──────────────────────────────┐  postMessage  ┌──────────────────────────────┐
 │ React app ── @engine client  │ ────────────► │ worker.ts → mklight-wasm     │
 │   createEngine, EngineClient │ ◄──────────── │  Engine, Sessions, job poll  │
 └──────────────────────────────┘  typed arrays └───────────────┬──────────────┘
                                   (transferred)                │ shared memory
                     host workers (thread-host.ts), one per thread │
          ┌────────────────────┬───────────────────┬──────────────┴──────┐
          │ rayon pool × N     │ scheduler workers │ progress, events    │
          └────────────────────┴───────────────────┴─────────────────────┘
```

### Building

```bash
scripts/build-wasm.sh          # release: wasm-opt -O3, prints raw/gzip/brotli size
scripts/build-wasm.sh --dev    # debug, no wasm-opt (fast, ~5× larger)
```

Output: `apps/web/engine/pkg/` (wasm-bindgen `--target web`, gitignored):
`mklight_wasm.js` (glue), `mklight_wasm_bg.wasm`, type declarations and a
`build.json` marker.

Requirements (the script checks them and says what to install):

* rustup; the script installs the pinned nightly (`NIGHTLY` in the script,
  `nightly-2026-11-01`) with `rust-src` and the `wasm32-unknown-unknown`
  target on first use. Threads need the standard library rebuilt with
  `+atomics,+bulk-memory` (`-Z build-std`), which only nightly allows.
  Native builds keep the stable toolchain of `rust-toolchain.toml`; no
  toolchain file is needed for the wasm build, since the script selects the
  nightly explicitly (`cargo +nightly-…`). Bump the date deliberately, with
  a full run of the web tests.
* `wasm-bindgen-cli` of exactly the version of the `wasm-bindgen` crate in
  `Cargo.lock` (`cargo install wasm-bindgen-cli --version <v> --locked`).
* binaryen's `wasm-opt` (`brew install binaryen`) for release builds;
  `brotli` (optional) for the size report.

Build flags (all in the script): target features `atomics`, `bulk-memory`,
`mutable-globals`, `nontrapping-fptoint`, `sign-ext` and `simd128`
(fixed-width SIMD, supported by Chrome 91+ and Safari 16.4+, so by every
browser the app targets), plus rustfft's `wasm_simd` algorithms
(`mklight-wasm` feature `fft-simd`, default); linker
`--shared-memory --import-memory --max-memory=2 GiB` and the TLS exports
wasm-bindgen's thread transform needs; `panic = abort` (a panic traps the
thread; see "Known gaps"). The release profile is the workspace's (thin LTO)
without debug info.

#### Size

Release build, `nightly-2026-11-01`, wasm-opt -O3:

| | bytes |
|---|---|
| `mklight_wasm_bg.wasm` | 13.10 MB |
| gzip -9 | 4.23 MB |
| brotli -q 11 | 2.72 MB |

(The WebGPU backend, wgpu and its WGSL kernels, adds 3.91 MB raw and
0.73 MB brotli; the build without it measured 9.19 / 2.99 / 1.99 MB.)

(rustfft's SIMD algorithms add 0.15 MB raw, 0.02 MB brotli.)

The JavaScript (client, worker, host, glue) adds about 30 kB.

The whole engine is in it: every analysis, the optimisers, the stray-light
and ghost solvers, the built-in glass catalogs. Static hosts that serve
brotli or gzip (GitHub Pages serves gzip) send the compressed size.

### Engine crates on wasm32: one codebase

`mklight-engine` and every crate below it compile for
`wasm32-unknown-unknown` with the same algorithms. The only platform code is
plumbing, collected in **`mklight_core::platform`**:

| what | native | wasm32 |
|---|---|---|
| `Instant`, `SystemTime`, `UNIX_EPOCH` | `std::time` | the `web-time` crate (`performance.now()`, `Date.now()`); `std`'s clocks panic on wasm32 |
| `platform::spawn(name, f)` → `JoinHandle` | `std::thread::Builder::spawn` | queues `f` for a *host thread* (below) |
| `available_parallelism()` | `std::thread::available_parallelism` | `navigator.hardwareConcurrency`, set by the glue |
| `process_cpu_seconds()` (ghost stage profiles) | `getrusage` | 0 |
| `load_per_core()` (cost model prior) | `getloadavg` | 0 |
| `fma(a, b, c)` | unused (`mul_add` is the instruction) | linked as the `fma` symbol every `mul_add` calls: the same correctly rounded result, faster than the builtin (see "Speed") |

Changes in the engine crates (all behaviour-neutral natively):

* Every `std::time::Instant` / `SystemTime` use in `mklight-core`, `-trace`,
  `-analysis`, `-wave`, `-ghost`, `-stray`, `-optim`, `-gpu` and `-engine`
  reads `mklight_core::platform::{Instant, SystemTime}` instead.
* `mklight-engine::scheduler` starts its worker, progress and event threads
  with `platform::spawn` (the event thread also records its own id, since on
  wasm32 the id is known only once it runs); `available_parallelism` comes
  from `platform`.
* `Engine::prepare_gpu` starts its thread with `platform::spawn`.
* `analysis::optim::interruptible_for`: the cancellation watcher was a
  scoped thread; it is now a `platform::spawn` thread that owns clones of
  the tokens and is joined by a drop guard (also on unwind), with the same
  2 ms polling.
* `analysis::cost::load_per_core` and `mklight-ghost`'s `cpu_seconds` call
  `platform`; `libc` moved from `mklight-engine` and `mklight-ghost` to
  `mklight-core` (unix only).
* `Session::load_catalog_bytes` and `Session::save_to_string`: the
  path-free forms of `load_catalog_file` and `save` (split out of them,
  sharing their code), for a browser that has file contents but no paths.
* `mklight-bench`'s host statistics (`host.rs`) read the load average and
  CPU time only on unix, so the benches also build for wasm32-wasip1.
* File I/O stays as it is: on wasm32 `std::fs` returns errors, which the
  engine already reports (`io`; a project's catalog references by path come
  back in `open_warnings`).
* `mklight-metal` drops out on wasm32 as it does off macOS.
* The GPU paths' scoped threads (`mklight-analysis` aim, `mklight-wave`,
  `mklight-ghost` irradiance, `mklight-stray` hybrid, `mklight-optim` GPU
  scheduler) use `platform::scope`: `std::thread::scope` natively, the same
  API over host threads on wasm32.

`crates/mklight-wasm` is the wasm-bindgen API: the web's counterpart of
`mklight-ffi`, wrapping the same `mklight-engine` calls (engine, numbered
sessions, systems, catalogs, solves, import, projects, jobs, results). On
native targets it is empty, so `cargo test --workspace` is unaffected.

### Threads

Two kinds of threads run the engine on the web, both Web Workers that share
the module's memory:

* **The rayon pool** (the analyses' data parallelism), `N` threads, default
  `navigator.hardwareConcurrency` (`createEngine({threads})` overrides).
* **The engine's own threads**: the scheduler's workers
  (`min(cores, 4)`), its progress and event threads, and short-lived ones
  (an optimisation's cancellation watcher).

Both start the same way: Rust calls `platform::spawn`, which on wasm32 puts
the closure on a queue; *host workers* (`thread-host.ts`) take closures off
the queue and run them; the engine worker starts as many host workers as
`wanted_host_threads()` asks for, after creating the engine and after every
submission and poll. A queue rather than a worker per spawn because a worker
can only start a new worker while its event loop runs, and engine threads
block (`Atomics.wait` through std's mutexes and condvars), so a blocking
thread could never wait for a thread it just asked for. Spawns happen from
the engine worker (which returns to its event loop at once) or wait for an
idle host; a finished closure frees its host for the next one.

A browser may refuse to start a worker (an `error` event before it runs;
see "Isolation" for the case that happens in practice). Every engine thread
waits for its host, so the engine worker retries a refused start, with
growing delays (50 ms to 6.4 s, 8 attempts), and a start that fails for
good is counted as not starting (`host_start_failed`), so the next top-up
asks for it again. Without that, one refused host left a closure queued
for good: the optimiser joins a short-lived cancellation-watcher thread at
every step, so the run hung at the first step whose watcher got no host,
and Stop, which reaches the run through that watcher, was lost with it.

**Why not wasm-bindgen-rayon's worker glue.** rayon is used as it is, but
its pool is built with `ThreadPoolBuilder::spawn_handler` over
`platform::spawn` (`mklight_wasm::init_thread_pool`) instead of
wasm-bindgen-rayon's `initThreadPool`. That crate's JavaScript helper finds
the glue through `import.meta.url` (its `no-bundler` mode) or through the
package's own module path (its bundler mode); once Vite bundles the engine
worker, the first points at the bundled worker chunk (which has no
`default` init export) and the second needs a package layout wasm-bindgen
`--target web` does not produce. One host-worker mechanism for all threads
works the same bundled and unbundled, and is one code path less.

The engine worker is the wasm "main" thread: it creates the memory, the
engine and every host worker, and answers the page's calls. It blocks only
briefly (a session lock); analyses run on scheduler threads, never on it.

### Isolation (COOP/COEP)

Shared memory needs a **cross-origin isolated** page. Hosts that can set
headers send, for the app's HTML (and, simplest, every response):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

(`Cross-Origin-Resource-Policy: same-origin` on subresources is implied for
same-origin requests; cross-origin resources must send CORP or CORS.)

On hosts that cannot (GitHub Pages and other plain static hosts),
**`apps/web/engine/coi-serviceworker.js`** adds them: a service worker that
rewrites every same-origin response's headers. The same file is the page
script that installs it: placed first in `<head>`,

```html
<script src="coi-serviceworker.js"></script>
```

it registers itself and reloads the page once, after which the page is
isolated. It never loops: a second reload that still is not isolated stops,
and `ready()` explains why. Serve the file from the app's base path (copy it
into the app's `public/`), since a service worker controls only pages at or
below its own path. `ensureIsolation(swUrl?)` from `@engine` does the same
programmatically.

**Headers on every response, `304`s included.** WebKit checks a worker
script's COEP on the headers of the response it gets, and when the
browser revalidates a cached script, that is the `304 Not Modified`, which
many servers send without the configured headers (Vite's `server.headers`
and `preview.headers` among them; a `Cache-Control: no-cache` asset is
revalidated on every load). WebKit then refuses that worker ("Worker load
was blocked by Cross-Origin-Embedder-Policy") while other loads of the same
script start, so a few of the engine's host workers fail at random.
Chromium takes the headers of the cached `200` the `304` refreshes, and
Safari 27.0.1 started every worker in the same test, but Playwright's
WebKit (build 2388) does not. Hosts that set the headers should send them
on `304`s too; the engine survives a refused start either way (see
"Threads"), and `tests/static-server.mjs --isolate-200` serves exactly that
case to the engine's tests. The service worker always answers with a full
response, so pages it isolates are not affected.

`ready()` (no engine needed) reports `{isolated, threaded, threads,
hardwareConcurrency, webgpu, reason}`; `reason` is a user-facing sentence
when the engine cannot run.

**No single-threaded fallback build.** A second, non-atomic build would
double the build and the download variants, and the engine's scheduler
needs threads (it would need a separate synchronous job path). With the
service worker, isolation works on every static host over HTTPS in current
Chrome, Safari and Firefox; what remains (plain HTTP on a non-localhost
address, browsers or private modes that disable service workers, the page
embedded in a non-isolated frame) gets a clear message instead:
`createEngine` rejects with `EngineError` `unavailable` and `ready().reason`.

### The client (`@engine`)

`apps/web/engine/src/`:

| file | role |
|---|---|
| `types.ts` | The contract the web UI codes against: `EngineClient`, `AnalysisJob`, `EngineInfo`, `ModelSnapshot`, `AnalysisResult`, `EngineError`, the JSON shapes |
| `index.ts` | Exports: `createEngine`, `ready`, `ensureIsolation`, `decodeResult`, all types |
| `client.ts` | The page-side `EngineClient` (one per session) over one worker connection |
| `worker.ts` | The engine worker: owns the wasm engine, starts the host workers, polls jobs |
| `thread-host.ts` | A host worker (runs engine and rayon threads) |
| `protocol.ts` | The typed messages between `client.ts` and `worker.ts` |
| `ready.ts`, `isolation.ts` | Readiness report; registering the isolation service worker |

The web app imports it through the alias `@engine` →
`apps/web/engine/src/index.ts` and needs, in its Vite config,
`worker: { format: 'es' }` (the workers are ES modules) and the isolation
headers on the dev and preview servers through the plugin
`crossOriginIsolation()` of `apps/web/engine/vite-isolation.ts`, as
`apps/web/engine/vite.config.ts` shows (not `server.headers` /
`preview.headers`, which leave the headers off `304` revalidations and off
some of Vite's own dev modules; see "Isolation"). The module has no npm runtime
dependencies; its `pkg/` must be built (`scripts/build-wasm.sh`) before the
app builds.

```ts
import { createEngine, ready } from '@engine'

const r = await ready()                   // r.reason when the engine cannot run
const engine = await createEngine()       // one worker, threads = hardwareConcurrency
const snap = await engine.openSnapshot(system)   // throws EngineError{kind:'invalidModel', issues}
const job = engine.submit({ analysis: 'spot_diagram', settings: { rings: 8 } }, snap)
job.onProgress(p => console.log(p.fraction, p.stage))
const r = await job.result                // {json, buffers: {points: Float64Array, …}, runInfo, …}
```

Contract, against the Swift `EngineClient` (`apps/macos/Sources/
MKLightEngine/EngineClient.swift`):

* `info`: name, version, backends (`mk_engine_list_backends`), threads,
  isolated, threaded, webgpu.
* `openSnapshot(system)` sets the session's system and returns
  `{hash, system, libraryHash}` (the system as the engine holds it). What
  the engine cannot take as a system is `invalidModel` with `issues`.
* `analysisDescriptors()`: `mk_engine_list_analyses`, verbatim.
* `submit(req, snap)`: the session is set to `snap` (if it holds another
  snapshot) and the job submitted in one step, as in the Swift client.
  `req.backend` is `'auto'` (default), `'cpu_reference'`, `'gpu'` (the
  browser GPU backend with CPU fallback) or the engine's JSON object;
  `fresh`, `tier`, `aim_stop`, `route` as in FFI.md. Never throws: an
  unknown analysis rejects `result` with `notFound`, bad settings with
  `invalidSettings`. `cancel()` rejects with `cancelled`; `stop()` ends a
  stoppable analysis with its best result.
* `result`: `{json, buffers, bufferInfo, runInfo, libraryHash, backend,
  derivedSnapshotHash, resultVersion}`; `json` is the envelope's `result`.
* `materialNames()`, `glasses(catalog?)`, `catalogs()`,
  `loadCatalog(fileName, text)` (AGF, or GlassCatalog JSON for `.json`;
  recorded in the project by file name, relative to the project folder).
* `applySolves(settings, snap)` → `{report, snapshot}`.
* `importPrescription(data, 'zmx'|'seq')`: glasses resolve against the
  built-in catalog and the session's catalogs; the session is unchanged.
* `openProject(text)` / `saveProject()`: `.mklight` text in and out
  (FFI.md "Project file"). Catalog references by path cannot be read in a
  browser and come back in `info.open_warnings`.
* `makeDocumentEngine()`: a new session on the same worker; `dispose()`
  frees a session.

Refinements over the original sketch, all additive: `AnalysisResult` also
carries `bufferInfo`, `libraryHash`, `backend`, `derivedSnapshotHash`,
`resultVersion`; `AnalysisJob` carries `analysis` and `snapshotHash`;
`EngineClient` adds `glasses`, `openProject`, `saveProject`, `sessionInfo`,
`dispose`; `ModelSnapshot` carries `libraryHash`; `EngineInfo` adds
`threaded`. Engine JSON keeps its snake_case keys; only the TS wrappers are
camelCase.

#### Messages and buffers

Calls are typed request/reply messages (`protocol.ts`), handled strictly in
order by the worker. Progress is a 30 ms poll of the active jobs' status in
the worker (`job_statuses`), posted only when it changes; when a job ends
the worker takes its result and posts it. Bulk buffers leave the shared wasm
memory with **one copy** (a fresh, unshared typed array per buffer, made in
Rust by `job_take_result`) and are **transferred** to the page (no second
copy); never JSON. `u64`/`i64` buffers arrive as `BigUint64Array` /
`BigInt64Array`.

### Memory

* Shared memory: initial 2.5 MiB (grows on demand), **maximum 2 GiB** (`--max-memory`). A
  shared memory's maximum is reserved up front by some engines; 4 GiB fails
  on iOS/iPadOS Safari.
* Each thread's shadow stack is 2 MiB (main and host threads).
* The analysis graph's budget on the web is **64 MiB**
  (`analysis_graph_bytes`; native default 256 MiB); `createEngine({config})`
  overrides any engine configuration key.

### WebGPU

The WebGPU backend (`crates/mklight-webgpu`, docs/GPU.md §11) is the web's
GPU: `gpu:webgpu`, the backend of the client's `backend: 'gpu'`.

* **The GPU worker** (`src/gpu-host.ts`): started by the engine worker,
  next to the host workers, when `navigator.gpu` exists; it shares the
  module and memory and runs `gpu_thread` (`mklight-wasm`), an async loop
  that owns the `GPUDevice` (wgpu objects are not `Send`), serves requests
  from a channel in the shared memory and sleeps in `Atomics.waitAsync`. It
  never blocks.
* **The proxy**: engine and rayon threads reach the device through
  `mklight_webgpu::proxy::GpuProxy`: a call puts a request pointing at the
  caller's batch (uploads and readback destinations stay in the shared
  heap) on the channel, notifies the GPU worker and blocks in `Atomics.wait`
  until the batch's readbacks are written.
* **Registration**: after `engine_create` the engine worker calls
  `register_webgpu` (feature `webgpu`, on by default), which registers
  `GpuProvider::webgpu_browser()`. The device opens on first use (listing
  the backends opens it): the proxy waits up to 10 s for the GPU worker and
  runs the numerics probe. A browser without an adapter lists the backend
  unavailable with the reason.
* **Threads**: the GPU paths' scoped threads use
  `mklight_core::platform::scope` (`std::thread::scope` natively, host
  threads here), and the backends' plan workers `platform::spawn`.
* **Routing**: `auto` decides from a calibration of this device: right
  after the engine is created, the worker installs the one cached in
  IndexedDB (`mklight-webgpu-calibration`, keyed by the adapter's
  vendor/architecture/device, the thread count and the engine version) or
  starts a measurement in the background (spot diagram, ray grid and
  geometric MTF at two small sizes on both backends; a few seconds with
  pipeline compilation) and caches it. Meanwhile the prior routes: ray
  grids from 250,000 rays, spot diagrams and geometric MTFs from 400,000
  rays to WebGPU, everything else to the CPU reference. An explicit
  `'gpu'` request runs every WebGPU analysis on the device. GPU.md §11.6
  has the rule and the browser measurements; `createEngine({
  webgpuCalibration: 'off' })` keeps the prior, and the client's
  `webgpuCalibration()` / `setWebgpuCalibration(c)` read and force it.

`ready().webgpu` and `info.webgpu` report whether the browser has a WebGPU
adapter; `info.backends` lists `gpu:webgpu` with its availability. Tests:
`tests/webgpu.spec.ts` (Chromium runs the full browser in new-headless mode
with WebGPU enabled, since the headless shell has no adapter; WebKit has
WebGPU as is; it checks the calibration is measured, cached, reused and
changes routing when forced). `MKLIGHT_WEB_PORT` (default 4710) moves the
harness's two ports, for test runs of two checkouts at once.

### Tests

`apps/web/engine/tests/engine.spec.ts` (Playwright, headless **Chromium**
and **WebKit**, the latter standing in for Safari):

```bash
scripts/build-wasm.sh
cd apps/web/engine
npm install
npx playwright install chromium webkit   # once
npm test                                 # or test:chromium / test:webkit
npm run typecheck
```

The global setup builds the native CLI and writes reference results for a
Cooke triplet (`mklight new --sample cooke_triplet`; `mklight run … --backend
cpu --buffers`). The harness (`harness/`) is built with Vite and served by a
plain static server **without** isolation headers (port 4710, the
GitHub-Pages case, isolated by the service worker), **with** them (4711),
and with them on `200`s but bare `304` revalidations (4712, on which WebKit
refuses some worker starts). The tests check: isolation both ways; that
refused worker starts do not stall an optimisation run next to the app's
other jobs, and that Stop reaches a running one (4712); the engine loads and lists its
backends and analyses; the Cooke triplet opens with the native snapshot
hash; `spot_diagram`, `mtf`, `psf` and `ray_fan` on the CPU f64 reference
match the native results (below); `run_info` equals the native one except
`started_unix_ms` and `duration_ms`; errors (`invalidModel` with issues,
`notFound`, `invalidSettings`), project save, catalogs, prescription import,
document sessions and cancellation; and the rayon speed-up (a 250-ring spot
diagram at 1 thread and at `hardwareConcurrency` threads, required > 2× with
≥ 4 cores).

**Agreement with native.** The CPU f64 reference on the web computes with
the same code as natively; the differences are in the elementary functions
(`sin`, `cos`, `exp`, `pow`, … come from Rust's `libm` port on wasm32 and
from the system's libm natively) and stay in the last bits. Measured
(Cooke triplet, default settings, Chromium and WebKit alike):

| analysis | numbers | bit-identical | max abs diff | max rel diff |
|---|---|---|---|---|
| `spot_diagram` | 4 796 | 4 283 | 1.8e-14 | 1.8 (a centroid of ±3e-17, i.e. zero) |
| `mtf` | 446 | 320 | 1.3e-12 | 7.3e-12 |
| `psf` | 253 853 | 6 168 | 2.3e-13 | 5.7e-9 (a 7.7e-8 value) |
| `ray_fan` | 2 244 | 2 076 | 1.6e-14 | 1.0e-9 (a 5.4e-6 value) |

The tests require, per number, an absolute difference ≤ 1e-12 or a
relative one ≤ 1e-9; the snapshot hash, `library_hash` and every `run_info`
field but `started_unix_ms` and `duration_ms` are equal. Chromium and
WebKit produce bit-identical results (same wasm, deterministic engine).
`f64::mul_add` is a correctly rounded `fma` on both sides (hardware natively,
`mklight_core::platform::fma` on wasm), so it adds no difference.

**Speed.** On a 12-core Apple Silicon Mac (8 performance cores), release
build as shipped (simd128, rustfft SIMD, `platform::fma`):

| | 1 thread | all threads | speed-up |
|---|---|---|---|
| Chromium, spot diagram 250 rings | 3.06 s | 0.58 s (12) | 5.3× |
| WebKit, same | 3.48 s | 0.65 s (8, WebKit's cap) | 5.4× |
| native CLI, same | 0.40–0.45 s | 0.12–0.25 s (12) | |

Other analyses, Chromium, 1 thread → 12: geometric MTF 3.4–4×,
through-focus MTF 3.1–3.6×, relative illumination map 1.9×, PSF 512²
1.3–1.6×, MTF vs field 1.1–1.4× (their serial parts dominate).

Single-threaded, the web engine is 1.4–1.9× slower than native for FFT-heavy
work (PSF, through-focus MTF) and about 7× slower for ray tracing (spot
diagrams, geometric MTF). `mklight-bench`'s `packet_bench` (Cooke triplet,
48 rings, one thread; the wasm32-wasip1 build under wasmtime, which runs
the same code as the browser's engine) shows why:

| `trace_rays`, Cooke, 1 thread | ms |
|---|---|
| native (aarch64, `fmadd`, NEON) | 9.6 |
| wasm, compiler-builtins' integer `fma` | 110 |
| wasm, `platform::fma` (shipped) | 100.7 |
| wasm, `mul_add` unfused (`a*b + c`; wrong results, a lower bound only) | 31.7 |

The tracer's `mul_add` (about 120 calls in the packet kernel alone) is
70 % of wasm ray-tracing time: wasm has no fused multiply-add, so every
one is a call to a software `fma`. The ways around it are closed by bitwise
identity with native: relaxed SIMD's `madd` may or may not fuse depending
on the machine, and an unfused `a*b + c` gives other numbers. What the web
build does instead is replace the builtin `fma` (a 128-bit integer
algorithm that spends half its time in `scalbn`) by
**`mklight_core::platform::fma`**: Dekker's exact product, TwoSum and
Boldo–Melquiond's round-to-odd, correctly rounded and so bitwise identical
to `mul_add`, branch-free on the tracer's inputs (zero products and exact
cancellations, 22 % of the calls on the Cooke triplet, take the general
path; operands it cannot handle exactly take the integer algorithm). It
is linked as the `fma` symbol on wasm32 only, so no tracer code changes;
natively nothing changes. It saves about 9 % of tracing in wasmtime and
13 % in Chromium (spot diagram, 250 rings, 1 thread: 3.57 → 3.06 s). The
remaining cost is the emulation's ~30 dependent operations per call;
inlining it into the kernel (by calling it instead of `mul_add`) gained only
2 % more and is not done. `crates/mklight-core/tests/fma.rs` checks it bit
for bit against the hardware's `fmadd` on 13 million inputs (random bit
patterns over every exponent, the fast path's edges, tracer-sized values,
cancellations and near-ties, products at the underflow and overflow
boundaries, infinities, NaN and signed zeros);
`MKLIGHT_FMA_STRESS=100` runs 1.3 billion.

**SIMD.** Measured with five builds (`apps/web/engine/tests/
measure-variants.mjs`; `scripts/build-wasm.sh --no-simd`, `--no-fft-simd`,
`--libm-fma` make the variants), Chromium, ms at 1 thread / 12 threads:

| build | spot 250 rings | geometric MTF | PSF 512² | through-focus MTF |
|---|---|---|---|---|
| A: no SIMD, builtin fma | 3657 / 622 | 116 / 30 | 372 / 241 | 1108 / 328 |
| C: simd128, scalar FFT, builtin fma | 3639 / 651 | 120 / 60 | 390 / 221 | 1020 / 332 |
| B: simd128 + FFT SIMD, builtin fma | 3566 / 635 | 114 / 26 | 332 / 239 | 1020 / 306 |
| E: no SIMD, `platform::fma` | 4143 / 690 | 117 / 28 | 388 / 238 | 1079 / 326 |
| shipped: simd128 + FFT SIMD + `platform::fma` | 3085 / 566 | 97 / 29 | 310 / 240 | 1020 / 327 |

(Repeated runs vary by about ±10 %; best of three.) simd128 alone changes
little: in the packet kernel LLVM emits about 50 `f64x2` arithmetic
operations against 285 scalar ones and ~120 scalar `fma` calls, because
every `mul_add` is a scalar call that splits the vector chains (natively
the kernel has about 700 NEON `f64x2` instructions; CONVENTIONS §7.7).
rustfft's SIMD speeds the FFT-bound PSF by about 10 %.

Results: every analysis checked (spot diagram, ray fan, MTF, geometric
MTF, through-focus MTF, OPD fan, wavefront map, PSF) is bit-identical
across all five builds, except that rustfft's SIMD algorithms change the
PSF's last bits (as on native, where rustfft uses NEON; against native the
PSF gets slightly closer: 6 168 bit-identical values instead of 5 060).
`mklight-trace`'s integration tests, the packet ≡ scalar contract
(`tests/packets.rs`) and the committed reference fixtures included, pass
on wasm32 with simd128 and `platform::fma` (117 of 123; wasm32-wasip1
under wasmtime; the 5 skipped cancel from a second thread, which WASI
lacks, and 1 is ignored everywhere):

```bash
CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=/" RUSTFLAGS="-C target-feature=+simd128" \
  CARGO_TARGET_DIR=target/wasi cargo test --release -p mklight-trace --test tests \
  --target wasm32-wasip1 -- --skip semi_diameters_cancel_within_a_task --skip cancel
```

### Known gaps

* No single-threaded build (see "Isolation").
* `panic = abort`: an engine panic traps the thread it runs on. Natively the
  scheduler turns a job's panic into a failed job; on the web the job's
  thread dies with it, the job stays `running` and its `result` never
  settles (the console shows the panic message), and the engine should be
  recreated. A panic in a call the worker answers itself becomes an
  `EngineError` `panic`. Building with `panic = unwind` needs wasm
  exception handling in the build-std standard library (not yet tried).
* Ray tracing is about 7× slower than native single-threaded: 70 % of it is
  the software `fma` that bitwise identity requires (see "Speed").
* Real Safari is not driven by the test suites (Playwright's WebKit stands
  in for it; driving Safari needs "Allow Remote Automation" switched on).
  The optimisation and Stop were checked by hand in Safari 27.0.1 (a page
  that posts its result to a local server, against a server answering
  revalidations with bare `304`s).
* Catalog and project files by path: the browser has contents, not paths;
  `loadCatalog(fileName, text)` and `openProject(text)` cover the app's
  needs, and a project's references to catalog files are reported in
  `open_warnings` rather than read.
* `prepareGPU` / `gpuPreparation` are not in the web client yet: the first
  use of a WebGPU kernel compiles its pipeline (30–200 ms per kernel, several
  seconds for the trace and aiming kernels on a cold driver cache).
* Job progress is polled (30 ms), not pushed by the engine's event thread.
