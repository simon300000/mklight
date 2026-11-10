<p align="center">
  <img src="docs/images/icon.png" width="112" alt="MKLight icon">
</p>

<h1 align="center">MKLight</h1>

<p align="center">
  <b>GPU-accelerated optical design for imaging systems</b><br>
  Sequential lens design, analysis, optimisation and tolerancing, with ghost,<br>
  stray-light and wave-optics solvers that all share one physical model.
</p>

<p align="center">
  <img alt="macOS 27" src="https://img.shields.io/badge/macOS-27-000000?logo=apple&logoColor=white">
  <img alt="Rust" src="https://img.shields.io/badge/engine-Rust-B7410E?logo=rust&logoColor=white">
  <img alt="Swift 6" src="https://img.shields.io/badge/app-SwiftUI-F05138?logo=swift&logoColor=white">
  <img alt="Metal" src="https://img.shields.io/badge/GPU-Metal-0A84FF">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
</p>

<p align="center">
  <img src="docs/images/main-window.png" alt="MKLight main window: a Double Gauss lens with lens data, layout, spot diagram, MTF and glass map">
</p>

## Highlights

- **One physical model.** Geometric, wave-optics, ghost and stray-light
  solvers read the same `OpticalSystem` and its materials and coatings. The
  sequential, ghost and stray-light solvers each own a class of light paths,
  so their results add up without counting any light twice.
- **A reference you can trust.** A CPU double-precision tracer is the
  reference. The Metal GPU backend switches to double-float arithmetic where
  single precision would not meet the accuracy budget, and it is checked
  against the reference on every sample design.
- **Fast where it matters.** The GPU takes the throughput work, such as
  dense wave-optics transforms, image simulation and large ghost maps. A
  calibrated cost model sends a job to the GPU only when it will finish
  sooner there, so small, latency-bound work stays on the CPU. A content-hashed analysis cache
  rebuilds only what an edit changed.
- **Reproducible results.** Every result records the engine version, the
  design's content hash, its settings, the backend and precision it ran on,
  and the random seed.
- **A native Mac app.** SwiftUI with Liquid Glass and a Metal viewport. Every
  analysis updates live as you edit, and every panel can be tabbed, split or
  moved into a window of its own.
- **Scriptable.** A command line and Python bindings run on the same engine
  as the app.

## Features

<table>
<tr><td width="50%" valign="top">

**Design**
- Lens data editor with inline variables, solves and validation
- Surfaces: sphere and conic, even and odd aspheres, Forbes Q-con and
  Q-bfs, toroid, biconic and cylinder, XY polynomial, Zernike sag, and
  diffractive phase on a plane, conic or asphere base
- Decentres, tilts, mechanics, coatings and multiple configurations
- Paraxial solves and pickups
- Glass catalogs (AGF) with melt offsets, temperature and pressure
- Import of `.zmx` and `.seq` prescriptions

</td><td width="50%" valign="top">

**Analysis** (more than 40)
- Spot diagrams, ray and OPD fans, field curvature and distortion, Seidel
  sums, longitudinal and lateral colour
- MTF, through-focus MTF and MTF vs field; PSF (FFT and Huygens), line
  spread, encircled energy, wavefront maps with Zernike fits
- Image simulation of test charts
- Relative illumination, chief ray angle and pupil aberration maps
- Ghost paths and ghost irradiance, non-sequential stray light
- Polarisation ray tracing and system transmission

</td></tr>
<tr><td valign="top">

**Optimisation and tolerancing**
- Merit functions from a default generator (RMS spot, wavefront or MTF),
  or custom operands
- Damped least squares and trust regions with bounds and constraints;
  global search with differential evolution, CMA-ES, basin hopping and more
- Glass substitution that re-optimises every combination it tries
- Tolerancing by sensitivity and Monte Carlo, with compensators, surface
  irregularity, element motions and catalogue index and Abbe tolerances

</td><td valign="top">

**The app**
- Metal layout: y–z and x–z sections and a 3-D orbit view, with surface
  meshes and beam footprints
- Live analyses that redraw as you edit, with full undo
- Candidate designs from optimisation, glass substitution or tolerancing
  previewed in every panel before you apply them
- Panels that can be tabbed, split in any direction or moved to their own
  windows, all saved with the project
- Light and dark appearance

</td></tr>
</table>

## Panels and windows

Lens data, the layout and every analysis are panels. Drag a panel by its
title onto another panel to make a tab, near an edge to split it, or out of
the window to open it in a window of its own. Separate windows stay live
views of the same design: they follow every edit, selection and preview, and
they close with the main window.

<p align="center">
  <img src="docs/images/separate-windows.png" alt="A Cooke triplet in the main window, with through-focus MTF and a relative illumination map in a separate window">
</p>

## Gallery

<table>
<tr>
<td width="50%"><img src="docs/images/optimize.png" alt="Optimising a Double Gauss: the result previewed in the window before it is applied"><br><sub><b>Optimisation.</b> The result is previewed in the layout and the analyses before you apply it.</sub></td>
<td width="50%"><img src="docs/images/glass-substitution-dark.png" alt="Glass substitution on an achromat: combinations ranked by merit, one previewed"><br><sub><b>Glass substitution.</b> Combinations are ranked by their re-optimised merit; select one to preview it.</sub></td>
</tr>
<tr>
<td><img src="docs/images/tolerance.png" alt="Tolerancing a Cooke triplet: predicted performance, sensitivities and a previewed Monte Carlo trial"><br><sub><b>Tolerancing.</b> Sensitivities and Monte Carlo yields; any trial can be rebuilt and previewed.</sub></td>
<td><img src="docs/images/image-quality.png" alt="Image simulation of a Siemens star, through-focus MTF and a relative illumination map"><br><sub><b>Image quality.</b> Image simulation, through-focus MTF and a relative illumination map.</sub></td>
</tr>
<tr>
<td><img src="docs/images/layout-3d-dark.png" alt="A Double Gauss in the 3-D view with the selected surface highlighted, ray fans and a wavefront map"><br><sub><b>3-D layout.</b> The selected surface is highlighted; ray fans and a wavefront map beside it.</sub></td>
<td><img src="docs/images/freeform-3d.png" alt="An XY freeform corrector in 3-D with the surface mesh and the inspector's departure map"><br><sub><b>Freeform surfaces.</b> An XY polynomial corrector with its surface mesh and departure map.</sub></td>
</tr>
</table>

## Getting started

### Requirements

- Apple Silicon Mac with macOS 27 or later
- Xcode 27.2 or later (Swift 6.4); optionally the Metal Toolchain
  (`xcodebuild -downloadComponent MetalToolchain`) to precompile the viewport
  shaders
- Rust stable (1.90 or later)
- Python 3.11 or later, for the Python bindings

### Engine and tests

```bash
cargo test --workspace
```

### The Mac app

```bash
scripts/build-macos.sh          # release build → build/macos/MKLight.app
open build/macos/MKLight.app
```

`scripts/build-macos.sh --debug` builds a debug app. See
[apps/macos/README.md](apps/macos/README.md) for the package layout and
[docs/APP.md](docs/APP.md) for the design of the app.

### Command line

```bash
cargo build -p mklight-cli --release      # → target/release/mklight
mklight new lens.mklight --sample double_gauss
mklight show lens.mklight
mklight run lens.mklight spot_diagram --json
```

### Python

```bash
cd crates/mklight-py
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
```

```python
import mklight

s = mklight.Engine().new(sample="achromat")
print(mklight.paraxial(s)["efl"])     # 100.0009…
r = s.run("spot_diagram")
print(r.run_info["snapshot_hash"])
```

## Repository layout

| path | contents |
|---|---|
| `crates/mklight-core` | The physical model, units, materials and glass catalogs, coatings, jobs |
| `crates/mklight-trace` | The CPU double-precision reference tracer, paraxial optics, solves, polarisation |
| `crates/mklight-analysis` | Geometric analyses |
| `crates/mklight-wave` | Wave optics: PSF, MTF, wavefront, image simulation |
| `crates/mklight-ghost` | Explicit ghost paths |
| `crates/mklight-stray` | Non-sequential Monte Carlo stray light |
| `crates/mklight-optim` | Merit functions, optimisers, glass substitution, tolerancing |
| `crates/mklight-gpu`, `crates/mklight-metal` | The GPU backend interface and the Metal backend |
| `crates/mklight-engine` | Sessions, projects, the analysis registry and job scheduler |
| `crates/mklight-ffi`, `crates/mklight-py`, `crates/mklight-cli` | The C ABI, Python bindings and command line |
| `apps/macos` | The SwiftUI app |

## Documentation

| | |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Crates, state model, solvers and backends |
| [Conventions](docs/CONVENTIONS.md) | Units, signs, definitions and the precision budget |
| [Decisions](docs/DECISIONS.md) | Decisions made and open questions |
| [Surfaces](docs/SURFACES.md) · [Materials](docs/MATERIALS.md) · [Solves](docs/SOLVES.md) · [Import](docs/IMPORT.md) | The model |
| [Analysis](docs/ANALYSIS.md) · [Wave optics](docs/WAVE.md) · [Polarisation](docs/POLARIZATION.md) | Image quality |
| [Ghosts](docs/GHOST.md) · [Stray light](docs/STRAY.md) | Ghost and stray-light solvers |
| [Optimisation](docs/OPTIMIZATION.md) | Merit functions, optimisers and tolerancing |
| [GPU](docs/GPU.md) · [Results](docs/RESULTS.md) | The Metal backend, its accuracy and the result format |
| [App](docs/APP.md) · [CLI](docs/CLI.md) · [Python](docs/PYTHON.md) · [FFI](docs/FFI.md) | Front ends |

## Status

MKLight is in active development and has not been released yet. The macOS
app and the Metal backend are the current focus. iPad and iPhone versions
of the app, Windows and Linux front ends, and Vulkan and CUDA backends are
planned.

## License

Proprietary. All rights reserved.
