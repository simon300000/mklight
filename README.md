<p align="center">
  <img src="docs/images/icon.png" width="128" alt="MKLight icon">
</p>

<h1 align="center">MKLight</h1>

<p align="center">
  Optical design for imaging systems.<br>
  GPU-accelerated, native on the Mac and on Windows, and in the browser.
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#the-apps">The apps</a> ·
  <a href="#a-closer-look">A closer look</a> ·
  <a href="#command-line-and-python">Scripting</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="docs/DEVELOPMENT.md">Development</a>
</p>

<p align="center">
  <img src="docs/images/main-window.png" alt="MKLight main window: a Double Gauss lens with lens data, layout, spot diagram, MTF and glass map">
</p>

MKLight designs, analyses, optimises and tolerances sequential lens systems:
camera lenses, relays, telescopes. Its geometric, wave-optics, ghost and
stray-light solvers all read one physical model, and every analysis redraws
as you edit.

## Features

**Design**<br>
A lens data editor with inline variables, solves and pickups. Surfaces run
from spheres and aspheres to Forbes Q-types, toroids, freeforms and
diffractive phase, with coatings, tilts and decentres, and multiple
configurations.

**Analysis**<br>
More than 40 analyses: spot diagrams and ray fans, MTF and PSF, wavefront
maps, image simulation, relative illumination, polarisation and
transmission.

**Ghosts and stray light**<br>
Ghost paths and ghost irradiance, and non-sequential Monte Carlo stray light
on the optics and the mechanics.

**Optimisation**<br>
Merit functions from a default generator or your own operands. Damped least
squares and trust regions with bounds and constraints, global search, and
glass substitution that re-optimises every combination it tries.

**Tolerancing**<br>
Sensitivities and Monte Carlo yields, with compensators, surface
irregularity and element motions.

**Import and scripting**<br>
Opens `.zmx` and `.seq` prescriptions and AGF glass catalogues. A command
line and Python bindings run on the same engine as the apps.

## The apps

Three front ends share one Rust engine and the same project files, so a
design opens unchanged on every platform.

**macOS.** A native app in SwiftUI, with Liquid Glass and a Metal viewport.
Lens data, the layout and every analysis are panels: drag one onto another
to make a tab, near an edge to split it, or out of the window to give it a
window of its own. Candidate designs from optimisation, glass substitution
or tolerancing are previewed in every panel before you apply them, and
every edit can be undone. It is the reference: features land here first.
[App](docs/APP.md)

<p align="center">
  <img src="docs/images/separate-windows.png" alt="A Cooke triplet in the main window, with through-focus MTF and a relative illumination map in a separate window">
</p>

**Windows.** A native WinUI 3 app with Fluent design and Mica, a Direct3D
3-D view, and CUDA on NVIDIA GPUs. It follows the Mac app feature for
feature (panels and separate windows, the inspector, design jobs with
previews, the glass library) and is close to parity.
[Windows](docs/WINDOWS.md)

<table>
<tr>
<td width="50%"><img src="docs/images/windows-main.png" alt="MKLight for Windows: a Double Gauss with lens data, layout, spot diagram, MTF, glass map and the surface inspector"></td>
<td width="50%"><img src="docs/images/windows-3d-dark.png" alt="MKLight for Windows in the dark theme: a Double Gauss in the Direct3D 3-D view with ray fans and a wavefront map"></td>
</tr>
</table>

**Web.** The same engine compiled to WebAssembly and run in a Web Worker,
with WebGPU where the browser offers it, so nothing has to be installed.
[Web](docs/WEB.md)

<p align="center">
  <img src="docs/images/web-main.png" alt="MKLight in the browser: a Double Gauss with lens data, layout, spot diagram, ray fan and the surface inspector">
</p>

## A closer look

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

## Under the hood

- **One physical model.** Every solver reads the same system, materials and
  coatings. The sequential, ghost and stray-light solvers each own a class
  of light paths, so their results add up without counting any light twice.
- **A reference behind every result.** A CPU double-precision tracer is the
  reference. The GPU backends (Metal on the Mac, CUDA on NVIDIA, WebGPU in
  the browser) share one kernel specification and switch to double-float
  arithmetic where single precision would miss the accuracy budget; each is
  checked against the reference on every sample design.
- **The GPU where it pays.** A calibrated cost model sends a job to the GPU
  only when it will finish sooner there. Small, latency-bound work stays on
  the CPU, and a content-hashed cache rebuilds only what an edit changed.
- **Reproducible results.** Every result records the engine version, the
  design's content hash, its settings, the backend and precision it ran on,
  and the random seed.

## Command line and Python

The engine behind the app is also a command-line tool and a Python package.

```bash
mklight new lens.mklight --sample double_gauss
mklight show lens.mklight
mklight run lens.mklight spot_diagram --json
```

```python
import mklight

s = mklight.Engine().new(sample="achromat")
print(mklight.paraxial(s)["efl"])     # 100.0009…
r = s.run("spot_diagram")
print(r.run_info["snapshot_hash"])
```

## Documentation

- **Overview:** [Architecture](docs/ARCHITECTURE.md) ·
  [Conventions](docs/CONVENTIONS.md) · [Decisions](docs/DECISIONS.md)
- **The model:** [Surfaces](docs/SURFACES.md) ·
  [Materials](docs/MATERIALS.md) · [Solves](docs/SOLVES.md) ·
  [Import](docs/IMPORT.md)
- **Image quality:** [Analysis](docs/ANALYSIS.md) ·
  [Wave optics](docs/WAVE.md) · [Polarisation](docs/POLARIZATION.md)
- **Ghosts and stray light:** [Ghosts](docs/GHOST.md) ·
  [Stray light](docs/STRAY.md)
- **Optimisation and tolerancing:** [Optimisation](docs/OPTIMIZATION.md)
- **GPU and results:** [GPU](docs/GPU.md) · [Results](docs/RESULTS.md)
- **Front ends:** [macOS app](docs/APP.md) · [Windows app](docs/WINDOWS.md) ·
  [Web](docs/WEB.md) · [CLI](docs/CLI.md) · [Python](docs/PYTHON.md) ·
  [FFI](docs/FFI.md)
- **Working on MKLight:** [Development](docs/DEVELOPMENT.md)

## Status

MKLight is in active development and has not been released yet; for now it
builds from source, as [Development](docs/DEVELOPMENT.md) describes. The
macOS app leads; the Windows app is close to parity, and the web version
follows. GPU backends: Metal on the Mac, CUDA on NVIDIA (Windows; Linux
untested), and WebGPU in the browser. iPad and iPhone versions of the app, a
Linux front end, and a Vulkan backend for other GPUs are planned.

## License

Proprietary. All rights reserved.
