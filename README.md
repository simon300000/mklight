# MKLight

GPU-accelerated sequential optical design and optimisation for imaging systems
such as camera lenses, with ghost, stray-light and wave-optics solvers that all
share one physical model.

* **Engine:** headless Rust workspace (`crates/`). The CPU double-precision
  reference sits beside platform GPU backends: Metal on macOS today, with
  Vulkan and CUDA planned.
* **macOS app:** Swift/SwiftUI/AppKit with a Metal viewport (`apps/macos`).
* **Python and CLI:** headless access to the same engine.

Start with:

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): crates, state model, solvers, backends
* [docs/CONVENTIONS.md](docs/CONVENTIONS.md): units, signs, definitions, precision budget
* [docs/DECISIONS.md](docs/DECISIONS.md): decisions made and open questions

## Building

```bash
source ~/.cargo/env
cargo test --workspace
```

```bash
scripts/build-macos.sh   # builds MKLight.app
```

Requires Rust stable, Xcode with the Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`), and macOS 15 or later on
Apple Silicon.
