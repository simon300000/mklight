# Developing MKLight

How to build and test MKLight, and where things are. The design of the
system is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

- Apple Silicon Mac with macOS 27 or later
- Xcode 27.2 or later (Swift 6.4); optionally the Metal Toolchain
  (`xcodebuild -downloadComponent MetalToolchain`) to precompile the viewport
  shaders
- Rust stable, 1.90 or later (`rust-toolchain.toml` selects it, with rustfmt
  and clippy)
- Python 3.11 or later, for the Python bindings

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
| `crates/mklight-bench` | Measurement and qualification harnesses and the result comparator they share |
| `crates/mklight-ffi`, `crates/mklight-py`, `crates/mklight-cli` | The C ABI, Python bindings and command line |
| `apps/macos` | The SwiftUI app ([apps/macos/README.md](../apps/macos/README.md) has its package layout) |
| `docs` | The reference documents; `docs/history` keeps two measurement records they cite |
| `scripts` | `build-macos.sh` builds the app, `test.sh` runs the Rust tests |

ARCHITECTURE.md has the diagram of how the crates depend on each other and
what each one holds.

## Building

### The app

```bash
scripts/build-macos.sh                  # release build → build/macos/MKLight.app
open build/macos/MKLight.app
```

`scripts/build-macos.sh --debug` builds a debug app, which accepts launch
options for UI checks, and `--engine-only` builds only the engine library
the Swift package links. The build integration and the launch options are
in [APP.md](APP.md#building).

### The command line

```bash
cargo build -p mklight-cli --release    # → target/release/mklight
```

[CLI.md](CLI.md) describes the commands.

### The Python package

```bash
cd crates/mklight-py
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
```

Wheels and the package layout are in
[PYTHON.md](PYTHON.md#building-and-installing).

## Tests

### Rust

```bash
scripts/test.sh
```

builds the workspace and runs every test with the test programs side by
side: about two minutes on an M2 Pro. It runs a program that failed once
more on its own, so a test that fails only because the machine is busy does
not fail the run. `cargo test --workspace` runs the same tests one program
after another, in about four and a half minutes.

Each crate has one program for its unit tests and one for its integration
tests (`tests/main.rs`, with every file of `tests/` as a module), so the
tests of one file are

```bash
cargo test -p mklight-trace --test tests geometry::
```

`scripts/test.sh` takes the same selections as `cargo test`, and passes what
follows `--` to every test program:

```bash
scripts/test.sh -p mklight-trace -p mklight-wave
scripts/test.sh -- --ignored
JOBS=4 scripts/test.sh                  # programs at a time (default 8)
```

After a change that rebuilds everything allow about a minute to build and
three to run: macOS scans every newly built program when it first starts
(XProtect). macOS has a setting for this, *System Settings → Privacy &
Security → Developer Tools*: the apps listed there (a terminal, or the app
that starts the tests) may run locally built software without that check.
It is a security setting to decide on yourself.

### Swift

```bash
scripts/build-macos.sh --engine-only
cd apps/macos && swift test
```

Without the engine library the package still builds and the tests against
the real engine are skipped ([APP.md](APP.md#tests)).

### Python

```bash
cd crates/mklight-py && source .venv/bin/activate
python -m pytest
```

after the package is built as above.

### Before a commit

```bash
cargo fmt --check
cargo clippy --workspace --all-targets
```

Both are clean on `master`.

## Measurements and qualification

The harnesses that measure the engine and qualify the Metal backend against
the CPU reference are ignored tests and example programs of `mklight-bench`.
[Its README](../crates/mklight-bench/README.md) lists every command and the
environment variables each one reads; the variables the engine itself reads
are in [GPU.md](GPU.md) §10.

## Documentation

- **One owner per fact.** Each topic is complete in one document and the
  others link to it: routing, route options and environment variables in
  GPU.md; each analysis family's division of work between the CPU and the
  GPU in its solver document; units, signs and the precision budget in
  CONVENTIONS.md; the C ABI in FFI.md; the app in APP.md.
- **RESULTS.md §8 is generated** from the analysis schemas, and a test
  fails when the file is out of date. After changing a schema, rewrite it
  with

  ```bash
  MKLIGHT_UPDATE_RESULTS_DOC=1 cargo test -p mklight-engine --test tests results_doc::
  ```

- **Section numbers are cited** from other documents and from code comments
  (`GPU.md §9.2`, `CONVENTIONS §7.3`), so a section keeps its number and
  title when its text changes.
