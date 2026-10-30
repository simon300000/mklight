# MKLight command line (`mklight-cli`)

`mklight` is the headless front end of the engine. It is a thin client of
`mklight-engine`, the facade the macOS app (through `mklight-ffi`) and the
Python bindings use too: projects are opened as engine `Session`s (with
schema migration and catalog loading), analyses come from the engine's
analysis registry and run on its job scheduler, and every result is the same
envelope with a `RunInfo` that the app receives. `trace` and `paraxial` also
call the CPU f64 reference tracer (`mklight-trace`) directly, on the
session's cached `ResolvedSystem`.

```sh
cargo build -p mklight-cli --release      # → target/release/mklight
mklight new lens.mklight --sample achromat
mklight show lens.mklight
mklight paraxial lens.mklight
mklight trace lens.mklight --field 0 --pupil 0,1 --surfaces
mklight run lens.mklight rng_stream --seed 7 --progress --json
```

Units and signs follow `docs/CONVENTIONS.md`: mm, µm, degrees; field,
wavelength and surface indices are **0-based** (surface 0 is the object).

## Output

* By default, human-readable key/value lines and tables on stdout.
* `--json` (any command) prints one pretty-printed JSON document on stdout.
  `±∞` are the strings `"inf"`/`"-inf"`, NaN is `null` (as in `docs/FFI.md`).
  Object keys are in a fixed (sorted) order.
* `--no-timing` removes the wall-clock fields `started_unix_ms`,
  `duration_ms` and `elapsed_ms` everywhere, and `created`/`modified` inside
  `metadata` objects. With it, two runs on the same inputs print
  **byte-identical** JSON (tested for `run`, `show`, `paraxial`, `trace`,
  `analyses`, `info`). It also applies to files written with `run --out`.
* Errors are printed to stderr as `mklight: error: …`. With `--json`, stdout
  additionally carries `{"error": {"exit_code", "kind", "message"}}`.
* Warnings (missing catalogs, migration notes, glass data ranges, tracer
  warnings) are printed as `warning: …` lines and listed in the JSON.

## Exit codes

| code | kind | when |
|---|---|---|
| 0 | `ok` | success |
| 1 | `usage` | bad command line; unknown analysis, sample, catalog or parameter path; invalid settings JSON or values; field/wavelength index out of range; file exists (`new` without `--force`) |
| 2 | `invalid_model` | the project file cannot be parsed or migrated (including a newer `schema_version`); `validate` finds errors; materials or coatings do not resolve; an edit would make the system invalid |
| 3 | `analysis_failed` | the job failed; the analysis or requested backend is not available; a trace or numerical failure |
| 4 | `io` | a file cannot be read or written (missing project, catalog or settings file) |
| 130 | `cancelled` | the job was cancelled by Ctrl-C (SIGINT/SIGTERM) or `--timeout` |

clap's own parse errors are mapped to 1 (clap would use 2, which here means
"invalid model"); `--help` and `--version` exit with 0.

## Commands

### `mklight info`

Engine version, C ABI version (`major.minor` and the packed
`(major << 16) | minor` of `mk_abi_version()`), project schema version,
platform, default backend and every registered compute backend with its
availability, precision (`f64_reference` or `f32_gpu`), device and
capabilities, and the number of registered/available analyses.

### `mklight new <file> [--sample <name>] [--force]`

Creates a project from a built-in sample (`mklight_core::samples`:
`singlet`, `achromat`, `cooke_triplet`, `double_gauss`, `newtonian`,
`petzval_asphere`, `relay`) or, without `--sample`, the minimal empty system.
Saved atomically through the session (metadata stamped). Refuses to replace
an existing file unless `--force`.

### `mklight validate <project>`

Opens the project (migrating older versions in memory) and runs the engine's
`validate` analysis. JSON: `ok`, `errors`, `warnings`, `issues` (severity,
code, surface, message), `open_warnings`, `migration`, `snapshot_hash`,
`library_hash`. Exit 2 when `ok` is false (the report is still printed).

### `mklight show <project> [--no-semi-diameters]`

System settings (aperture, stop, fields, wavelengths with the primary marked,
ray aiming, environment) and the lens data table: index, label, type,
radius (`inf` for a plane), thickness, material, `n_d` after the surface,
semi-diameter, semi-diameter mode, conic, and notes (stop, mirror, ideal-lens
focal length, asphere coefficients, decenter/tilt, coating).

Semi-diameter modes: `fixed` (user value, clips rays), `aperture` (explicit
aperture, its bounding radius), `stop` (the stop, sized by the aperture
specification), `auto` (traced by `SequentialTracer::semi_diameters`: the
largest ray height over all fields, wavelengths and the pupil rim), `none`
(object at infinity, coordinate break). When the system does not resolve
(for example an unknown glass), the model data are still shown, indices and
traced values are `null`, and a warning says why. `--no-semi-diameters`
skips the ray trace.

### `mklight migrate <project> [--dry-run] [--no-backup]`

Upgrades the file to the current `schema_version` through the engine's
migration chain (`docs/FFI.md`, *Project file*). The original bytes are kept
as `<file>.v<from>.bak` (or `….bak.<k>` if that exists) before the migrated
project is saved atomically. A file that is already current is left alone;
`--dry-run` reports the migration and its notes without writing.

### `mklight catalogs list|show|import`

Glass catalogs are those of the session's material library: the built-in
catalog plus the user catalogs a project references. MKLight never ships
vendor catalogs; users import their own files.

* `catalogs list [--project p]` — name, source (`builtin`/`user`), glass
  count and path.
* `catalogs show <name> [--project p]` — the glasses of one catalog
  (case-insensitive name), sorted by name: `n_d`, `V_d`, wavelength range.
* `catalogs import <file> --project p` — loads an AGF file (any common
  encoding) or a `GlassCatalog` `.json` file through the session, records a
  reference (absolute path, and the path relative to the project folder)
  in the project and saves it. Catalog data are never copied into the
  project. Reader warnings (skipped glasses, `n_d`/`V_d` mismatches) are
  reported. An AGF catalog is named after its file stem.

### `mklight analyses [<name>]`

Lists every analysis in the engine registry with category, availability
(placeholders of later phases are listed as unavailable with the reason),
and its settings with defaults. With a name: the description and a settings
table (type, default, range, unit, description). JSON: the registry's
`AnalysisInfo` objects including `settings_schema` and `default_settings`.

### `mklight run <project> <analysis> [options]`

Runs any registered analysis through the job scheduler, so analyses
registered later by solver crates are available without CLI changes.

| option | meaning |
|---|---|
| `--settings <json>` / `--settings @file` | settings object (partial; missing keys take the schema defaults, unknown keys are rejected) |
| `--seed <n>` | sets `settings.seed`; only for analyses whose schema has a `seed` |
| `--backend cpu\|metal\|auto` | `auto` (default): the first available non-reference backend that implements the analysis, else the CPU reference. `metal`: explicit; exit 3 if the Metal backend is unavailable or does not implement the analysis (no silent fallback). `cpu`: the f64 reference |
| `--out <file>` | write the result envelope to a file (atomically); stdout gets `{"out", "run_info"}` |
| `--buffers` | inline each buffer's contents as a `data` array in the envelope |
| `--progress` | progress on stderr while the job runs (`progress  42.0%  stage`, or one JSON object per line with `--json`) |
| `--timeout <s>` | cancel the job after `s` seconds (exit 130) |

The JSON output is the engine's result envelope:
`{run_info, library_hash, result, buffers: [{index, name, dtype, shape}]}`.
The human output shows the run info, the result JSON and the buffer list.

**Cancellation.** The first Ctrl-C (SIGINT, or SIGTERM) cancels the job
through the engine; the analysis stops at its next cancellation check and
`mklight` reports the partial status: on stderr
`… cancelled (interrupt) at 42.0% during 'drawing' after 0.81 s`, and with
`--json` on stdout `{"cancelled": true, "reason": "interrupt"|"timeout",
"analysis", "status": {id, state, progress, stage, …}}`. Exit code 130. A
second Ctrl-C terminates immediately (also 130).

### `mklight trace <project> [--field i] [--wavelength j] [--pupil px,py] [--surfaces] [--no-clip]`

Traces one real ray with the reference tracer: system field `i` (default 0,
with its vignetting factors), system wavelength `j` (default: the primary),
normalised pupil point `(px, py)` (default `0,0`; with ray aiming off on the
paraxial entrance pupil, with real aiming on the stop). Fixed apertures clip
unless `--no-clip`.

JSON: `status` (`ok`, `miss`, `clipped`, `total_internal_reflection`,
`failed`), `stopped_at_surface`, `fail_reason`, `image` (`point`, `dir` in
the image surface's local frame), `opl_mm`, `transmittance` (bulk
absorption only), `chief_image_point`, `transverse_aberration` (ray − chief
on the image surface, CONVENTIONS §4) and, with `--surfaces`, per-surface
records: `point`, `dir` (after the interaction) and `normal` in the
surface's local frame (surface 0: the starting frame), `aoi_deg`,
cumulative `opl_mm` and `transmittance`, and `global_point`/`global_dir`.

### `mklight paraxial <project> [--rays]`

First-order properties at every system wavelength (columns; `*` marks the
primary): EFL, BFL, FFL, paraxial image distance, entrance/exit pupil
diameter and position, image-space and working F/#, object/image NA,
magnification, Lagrange invariant, total track, stop semi-diameter and
paraxial image height. JSON adds `image_height_per_tan`,
`image_height_per_object_height` and the primary-wavelength marginal and
chief rays (`y`, `u`, `nu` per surface); `--rays` prints those as a table.
Telecentric pupil positions are `"inf"`.

### `mklight set <project> <param-path-json> <value> [--dry-run] [--out file]`

Edits one parameter through a `ParamPath` (the JSON of
`mklight_core::param::ParamPath`, listed in `docs/FFI.md` under *Optical
system and parameter edits*; `surface` is the stable `SurfaceId`, not the
index):

```sh
mklight set lens.mklight '{"param":"thickness","surface":3}' 90
mklight set lens.mklight '{"param":"semi_diameter","surface":2}' 9.5
mklight set lens.mklight '{"param":"field_y","index":2}' 2.5
```

The value is a number or `inf`/`-inf`. The edit goes through
`Session::apply_edits` (structural validation, new snapshot), then the
`validate` analysis runs; an edit that introduces a validation error the
project did not already have is rejected with exit 2 and nothing is
written. Otherwise the project is saved atomically (to `--out` if given).
JSON: `param`, `old` (`null` when the parameter had no value, e.g. an
automatic semi-diameter), `new`, `changed`, `saved`, `saved_to`,
`snapshot_hash_before`, `snapshot_hash`.

## Tests

`cargo test -p mklight-cli` runs unit tests (formatting, exit-code mapping,
argument parsing) and integration tests that execute the built binary on
the sample systems: every command, JSON parsed and checked against known
answers (achromat EFL ≈ 100 mm at F/5, the axial marginal ray within 10 µm
of the axis at the paraxial focus, chief-ray height against the paraxial
image height, a user AGF glass with N-BK7 coefficients reproducing N-BK7's
first-order data), exit codes, migration backups, atomic `set`, Ctrl-C and
`--timeout` cancellation with monotone progress, and byte-identical
`--no-timing` output.
