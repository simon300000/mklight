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
| 1 | `usage` | bad command line; unknown analysis, sample or catalog; a parameter path that does not address a parameter of the system; invalid settings JSON or values; field/wavelength index out of range; a pupil point outside the unit disk; file exists (`new` without `--force`) |
| 2 | `invalid_model` | the project file cannot be parsed (including content that is not UTF-8) or migrated (including a newer `schema_version`); `validate` finds errors; materials or coatings do not resolve (also a glass named in analysis settings); `set` with a value the parameter cannot take, or an edit that would make the system invalid |
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
`petzval_asphere`, `relay`, `hybrid_doe`, `anamorphic`, `biconic`, `toroid`,
`q_con`, `q_bfs`, `odd_asphere`, `xy_freeform`, `zernike_sag`) or, without
`--sample`, the minimal empty system.
Saved atomically through the session (metadata stamped). Refuses to replace
an existing file unless `--force`.

### `mklight validate <project>`

Opens the project (migrating older versions in memory) and runs the engine's
`validate` analysis, followed by checks that make `validate` agree with the
ray-based commands (a project reported valid can be traced):

| code | severity | check |
|---|---|---|
| `paraxial` | error | the system resolves, but the reference tracer cannot set it up (no finite marginal ray for the aperture specification, an afocal system with an F/# aperture, …); `paraxial`, `trace` and the ray-based analyses would fail with exit 2 |
| `chief_ray` | warning | a chief field cannot be prepared or its ray (primary wavelength, unclipped) does not reach the image |
| `marginal_ray` | warning | the axial marginal ray (field 0, pupil `(0, 1)`, clipped) does not reach the image |

Real-ray failures are warnings: a system in the middle of a redesign, or one
vignetted on purpose, is still a valid model. When the system does not
resolve (unknown glass) the engine's issue reports the cause and the tracer
checks are skipped.

**Structurally invalid files.** A file that parses but fails the engine's
structural validation (`OpticalSystem::validation_issues`, codes in
CONVENTIONS §13: an F/# of 0, a negative weight, a NaN, a reference to a
missing surface, …) cannot be opened as a session, so `paraxial`, `trace`
and the analyses refuse it (exit 2). `validate` then lists *every*
structural issue as an error with its code (`invalid_aperture_value`,
`negative_weight`, …), its JSON `path` (`aperture.value`,
`fields.fields[1].weight`) and the surface index when the path is in
`surfaces`; `structural_only` is true and the other checks, which need a
valid model, do not run. Fix the issues with `mklight set` (below) or by
hand. A file that does not parse (unknown keys, wrong types, not a
project) keeps its load error.

JSON: `ok`, `structural_only`, `errors`, `warnings`, `issues` (severity,
code, surface, message, and `path` for structural issues; engine issues
first), `open_warnings`, `migration`, `snapshot_hash`, `library_hash`
(`null` when structurally invalid). Exit 2 when `ok` is false (the report
is still printed).

### `mklight show <project> [--no-semi-diameters]`

System settings (aperture, stop, fields, wavelengths with the primary marked,
ray aiming, environment) and the lens data table: index, label, type,
radius (`inf` for a plane), thickness, material, `n_d` after the surface,
semi-diameter, semi-diameter mode, conic, and notes (stop, mirror, ideal-lens
focal length, asphere coefficients, decenter/tilt, coating).

The type column uses the model surface type names. After the lens table, each
non-legacy surface lists its coefficients with mathematical labels (such as
`A4 r^4`, `C(x^2 y)`, `Z7`, or `Q2`), normalisation radius where applicable,
and diffractive design order, design wavelength and trace order. `--json`
continues to print the model report with each full surface shape in `surfaces`.

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
| `--backend cpu\|metal\|auto` | `auto` (default): the engine's `auto`, i.e. the first available GPU backend that implements the analysis and passes precision qualification for the system, else the CPU reference; ray grids below 65 536 rays stay on the CPU reference, docs/RESULTS.md §5 (the envelope's `backend` object records which and why). `metal`: explicit; exit 3 if the Metal backend is unavailable or does not implement the analysis (no silent fallback). `cpu`: the f64 reference |
| `--out <file>` | write the result envelope to a file (atomically); stdout gets `{"out", "run_info"}` |
| `--buffers` | inline each buffer's contents as a `data` array in the envelope |
| `--progress` | progress on stderr while the job runs (`progress  42.0%  stage`, or one JSON object per line with `--json`) |
| `--timeout <s>` | cancel the job after `s` seconds (exit 130) |

The JSON output is the engine's result envelope:
`{run_info, library_hash, result, buffers: [{index, name, dtype, shape}]}`.
The human output shows the run info, the result JSON and the buffer list.

**Stop and cancellation.** For `optimize` and `tolerance`, the first Ctrl-C
(SIGINT, or SIGTERM) requests Stop through the engine: the job finishes with
its best-so-far result, `result.stopped: true`, and normal output (exit 0).
A second signal cancels and discards the result (exit 130). Analyses without
Stop support cancel on the first signal. `--timeout` always cancels. Cancelled
runs report partial status on stderr and, with `--json`, on stdout:
`{"cancelled": true, "reason": "interrupt"|"timeout", "analysis", "status": …}`.

### `mklight trace <project> [--field i] [--wavelength j] [--pupil px,py] [--surfaces] [--no-clip]`

Traces one real ray with the reference tracer: system field `i` (default 0,
with its vignetting factors), system wavelength `j` (default: the primary),
normalised pupil point `(px, py)` on the unit disk (default `0,0`; exit 1
for `|p| > 1`; with ray aiming off on the
paraxial entrance pupil, with real aiming on the stop). Fixed apertures clip
unless `--no-clip`.

JSON: `status` (`ok`, `miss`, `clipped`, `total_internal_reflection`,
`failed`), `stopped_at_surface`, `fail_reason`, `image` (`point`, `dir` in
the image surface's local frame), `opl_mm`, `transmittance` (bulk
absorption only), `chief_image_point`, `transverse_aberration` (ray − chief
on the image surface, CONVENTIONS §4) and, with `--surfaces`, per-surface
records: `type` (surface shape), `is_coordinate_break`, `point`, `dir`
(after the interaction) and `normal` in the surface's local frame (surface
0: the starting point in its frame), `aoi_deg`, cumulative `opl_mm` and
`transmittance`, and `global_point`/`global_dir`. Surface 0 and coordinate
breaks have no intersection: their `normal` and `aoi_deg` are `null` (`-`
in the table), and a coordinate break's `point` is the incoming ray, at its
previous point, re-expressed in the break's frame.

### `mklight paraxial <project> [--rays]`

First-order properties at every system wavelength (columns; `*` marks the
primary): EFL, BFL, FFL, paraxial image distance, entrance/exit pupil
diameter and position, image-space and working F/#, object/image NA,
magnification, Lagrange invariant, total track, stop semi-diameter and
paraxial image height. JSON adds `image_height_per_tan`,
`image_height_per_object_height` and the primary-wavelength marginal and
chief rays (`y`, `u`, `nu` per surface); `--rays` prints those as a table.
Telecentric pupil positions are `"inf"`. Anamorphic systems also print the
x-axis and y-axis first-order tables at the primary wavelength and signed
squeeze (`EFL_y / EFL_x`); JSON includes both axes and squeeze at every
wavelength in `paraxial_xy`.

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

The value is a number or `inf`/`-inf`. A path that does not address a
parameter of this system (no such surface, field or wavelength; a conic on
a plane; the aperture value when the aperture floats by stop size) is a
usage error (1). The edit goes through `Session::apply_edits` (structural
validation, new snapshot); a value the parameter cannot take (a
semi-diameter ≤ 0, a negative wavelength, an infinite gap after the object)
is rejected with exit 2. Then the checks of `mklight validate` run (engine
analysis and the tracer checks above); an edit that introduces an error the
project did not already have is rejected with exit 2 and nothing is
written. So `set` never turns a working project into one that `paraxial` or
`trace` refuse (an F/# of 0, a negative field weight). An edit to a project
that already fails a check is allowed when it adds no new error, so a
broken file can be repaired step by step. That includes a structurally
invalid file (see `validate`): the edit is applied to the model directly
and refused (exit 2, nothing written) only if it adds a structural issue
(a code at a JSON path) the file did not have; the result has `valid`
(structurally valid after the edit) and `remaining_issues` (the remaining
structural issues, or, once valid, the full `validate` issues of the
repaired project, whose `snapshot_hash` is then reported).
`snapshot_hash_before` is `null` for such a file. New warnings (a real ray that no
longer reaches the image, a wavelength outside a glass's data range, a gap
sign opposite to the propagation direction) are reported but do not block
the edit. Otherwise the project is saved atomically (to `--out` if given).
JSON: `param`, `old` (`null` when the parameter had no value, e.g. an
automatic semi-diameter), `new`, `changed`, `saved`, `saved_to`,
`snapshot_hash_before`, `snapshot_hash`, `new_warnings`.

`set` does not judge whether a valid, traceable value is sensible (a
radius of 10⁻⁹ mm on a lens with an F/# aperture traces, because the
pupil shrinks with the focal length).

## Tests

`cargo test -p mklight-cli` runs unit tests (formatting, exit-code mapping,
argument parsing) and integration tests that execute the built binary on
the sample systems: every command, JSON parsed and checked against known
answers (paraxial EFL of every sample, the axial marginal ray of every
sample near the axis at the paraxial focus with its image-space slope
matching the working F/# (sine condition), achromat EFL ≈ 100 mm at F/5,
chief-ray height against the paraxial image height, the Newtonian's folded
image at global (0, −100, −400) for any pupil point, the relay's m = −1 and
5 mm edge field imaged at −5 mm, coordinate breaks reported without AOI, a user AGF glass with N-BK7 coefficients reproducing N-BK7's
first-order data), exit codes (including non-UTF-8 projects for every command), `validate` and
`set` agreeing with `paraxial`/`trace` (F/# 0, negative field weight:
every structural issue listed, step-by-step repair with `set`),
migration backups, atomic `set`, Ctrl-C and
`--timeout` cancellation with monotone progress, and byte-identical
`--no-timing` output.
