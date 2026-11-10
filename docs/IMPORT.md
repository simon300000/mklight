# Importing lens prescription files

MKLight reads two common lens prescription formats into its own model
(`mklight_core::import`):

* **`.zmx` files**: one record per line, a four-letter keyword and its
  fields; a surface block starts with `SURF n`.
* **`.seq` files**: a command script; commands are separated by `;` or line
  ends, `!` starts a comment.

Both are read in UTF-8, UTF-16 LE or BE (with or without a byte-order mark)
or Latin-1; the encoding is detected as for AGF catalogs
(`agf::decode_bytes`). Lengths are converted from the file's unit to mm,
curvatures and sag coefficients accordingly.

An import **never fails on content it cannot map**. Solves, pickups,
macros, multi-configuration and tolerance data, non-sequential data,
unknown surface types and unknown glasses are dropped or replaced, and each
one is reported as a structured warning. Only text that is not a
prescription at all (no surfaces) is an error. MKLight does not write
either format.

## Entry points

| where | call |
|---|---|
| Rust | `import_prescription(bytes, format, &library)`, `read_prescription_file(path, &library)` → `PrescriptionImport { format, encoding, system, warnings }` |
| engine | `Session::import_prescription(path)`: imports with the session's catalogs and replaces its system |
| C ABI (1.3) | `mk_import_prescription(bytes, len, "zmx"|"seq", &json)` (built-in catalog, no session); `mk_session_import_prescription(session, path, &json)` |
| CLI | `mklight import <file> <project> [--catalog agf]… [--force]` |
| Python | `Session.import_prescription(path)` |
| app | Open… (and the Welcome window's Open…), drag and drop, Finder's Open With; see `APP.md` |

## Warnings

Each warning is `{"kind", "line", "surface", "message"}`: `line` is the
1-based line in the file, `surface` the index in the imported system (both
may be `null`). Records that repeat (every `MOFF` line, every variable flag)
are reported once with a count. Warnings are in file order; those about the
whole system come last.

| kind | meaning |
|---|---|
| `unsupported_record` | a record or command that is not mapped; dropped |
| `unsupported_surface` | a surface type, or terms of a type, MKLight cannot hold; the base sphere or conic is kept |
| `unknown_glass` | a glass in none of the loaded catalogs |
| `solve_dropped` | a solve, pickup or optimisation-variable flag; the current value is kept |
| `non_sequential` | non-sequential data; dropped (a non-sequential component becomes a plane) |
| `approximated` | a setting MKLight expresses differently (ray aiming, theodolite fields, an image-space NA) |
| `malformed` | a line whose fields cannot be read; ignored |
| `defaulted` | something the file does not state (no stop, no aperture, no wavelengths, no image surface); a default is used |
| `validation` | the imported system has a structural problem (CONVENTIONS §13) |

## Glasses

Glasses resolve against the loaded catalogs (built-in plus user catalogs),
case-insensitively: first the catalog the file names for the glass
(`.seq`: `N-BK7_SCHOTT`), then the catalogs the file prefers (`.zmx`:
`GCAT`), then every catalog. A found glass is stored with the catalog it
was found in. A glass that is not found becomes a **model glass** when the
file gives its `n_d`/`V_d` (`.zmx` `GLAS` records always do), with an
`unknown_glass` warning; otherwise it stays a catalog reference by name,
which resolves once a catalog containing it is loaded. Model glasses
(`.zmx` `___BLANK`, `.seq` `1.5168:64.17` or a six-digit glass code) map to
`model` materials. `MIRROR` and `REFL` make a reflecting surface.

To resolve vendor glasses, load the catalogs first (CLI `--catalog`, Python
`load_catalog`, the app's Load Glass Catalog…, which then imports the file
again).

## `.zmx` mapping

| record | MKLight |
|---|---|
| `NAME`, `NOTE` | `name`, `notes` |
| `UNIT` (MM, CM, IN, M) | length scale to mm |
| `ENPD`, `FNUM`, `OBNA`, `FLOA` | entrance pupil diameter, image-space F/#, object-space NA, float by stop size (the stop's `DIAM` becomes its fixed semi-diameter) |
| `FTYP` | field type (angle, object height, paraxial or real image height) and the numbers of fields and wavelengths (never more than the file lists) |
| `XFLN`, `YFLN`, `FWGN`/`FWGT`, `VDXN`, `VDYN`, `VCXN`, `VCYN` | fields, weights, vignetting factors |
| `WAVM` (or `WAVL`/`WAVN` with `WWGT`/`WWGN`), `PWAV` | wavelengths (µm), weights, primary |
| `ENVD` | temperature and pressure |
| `GCAT` | preferred catalogs |
| `SURF`, `TYPE`, `CURV`, `CONI`, `DISZ`, `STOP`, `COMM` | surfaces, shape, thickness (`INFINITY` for the object), stop, label |
| `GLAS` | material (see *Glasses*) |
| `PARM`, `XDAT` | shape parameters and extra data, per type below |
| `CLAP`, `SQAP`, `ELAP` | circular or annular, rectangular, elliptical apertures |

`DIAM` sizes are computed by the program that wrote the file; MKLight
computes its own automatic semi-diameters instead, except for the stop
when the aperture floats by the stop size.

| `TYPE` | MKLight shape |
|---|---|
| `STANDARD` | `plane` or `conic` |
| `EVENASPH` | `even_asphere` (`PARM 2…8` on r⁴…r¹⁶); a nonzero r² term (`PARM 1`) gives an `odd_asphere` with only even powers, the same sag |
| `ODDASPHE` | `odd_asphere` (`PARM 1…8` on r¹…r⁸) |
| `COORDBRK` | `coordinate_break`: decentres `PARM 1, 2`, tilts `PARM 3…5` (degrees, right-handed), order `PARM 6` |
| `PARAXIAL` | `paraxial` (focal length `PARM 1`) |
| `TOROIDAL` | `toroid` (radius of rotation `PARM 1`, y⁴… coefficients `PARM 3…8`; the y² term and Zernike terms are dropped) |
| `BICONICX` | `biconic` (x radius `PARM 1`, x conic `PARM 2`) |
| `XPOLYNOM` | `xy_polynomial` (`XDAT 1` terms, `XDAT 2` normalisation radius, coefficients from `XDAT 3`) |
| `SZERNSAG`, `FZERNSAG` | `zernike_sag`, standard (Noll) or fringe ordering, as `XPOLYNOM`; the extra even-asphere terms are dropped |
| `BINARY_2` | `diffractive` on the even-asphere base; order `PARM 0`, phase terms as `XPOLYNOM` (radians), design wavelength the primary |
| `NONSEQCO` | a plane, with a `non_sequential` warning |
| others | the base sphere or conic, with an `unsupported_surface` warning |

Not imported (reported): solves, pickups and other unknown surface records,
coatings, obscurations and aperture decentres, ray aiming, multi-
configuration (`MNUM`, `MOFF`) and tolerance data, non-sequential objects.
Records that only affect drawing or bookkeeping (`HIDE`, `SLAB`, `PFIL`, …)
are ignored silently.

## `.seq` mapping

| command | MKLight |
|---|---|
| `TIT`, `DIM` (M, C, I), `RDM` / `CUM` | name, length unit, radius or curvature mode |
| `EPD`, `FNO`, `NAO` | entrance pupil diameter, image-space F/#, object-space NA; `NA` (image space) becomes the paraxial working F/# `1/(2 NA)` (approximated) |
| `WL` (nm), `WTW`, `REF` | wavelengths, weights, primary (default: the middle one) |
| `XAN`/`YAN`, `XOB`/`YOB`, `XIM`/`YIM`, `XRI`/`YRI`, `WTF` | fields (angle, object height, paraxial or real image height), weights |
| `VUX`, `VLX`, `VUY`, `VLY` | vignetting, as decentre `(VL − VU)/2` and compression `(VL + VU)/2` |
| `SO`, `S`, `SI` | object, lens and image surfaces: radius (or curvature), thickness, glass |
| `STO`, `RDY`, `CUY`, `THI`, `GLA`, `K`, `A`…`J`, `REFL`, `RMD` | stop, radius, curvature, thickness, glass, conic, asphere coefficients (r⁴…r²⁰), reflection; on the latest surface or the one a qualifier names (`S3`, `SO`, `SI`) |
| `CIR` | the stop's size when the aperture floats by it; other clear apertures are computed |
| `REX`/`REY`, `ELX`/`ELY` | rectangular and elliptical apertures |
| `XDE`, `YDE`, `DAR` | x/y decentres, returned after the surface with `DAR` |
| `PIM` | a marginal-ray-height solve (height 0) on the last thickness |

Not imported (reported): tilts and axial decentres (`ADE`, `BDE`, `CDE`,
`ZDE`, `BEN`, `REV`, `RET`), special surfaces (`SPS`; the base conic is
kept), solves and pickups other than `PIM`, variable flags (`CCY`, `THC`,
…), private glass catalogs (`PRV` … `END`), holes and obscurations, and any
other command.

## Tests

`crates/mklight-core/tests/fixtures/prescriptions` holds hand-written
fixtures in both formats: the singlet, the Cooke triplet (the `.zmx` file in
UTF-16 LE with a byte-order mark) and the Petzval lens with an aspheric
surface, matching the native samples, plus `awkward.zmx` with content that
does not map. `mklight-core/tests/prescription_import.rs` checks the
mapping, every encoding and the warnings;
`mklight-trace/tests/prescription_round_trip.rs` checks that each fixture's
first-order properties (EFL, BFL, image distance, pupils, F/#, track, image
height) equal the native sample's to 1e-9.
