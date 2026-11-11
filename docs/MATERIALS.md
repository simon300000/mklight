# MKLight Materials

How MKLight represents optical materials, which dispersion formulas it
supports, and how to load glass catalogs. Code: `crates/mklight-core/src/materials/`.

## Index conventions

* Wavelengths are in **µm**. Glass data are indices **relative to air** at the
  catalog reference conditions: the glass's `T_ref` (from its `TD` record;
  `CATALOG_T_C` = 20 °C when it has none, which is also what nearly every
  catalog uses) and **1 atm** (`CATALOG_P_ATM`). This is the convention glass
  catalogs use, and it matches `docs/CONVENTIONS.md` §2.
* Air is `n = 1` at the system temperature and pressure by definition.
  Vacuum is `1/n_air(λ, T, P)`. `air_index` is the Kohlrausch/Edlén formula at
  15 °C with ideal-gas T/P scaling.
* **Thermal model.** For a system at temperature `T` and pressure `P`,
  `ResolvedMaterial::index` does the following:
  1. `n_rel,ref = n_catalog(λ)` (relative to air at `T_ref`, 1 atm);
  2. `n_abs,ref = n_rel,ref · n_air(λ, T_ref, 1 atm)`;
  3. `Δn_abs = (n_abs,ref² − 1)/(2 n_abs,ref) · (D0 ΔT + D1 ΔT² + D2 ΔT³ + (E0 ΔT + E1 ΔT²)/(λ² − λ_tk²))`,
     with `ΔT = T − T_ref`;
  4. `n = (n_abs,ref + Δn_abs) / n_air(λ, T, P)`.

  At `T_ref` and 1 atm this returns the catalog index unchanged. A glass
  without `TD` data (`T_ref` = 20 °C, `Δn_abs` = 0) still gets the air-index
  correction for `T` and `P`.
* Internal absorption uses the catalog internal transmittance `τ` of a sample
  of thickness `d`. The absorption coefficient `α = −ln τ / d` is interpolated
  linearly in λ, and a path of length `L` transmits `exp(−α L)` (Beer–Lambert).
* Thermal expansion (`Glass::cte`, 1/K) is the catalog TCE for −30…+70 °C.
  When a catalog sets the "ignore thermal expansion" flag, `cte` is `None` and
  the raw value is kept in the AGF metadata.

## Catalogue index and Abbe offsets

A catalogue reference may carry a melt offset; both components default to
zero, and a zero offset is omitted from JSON:

```json
{"type":"catalog","catalog":"BUILTIN","name":"N-BK7",
 "offset":{"dn_d":0.002,"dv_d":-3.0}}
```

With `n(λ)` the glass's resolved index at the system temperature and
pressure (thermal and air corrections above), and `n_d`, `V_d` computed from
it at d, F and C:

```
s     = (n_d + Δn_d − 1)/(n_d − 1) · V_d/(V_d + ΔV_d)
n'(λ) = n_d + Δn_d + s · (n(λ) − n_d)
```

so `n'_d = n_d + Δn_d` and `V'_d = V_d + ΔV_d` exactly (to rounding). The
dispersion is scaled about d, which keeps its shape and every relative
partial dispersion such as `P_g,F`; `ΔP_g,F` against the normal line moves
with `V_d`. The anchors are recomputed at each environment, so the offset is
the same change to the nominal glass at any temperature; catalogue thermal
and absorption data are kept. A zero offset resolves to the nominal glass
bit for bit. Resolution refuses an offset unless `n_d > 1` and `V_d > 0`
before and after it.

Validation (CONVENTIONS §13) reports `non_finite` and, for
`|Δn_d| > 0.05` or `|ΔV_d| > 10`, `invalid_material` at
`….material.offset.dn_d` / `.dv_d`. Tracing (CPU, GPU, paraxial) sees the
offset through the resolved indices; catalogue files are never modified.
A model glass (`{"type": "model", "nd", "vd", "dpgf"}`, on a surface or in
a coating layer) needs `n_d > 1` and `V_d > 0`, the condition of its Cauchy
fit: validation reports `invalid_material` at `….material.nd` / `.vd`.

* `glass_map` keeps the nominal point and adds a `kind: "offset"` point per
  offset surface (at the glass's catalogue conditions); only that point
  lists the surface.
* `glass_info` takes the same `offset`; computed indices, `nd`, `vd`,
  `pgf` and the curve include it, while `catalog_nd`, `catalog_vd` and
  `dispersion` stay nominal.
* `glass_substitution` measures windows and distances from the offset
  position and installs the replacement **without** an offset, even when it
  is the same glass: a melt offset belongs to the original melt.
  Configuration overrides and pickups still apply. Continuous material
  variables on a substituted surface are refused.
* `ParamPath` `material_dn_d` / `material_dv_d` (`{"param":"material_dn_d",
  "surface":1}`) address the offset for variables, overrides, pickups and
  tolerances; a tolerance adds to the saved offset. `model_nd` /
  `model_vd` still address model glasses.

## Dispersion formulas

`Dispersion` variants (`dispersion.rs`), with the standard AGF formula numbers.
`λ` is in µm, and `C`-type Sellmeier constants are in µm².

| # | name | formula | variant |
|---|---|---|---|
| 1 | Schott | `n² = a0 + a1λ² + a2λ⁻² + a3λ⁻⁴ + a4λ⁻⁶ + a5λ⁻⁸` | `Schott` |
| 2 | Sellmeier 1 | `n² − 1 = Σ₁³ Kᵢλ²/(λ² − Lᵢ)`, CD order `K1 L1 K2 L2 K3 L3` | `Sellmeier` (3 terms) |
| 3 | Herzberger | `n = A + B·L + C·L² + Dλ² + Eλ⁴ + Fλ⁶`, `L = 1/(λ² − 0.028)` | `Herzberger` |
| 4 | Sellmeier 2 | `n² − 1 = A + B1λ²/(λ² − λ1²) + B2/(λ² − λ2²)` (no `λ²` in the second numerator), CD order `A B1 λ1 B2 λ2` ⚠ | `Sellmeier2` |
| 5 | Conrady | `n = n0 + A/λ + B/λ^3.5` | `Conrady` |
| 6 | Sellmeier 3 | as 2 with 4 terms | `Sellmeier` (4 terms) |
| 7 | Handbook of Optics 1 | `n² = A + B/(λ² − C) − Dλ²` | `Handbook1` |
| 8 | Handbook of Optics 2 | `n² = A + Bλ²/(λ² − C) − Dλ²` | `Handbook2` |
| 9 | Sellmeier 4 | `n² = A + Bλ²/(λ² − C) + Dλ²/(λ² − E)` (no implicit 1) | `Sellmeier4` |
| 10 | Extended | `n² = a0 + a1λ² + a2λ⁻² + … + a7λ⁻¹²` | `Extended` |
| 11 | Sellmeier 5 | as 2 with 5 terms | `Sellmeier` (5 terms) |
| 12 | Extended 2 | `n² = a0 + a1λ² + a2λ⁻² + a3λ⁻⁴ + a4λ⁻⁶ + a5λ⁻⁸ + a6λ⁴ + a7λ⁶` ✓ (`a7`: ⚠) | `Extended2` |
| 13 | Extended 3 | `n² = a0 + a1λ² + a2λ⁴ + a3λ⁻² + a4λ⁻⁴ + a5λ⁻⁶ + a6λ⁻⁸ + a7λ⁻¹⁰ + a8λ⁻¹²` ✓ | `Extended3` |
| – | Cauchy | `n = A + B/λ² + C/λ⁴` (model glasses), exported exactly as formula 1 | `Cauchy` |
| – | Tabulated | monotone cubic interpolation of `(λ, n)` pairs; no AGF form | `Tabulated` |

✓ Verified against manufacturer data (below). ⚠ Not checked against a
primary specification, and flagged in the source:

* **Formula 4**: the second term `B2/(λ² − λ2²)` has no `λ²` in its
  numerator. This agrees with most independent open-source AGF readers; at
  least one puts `λ²` there too. For `B2 ≠ 0` the two readings differ by
  ~0.1 in index, so a formula-4 glass that gets an `IndexMismatch` warning
  should be inspected. No manufacturer catalog checked uses formula 4 (the
  Nikon/Hikari catalog below has formulas 6, 12 and 13 only), so there is
  no published glass to settle it.
* **Formula 12, coefficient `a7`** (`λ⁶`): every formula-12 glass of the
  catalog below has `a7 = 0`, so its position rests on secondary
  documentation. `a0 … a6` are verified.

**Open requirement**: one real glass per formula with a known answer.
Formula 4 and formula 12's `a7` still lack one (no published glass with its
catalog indices is known); until one is found, both stay flagged here and
in `dispersion.rs`.

**Verification of formulas 12 and 13** against the manufacturer's own data.
Source: Nikon Corporation, glass data in AGF format
(`NIKON-HIKARI_201911.AGF`, updated November 2019), the AGF download on the
Nikon optical-components downloads page
(nikon.com/business/components/downloads).
Each glass's `NM` record carries the manufacturer's catalog `n_d` and `V_d`;
the coefficients were evaluated at the d, F and C lines
(0.5875618 / 0.4861327 / 0.6562725 µm):

| formula | glasses | max \|Δn_d\| | max \|ΔV_d\| | alternative orders |
|---|---|---|---|---|
| 12 | 21 | 4.9·10⁻⁷ | 4.9·10⁻⁷ | `a6`, `a7` on `λ⁶`, `λ⁴` swapped: Δn_d up to 1.1·10⁻⁴; `a6` on `λ⁻¹⁰`: 8.6·10⁻³ |
| 13 | 164 (60 with `a7 ≠ 0`, 32 with `a8 ≠ 0`) | 4.1·10⁻⁸ | 5.0·10⁻⁷ | `λ⁴` term last or read as formula 12: Δn_d ≥ 0.086 |

The test `agf::tests::manufacturer_catalog_verifies_formulas_12_and_13`
holds four of these glasses record for record (J-FK5 and J-SFH1, formula 13;
E-LAKH1 and E-KZFH1, formula 12), checks n_d to 10⁻⁶ and V_d to 2·10⁻⁶, and
checks that the permuted orders fail. Test values, from the catalog:

| glass | formula | catalog n_d | catalog V_d |
|---|---|---|---|
| J-FK5 | 13 | 1.487490 | 70.318793 |
| J-SFH1 | 13 | 1.808090 | 22.738831 |
| E-LAKH1 | 12 | 1.748099 | 52.304982 |
| E-KZFH1 | 12 | 1.612660 | 44.461379 |

The same catalog wraps its `CD` and `TD` records over several lines
(continuation lines, "Tolerance and policy" below); all 379 glasses of the
file reproduce their catalog n_d and V_d.

In the tests, formulas 1 and 2 reproduce published glass data (N-BK7, F2,
BK7) to the catalog n_d/V_d, and formulas 12 and 13 manufacturer data as
above. Every formula is checked against hand-evaluated
values at λ ≠ 1 µm with distinct coefficients (so a permuted coefficient
mapping fails), and the resonance forms (4, 6–9, 11) are also checked against
exact algebraic identities with the plain Sellmeier sum.

`n²`-type formulas are evaluated as a compensated (Neumaier) sum of their
terms and then square-rooted. A negative `n²` gives NaN rather than a wrong
number.

## Loading catalogs

MKLight ships only a small built-in catalog (`BUILTIN`: N-BK7, F2, N-SF6,
fused silica, CaF₂ and a few others, with published Sellmeier data). Vendor
catalogs are user-supplied AGF files:

```rust
use mklight_core::materials::{GlassCatalog, MaterialLibrary, agf};
use std::sync::Arc;

// Simple: glasses only, default policy.
let cat = GlassCatalog::from_agf_file("catalogs/VENDOR.agf")?;

// Detailed: warnings, AGF metadata (MIL code, status, TCE 100/300, ΔPgF,
// resistance codes, unknown records) and the detected encoding.
let detailed = agf::read_agf_file("catalogs/VENDOR.agf", &agf::AgfOptions::default())?;
for w in &detailed.warnings {
    eprintln!("{w}");
}

let mut lib = MaterialLibrary::with_builtin();
lib.add_catalog(Arc::new(cat));
```

`GlassCatalog::from_agf_file` and `MaterialLibrary` retain availability,
exclude-substitution and climate/stain resistance in `Glass::substitution`.
`Glass::relative_cost` and `Glass::transmittance` retain OD cost and IT
samples. `glass_map` reports all these material fields without separate
metadata, and the `glass_substitution` job filters on the same values (its
candidate settings and distance metric are in `OPTIMIZATION.md`). The
complete AGF metadata (including ΔP_g,F and other resistance codes) stays in
`AgfCatalog::meta`.

The catalog is named after the file stem. Glass lookups are case-insensitive,
and a `MaterialRef::Catalog` with `catalog: None` searches the catalogs in
load order.

**Encodings.** The reader accepts UTF-8 (with or without BOM), Latin-1 and
UTF-16 LE/BE (with or without BOM; many vendor files are UTF-16 LE), with
CRLF, LF or CR line endings.

**Tolerance and policy.** Unknown records are kept verbatim and written back
on export. `-` means a missing value, and missing trailing fields are allowed.
A line that starts with a number (or `-`) after an `ED`, `CD`, `TD` or `OD`
record continues that record (wrapped coefficient lists).
A glass is **skipped** with a warning when its dispersion cannot be evaluated
(no `CD`, formula outside 1–13, or a non-numeric coefficient). A glass whose
coefficients do not reproduce its catalog `n_d` within 1·10⁻⁴ or `V_d` within
0.2 is **kept** by default with an `IndexMismatch` warning, because the
coefficients are the authoritative data. Set `AgfOptions::mismatch =
MismatchPolicy::Skip` to drop such glasses instead. The check runs only when
the d line lies in the glass's `LD` range and the catalog lists `n_d, V_d > 0`.

## Exporting glasses

`agf::write_agf` (or `write_agf_file`, with a choice of encoding) writes any
catalog, including edited glasses and model glasses (`agf::model_glass(name,
nd, vd, dpgf)`, the same Cauchy fit `MaterialRef::Model` uses). Numbers are
written in their shortest exactly-round-tripping form, so a write → read cycle
reproduces every glass bit for bit. Anything the writer cannot express exactly
is reported with a `NotExportable` warning:

* tabulated dispersions, Sellmeier sums with more than five terms and
  non-finite coefficients: the glass is left out;
* whitespace in a name becomes `_`, and a name collision gets a `_2`, `_3`, …
  suffix;
* unknown `ED` values are written as `-`, so a missing CTE never reads back
  as 0; unknown `OD` values as −1; non-finite values as unknown;
* free text is split at every line break, and malformed MIL codes or kept
  unknown records are replaced or dropped, so no field can inject records;
* an edited `Glass::cte` clears a stale "ignore thermal expansion" flag;
* an unrestricted wavelength range (a glass read without `LD`) is written
  without `LD`.
