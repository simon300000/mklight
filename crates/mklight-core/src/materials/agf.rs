//! Reader and writer for AGF-format glass catalog text files (the de-facto
//! interchange format glass manufacturers publish). Users supply the catalog
//! files, and MKLight does not ship vendor catalogs.
//!
//! An AGF file is a list of whitespace-separated records, each starting with a
//! two-letter tag. A glass starts at its `NM` record and owns every record up
//! to the next `NM`:
//!
//! | tag | fields |
//! |---|---|
//! | `CC` | catalog comment (free text) |
//! | `NM` | name, formula number, MIL code, n_d, V_d, exclude-substitution, status, melt frequency |
//! | `GC` | glass comment (free text) |
//! | `ED` | TCE −30…+70 °C (10⁻⁶/K), TCE +100…+300 °C (10⁻⁶/K), density (g/cm³), ΔP_g,F, ignore-thermal-expansion flag |
//! | `CD` | dispersion coefficients, in the order of the formula (see `docs/MATERIALS.md`) |
//! | `TD` | D0 D1 D2 E0 E1 λ_tk (µm) T_ref (°C) |
//! | `OD` | relative cost, climate CR, stain FR, acid SR, alkali AR, phosphate PR |
//! | `LD` | min and max wavelength (µm) of the dispersion data |
//! | `IT` | λ (µm), internal transmittance, sample thickness (mm) |
//!
//! The reader is tolerant: unknown records are kept verbatim (and written back
//! out), `-` stands for a missing value, trailing fields may be absent, and
//! the text may be UTF-8, Latin-1 or UTF-16 (LE/BE, with or without a BOM)
//! with any line-ending convention. Problems are reported as [`AgfWarning`]s
//! rather than errors. The parse fails only when no glass is found at all.
//!
//! **Skip-or-keep policy.** A glass is *skipped* (with a warning) when its
//! dispersion cannot be evaluated: no `CD` record, an unknown formula number,
//! or a non-numeric coefficient. A glass whose coefficients do not reproduce
//! its catalog `n_d` within `nd_tolerance` or `V_d` within `vd_tolerance` is
//! *kept* by default, with an [`AgfWarningKind::IndexMismatch`] warning, because
//! the coefficients are the authoritative data and `n_d`/`V_d` are only
//! display values. Set [`AgfOptions::mismatch`] to [`MismatchPolicy::Skip`] to
//! drop such glasses instead. The check runs only when the d line lies inside
//! the glass's `LD` range and the catalog gives `n_d > 0` and `V_d > 0`
//! (IR materials often list zeros).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{MkError, MkResult};
use crate::materials::{
    Dispersion, Glass, GlassCatalog, ThermalCoefficients, TransmittancePoint, lines,
    model_glass_cauchy,
};

/// Default tolerance on `|n_d(coefficients) − n_d(catalog)|`.
pub const DEFAULT_ND_TOLERANCE: f64 = 1e-4;
/// Default tolerance on `|V_d(coefficients) − V_d(catalog)|`. Catalogs round
/// `V_d` to two decimals, and `V_d` amplifies index errors by about `V_d/(n_d−1)·10`.
pub const DEFAULT_VD_TOLERANCE: f64 = 0.2;

/// Number of `CD` coefficients the writer emits (the conventional width).
const CD_WIDTH: usize = 10;

// ---------------------------------------------------------------------------
// Text encoding
// ---------------------------------------------------------------------------

/// Text encoding of a catalog file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextEncoding {
    Utf8,
    /// UTF-8 with a byte-order mark.
    Utf8Bom,
    /// UTF-16 little-endian. Written with a BOM.
    Utf16Le,
    /// UTF-16 big-endian. Written with a BOM.
    Utf16Be,
    /// ISO-8859-1. Used when the bytes are not valid UTF-8.
    Latin1,
}

/// Decodes catalog bytes, detecting the encoding:
///
/// 1. a byte-order mark (UTF-8, UTF-16 LE or BE) wins;
/// 2. otherwise, if NUL bytes dominate the odd (or even) byte positions of the
///    first 1 KiB, the text is UTF-16 LE (or BE). AGF text is almost entirely
///    ASCII, so the high byte of each UTF-16 unit is zero;
/// 3. otherwise UTF-8 if the bytes are valid UTF-8, else Latin-1.
pub fn decode_bytes(bytes: &[u8]) -> (String, TextEncoding) {
    match bytes {
        [0xEF, 0xBB, 0xBF, rest @ ..] => return (latin1_fallback(rest).0, TextEncoding::Utf8Bom),
        [0xFF, 0xFE, rest @ ..] => return (decode_utf16(rest, false), TextEncoding::Utf16Le),
        [0xFE, 0xFF, rest @ ..] => return (decode_utf16(rest, true), TextEncoding::Utf16Be),
        _ => {}
    }
    let sample = &bytes[..bytes.len().min(1024) & !1];
    let pairs = sample.len() / 2;
    if pairs > 0 {
        let zeros_even = sample.iter().step_by(2).filter(|&&b| b == 0).count();
        let zeros_odd = sample
            .iter()
            .skip(1)
            .step_by(2)
            .filter(|&&b| b == 0)
            .count();
        // More than 30 % NULs in one byte lane, and few in the other.
        if 10 * zeros_odd > 3 * pairs && 4 * zeros_even < zeros_odd {
            return (decode_utf16(bytes, false), TextEncoding::Utf16Le);
        }
        if 10 * zeros_even > 3 * pairs && 4 * zeros_odd < zeros_even {
            return (decode_utf16(bytes, true), TextEncoding::Utf16Be);
        }
    }
    latin1_fallback(bytes)
}

fn latin1_fallback(bytes: &[u8]) -> (String, TextEncoding) {
    match std::str::from_utf8(bytes) {
        Ok(s) => (s.to_string(), TextEncoding::Utf8),
        Err(_) => (
            bytes.iter().map(|&b| b as char).collect(),
            TextEncoding::Latin1,
        ),
    }
}

fn decode_utf16(bytes: &[u8], big_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&p| {
            if big_endian {
                u16::from_be_bytes(p)
            } else {
                u16::from_le_bytes(p)
            }
        })
        .collect();
    let s = String::from_utf16_lossy(&units);
    s.strip_prefix('\u{FEFF}').map(str::to_string).unwrap_or(s)
}

/// Encodes text for writing. UTF-16 output gets a BOM. Latin-1 output replaces
/// characters outside ISO-8859-1 with `?`.
pub fn encode_text(text: &str, encoding: TextEncoding) -> Vec<u8> {
    match encoding {
        TextEncoding::Utf8 => text.as_bytes().to_vec(),
        TextEncoding::Utf8Bom => [&[0xEF, 0xBB, 0xBF][..], text.as_bytes()].concat(),
        TextEncoding::Utf16Le => std::iter::once(0xFEFF)
            .chain(text.encode_utf16())
            .flat_map(u16::to_le_bytes)
            .collect(),
        TextEncoding::Utf16Be => std::iter::once(0xFEFF)
            .chain(text.encode_utf16())
            .flat_map(u16::to_be_bytes)
            .collect(),
        TextEncoding::Latin1 => text
            .chars()
            .map(|c| u8::try_from(u32::from(c)).unwrap_or(b'?'))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Catalog availability status (`NM` field 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlassStatus {
    #[default]
    Standard,
    Preferred,
    Obsolete,
    Special,
    Melt,
    Other(i32),
}

impl GlassStatus {
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => GlassStatus::Standard,
            1 => GlassStatus::Preferred,
            2 => GlassStatus::Obsolete,
            3 => GlassStatus::Special,
            4 => GlassStatus::Melt,
            c => GlassStatus::Other(c),
        }
    }
    pub fn code(self) -> i32 {
        match self {
            GlassStatus::Standard => 0,
            GlassStatus::Preferred => 1,
            GlassStatus::Obsolete => 2,
            GlassStatus::Special => 3,
            GlassStatus::Melt => 4,
            GlassStatus::Other(c) => c,
        }
    }
}

/// Catalog data from `OD`. `None` means the field was `-` or negative (the
/// catalogs' "unknown" marker).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct OrderingData {
    pub relative_cost: Option<f64>,
    pub climate_resistance: Option<f64>,
    pub stain_resistance: Option<f64>,
    pub acid_resistance: Option<f64>,
    pub alkali_resistance: Option<f64>,
    pub phosphate_resistance: Option<f64>,
}

/// AGF fields that have no place in [`Glass`] itself. They are kept so that a
/// catalog survives a read → edit → write cycle without losing data.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct AgfGlassMeta {
    /// Formula number of the `NM` record (1…13).
    pub formula: u32,
    pub mil_code: Option<String>,
    pub exclude_substitution: bool,
    pub status: GlassStatus,
    pub melt_frequency: Option<i32>,
    /// TCE −30…+70 °C as written (10⁻⁶/K). `Glass::cte` is this × 10⁻⁶,
    /// unless `ignore_thermal_expansion` is set, in which case `Glass::cte` is `None`.
    pub tce_m30_70: Option<f64>,
    /// TCE +100…+300 °C (10⁻⁶/K).
    pub tce_100_300: Option<f64>,
    /// Deviation of the relative partial dispersion `P_g,F` from the normal line.
    pub dpgf: Option<f64>,
    /// The catalog asks that thermal expansion of this material be ignored
    /// (typical for liquids, gases and some crystals).
    pub ignore_thermal_expansion: bool,
    pub ordering: OrderingData,
    /// Unknown records, verbatim, in file order.
    pub extra_records: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgfWarningKind {
    /// A record tag the reader does not know. Kept verbatim. Reported once per tag.
    UnknownRecord,
    /// A record with unparseable or out-of-range fields. The field is ignored.
    MalformedRecord,
    /// A glass record before the first `NM`. Ignored.
    OrphanRecord,
    /// No `CD` record or bad coefficients. The glass is skipped.
    MissingDispersion,
    /// A formula number outside 1…13. The glass is skipped.
    UnsupportedFormula,
    /// No `LD` record. The glass is kept with an unrestricted range.
    MissingRange,
    /// A later glass with the same name replaced an earlier one.
    DuplicateGlass,
    /// The coefficients do not reproduce the catalog `n_d`/`V_d`.
    IndexMismatch,
    /// The writer could not express a glass in AGF (or had to alter its name).
    NotExportable,
}

/// A non-fatal problem found while reading or writing a catalog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgfWarning {
    /// 1-based line number in the input, when there is one.
    pub line: Option<usize>,
    pub glass: Option<String>,
    pub kind: AgfWarningKind,
    pub message: String,
}

impl fmt::Display for AgfWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(l) = self.line {
            write!(f, "line {l}: ")?;
        }
        if let Some(g) = &self.glass {
            write!(f, "{g}: ")?;
        }
        f.write_str(&self.message)
    }
}

/// What to do with a glass whose coefficients disagree with its `n_d`/`V_d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MismatchPolicy {
    #[default]
    Keep,
    Skip,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgfOptions {
    pub nd_tolerance: f64,
    pub vd_tolerance: f64,
    pub mismatch: MismatchPolicy,
}

impl Default for AgfOptions {
    fn default() -> Self {
        AgfOptions {
            nd_tolerance: DEFAULT_ND_TOLERANCE,
            vd_tolerance: DEFAULT_VD_TOLERANCE,
            mismatch: MismatchPolicy::Keep,
        }
    }
}

/// A parsed AGF catalog: the glasses, their AGF-only metadata (keyed by glass
/// name), the catalog comments and the warnings produced while reading.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AgfCatalog {
    pub catalog: GlassCatalog,
    pub meta: BTreeMap<String, AgfGlassMeta>,
    /// `CC` lines, in file order.
    pub comments: Vec<String>,
    /// Encoding of the input, when it was read from bytes.
    pub encoding: Option<TextEncoding>,
    pub warnings: Vec<AgfWarning>,
}

impl AgfCatalog {
    /// Wraps a catalog without AGF metadata (e.g. built-in or edited glasses)
    /// so it can be written.
    pub fn from_catalog(catalog: GlassCatalog) -> Self {
        AgfCatalog {
            catalog,
            ..Default::default()
        }
    }

    /// AGF text of this catalog (see [`write_agf`]).
    pub fn to_agf_string(&self) -> (String, Vec<AgfWarning>) {
        write_agf(self)
    }
}

// ---------------------------------------------------------------------------
// Formula mapping
// ---------------------------------------------------------------------------

/// Builds the [`Dispersion`] for AGF formula number `formula` from its `CD`
/// coefficients. Missing trailing coefficients are taken as zero.
pub fn dispersion_from_agf(formula: u32, cd: &[f64]) -> MkResult<Dispersion> {
    let k = |i: usize| cd.get(i).copied().unwrap_or(0.0);
    let arr = |n: usize| -> Vec<f64> { (0..n).map(k).collect() };
    let sellmeier = |terms: usize| Dispersion::Sellmeier {
        b: (0..terms).map(|i| k(2 * i)).collect(),
        c: (0..terms).map(|i| k(2 * i + 1)).collect(),
    };
    Ok(match formula {
        1 => Dispersion::Schott {
            a: arr(6).try_into().unwrap(),
        },
        2 => sellmeier(3),
        3 => Dispersion::Herzberger {
            a: arr(6).try_into().unwrap(),
        },
        4 => Dispersion::Sellmeier2 {
            a: k(0),
            b: [k(1), k(3)],
            lambda_um: [k(2), k(4)],
        },
        5 => Dispersion::Conrady {
            n0: k(0),
            a: k(1),
            b: k(2),
        },
        6 => sellmeier(4),
        7 => Dispersion::Handbook1 {
            a: k(0),
            b: k(1),
            c: k(2),
            d: k(3),
        },
        8 => Dispersion::Handbook2 {
            a: k(0),
            b: k(1),
            c: k(2),
            d: k(3),
        },
        9 => Dispersion::Sellmeier4 {
            a: k(0),
            b: k(1),
            c: k(2),
            d: k(3),
            e: k(4),
        },
        10 => Dispersion::Extended {
            a: arr(8).try_into().unwrap(),
        },
        11 => sellmeier(5),
        12 => Dispersion::Extended2 {
            a: arr(8).try_into().unwrap(),
        },
        13 => Dispersion::Extended3 {
            a: arr(9).try_into().unwrap(),
        },
        f => {
            return Err(MkError::Unsupported(format!(
                "AGF dispersion formula {f} (supported: 1–13)"
            )));
        }
    })
}

/// The AGF formula number and `CD` coefficients for `d`. A Cauchy formula is
/// converted exactly to Schott form. Tabulated data and Sellmeier sums with
/// more than five terms have no AGF form.
pub fn dispersion_to_agf(d: &Dispersion) -> MkResult<(u32, Vec<f64>)> {
    Ok(match d {
        Dispersion::Schott { a } => (1, a.to_vec()),
        Dispersion::Sellmeier { b, c } => {
            if b.len() != c.len() {
                return Err(MkError::InvalidModel(format!(
                    "Sellmeier with {} B and {} C coefficients",
                    b.len(),
                    c.len()
                )));
            }
            let (formula, terms) = match b.len() {
                0..=3 => (2, 3),
                4 => (6, 4),
                5 => (11, 5),
                n => {
                    return Err(MkError::Unsupported(format!(
                        "{n}-term Sellmeier has no AGF formula (max 5 terms)"
                    )));
                }
            };
            let mut cd = Vec::with_capacity(2 * terms);
            for i in 0..terms {
                cd.push(b.get(i).copied().unwrap_or(0.0));
                cd.push(c.get(i).copied().unwrap_or(0.0));
            }
            (formula, cd)
        }
        Dispersion::Herzberger { a } => (3, a.to_vec()),
        Dispersion::Sellmeier2 { a, b, lambda_um } => {
            (4, vec![*a, b[0], lambda_um[0], b[1], lambda_um[1]])
        }
        Dispersion::Conrady { n0, a, b } => (5, vec![*n0, *a, *b]),
        Dispersion::Handbook1 { a, b, c, d } => (7, vec![*a, *b, *c, *d]),
        Dispersion::Handbook2 { a, b, c, d } => (8, vec![*a, *b, *c, *d]),
        Dispersion::Sellmeier4 { a, b, c, d, e } => (9, vec![*a, *b, *c, *d, *e]),
        Dispersion::Extended { a } => (10, a.to_vec()),
        Dispersion::Extended2 { a } => (12, a.to_vec()),
        Dispersion::Extended3 { a } => (13, a.to_vec()),
        Dispersion::Cauchy { a, b, c } => {
            // (A + B u + C u²)² with u = λ⁻², expanded exactly.
            (
                1,
                vec![
                    a * a,
                    0.0,
                    2.0 * a * b,
                    b.mul_add(*b, 2.0 * a * c),
                    2.0 * b * c,
                    c * c,
                ],
            )
        }
        Dispersion::Tabulated { .. } => {
            return Err(MkError::Unsupported(
                "tabulated dispersion has no AGF formula".into(),
            ));
        }
    })
}

/// `(n_d, V_d)` computed from a glass's dispersion coefficients.
pub fn computed_nd_vd(d: &Dispersion) -> (f64, f64) {
    let nd = d.index(lines::D);
    (nd, (nd - 1.0) / (d.index(lines::F) - d.index(lines::C)))
}

/// A model glass (`n_d`, `V_d`, `ΔP_g,F`) as a catalog [`Glass`], so it can be
/// exported. It uses the same Cauchy fit as `MaterialRef::Model`, which is
/// anchored at the g, F, d and C lines; its range is set to 0.35–1.1 µm,
/// where that fit is meaningful.
pub fn model_glass(name: &str, nd: f64, vd: f64, dpgf: f64) -> MkResult<Glass> {
    let [a, b, c] = model_glass_cauchy(nd, vd, dpgf)?;
    Ok(Glass {
        name: name.to_string(),
        dispersion: Dispersion::Cauchy { a, b, c },
        nd,
        vd,
        min_um: 0.35,
        max_um: 1.1,
        thermal: None,
        cte: None,
        density: None,
        transmittance: Vec::new(),
        relative_cost: None,
        comment: format!("model glass nd={nd} vd={vd} dPgF={dpgf}"),
    })
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Parses AGF text into a catalog named `name`, with default options and
/// without the warnings and metadata (see [`parse_agf_with`]).
pub fn parse_agf(name: &str, text: &str) -> MkResult<GlassCatalog> {
    Ok(parse_agf_with(name, text, &AgfOptions::default())?.catalog)
}

/// Parses raw catalog bytes in any supported encoding (see [`decode_bytes`]).
pub fn parse_agf_bytes(name: &str, bytes: &[u8], opts: &AgfOptions) -> MkResult<AgfCatalog> {
    let (text, enc) = decode_bytes(bytes);
    let mut cat = parse_agf_with(name, &text, opts)?;
    cat.encoding = Some(enc);
    Ok(cat)
}

/// Reads an AGF file. The catalog is named after the file stem.
pub fn read_agf_file(path: impl AsRef<Path>, opts: &AgfOptions) -> MkResult<AgfCatalog> {
    let path = path.as_ref();
    let bytes =
        std::fs::read(path).map_err(|e| MkError::Io(format!("reading {}: {e}", path.display())))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "AGF".into());
    parse_agf_bytes(&name, &bytes, opts)
}

/// A numeric field. `-` (and an absent field) is `None`. Fortran-style `D`
/// exponents are accepted. `Err` means the field is present but not a finite number.
fn field(tok: Option<&&str>) -> Result<Option<f64>, ()> {
    let Some(t) = tok else { return Ok(None) };
    let t = t.trim();
    if t.is_empty() || t == "-" {
        return Ok(None);
    }
    let v = t
        .parse::<f64>()
        .or_else(|_| t.replace(['D', 'd'], "E").parse::<f64>())
        .map_err(|_| ())?;
    if v.is_finite() { Ok(Some(v)) } else { Err(()) }
}

/// Text after the record tag, with the separating whitespace removed.
fn rest_of_line(line: &str) -> &str {
    let t = line.trim_start();
    t.get(2..).unwrap_or("").trim()
}

/// A glass being assembled from its records.
struct Pending {
    line: usize,
    glass: Glass,
    meta: AgfGlassMeta,
    formula: Option<f64>,
    cd: Option<Vec<f64>>,
    bad_cd: bool,
    has_ld: bool,
}

struct Reader<'a> {
    opts: &'a AgfOptions,
    out: AgfCatalog,
    unknown_tags: BTreeSet<String>,
}

impl Reader<'_> {
    fn warn(&mut self, line: usize, glass: Option<&str>, kind: AgfWarningKind, msg: String) {
        self.out.warnings.push(AgfWarning {
            line: Some(line),
            glass: glass.map(str::to_string),
            kind,
            message: msg,
        });
    }

    fn finish(&mut self, p: Pending) {
        let Pending {
            line,
            mut glass,
            mut meta,
            formula,
            cd,
            bad_cd,
            has_ld,
        } = p;
        let name = glass.name.clone();
        let n = Some(name.as_str());
        let formula = match formula {
            Some(f) if f.fract() == 0.0 && (1.0..=13.0).contains(&f) => f as u32,
            Some(f) => {
                self.warn(
                    line,
                    n,
                    AgfWarningKind::UnsupportedFormula,
                    format!("unsupported dispersion formula {f}; glass skipped"),
                );
                return;
            }
            None => {
                self.warn(
                    line,
                    n,
                    AgfWarningKind::UnsupportedFormula,
                    "missing dispersion formula number; glass skipped".into(),
                );
                return;
            }
        };
        let cd = match cd {
            Some(cd) if !bad_cd => cd,
            _ => {
                self.warn(
                    line,
                    n,
                    AgfWarningKind::MissingDispersion,
                    "no usable CD record; glass skipped".into(),
                );
                return;
            }
        };
        meta.formula = formula;
        glass.dispersion = match dispersion_from_agf(formula, &cd) {
            Ok(d) => d,
            Err(e) => {
                self.warn(line, n, AgfWarningKind::UnsupportedFormula, e.to_string());
                return;
            }
        };
        if !has_ld {
            self.warn(
                line,
                n,
                AgfWarningKind::MissingRange,
                "no LD record; wavelength range left unrestricted".into(),
            );
        }
        // Sort transmittance data by wavelength. Where a wavelength repeats
        // (data for two sample thicknesses), keep the thicker sample, whose
        // absorption coefficient is better determined.
        let t = &mut glass.transmittance;
        t.sort_by(|a, b| {
            a.um.total_cmp(&b.um)
                .then(b.thickness_mm.total_cmp(&a.thickness_mm))
        });
        t.dedup_by(|later, earlier| later.um == earlier.um);

        if glass.nd > 0.0 && glass.vd > 0.0 && (glass.min_um..=glass.max_um).contains(&lines::D) {
            let (nd, vd) = computed_nd_vd(&glass.dispersion);
            // NaN (n² < 0 at a reference line) counts as a mismatch.
            let bad_nd = nd.is_nan() || (nd - glass.nd).abs() > self.opts.nd_tolerance;
            let bad_vd = vd.is_nan() || (vd - glass.vd).abs() > self.opts.vd_tolerance;
            if bad_nd || bad_vd {
                let skip = self.opts.mismatch == MismatchPolicy::Skip;
                self.warn(
                    line,
                    n,
                    AgfWarningKind::IndexMismatch,
                    format!(
                        "coefficients give nd = {nd:.6}, vd = {vd:.3}; catalog says nd = {}, vd = {} \
                         (Δnd = {:.1e}, ΔVd = {:.3}); glass {}",
                        glass.nd,
                        glass.vd,
                        nd - glass.nd,
                        vd - glass.vd,
                        if skip { "skipped" } else { "kept" }
                    ),
                );
                if skip {
                    return;
                }
            }
        }
        if self.out.catalog.glasses.contains_key(&name) {
            self.warn(
                line,
                n,
                AgfWarningKind::DuplicateGlass,
                "duplicate glass name; the later definition replaces the earlier".into(),
            );
        }
        self.out.meta.insert(name.clone(), meta);
        self.out.catalog.glasses.insert(name, glass);
    }
}

/// Parses AGF text into a catalog named `name`, returning the glasses, their
/// AGF metadata, the catalog comments and all warnings.
pub fn parse_agf_with(name: &str, text: &str, opts: &AgfOptions) -> MkResult<AgfCatalog> {
    use AgfWarningKind::*;
    let mut r = Reader {
        opts,
        out: AgfCatalog {
            catalog: GlassCatalog {
                name: name.to_string(),
                ..Default::default()
            },
            ..Default::default()
        },
        unknown_tags: BTreeSet::new(),
    };
    let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
    let normalized = text.replace("\r\n", "\n");
    let mut cur: Option<Pending> = None;
    // After a malformed NM, the rest of that glass's records are ignored silently.
    let mut skipping = false;
    let mut nm_seen = 0usize;

    for (i, raw) in normalized.split(['\n', '\r']).enumerate() {
        let ln = i + 1;
        let line = raw.trim_matches(|c: char| c.is_whitespace() || c == '\0');
        if line.is_empty() {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        let tag = toks[0].to_ascii_uppercase();
        let f = &toks[1..];

        if tag == "CC" {
            r.out.comments.push(rest_of_line(line).to_string());
            continue;
        }
        if tag == "NM" {
            nm_seen += 1;
            if let Some(p) = cur.take() {
                r.finish(p);
            }
            let Some(gname) = f.first() else {
                r.warn(
                    ln,
                    None,
                    MalformedRecord,
                    "NM record without a glass name".into(),
                );
                skipping = true;
                continue;
            };
            skipping = false;
            let gname = gname.to_string();
            let g = Some(gname.as_str());
            let nm_field = |k: usize, what: &str, r: &mut Reader| match field(f.get(k)) {
                Ok(v) => v,
                Err(()) => {
                    r.warn(ln, g, MalformedRecord, format!("NM: bad {what} '{}'", f[k]));
                    None
                }
            };
            let formula = nm_field(1, "formula number", &mut r);
            let nd = nm_field(3, "nd", &mut r).unwrap_or(0.0);
            let vd = nm_field(4, "vd", &mut r).unwrap_or(0.0);
            let excl = nm_field(5, "exclude-substitution flag", &mut r);
            let status = nm_field(6, "status", &mut r);
            let melt = nm_field(7, "melt frequency", &mut r);
            let mil = f.get(2).filter(|s| **s != "-").map(|s| s.to_string());
            cur = Some(Pending {
                line: ln,
                glass: Glass {
                    name: gname.clone(),
                    dispersion: Dispersion::Cauchy {
                        a: 1.0,
                        b: 0.0,
                        c: 0.0,
                    },
                    nd,
                    vd,
                    min_um: 0.0,
                    max_um: f64::MAX,
                    thermal: None,
                    cte: None,
                    density: None,
                    transmittance: Vec::new(),
                    relative_cost: None,
                    comment: String::new(),
                },
                meta: AgfGlassMeta {
                    mil_code: mil,
                    exclude_substitution: excl.is_some_and(|v| v != 0.0),
                    status: GlassStatus::from_code(status.map_or(0, |v| v as i32)),
                    melt_frequency: melt.map(|v| v as i32),
                    ..Default::default()
                },
                formula,
                cd: None,
                bad_cd: false,
                has_ld: false,
            });
            continue;
        }

        let Some(p) = cur.as_mut() else {
            if !skipping {
                let kind = if nm_seen == 0 {
                    OrphanRecord
                } else {
                    MalformedRecord
                };
                r.warn(
                    ln,
                    None,
                    kind,
                    format!("{tag} record outside any glass; ignored"),
                );
            }
            continue;
        };
        let gname = p.glass.name.clone();
        let g = Some(gname.as_str());
        // Parses fields `0..n`; on a bad field, warns and returns `None` for the record.
        let fields = |n: usize, r: &mut Reader| -> Option<Vec<Option<f64>>> {
            let mut v = Vec::with_capacity(n);
            for k in 0..n {
                match field(f.get(k)) {
                    Ok(x) => v.push(x),
                    Err(()) => {
                        r.warn(
                            ln,
                            g,
                            MalformedRecord,
                            format!(
                                "{tag}: field {} '{}' is not a number; record ignored",
                                k + 1,
                                f[k]
                            ),
                        );
                        return None;
                    }
                }
            }
            Some(v)
        };
        match tag.as_str() {
            "GC" => {
                if !p.glass.comment.is_empty() {
                    p.glass.comment.push('\n');
                }
                p.glass.comment.push_str(rest_of_line(line));
            }
            "ED" => {
                if let Some(v) = fields(5, &mut r) {
                    p.meta.tce_m30_70 = v[0];
                    p.meta.tce_100_300 = v[1];
                    p.glass.density = v[2].filter(|d| *d > 0.0);
                    p.meta.dpgf = v[3];
                    p.meta.ignore_thermal_expansion = v[4].is_some_and(|x| x != 0.0);
                    p.glass.cte = if p.meta.ignore_thermal_expansion {
                        None
                    } else {
                        v[0].map(|t| t / 1e6)
                    };
                }
            }
            "CD" => match fields(f.len(), &mut r) {
                Some(v) => p.cd = Some(v.into_iter().map(|x| x.unwrap_or(0.0)).collect()),
                None => p.bad_cd = true,
            },
            "TD" => {
                if let Some(v) = fields(7, &mut r)
                    && v.iter().any(Option::is_some)
                {
                    let z = |k: usize| v[k].unwrap_or(0.0);
                    p.glass.thermal = Some(ThermalCoefficients {
                        d0: z(0),
                        d1: z(1),
                        d2: z(2),
                        e0: z(3),
                        e1: z(4),
                        lambda_tk_um: z(5),
                        t_ref_c: v[6].unwrap_or(20.0),
                    });
                }
            }
            "OD" => {
                if let Some(v) = fields(6, &mut r) {
                    let pos = |k: usize| v[k].filter(|x| *x >= 0.0);
                    p.meta.ordering = OrderingData {
                        relative_cost: pos(0),
                        climate_resistance: pos(1),
                        stain_resistance: pos(2),
                        acid_resistance: pos(3),
                        alkali_resistance: pos(4),
                        phosphate_resistance: pos(5),
                    };
                    p.glass.relative_cost = pos(0).filter(|c| *c > 0.0);
                }
            }
            "LD" => {
                if let Some(v) = fields(2, &mut r) {
                    match (v[0], v[1]) {
                        (Some(lo), Some(hi)) if lo >= 0.0 && hi > lo => {
                            p.glass.min_um = lo;
                            p.glass.max_um = hi;
                            p.has_ld = true;
                        }
                        _ => r.warn(
                            ln,
                            g,
                            MalformedRecord,
                            "LD: needs 0 ≤ min < max; record ignored".into(),
                        ),
                    }
                }
            }
            "IT" => {
                if let Some(v) = fields(3, &mut r) {
                    match (v[0], v[1], v[2]) {
                        (Some(um), Some(tau), Some(th))
                            if um > 0.0 && (0.0..=1.0).contains(&tau) && th > 0.0 =>
                        {
                            p.glass.transmittance.push(TransmittancePoint {
                                um,
                                tau,
                                thickness_mm: th,
                            })
                        }
                        // Blank or partial IT lines are common filler in catalogs.
                        (None, None, None) => {}
                        _ => r.warn(
                            ln,
                            g,
                            MalformedRecord,
                            "IT: needs λ > 0, 0 ≤ τ ≤ 1 and thickness > 0; record ignored".into(),
                        ),
                    }
                }
            }
            _ => {
                p.meta.extra_records.push(line.to_string());
                if r.unknown_tags.insert(tag.clone()) {
                    r.warn(
                        ln,
                        g,
                        UnknownRecord,
                        format!("unknown record type '{tag}' kept verbatim"),
                    );
                }
            }
        }
    }
    if let Some(p) = cur.take() {
        r.finish(p);
    }
    if nm_seen == 0 {
        return Err(MkError::Parse(format!(
            "'{name}' contains no glass (NM) records; is it an AGF catalog?"
        )));
    }
    Ok(r.out)
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Shortest decimal text that reads back to exactly `v`.
fn num(v: f64) -> String {
    let a = v.abs();
    if v == 0.0 {
        "0".into()
    } else if (1e-3..1e7).contains(&a) {
        format!("{v}")
    } else {
        format!("{v:E}")
    }
}

fn opt(v: Option<f64>) -> String {
    v.map_or_else(|| "-1".into(), num)
}

/// Writes a catalog as AGF text (UTF-8, CRLF line endings, as the format
/// customarily uses). Every number is written in its shortest exactly
/// round-tripping form, so write → read reproduces the glasses bit for bit.
///
/// Glasses whose dispersion has no AGF form (see [`dispersion_to_agf`]) are
/// left out with a [`AgfWarningKind::NotExportable`] warning. Whitespace in a
/// glass name is replaced by `_`, also with a warning. `ED` is written only
/// when the glass has some of its data, with the missing values as 0 (a
/// density of 0 reads back as "unknown", a TCE of 0 as 0). `TD` is written only
/// when the glass has thermal coefficients. Missing `OD` values are written as −1.
pub fn write_agf(cat: &AgfCatalog) -> (String, Vec<AgfWarning>) {
    use std::fmt::Write;
    let mut s = String::new();
    let mut warnings = Vec::new();
    let mut warn = |glass: &str, msg: String| {
        warnings.push(AgfWarning {
            line: None,
            glass: Some(glass.to_string()),
            kind: AgfWarningKind::NotExportable,
            message: msg,
        })
    };
    let nl = "\r\n";
    for c in &cat.comments {
        let _ = write!(s, "CC {c}{nl}");
    }
    let empty = AgfGlassMeta::default();
    for (key, g) in &cat.catalog.glasses {
        let meta = cat.meta.get(key).unwrap_or(&empty);
        let (formula, mut cd) = match dispersion_to_agf(&g.dispersion) {
            Ok(x) => x,
            Err(e) => {
                warn(&g.name, format!("not written: {e}"));
                continue;
            }
        };
        let mut name: String = g
            .name
            .chars()
            .map(|c| if c.is_whitespace() { '_' } else { c })
            .collect();
        if name.is_empty() {
            name = key
                .chars()
                .map(|c| if c.is_whitespace() { '_' } else { c })
                .collect();
        }
        if name.is_empty() {
            warn(&g.name, "not written: empty glass name".into());
            continue;
        }
        if name != g.name {
            warn(
                &g.name,
                format!("written as '{name}' (AGF names cannot contain whitespace)"),
            );
        }
        let mil = meta.mil_code.clone().unwrap_or_else(|| {
            format!(
                "{:03}{:03}",
                ((g.nd - 1.0) * 1000.0).round().max(0.0) as i64,
                (g.vd * 10.0).round().max(0.0) as i64
            )
        });
        let _ = write!(
            s,
            "NM {name} {formula} {mil} {} {} {} {} {}{nl}",
            num(g.nd),
            num(g.vd),
            u8::from(meta.exclude_substitution),
            meta.status.code(),
            meta.melt_frequency.unwrap_or(0),
        );
        for c in g.comment.lines() {
            let _ = write!(s, "GC {c}{nl}");
        }
        // Prefer the TCE exactly as read, so that it round-trips bit for bit.
        let tce = match (g.cte, meta.tce_m30_70) {
            (Some(cte), Some(raw)) if raw / 1e6 == cte => raw,
            (Some(cte), _) => cte * 1e6,
            (None, raw) if meta.ignore_thermal_expansion => raw.unwrap_or(0.0),
            (None, _) => 0.0,
        };
        let has_ed = g.cte.is_some()
            || g.density.is_some()
            || meta.tce_m30_70.is_some()
            || meta.tce_100_300.is_some()
            || meta.dpgf.is_some()
            || meta.ignore_thermal_expansion;
        if has_ed {
            let _ = write!(
                s,
                "ED {} {} {} {} {}{nl}",
                num(tce),
                num(meta.tce_100_300.unwrap_or(0.0)),
                num(g.density.unwrap_or(0.0)),
                num(meta.dpgf.unwrap_or(0.0)),
                u8::from(meta.ignore_thermal_expansion),
            );
        }
        cd.resize(cd.len().max(CD_WIDTH), 0.0);
        let cd: Vec<String> = cd.into_iter().map(num).collect();
        let _ = write!(s, "CD {}{nl}", cd.join(" "));
        if let Some(th) = &g.thermal {
            let _ = write!(
                s,
                "TD {} {} {} {} {} {} {}{nl}",
                num(th.d0),
                num(th.d1),
                num(th.d2),
                num(th.e0),
                num(th.e1),
                num(th.lambda_tk_um),
                num(th.t_ref_c)
            );
        }
        let o = &meta.ordering;
        // The glass's own (possibly edited) relative cost wins over the stored ordering data.
        let cost = g.relative_cost.or(o.relative_cost);
        let _ = write!(
            s,
            "OD {} {} {} {} {} {}{nl}",
            opt(cost),
            opt(o.climate_resistance),
            opt(o.stain_resistance),
            opt(o.acid_resistance),
            opt(o.alkali_resistance),
            opt(o.phosphate_resistance),
        );
        let _ = write!(s, "LD {} {}{nl}", num(g.min_um), num(g.max_um));
        for t in &g.transmittance {
            let _ = write!(
                s,
                "IT {} {} {}{nl}",
                num(t.um),
                num(t.tau),
                num(t.thickness_mm)
            );
        }
        for x in &meta.extra_records {
            let _ = write!(s, "{x}{nl}");
        }
    }
    (s, warnings)
}

/// Writes a catalog to `path` in `encoding`. Returns the writer's warnings.
pub fn write_agf_file(
    path: impl AsRef<Path>,
    cat: &AgfCatalog,
    encoding: TextEncoding,
) -> MkResult<Vec<AgfWarning>> {
    let path = path.as_ref();
    let (text, warnings) = write_agf(cat);
    std::fs::write(path, encode_text(&text, encoding))
        .map_err(|e| MkError::Io(format!("writing {}: {e}", path.display())))?;
    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::builtin;

    /// Published catalog data for N-BK7 and F2 (Sellmeier, formula 2), and
    /// BK7 in the older Schott power series (formula 1), written the way
    /// vendor files write them (E-notation, 3-digit exponents, unsorted IT
    /// data, an unknown record, a repeated IT wavelength).
    const SNIPPET: &str = "\
CC Synthetic test catalog
CC Coefficients are published manufacturer data
NM N-BK7 2 517642.251 1.5168 64.17 0 1 0
GC Borosilicate crown
ED 7.1 8.3 2.51 -0.0009 0
CD 1.03961212E+000 6.00069867E-003 2.31792344E-001 2.00179144E-002 1.01046945E+000 1.03560653E+002 0.0 0.0 0.0 0.0
TD 1.86E-006 1.31E-008 -1.37E-011 4.34E-007 6.27E-010 1.70E-001 2.00E+001
OD 1.0 1 0 1 2.3 2
LD 3.0E-001 2.5E+000
IT 0.5 0.998 25
IT 0.35 0.97 10
IT 0.35 0.92 25
IT 2.5 0.67 25
IT 0.4 0.997 25
BD 1 2 3
NM F2 2 620364.360 1.62004 36.37 0 0 0
ED 8.2 9.2 3.6 0.0026 0
CD 1.34533359E+000 9.97743871E-003 2.09073176E-001 4.70450767E-002 9.37357162E-001 1.11886764E+002 0 0 0 0
TD 1.51E-006 1.56E-008 -2.78E-011 9.34E-007 1.04E-009 2.50E-001 2.00E+001
OD 1.6 1 0 1 1 1
LD 3.2E-001 2.5E+000
NM BK7_SCHOTT 1 517642 1.5168 64.17 0 2 0
CD 2.2718929 -1.0108077E-002 1.0592509E-002 2.0816965E-004 -7.6472538E-006 4.9240991E-007
LD 0.31 2.325
";

    fn parse(text: &str) -> AgfCatalog {
        parse_agf_with("TEST", text, &AgfOptions::default()).unwrap()
    }

    fn kinds(c: &AgfCatalog) -> Vec<AgfWarningKind> {
        c.warnings.iter().map(|w| w.kind).collect()
    }

    #[test]
    fn parses_published_glasses() {
        let c = parse(SNIPPET);
        assert_eq!(c.catalog.glasses.len(), 3);
        // Only the unknown `BD` record is reported. In particular, all three
        // glasses reproduce their catalog nd/vd.
        assert_eq!(kinds(&c), vec![AgfWarningKind::UnknownRecord]);
        assert_eq!(c.comments.len(), 2);

        let b = &c.catalog.glasses["N-BK7"];
        let reference = &builtin::catalog().glasses["N-BK7"];
        // Same decimal coefficients → bit-identical dispersion.
        assert_eq!(b.dispersion, reference.dispersion);
        assert_eq!((b.nd, b.vd, b.min_um, b.max_um), (1.5168, 64.17, 0.3, 2.5));
        assert_eq!(b.cte, Some(7.1e-6));
        assert_eq!(b.density, Some(2.51));
        assert_eq!(b.relative_cost, Some(1.0));
        assert_eq!(b.comment, "Borosilicate crown");
        let th = b.thermal.unwrap();
        assert_eq!(
            (th.d0, th.e1, th.lambda_tk_um, th.t_ref_c),
            (1.86e-6, 6.27e-10, 0.17, 20.0)
        );
        // IT is sorted, and the repeated 0.35 µm point keeps the 25 mm sample.
        let it: Vec<_> = b
            .transmittance
            .iter()
            .map(|t| (t.um, t.tau, t.thickness_mm))
            .collect();
        assert_eq!(
            it,
            vec![
                (0.35, 0.92, 25.0),
                (0.4, 0.997, 25.0),
                (0.5, 0.998, 25.0),
                (2.5, 0.67, 25.0)
            ]
        );
        let m = &c.meta["N-BK7"];
        assert_eq!(m.formula, 2);
        assert_eq!(m.mil_code.as_deref(), Some("517642.251"));
        assert_eq!(m.status, GlassStatus::Preferred);
        assert_eq!(m.tce_100_300, Some(8.3));
        assert_eq!(m.dpgf, Some(-0.0009));
        assert_eq!(m.ordering.alkali_resistance, Some(2.3));
        assert_eq!(m.extra_records, vec!["BD 1 2 3".to_string()]);

        for name in ["N-BK7", "F2", "BK7_SCHOTT"] {
            let g = &c.catalog.glasses[name];
            let (nd, vd) = computed_nd_vd(&g.dispersion);
            assert!(
                (nd - g.nd).abs() < 1e-4 && (vd - g.vd).abs() < 0.05,
                "{name}: {nd} {vd}"
            );
        }
        // The formula-1 and formula-2 descriptions of BK7 are independent fits
        // to similar melt data and agree to the catalogs' precision.
        let s = &c.catalog.glasses["BK7_SCHOTT"];
        assert_eq!(c.meta["BK7_SCHOTT"].status, GlassStatus::Obsolete);
        for l in [0.4047, 0.48613, 0.58756, 0.65627, 0.85211] {
            let d = s.catalog_index(l) - b.catalog_index(l);
            assert!(d.abs() < 2e-5, "λ = {l}: Δn = {d:e}");
        }
        // The library resolves a parsed glass (case-insensitively) at the
        // reference environment.
        let mut lib = crate::materials::MaterialLibrary::new();
        lib.add_catalog(std::sync::Arc::new(c.catalog.clone()));
        let r = lib
            .resolve(&crate::model::MaterialRef::glass("f2"), &Default::default())
            .unwrap();
        assert!((r.index(lines::D) - 1.62004).abs() < 2e-5);
    }

    #[test]
    fn encodings_decode_to_the_same_catalog() {
        let reference = parse(SNIPPET).catalog;
        let crlf = SNIPPET.replace('\n', "\r\n");
        let cr_only = SNIPPET.replace('\n', "\r");
        let le = encode_text(&crlf, TextEncoding::Utf16Le);
        let be = encode_text(SNIPPET, TextEncoding::Utf16Be);
        // Hand-checked prefix: BOM FF FE, then "CC" as 43 00 43 00.
        assert_eq!(&le[..6], &[0xFF, 0xFE, b'C', 0, b'C', 0]);
        assert_eq!(&be[..6], &[0xFE, 0xFF, 0, b'C', 0, b'C']);
        let cases: Vec<(Vec<u8>, TextEncoding)> = vec![
            (SNIPPET.as_bytes().to_vec(), TextEncoding::Utf8),
            (crlf.as_bytes().to_vec(), TextEncoding::Utf8),
            (cr_only.as_bytes().to_vec(), TextEncoding::Utf8),
            (
                encode_text(SNIPPET, TextEncoding::Utf8Bom),
                TextEncoding::Utf8Bom,
            ),
            (le.clone(), TextEncoding::Utf16Le),
            (le[2..].to_vec(), TextEncoding::Utf16Le), // no BOM
            (be.clone(), TextEncoding::Utf16Be),
            (be[2..].to_vec(), TextEncoding::Utf16Be), // no BOM
        ];
        for (i, (bytes, enc)) in cases.into_iter().enumerate() {
            let c = parse_agf_bytes("TEST", &bytes, &AgfOptions::default()).unwrap();
            assert_eq!(c.encoding, Some(enc), "case {i}");
            assert_eq!(c.catalog, reference, "case {i}");
        }
    }

    #[test]
    fn latin1_fallback() {
        let mut bytes = SNIPPET.as_bytes().to_vec();
        // Append " 20 °C" with a Latin-1 degree sign (invalid as UTF-8) to the GC line.
        let at = SNIPPET.find("GC Borosilicate crown").unwrap() + "GC Borosilicate crown".len();
        bytes.splice(at..at, b" 20 \xB0C".iter().copied());
        let c = parse_agf_bytes("TEST", &bytes, &AgfOptions::default()).unwrap();
        assert_eq!(c.encoding, Some(TextEncoding::Latin1));
        assert_eq!(
            c.catalog.glasses["N-BK7"].comment,
            "Borosilicate crown 20 °C"
        );
    }

    #[test]
    fn tolerant_of_messy_input() {
        let text = "\
LD 0.3 2.5
nm   GLASS_A\t2  -  1.5168   64.17
CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653
ED - - 0 - 1
TD 1.86e-6
OD - -1 - 1
IT
IT 0.4 0.99
  LD   0.3    2.5
XY something new
XY again
NM GLASS_B 2 - 1.62004 36.37
CD 1.34533359 9.97743871D-3 0.209073176 0.0470450767 0.937357162 111.886764
";
        let c = parse(text);
        use AgfWarningKind::*;
        assert_eq!(
            kinds(&c),
            vec![OrphanRecord, MalformedRecord, UnknownRecord, MissingRange]
        );
        let a = &c.catalog.glasses["GLASS_A"];
        let m = &c.meta["GLASS_A"];
        assert_eq!(m.mil_code, None);
        assert_eq!(m.melt_frequency, None);
        assert!(m.ignore_thermal_expansion);
        assert_eq!((a.cte, a.density), (None, None));
        assert_eq!(a.relative_cost, None);
        assert_eq!(m.ordering.acid_resistance, Some(1.0));
        let th = a.thermal.unwrap();
        assert_eq!((th.d0, th.d1, th.t_ref_c), (1.86e-6, 0.0, 20.0));
        assert!(a.transmittance.is_empty());
        assert_eq!((a.min_um, a.max_um), (0.3, 2.5));
        assert_eq!(m.extra_records, vec!["XY something new", "XY again"]);
        // Fortran `D` exponent; no LD means an unrestricted range.
        let b = &c.catalog.glasses["GLASS_B"];
        assert_eq!(b.dispersion, builtin::catalog().glasses["F2"].dispersion);
        assert_eq!((b.min_um, b.max_um), (0.0, f64::MAX));
    }

    #[test]
    fn unusable_glasses_are_skipped() {
        let text = "\
NM NO_CD 2 - 1.5 60
LD 0.4 0.7
NM BAD_FORMULA 14 - 1.5 60
CD 1 2 3
NM BAD_COEF 2 - 1.5 60
CD 1.0 0.006 abc
NM OK 5 - - -
CD 1.5 0.01 0.001
";
        let c = parse(text);
        use AgfWarningKind::*;
        assert_eq!(
            kinds(&c),
            vec![
                MissingDispersion,
                UnsupportedFormula,
                MalformedRecord,
                MissingDispersion,
                MissingRange
            ]
        );
        assert_eq!(c.catalog.glasses.keys().collect::<Vec<_>>(), vec!["OK"]);
        assert!(parse_agf("EMPTY", "CC nothing here\n").is_err());
        assert!(parse_agf_bytes("EMPTY", b"", &AgfOptions::default()).is_err());
    }

    #[test]
    fn index_mismatch_policy() {
        // N-BK7 coefficients with nd off by 2e-4.
        let text = SNIPPET.replace("1.5168 64.17 0 1 0", "1.5170 64.17 0 1 0");
        let kept = parse(&text);
        let w: Vec<_> = kept
            .warnings
            .iter()
            .filter(|w| w.kind == AgfWarningKind::IndexMismatch)
            .collect();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].glass.as_deref(), Some("N-BK7"));
        assert_eq!(w[0].line, Some(3));
        assert!(kept.catalog.glasses.contains_key("N-BK7"));

        let opts = AgfOptions {
            mismatch: MismatchPolicy::Skip,
            ..Default::default()
        };
        let skipped = parse_agf_with("TEST", &text, &opts).unwrap();
        assert!(!skipped.catalog.glasses.contains_key("N-BK7"));
        assert_eq!(skipped.catalog.glasses.len(), 2);

        // A wrong vd alone also triggers it.
        let text = SNIPPET.replace("1.62004 36.37", "1.62004 36.9");
        assert!(kinds(&parse(&text)).contains(&AgfWarningKind::IndexMismatch));
    }

    #[test]
    fn round_trip_parsed_catalog() {
        let c = parse(SNIPPET);
        let (text, warnings) = write_agf(&c);
        assert!(warnings.is_empty());
        assert!(text.contains("\r\n"));
        let back = parse(&text);
        assert_eq!(back.catalog, c.catalog);
        assert_eq!(back.meta, c.meta);
        assert_eq!(back.comments, c.comments);
        // And once more through UTF-16LE bytes.
        let bytes = encode_text(&text, TextEncoding::Utf16Le);
        let again = parse_agf_bytes("TEST", &bytes, &AgfOptions::default()).unwrap();
        assert_eq!(again.catalog, c.catalog);
    }

    #[test]
    fn round_trip_builtin_catalog() {
        let cat = builtin::catalog();
        let (text, warnings) = write_agf(&AgfCatalog::from_catalog(cat.clone()));
        assert!(warnings.is_empty());
        let back = parse_agf_with(&cat.name, &text, &AgfOptions::default()).unwrap();
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
        assert_eq!(back.catalog, cat);
        assert_eq!(back.meta["N-BK7"].mil_code.as_deref(), Some("517642"));
    }

    /// Every formula number survives formula → Dispersion → formula, and a
    /// write → read cycle reproduces each dispersion exactly.
    #[test]
    fn all_formulas_round_trip() {
        let cd: Vec<f64> = vec![
            2.25, 0.0123, 0.0456, 1.5e-4, -2.5e-6, 3.75e-7, 1.25e-8, -6.5e-10, 7.0e-11, 1.0e-3,
        ];
        let mut cat = GlassCatalog {
            name: "FORMULAS".into(),
            ..Default::default()
        };
        for f in 1..=13u32 {
            let d = dispersion_from_agf(f, &cd).unwrap();
            let (f2, cd2) = dispersion_to_agf(&d).unwrap();
            assert_eq!(f2, f);
            assert_eq!(dispersion_from_agf(f2, &cd2).unwrap(), d);
            let mut g = model_glass(&format!("F{f}"), 1.5, 60.0, 0.0).unwrap();
            g.dispersion = d;
            g.nd = 0.0; // skip the nd/vd check: these are not real glasses
            cat.glasses.insert(g.name.clone(), g);
        }
        let (text, w) = write_agf(&AgfCatalog::from_catalog(cat.clone()));
        assert!(w.is_empty());
        let back = parse(&text);
        assert_eq!(back.catalog.glasses, cat.glasses);
        for f in 1..=13u32 {
            assert_eq!(back.meta[&format!("F{f}")].formula, f);
        }
        assert!(dispersion_from_agf(0, &cd).is_err());
        assert!(dispersion_from_agf(14, &cd).is_err());
    }

    /// Spot values of each AGF formula, written out by hand from the formula
    /// definitions with `cd = [k0, k1, …]`.
    #[test]
    fn formula_known_values() {
        let k = [
            2.0, 0.01, 0.02, 0.003, 0.0004, 0.00005, 0.000006, 0.0000007, 0.00000008, 0.0,
        ];
        let n = |f: u32, l: f64| dispersion_from_agf(f, &k).unwrap().index(l);
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        let l2: f64 = 4.0;
        // 2 Sellmeier1: interleaved K1 L1 K2 L2 K3 L3.
        let s1 = 1.0 + k[0] * l2 / (l2 - k[1]) + k[2] * l2 / (l2 - k[3]) + k[4] * l2 / (l2 - k[5]);
        close(n(2, 2.0), s1.sqrt());
        // 4 Sellmeier2: n² = 1 + A + B1 λ²/(λ² − λ1²) + B2 λ²/(λ² − λ2²).
        let s2 = 1.0 + k[0] + k[1] * l2 / (l2 - k[2] * k[2]) + k[3] * l2 / (l2 - k[4] * k[4]);
        close(n(4, 2.0), s2.sqrt());
        // 6 Sellmeier3: four interleaved terms.
        let s3 = s1 + k[6] * l2 / (l2 - k[7]);
        close(n(6, 2.0), s3.sqrt());
        // 7 Handbook1: n² = A + B/(λ² − C) − Dλ².
        close(n(7, 2.0), (k[0] + k[1] / (l2 - k[2]) - k[3] * l2).sqrt());
        // 8 Handbook2: n² = A + Bλ²/(λ² − C) − Dλ².
        close(
            n(8, 2.0),
            (k[0] + k[1] * l2 / (l2 - k[2]) - k[3] * l2).sqrt(),
        );
        // 9 Sellmeier4: n² = A + Bλ²/(λ² − C) + Dλ²/(λ² − E).
        close(
            n(9, 2.0),
            (k[0] + k[1] * l2 / (l2 - k[2]) + k[3] * l2 / (l2 - k[4])).sqrt(),
        );
        // 11 Sellmeier5: five interleaved terms.
        let s5 = s3 + k[8] * l2 / (l2 - k[9]);
        close(n(11, 2.0), s5.sqrt());
        // 5 Conrady: n = n0 + A/λ + B/λ^3.5, at λ = 4 (4^3.5 = 128).
        close(n(5, 4.0), k[0] + k[1] / 4.0 + k[2] / 128.0);
        // 1, 10, 12, 13: at λ = 1 the polynomial forms are plain coefficient sums.
        let sum = |m: usize| k[..m].iter().sum::<f64>().sqrt();
        close(n(1, 1.0), sum(6));
        close(n(10, 1.0), sum(8));
        close(n(12, 1.0), sum(8));
        close(n(13, 1.0), sum(9));
        // 3 Herzberger at λ = 1: L = 1/0.972.
        let lh = 1.0 / 0.972;
        close(
            n(3, 1.0),
            k[0] + k[1] * lh + k[2] * lh * lh + k[3] + k[4] + k[5],
        );
    }

    #[test]
    fn model_glass_exports_exactly() {
        let g = model_glass("MY_GLASS", 1.6, 40.0, 0.005).unwrap();
        let mut cat = GlassCatalog {
            name: "EDITED".into(),
            ..Default::default()
        };
        cat.glasses.insert(g.name.clone(), g.clone());
        let (text, w) = write_agf(&AgfCatalog::from_catalog(cat));
        assert!(w.is_empty());
        assert!(text.contains("NM MY_GLASS 1 600400 1.6 40 0 0 0"), "{text}");
        let back = parse(&text);
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
        let b = &back.catalog.glasses["MY_GLASS"];
        assert!(matches!(b.dispersion, Dispersion::Schott { .. }));
        for i in 0..=40 {
            let l = 0.35 + 0.75 * i as f64 / 40.0;
            let d = b.catalog_index(l) - g.catalog_index(l);
            assert!(d.abs() < 1e-14, "λ = {l}: Δn = {d:e}");
        }
        let (nd, vd) = computed_nd_vd(&b.dispersion);
        assert!((nd - 1.6).abs() < 1e-12 && (vd - 40.0).abs() < 1e-9);
    }

    #[test]
    fn writer_reports_unexportable_glasses() {
        let mut cat = builtin::catalog();
        let mut t = cat.glasses["F2"].clone();
        t.name = "TABLE GLASS".into();
        t.dispersion = Dispersion::Tabulated {
            points: vec![(0.4, 1.6), (0.7, 1.58)],
        };
        cat.glasses.insert(t.name.clone(), t);
        let mut s = cat.glasses["F2"].clone();
        s.name = "MY F2".into();
        cat.glasses.insert(s.name.clone(), s);
        let (text, w) = write_agf(&AgfCatalog::from_catalog(cat));
        assert_eq!(w.len(), 2);
        assert!(w.iter().all(|w| w.kind == AgfWarningKind::NotExportable));
        assert!(!text.contains("TABLE"));
        assert!(parse(&text).catalog.glasses.contains_key("MY_F2"));
    }

    #[test]
    fn load_from_file() {
        let dir = std::env::temp_dir().join(format!("mklight-agf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("MYCAT.agf");
        let c = parse(SNIPPET);
        let w = write_agf_file(&path, &c, TextEncoding::Utf16Le).unwrap();
        assert!(w.is_empty());
        let loaded = GlassCatalog::from_agf_file(&path).unwrap();
        assert_eq!(loaded.name, "MYCAT");
        assert_eq!(loaded.glasses, c.catalog.glasses);
        let detailed = read_agf_file(&path, &AgfOptions::default()).unwrap();
        assert_eq!(detailed.encoding, Some(TextEncoding::Utf16Le));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(GlassCatalog::from_agf_file(dir.join("missing.agf")).is_err());
    }
}
