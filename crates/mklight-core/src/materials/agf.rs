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
/// 1. a byte-order mark wins. UTF-16 LE/BE are decoded as such; after a UTF-8
///    BOM the rest is UTF-8 if valid, else Latin-1 (and reported as Latin-1);
/// 2. otherwise, if at least 5 % of the 2-byte units over the *whole* buffer
///    are ASCII characters in one UTF-16 byte order (a NUL high byte) and they
///    outnumber the other byte order 4 : 1, the text is UTF-16 in that order.
///    UTF-8 and Latin-1 text contain no NUL bytes, and AGF records are ASCII,
///    so this holds even when a long header of non-ASCII comments comes first;
/// 3. otherwise UTF-8 if the bytes are valid UTF-8, else Latin-1.
pub fn decode_bytes(bytes: &[u8]) -> (String, TextEncoding) {
    match bytes {
        [0xEF, 0xBB, 0xBF, rest @ ..] => {
            return match std::str::from_utf8(rest) {
                Ok(s) => (s.to_string(), TextEncoding::Utf8Bom),
                Err(_) => (latin1(rest), TextEncoding::Latin1),
            };
        }
        [0xFF, 0xFE, rest @ ..] => return (decode_utf16(rest, false), TextEncoding::Utf16Le),
        [0xFE, 0xFF, rest @ ..] => return (decode_utf16(rest, true), TextEncoding::Utf16Be),
        _ => {}
    }
    let (mut le, mut be) = (0usize, 0usize);
    for &[lo, hi] in bytes.as_chunks::<2>().0 {
        match (lo, hi) {
            (a, 0) if a != 0 && a.is_ascii() => le += 1,
            (0, b) if b != 0 && b.is_ascii() => be += 1,
            _ => {}
        }
    }
    let units = bytes.len() / 2;
    if 20 * le >= units && le > 4 * be && le > 0 {
        return (decode_utf16(bytes, false), TextEncoding::Utf16Le);
    }
    if 20 * be >= units && be > 4 * le && be > 0 {
        return (decode_utf16(bytes, true), TextEncoding::Utf16Be);
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => (s.to_string(), TextEncoding::Utf8),
        Err(_) => (latin1(bytes), TextEncoding::Latin1),
    }
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
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

/// Reader options. In JSON every field is optional and unknown keys are
/// rejected.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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

    /// Glasses a discrete glass-substitution search may use: neither flagged
    /// "exclude substitution" nor obsolete. This metadata lives only here:
    /// [`GlassCatalog::from_agf_file`] and `MaterialLibrary` keep the glasses
    /// alone, so a caller that needs the filter keeps the `AgfCatalog` from
    /// [`read_agf_file`].
    pub fn substitution_candidates(&self) -> impl Iterator<Item = &Glass> {
        self.catalog.glasses.iter().filter_map(|(k, g)| {
            let m = self.meta.get(k);
            let excluded =
                m.is_some_and(|m| m.exclude_substitution || m.status == GlassStatus::Obsolete);
            (!excluded).then_some(g)
        })
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

/// Records whose numbers may continue on the following lines.
const WRAPPING_TAGS: [&str; 4] = ["ED", "CD", "TD", "OD"];

/// The non-empty, trimmed lines of `text` with their 1-based line numbers,
/// where a line that starts with a number (or `-`) and follows a numeric
/// record (`ED`, `CD`, `TD`, `OD`) is appended to that record. Manufacturer
/// catalogs wrap long `CD` and `TD` records over several lines this way (for
/// example the 10 `CD` coefficients of formula 13 over three lines).
fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    for (i, raw) in text.split(['\n', '\r']).enumerate() {
        let line = raw.trim_matches(|c: char| c.is_whitespace() || c == '\0');
        let Some(first) = line.split_whitespace().next() else {
            continue;
        };
        let numeric = first == "-" || first.parse::<f64>().is_ok();
        if numeric
            && let Some((_, prev)) = out.last_mut()
            && prev
                .split_whitespace()
                .next()
                .is_some_and(|t| WRAPPING_TAGS.contains(&t.to_ascii_uppercase().as_str()))
        {
            prev.push(' ');
            prev.push_str(line);
            continue;
        }
        out.push((i + 1, line.to_string()));
    }
    out
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

    for (ln, line) in logical_lines(&normalized) {
        let line = line.as_str();
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

/// Record tags the reader interprets. An unknown record written back verbatim
/// must not start with one of these, or it would change the catalog on re-read.
const KNOWN_TAGS: [&str; 9] = ["CC", "NM", "GC", "ED", "CD", "TD", "OD", "LD", "IT"];

/// Free text split at every line break the reader recognises (CR, LF, CRLF),
/// so that text written after a record tag can never start a new record.
fn text_lines(s: &str) -> Vec<String> {
    s.replace("\r\n", "\n")
        .split(['\r', '\n'])
        .map(str::to_string)
        .collect()
}

/// Collects the writer's [`AgfWarningKind::NotExportable`] warnings.
struct WriteLog(Vec<AgfWarning>);

impl WriteLog {
    fn warn(&mut self, glass: &str, message: String) {
        self.0.push(AgfWarning {
            line: None,
            glass: Some(glass.to_string()),
            kind: AgfWarningKind::NotExportable,
            message,
        });
    }

    /// `v` if it is finite. A non-finite value is reported and treated as unknown.
    fn finite(&mut self, glass: &str, what: &str, v: Option<f64>) -> Option<f64> {
        match v {
            Some(x) if !x.is_finite() => {
                self.warn(glass, format!("{what} is {x}; written as unknown"));
                None
            }
            v => v,
        }
    }
}

/// `-` for an unknown value. The reader maps `-` back to `None`.
fn dash(v: Option<f64>) -> String {
    v.map_or_else(|| "-".into(), num)
}

/// `-1` for an unknown `OD` value (the catalogs' own convention).
fn minus_one(v: Option<f64>) -> String {
    v.map_or_else(|| "-1".into(), num)
}

/// Writes a catalog as AGF text (UTF-8, CRLF line endings, as the format
/// customarily uses). Every number is written in its shortest exactly
/// round-tripping form, so write → read reproduces the glasses bit for bit.
///
/// What the writer cannot express exactly is reported as an
/// [`AgfWarningKind::NotExportable`] warning, never silently altered:
///
/// * glasses whose dispersion has no AGF form (see [`dispersion_to_agf`]) or
///   has non-finite coefficients are left out;
/// * whitespace in a glass name becomes `_`; if that collides with another
///   glass's name, a `_2`, `_3`, … suffix is added;
/// * unknown values are written as `-` in `ED` (so a missing CTE never reads
///   back as 0) and as −1 in `OD`. Non-finite values are written as unknown;
///   non-finite `n_d`/`V_d` as 0 (which disables the consistency check);
/// * free text (`CC`, `GC`) is split at every line break, a MIL code that is
///   empty or contains whitespace is replaced by the one computed from
///   `n_d`/`V_d`, and kept unknown records that span lines or use a known tag
///   are dropped, so no field can inject records;
/// * the "ignore thermal expansion" flag of the metadata is written only while
///   `Glass::cte` is `None`: an edited CTE wins over a stale flag;
/// * `LD` is omitted for an unrestricted range (`0…f64::MAX`, what the reader
///   produces without `LD`), and also (with a warning) for any other range
///   that is not a finite `0 ≤ min < max`.
///
/// `TD` is written only when the glass has thermal coefficients.
pub fn write_agf(cat: &AgfCatalog) -> (String, Vec<AgfWarning>) {
    use std::fmt::Write;
    let mut s = String::new();
    let mut log = WriteLog(Vec::new());
    let nl = "\r\n";
    for c in &cat.comments {
        for line in text_lines(c) {
            let _ = write!(s, "CC {line}{nl}");
        }
    }
    let sanitize = |n: &str| -> String {
        n.chars()
            .map(|c| if c.is_whitespace() { '_' } else { c })
            .collect()
    };
    let mut emitted: BTreeSet<String> = BTreeSet::new();
    let empty = AgfGlassMeta::default();
    for (key, g) in &cat.catalog.glasses {
        let meta = cat.meta.get(key).unwrap_or(&empty);
        let (formula, mut cd) = match dispersion_to_agf(&g.dispersion) {
            Ok(x) => x,
            Err(e) => {
                log.warn(&g.name, format!("not written: {e}"));
                continue;
            }
        };
        if let Some(x) = cd.iter().find(|x| !x.is_finite()) {
            log.warn(
                &g.name,
                format!("not written: dispersion coefficient {x} is not finite"),
            );
            continue;
        }
        let mut name = sanitize(&g.name);
        if name.is_empty() {
            name = sanitize(key);
        }
        if name.is_empty() {
            log.warn(&g.name, "not written: empty glass name".into());
            continue;
        }
        if emitted.contains(&name) {
            let base = name.clone();
            name = (2..)
                .map(|k| format!("{base}_{k}"))
                .find(|n| !emitted.contains(n) && !cat.catalog.glasses.contains_key(n))
                .expect("unbounded suffix search");
        }
        if name != g.name {
            log.warn(
                &g.name,
                format!(
                    "written as '{name}' (AGF names cannot contain whitespace and must be unique)"
                ),
            );
        }
        emitted.insert(name.clone());

        let nd = log.finite(&g.name, "nd", Some(g.nd)).unwrap_or(0.0);
        let vd = log.finite(&g.name, "vd", Some(g.vd)).unwrap_or(0.0);
        let mil = match &meta.mil_code {
            Some(m) if !m.is_empty() && !m.contains(char::is_whitespace) => m.clone(),
            other => {
                let computed = format!(
                    "{:03}{:03}",
                    ((nd - 1.0) * 1000.0).round().max(0.0) as i64,
                    (vd * 10.0).round().max(0.0) as i64
                );
                if let Some(m) = other {
                    log.warn(
                        &g.name,
                        format!("MIL code {m:?} is not a single token; written as {computed}"),
                    );
                }
                computed
            }
        };
        let _ = write!(
            s,
            "NM {name} {formula} {mil} {} {} {} {} {}{nl}",
            num(nd),
            num(vd),
            u8::from(meta.exclude_substitution),
            meta.status.code(),
            meta.melt_frequency.unwrap_or(0),
        );
        if !g.comment.is_empty() {
            for line in text_lines(&g.comment) {
                let _ = write!(s, "GC {line}{nl}");
            }
        }

        // An edited CTE overrides a stale "ignore thermal expansion" flag.
        let ignore = meta.ignore_thermal_expansion && g.cte.is_none();
        let cte = log.finite(&g.name, "CTE", g.cte);
        let raw_tce = log.finite(&g.name, "TCE -30/+70", meta.tce_m30_70);
        // Prefer the TCE exactly as read, so that it round-trips bit for bit.
        let tce = match (cte, raw_tce) {
            (Some(cte), Some(raw)) if raw / 1e6 == cte => Some(raw),
            (Some(cte), _) => Some(cte * 1e6),
            (None, raw) if ignore => raw,
            (None, _) => None,
        };
        let tce_hi = log.finite(&g.name, "TCE 100/300", meta.tce_100_300);
        let density = log.finite(&g.name, "density", g.density);
        let dpgf = log.finite(&g.name, "dPgF", meta.dpgf);
        if tce.is_some() || tce_hi.is_some() || density.is_some() || dpgf.is_some() || ignore {
            let _ = write!(
                s,
                "ED {} {} {} {} {}{nl}",
                dash(tce),
                dash(tce_hi),
                dash(density),
                dash(dpgf),
                u8::from(ignore),
            );
        }
        cd.resize(cd.len().max(CD_WIDTH), 0.0);
        let cd: Vec<String> = cd.into_iter().map(num).collect();
        let _ = write!(s, "CD {}{nl}", cd.join(" "));
        if let Some(th) = &g.thermal {
            let v = [
                th.d0,
                th.d1,
                th.d2,
                th.e0,
                th.e1,
                th.lambda_tk_um,
                th.t_ref_c,
            ];
            if v.iter().all(|x| x.is_finite()) {
                let v: Vec<String> = v.into_iter().map(num).collect();
                let _ = write!(s, "TD {}{nl}", v.join(" "));
            } else {
                log.warn(&g.name, "TD not written: non-finite coefficient".into());
            }
        }
        let o = &meta.ordering;
        // The glass's own (possibly edited) relative cost wins over the stored ordering data.
        let od = [
            ("relative cost", g.relative_cost.or(o.relative_cost)),
            ("climate resistance", o.climate_resistance),
            ("stain resistance", o.stain_resistance),
            ("acid resistance", o.acid_resistance),
            ("alkali resistance", o.alkali_resistance),
            ("phosphate resistance", o.phosphate_resistance),
        ]
        .map(|(what, v)| minus_one(log.finite(&g.name, what, v)));
        let _ = write!(s, "OD {}{nl}", od.join(" "));
        let (lo, hi) = (g.min_um, g.max_um);
        if lo.is_finite() && hi.is_finite() && hi < f64::MAX && lo >= 0.0 && hi > lo {
            let _ = write!(s, "LD {} {}{nl}", num(lo), num(hi));
        } else if !(lo == 0.0 && hi == f64::MAX) {
            log.warn(
                &g.name,
                format!("LD not written: range {lo}…{hi} µm is not a finite 0 ≤ min < max"),
            );
        }
        for t in &g.transmittance {
            if [t.um, t.tau, t.thickness_mm].iter().all(|x| x.is_finite()) {
                let _ = write!(
                    s,
                    "IT {} {} {}{nl}",
                    num(t.um),
                    num(t.tau),
                    num(t.thickness_mm)
                );
            } else {
                log.warn(
                    &g.name,
                    format!(
                        "IT point ({}, {}, {}) not written: not finite",
                        t.um, t.tau, t.thickness_mm
                    ),
                );
            }
        }
        for x in &meta.extra_records {
            let tag = x
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_ascii_uppercase();
            if x.contains(['\r', '\n']) || tag.is_empty() || KNOWN_TAGS.contains(&tag.as_str()) {
                log.warn(
                    &g.name,
                    format!("kept record {x:?} not written: it spans lines or uses a known tag"),
                );
            } else {
                let _ = write!(s, "{}{nl}", x.trim());
            }
        }
    }
    (s, log.0)
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

    /// Four glasses of the manufacturer catalog `NIKON-HIKARI_201911.AGF`
    /// (Nikon, AGF-format glass data, nikon.com, November 2019), copied
    /// record for record: formula 13 (J-FK5; J-SFH1 with a7, a8 ≠ 0) and
    /// formula 12 (E-LAKH1, E-KZFH1), with `CD` and `TD` wrapped over
    /// several lines as in the file. `docs/MATERIALS.md` lists the source.
    const NIKON_HIKARI: &str = "CC  Updated 11/2019\r
NM J-FK5 13 1 1.487490 70.318793 0 0 1\r
GC \t\t\t\t\t\t\t\t\t\r
ED 8.900000000E+000 0.000000000E+000 2.450000000E+000 2.700000000E-003 0 0\r
CD 2.188268550E+000 -9.190447240E-003 -1.116210710E-004 9.263728150E-003 \r
7.349007330E-005 4.197242420E-006 -1.154122030E-007 0.000000000E+000 \r
0.000000000E+000 0.000000000E+000\r
TD -8.080000000E-006 1.810000000E-008 -1.840000000E-012 3.750000000E-007 \r
3.020000000E-010 1.690000000E-001 2.300000000E+001\r
OD -1.00000 -1.00000 -1.00000 -1.00000 -1.00000 3.00000\r
LD 3.65015000E-001 2.05809000E+000\r
NM J-SFH1 13 1 1.808090 22.738831 0 0 3\r
ED 9.100000000E+000 0.000000000E+000 3.310000000E+000 2.260000000E-002 0 0\r
CD 3.116370390E+000 -1.401032520E-002 0.000000000E+000 4.552954590E-002 \r
3.791295070E-003 -5.762037930E-004 1.537939770E-004 -1.657800290E-005 \r
8.814805000E-007 0.000000000E+000\r
LD 3.88865000E-001 2.05809000E+000\r
NM E-LAKH1 12 1 1.748099 52.304982 0 2 3\r
CD 2.982439420E+000 -2.024229460E-003 2.679066790E-002 -6.323014370E-004 \r
1.108986210E-004 -4.158410800E-006 -4.957928360E-003 0.000000000E+000 \r
0.000000000E+000 0.000000000E+000\r
LD 4.00000000E-001 7.00000000E-001\r
NM E-KZFH1 12 1 1.612660 44.461379 0 2 3\r
CD 2.546629040E+000 -1.229723320E-002 1.874646230E-002 4.602965830E-004 \r
7.853510660E-007 1.727209720E-006 -1.334768060E-004 0.000000000E+000 \r
0.000000000E+000 0.000000000E+000\r
LD 4.00000000E-001 7.00000000E-001\r
";

    #[test]
    fn manufacturer_catalog_verifies_formulas_12_and_13() {
        const D: f64 = 0.5875618;
        const F: f64 = 0.4861327;
        const C: f64 = 0.6562725;
        let c = parse(NIKON_HIKARI);
        // Every glass reproduces the manufacturer's n_d and V_d (no
        // IndexMismatch), which needs the wrapped CD lines joined.
        assert_eq!(kinds(&c), vec![]);
        let check = |d: &Dispersion, nd: f64, vd: f64| {
            let (n_d, n_f, n_c) = (d.index(D), d.index(F), d.index(C));
            ((n_d - nd).abs(), ((n_d - 1.0) / (n_f - n_c) - vd).abs())
        };
        for (name, formula) in [
            ("J-FK5", "Extended3"),
            ("J-SFH1", "Extended3"),
            ("E-LAKH1", "Extended2"),
            ("E-KZFH1", "Extended2"),
        ] {
            let g = &c.catalog.glasses[name];
            assert_eq!(g.dispersion.formula_name(), formula);
            let (e_nd, e_vd) = check(&g.dispersion, g.nd, g.vd);
            assert!(e_nd < 1e-6 && e_vd < 2e-6, "{name}: {e_nd:e} {e_vd:e}");
        }
        // J-FK5's thermal record, also wrapped: T_ref = 23 °C.
        let th = c.catalog.glasses["J-FK5"].thermal.unwrap();
        assert_eq!((th.d0, th.e1, th.t_ref_c), (-8.08e-6, 3.02e-10, 23.0));
        // Permuted coefficient orders miss the catalog values by far more:
        // formula 12 with λ⁴ and λ⁶ swapped, formula 13 read as formula 12.
        let Dispersion::Extended2 { a } = c.catalog.glasses["E-LAKH1"].dispersion else {
            unreachable!()
        };
        let swapped = Dispersion::Extended2 {
            a: [a[0], a[1], a[2], a[3], a[4], a[5], a[7], a[6]],
        };
        assert!(check(&swapped, 1.748099, 52.304982).0 > 1e-4);
        let Dispersion::Extended3 { a } = c.catalog.glasses["J-SFH1"].dispersion else {
            unreachable!()
        };
        let as12 = Dispersion::Extended2 {
            a: a[..8].try_into().unwrap(),
        };
        assert!(check(&as12, 1.808090, 22.738831).0 > 1e-2);
        // Without joining the continuation lines the coefficients are
        // truncated (the bug this reader had): n_d off by 2.4e-4.
        let first_line_only = dispersion_from_agf(
            13,
            &[2.18826855, -9.19044724e-3, -1.11621071e-4, 9.26372815e-3],
        )
        .unwrap();
        assert!(check(&first_line_only, 1.48749, 70.318793).0 > 2e-4);
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
    fn options_json_rejects_unknown_keys() {
        let o: AgfOptions = serde_json::from_str(r#"{"mismatch": "skip"}"#).unwrap();
        assert_eq!(o.mismatch, MismatchPolicy::Skip);
        assert_eq!(o.nd_tolerance, AgfOptions::default().nd_tolerance);
        assert!(serde_json::from_str::<AgfOptions>(r#"{"nd_tolerence": 1e-3}"#).is_err());
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
    /// definitions with `cd = [k0, k1, …]`. Evaluated at λ = 2 µm (λ² = 4,
    /// exact binary powers) and, for the polynomial forms, also at λ = 0.5 µm,
    /// with distinct coefficients, so a wrong CD → coefficient mapping fails.
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
        // 4 Sellmeier2: n² = 1 + A + B1 λ²/(λ² − λ1²) + B2/(λ² − λ2²), CD = A B1 λ1 B2 λ2.
        let s2 = 1.0 + k[0] + k[1] * l2 / (l2 - k[2] * k[2]) + k[3] / (l2 - k[4] * k[4]);
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

        // Polynomial forms. `p(x)` is λ^x at the wavelength under test.
        for l in [2.0_f64, 0.5] {
            let p = |x: i32| l.powi(x);
            // 1 Schott: a0 + a1λ² + a2λ⁻² + a3λ⁻⁴ + a4λ⁻⁶ + a5λ⁻⁸.
            let schott =
                k[0] + k[1] * p(2) + k[2] * p(-2) + k[3] * p(-4) + k[4] * p(-6) + k[5] * p(-8);
            close(n(1, l), schott.sqrt());
            // 10 Extended: Schott + a6λ⁻¹⁰ + a7λ⁻¹².
            close(n(10, l), (schott + k[6] * p(-10) + k[7] * p(-12)).sqrt());
            // 12 Extended2: Schott + a6λ⁴ + a7λ⁶.
            close(n(12, l), (schott + k[6] * p(4) + k[7] * p(6)).sqrt());
            // 13 Extended3: a0 + a1λ² + a2λ⁴ + a3λ⁻² + … + a8λ⁻¹².
            let e3 = k[0]
                + k[1] * p(2)
                + k[2] * p(4)
                + k[3] * p(-2)
                + k[4] * p(-4)
                + k[5] * p(-6)
                + k[6] * p(-8)
                + k[7] * p(-10)
                + k[8] * p(-12);
            close(n(13, l), e3.sqrt());
            // 3 Herzberger: A + B L + C L² + Dλ² + Eλ⁴ + Fλ⁶, L = 1/(λ² − 0.028).
            let lh = 1.0 / (p(2) - 0.028);
            close(
                n(3, l),
                k[0] + k[1] * lh + k[2] * lh * lh + k[3] * p(2) + k[4] * p(4) + k[5] * p(6),
            );
        }
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

    /// N-BK7 alone, without an `ED` record.
    const NBK7_NO_ED: &str = "\
NM N-BK7 2 517642.251 1.5168 64.17 0 1 0
CD 1.03961212E+000 6.00069867E-003 2.31792344E-001 2.00179144E-002 1.01046945E+000 1.03560653E+002
LD 0.3 2.5
";

    #[test]
    fn writer_keeps_unknown_values_unknown() {
        let mut c = parse(NBK7_NO_ED);
        assert!(!c.meta["N-BK7"].ignore_thermal_expansion);
        c.catalog.glasses.get_mut("N-BK7").unwrap().density = Some(2.51);
        let (text, w) = write_agf(&c);
        assert!(w.is_empty(), "{w:?}");
        assert!(text.contains("ED - - 2.51 - 0\r\n"), "{text}");
        let back = parse(&text);
        let g = &back.catalog.glasses["N-BK7"];
        // A missing CTE must not come back as "does not expand".
        assert_eq!((g.cte, g.density), (None, Some(2.51)));
        let m = &back.meta["N-BK7"];
        assert_eq!((m.tce_m30_70, m.tce_100_300, m.dpgf), (None, None, None));
        assert_eq!(back.catalog, c.catalog);
    }

    #[test]
    fn edited_cte_overrides_stale_ignore_flag() {
        let text = NBK7_NO_ED.replace("CD ", "ED 7.1 8.3 2.51 -0.0009 1\nCD ");
        let c = parse(&text);
        assert_eq!(c.catalog.glasses["N-BK7"].cte, None);
        // Unedited: the flag and the raw TCE survive.
        let back = parse(&write_agf(&c).0);
        assert_eq!(back.meta, c.meta);
        assert_eq!(back.catalog.glasses["N-BK7"].cte, None);
        // Edited: the user's CTE wins and the flag is cleared.
        let mut edited = c.clone();
        edited.catalog.glasses.get_mut("N-BK7").unwrap().cte = Some(9e-6);
        let back = parse(&write_agf(&edited).0);
        assert_eq!(back.catalog.glasses["N-BK7"].cte, Some(9e-6));
        assert!(!back.meta["N-BK7"].ignore_thermal_expansion);
    }

    #[test]
    fn writer_cannot_inject_records() {
        let mut c = parse(NBK7_NO_ED);
        c.comments
            .push("hello\nNM EVIL 2 - 1.5 60\nCD 0 0 0 0 0 0".into());
        let m = c.meta.get_mut("N-BK7").unwrap();
        m.mil_code = Some("517 642\r\nNM EVIL2 2 - 1.5 60".into());
        m.extra_records = vec![
            "XY fine".into(),
            "XY a\rNM EVIL3 2 - 1.5 60".into(),
            "nm EVIL4 2 - 1.5 60".into(),
        ];
        let g = c.catalog.glasses.get_mut("N-BK7").unwrap();
        g.comment = "first\rNM EVIL5 2 - 1.5 60\r\nCD 1 0 0 0 0 0".into();
        let (text, w) = write_agf(&c);
        // The MIL code and the two bad extra records are reported.
        assert_eq!(w.len(), 3, "{w:?}");
        let back = parse(&text);
        assert_eq!(
            back.catalog.glasses.keys().collect::<Vec<_>>(),
            vec!["N-BK7"]
        );
        // Free text is preserved, split at its line breaks.
        assert_eq!(
            &back.comments[..],
            &["hello", "NM EVIL 2 - 1.5 60", "CD 0 0 0 0 0 0"]
        );
        assert_eq!(
            back.catalog.glasses["N-BK7"].comment,
            "first\nNM EVIL5 2 - 1.5 60\nCD 1 0 0 0 0 0"
        );
        let m = &back.meta["N-BK7"];
        assert_eq!(m.mil_code.as_deref(), Some("517642"));
        assert_eq!(m.extra_records, vec!["XY fine"]);
    }

    #[test]
    fn writer_resolves_name_collisions_and_rejects_non_finite_data() {
        let base = parse(NBK7_NO_ED).catalog.glasses["N-BK7"].clone();
        let mut cat = GlassCatalog::default();
        for name in ["MY GLASS", "MY_GLASS", "NANGLASS", "NAN_ND"] {
            let mut g = base.clone();
            g.name = name.into();
            cat.glasses.insert(name.into(), g);
        }
        let g = cat.glasses.get_mut("NANGLASS").unwrap();
        g.dispersion = Dispersion::Schott {
            a: [f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let g = cat.glasses.get_mut("NAN_ND").unwrap();
        g.nd = f64::NAN;
        g.density = Some(f64::INFINITY);
        let (text, w) = write_agf(&AgfCatalog::from_catalog(cat.clone()));
        let who: Vec<_> = w.iter().map(|w| w.glass.as_deref().unwrap()).collect();
        // "MY GLASS" sorts first and is written as "MY_GLASS", so the real
        // MY_GLASS is renamed; both renames are reported.
        assert_eq!(
            who,
            vec!["MY GLASS", "MY_GLASS", "NANGLASS", "NAN_ND", "NAN_ND"]
        );
        assert!(w[1].message.contains("'MY_GLASS_2'"), "{}", w[1].message);
        let back = parse(&text);
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
        assert_eq!(
            back.catalog.glasses.keys().collect::<Vec<_>>(),
            vec!["MY_GLASS", "MY_GLASS_2", "NAN_ND"]
        );
        let n = &back.catalog.glasses["NAN_ND"];
        assert_eq!((n.nd, n.density), (0.0, None));
        assert_eq!(n.dispersion, base.dispersion);
    }

    #[test]
    fn unrestricted_range_is_not_written() {
        let text = NBK7_NO_ED.replace("LD 0.3 2.5\n", "");
        let c = parse(&text);
        assert_eq!(c.catalog.glasses["N-BK7"].max_um, f64::MAX);
        let (out, w) = write_agf(&c);
        assert!(w.is_empty());
        assert!(!out.contains("LD"), "{out}");
        assert_eq!(parse(&out).catalog, c.catalog);
        // A half-open range cannot be expressed: reported, not written.
        let mut half = c.clone();
        half.catalog.glasses.get_mut("N-BK7").unwrap().min_um = 0.3;
        let (out, w) = write_agf(&half);
        assert!(!out.contains("LD") && w.len() == 1, "{out} {w:?}");
    }

    #[test]
    fn utf16_without_bom_and_non_ascii_header() {
        let header = "CC 光学玻璃目录，供应商说明。此行全部为非拉丁字符注释内容\r\n".repeat(40);
        let text = format!("{header}{}", NBK7_NO_ED.replace('\n', "\r\n"));
        for (enc, bom_free) in [(TextEncoding::Utf16Le, true), (TextEncoding::Utf16Be, true)] {
            let bytes = encode_text(&text, enc);
            let bytes = if bom_free { &bytes[2..] } else { &bytes[..] };
            let c = parse_agf_bytes("T", bytes, &AgfOptions::default()).unwrap();
            assert_eq!(c.encoding, Some(enc));
            assert_eq!(c.comments.len(), 40);
            assert!(c.comments[0].starts_with("光学"));
            assert!(c.catalog.glasses.contains_key("N-BK7"));
        }
        // A UTF-8 BOM followed by bytes that are not UTF-8 is decoded, and
        // reported, as Latin-1.
        let mut bytes = encode_text(NBK7_NO_ED, TextEncoding::Utf8Bom);
        bytes.extend_from_slice(b"CC 20 \xB0C\n");
        let c = parse_agf_bytes("T", &bytes, &AgfOptions::default()).unwrap();
        assert_eq!(c.encoding, Some(TextEncoding::Latin1));
        assert_eq!(c.comments, vec!["20 °C"]);
    }

    #[test]
    fn substitution_candidates_skip_obsolete_and_excluded() {
        let text = SNIPPET.replace("1.62004 36.37 0 0 0", "1.62004 36.37 1 0 0");
        let c = parse(&text);
        assert!(c.meta["F2"].exclude_substitution);
        let names: Vec<_> = c
            .substitution_candidates()
            .map(|g| g.name.as_str())
            .collect();
        // BK7_SCHOTT is obsolete (status 2) and F2 is excluded.
        assert_eq!(names, vec!["N-BK7"]);
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
