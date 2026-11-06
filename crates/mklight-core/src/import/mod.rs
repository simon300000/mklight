//! Importers for lens prescription files written by other programs:
//! `.zmx` files (keyword records, `zmx`) and `.seq` files (command scripts,
//! `seq`). `docs/IMPORT.md` lists what each format maps to.
//!
//! An import never fails on content it cannot map. Solves, pickups,
//! macros, non-sequential data, unknown surface types and unknown glasses
//! are dropped or replaced, and each one is reported as an
//! [`ImportWarning`]. Only text that is not a prescription at all (no
//! surfaces) is an error.
//!
//! Glasses resolve against a [`MaterialLibrary`] (the built-in catalog plus
//! whatever user catalogs are loaded). A glass that is not found becomes a
//! model glass when the file gives its `n_d`/`V_d`, and otherwise stays a
//! catalog reference by name, which resolves once a catalog containing it is
//! loaded.

pub mod seq;
pub mod zmx;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

use crate::error::{MkError, MkResult};
use crate::materials::agf::{TextEncoding, decode_bytes};
use crate::materials::{MaterialLibrary, MaterialOffset};
use crate::model::{
    ApertureSpec, MaterialRef, OpticalSystem, SemiDiameter, Surface, SurfaceId, SurfaceShape,
};

/// The prescription formats MKLight reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrescriptionFormat {
    /// `.zmx` keyword records.
    Zmx,
    /// `.seq` command scripts.
    Seq,
}

impl PrescriptionFormat {
    /// The format of a file extension (case-insensitive, without the dot).
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "zmx" => Some(Self::Zmx),
            "seq" => Some(Self::Seq),
            _ => None,
        }
    }
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|e| e.to_str())
            .and_then(Self::from_extension)
    }
    pub fn extension(self) -> &'static str {
        match self {
            Self::Zmx => "zmx",
            Self::Seq => "seq",
        }
    }
}

impl std::str::FromStr for PrescriptionFormat {
    type Err = MkError;
    fn from_str(s: &str) -> MkResult<Self> {
        Self::from_extension(s.trim_start_matches('.')).ok_or_else(|| {
            MkError::Unsupported(format!("prescription format '{s}' (expected zmx or seq)"))
        })
    }
}

/// What kind of content an import dropped or changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportWarningKind {
    /// A record or command the importer does not map. Dropped.
    UnsupportedRecord,
    /// A surface type MKLight has no equivalent for, or terms of a type it
    /// cannot hold. The surface keeps its base sphere or conic.
    UnsupportedSurface,
    /// A glass that is in none of the loaded catalogs.
    UnknownGlass,
    /// A solve, pickup or optimisation-variable flag. The current value is
    /// kept; the rule is dropped.
    SolveDropped,
    /// Non-sequential data. Dropped.
    NonSequential,
    /// A setting MKLight expresses differently, mapped approximately.
    Approximated,
    /// A line whose fields could not be read. Ignored.
    Malformed,
    /// Something the file does not state; a default was used.
    Defaulted,
    /// The imported system has a structural problem (CONVENTIONS §13).
    Validation,
}

/// A non-fatal problem found while importing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportWarning {
    pub kind: ImportWarningKind,
    /// 1-based line number in the file, when there is one.
    pub line: Option<usize>,
    /// Index of the surface in the imported system, when there is one.
    pub surface: Option<usize>,
    pub message: String,
}

impl fmt::Display for ImportWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(l) = self.line {
            write!(f, "line {l}: ")?;
        }
        if let Some(s) = self.surface {
            write!(f, "surface {s}: ")?;
        }
        f.write_str(&self.message)
    }
}

/// The result of an import: the system and everything that did not map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrescriptionImport {
    pub format: PrescriptionFormat,
    /// Encoding the text was decoded from.
    pub encoding: TextEncoding,
    pub system: OpticalSystem,
    pub warnings: Vec<ImportWarning>,
}

/// Imports prescription bytes in any common encoding (UTF-8, UTF-16 LE/BE
/// with or without a byte-order mark, Latin-1).
pub fn import_prescription(
    bytes: &[u8],
    format: PrescriptionFormat,
    library: &MaterialLibrary,
) -> MkResult<PrescriptionImport> {
    let (text, encoding) = decode_bytes(bytes);
    let (system, warnings) = match format {
        PrescriptionFormat::Zmx => zmx::parse(&text, library)?,
        PrescriptionFormat::Seq => seq::parse(&text, library)?,
    };
    Ok(PrescriptionImport {
        format,
        encoding,
        system,
        warnings,
    })
}

/// Reads and imports a prescription file; the format follows the extension.
pub fn read_prescription_file(
    path: impl AsRef<Path>,
    library: &MaterialLibrary,
) -> MkResult<PrescriptionImport> {
    let path = path.as_ref();
    let format = PrescriptionFormat::from_path(path).ok_or_else(|| {
        MkError::Unsupported(format!(
            "{}: not a lens prescription file (expected .zmx or .seq)",
            path.display()
        ))
    })?;
    let bytes = std::fs::read(path).map_err(|e| MkError::Io(format!("{}: {e}", path.display())))?;
    let mut imported = import_prescription(&bytes, format, library)?;
    if imported.system.name.is_empty()
        && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
    {
        imported.system.name = stem.to_string();
    }
    Ok(imported)
}

// ---------------------------------------------------------------------------
// Shared by both readers

/// Collects warnings; records that repeat are reported once, with a count.
#[derive(Default)]
pub(crate) struct Warnings {
    list: Vec<ImportWarning>,
    /// Index into `list` and count of each aggregated key.
    grouped: Vec<(String, usize, usize)>,
}

impl Warnings {
    pub fn push(
        &mut self,
        kind: ImportWarningKind,
        line: Option<usize>,
        surface: Option<usize>,
        message: impl Into<String>,
    ) {
        self.list.push(ImportWarning {
            kind,
            line,
            surface,
            message: message.into(),
        });
    }

    /// One warning per `key`; later occurrences only raise its count, which
    /// `finish` appends to the message ("(3 times)").
    pub fn once(
        &mut self,
        key: &str,
        kind: ImportWarningKind,
        line: Option<usize>,
        message: impl Into<String>,
    ) {
        if let Some(g) = self.grouped.iter_mut().find(|g| g.0 == key) {
            g.2 += 1;
            return;
        }
        self.grouped.push((key.to_string(), self.list.len(), 1));
        self.push(kind, line, None, message);
    }

    pub fn finish(mut self) -> Vec<ImportWarning> {
        for (_, i, n) in &self.grouped {
            if *n > 1 {
                self.list[*i].message.push_str(&format!(" ({n} times)"));
            }
        }
        // In file order; warnings about the whole system last.
        self.list.sort_by_key(|w| w.line.unwrap_or(usize::MAX));
        self.list
    }
}

/// Length units of a file, as a factor to mm.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Units(pub f64);

impl Units {
    pub const MM: Units = Units(1.0);

    /// A length.
    pub fn len(self, v: f64) -> f64 {
        v * self.0
    }
    /// A curvature (1/length).
    pub fn curv(self, c: f64) -> f64 {
        c / self.0
    }
    /// A sag coefficient on `r^power` (length^(1−power)).
    pub fn coef(self, a: f64, power: i32) -> f64 {
        a * self.0.powi(1 - power)
    }
}

/// Parses a number; `INFINITY`/`INF` with an optional sign are infinite.
pub(crate) fn number(tok: &str) -> Option<f64> {
    let t = tok.trim().trim_matches(',');
    let (neg, body) = match t.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if body.eq_ignore_ascii_case("infinity") || body.eq_ignore_ascii_case("inf") {
        return Some(if neg {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    // Fortran-style exponents ("1.0D+03") appear in older files.
    let v: f64 = t.replace(['D', 'd'], "E").parse().ok()?;
    v.is_finite().then_some(v)
}

/// `1/R`, with a zero or infinite radius meaning a plane.
pub(crate) fn curvature_of_radius(r: f64) -> f64 {
    if r == 0.0 || !r.is_finite() {
        0.0
    } else {
        1.0 / r
    }
}

/// Removes trailing zero coefficients.
pub(crate) fn trim_zeros(mut v: Vec<f64>) -> Vec<f64> {
    while v.last() == Some(&0.0) {
        v.pop();
    }
    v
}

/// The shape of a rotationally symmetric surface with optional even
/// coefficients starting at `r²` (`even[0]` on r², `even[1]` on r⁴, …).
/// An `r²` term has no place in `even_asphere` (whose series starts at r⁴),
/// so such a surface becomes an `odd_asphere` with only the even powers
/// set, which is the same sag.
pub(crate) fn rotational_shape(c: f64, k: f64, even: &[f64]) -> SurfaceShape {
    let even = trim_zeros(even.to_vec());
    if even.is_empty() {
        return if c == 0.0 && k == 0.0 {
            SurfaceShape::Plane
        } else {
            SurfaceShape::Conic {
                curvature: c,
                conic: k,
            }
        };
    }
    if even[0] == 0.0 {
        return SurfaceShape::EvenAsphere {
            curvature: c,
            conic: k,
            coefficients: even[1..].to_vec(),
        };
    }
    let mut odd = vec![0.0; 2 * even.len()];
    for (i, a) in even.iter().enumerate() {
        odd[2 * i + 1] = *a;
    }
    SurfaceShape::OddAsphere {
        curvature: c,
        conic: k,
        coefficients: odd,
    }
}

/// Looks glasses up in the loaded catalogs: first the catalog the file
/// names for the glass, then the catalogs the file prefers, then all.
pub(crate) struct GlassResolver<'a> {
    pub library: &'a MaterialLibrary,
    /// Catalog names the file lists (`GCAT`), in its order.
    pub preferred: Vec<String>,
}

impl GlassResolver<'_> {
    /// `(catalog name, glass name)` as stored in the library.
    fn find(&self, catalog: Option<&str>, name: &str) -> Option<(String, String)> {
        self.library
            .catalogs()
            .iter()
            .filter(|c| catalog.is_none_or(|n| c.name.eq_ignore_ascii_case(n)))
            .find_map(|c| {
                c.glasses
                    .keys()
                    .find(|k| k.eq_ignore_ascii_case(name))
                    .map(|k| (c.name.clone(), k.clone()))
            })
    }

    /// The material for glass `name` from catalog `vendor` (if the file
    /// says), with the file's own `n_d`/`V_d` as a fallback.
    pub fn resolve(
        &self,
        name: &str,
        vendor: Option<&str>,
        nd_vd: Option<(f64, f64)>,
        line: Option<usize>,
        surface: usize,
        warnings: &mut Warnings,
    ) -> MaterialRef {
        let found = vendor
            .and_then(|v| self.find(Some(v), name))
            .or_else(|| self.preferred.iter().find_map(|p| self.find(Some(p), name)))
            .or_else(|| self.find(None, name));
        if let Some((catalog, glass)) = found {
            return MaterialRef::Catalog {
                catalog: Some(catalog),
                name: glass,
                offset: MaterialOffset::default(),
            };
        }
        let label = match vendor {
            Some(v) => format!("{name} ({v})"),
            None => name.to_string(),
        };
        match nd_vd {
            Some((nd, vd)) if nd > 1.0 && vd > 0.0 => {
                warnings.push(
                    ImportWarningKind::UnknownGlass,
                    line,
                    Some(surface),
                    format!(
                        "glass {label} is not in the loaded catalogs; replaced by a model glass \
                         with n_d = {nd}, V_d = {vd}. Load its catalog and import again to use \
                         the catalog data."
                    ),
                );
                MaterialRef::Model { nd, vd, dpgf: 0.0 }
            }
            _ => {
                warnings.push(
                    ImportWarningKind::UnknownGlass,
                    line,
                    Some(surface),
                    format!(
                        "glass {label} is not in the loaded catalogs; kept by name. Load a \
                         catalog that contains it."
                    ),
                );
                MaterialRef::glass(name)
            }
        }
    }
}

/// What the readers hand to [`finish_system`].
pub(crate) struct Draft {
    pub name: String,
    pub notes: String,
    pub surfaces: Vec<Surface>,
    pub stop: Option<usize>,
    pub aperture: Option<ApertureSpec>,
    /// Semi-diameter the file gives the stop (used when the aperture floats
    /// by the stop size).
    pub stop_semi_diameter: Option<f64>,
}

/// Turns the readers' surfaces into a system: sequential ids, a stop, an
/// aperture, an image surface. Structural problems become warnings.
pub(crate) fn finish_system(
    mut sys: OpticalSystem,
    draft: Draft,
    warnings: &mut Warnings,
) -> MkResult<OpticalSystem> {
    let Draft {
        name,
        notes,
        mut surfaces,
        stop,
        aperture,
        stop_semi_diameter,
    } = draft;
    if surfaces.len() < 2 {
        return Err(MkError::Parse(
            "no lens surfaces found: the file is not a sequential prescription".into(),
        ));
    }
    if surfaces.len() == 2 {
        warnings.push(
            ImportWarningKind::Defaulted,
            None,
            None,
            "the file has no image surface; one was added",
        );
        surfaces.push(Surface::new(0, SurfaceShape::Plane, 0.0, MaterialRef::Air));
    }
    for (i, s) in surfaces.iter_mut().enumerate() {
        s.id = SurfaceId(i as u64);
    }
    let image = surfaces.len() - 1;
    sys.stop = match stop {
        Some(k) if k > 0 && k < image => k,
        other => {
            let msg = match other {
                Some(k) => format!(
                    "the stop is on surface {k}, which cannot be the stop; surface 1 is used"
                ),
                None => "the file marks no aperture stop; surface 1 is used".into(),
            };
            warnings.push(ImportWarningKind::Defaulted, None, None, msg);
            1
        }
    };
    if surfaces[0].label.is_empty() {
        surfaces[0].label = "OBJ".into();
    }
    if surfaces[sys.stop].label.is_empty() {
        surfaces[sys.stop].label = "STOP".into();
    }
    if surfaces[image].label.is_empty() {
        surfaces[image].label = "IMA".into();
    }
    sys.aperture = match aperture {
        Some(a) => a,
        None => {
            warnings.push(
                ImportWarningKind::Defaulted,
                None,
                None,
                "the file states no system aperture; an entrance pupil diameter of 10 mm is used",
            );
            ApertureSpec::default()
        }
    };
    if sys.aperture == ApertureSpec::FloatByStopSize {
        match stop_semi_diameter {
            Some(sd) if sd > 0.0 => {
                surfaces[sys.stop].semi_diameter = SemiDiameter::Fixed { value: sd };
            }
            _ => {
                warnings.push(
                    ImportWarningKind::Defaulted,
                    None,
                    Some(sys.stop),
                    "the aperture floats by the stop size, but the stop has no size; an \
                     entrance pupil diameter of 10 mm is used",
                );
                sys.aperture = ApertureSpec::default();
            }
        }
    }
    if matches!(sys.aperture, ApertureSpec::ObjectSpaceNa { .. })
        && surfaces[0].thickness.is_infinite()
    {
        warnings.push(
            ImportWarningKind::Defaulted,
            None,
            None,
            "an object-space NA needs a finite object; an entrance pupil diameter of 10 mm is used",
        );
        sys.aperture = ApertureSpec::default();
    }
    sys.name = name.trim().to_string();
    sys.notes = notes.trim().to_string();
    sys.surfaces = surfaces;
    sys.solves.retain(|s| match &s.target {
        crate::param::ParamPath::Thickness { surface } => surface.0 < image as u64,
        _ => true,
    });
    for issue in sys.validation_issues() {
        warnings.push(
            ImportWarningKind::Validation,
            None,
            None,
            format!("{}: {}", issue.path, issue.message),
        );
    }
    Ok(sys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(number("1.5"), Some(1.5));
        assert_eq!(number("-2.0E+01"), Some(-20.0));
        assert_eq!(number("1.0D+03"), Some(1000.0));
        assert_eq!(number("INFINITY"), Some(f64::INFINITY));
        assert_eq!(number("-Infinity"), Some(f64::NEG_INFINITY));
        assert_eq!(number("abc"), None);
        assert_eq!(number("nan"), None);
    }

    #[test]
    fn r2_terms_become_an_odd_asphere_with_the_same_sag() {
        let even = [1e-3, 2e-5, -3e-8];
        let shape = rotational_shape(0.02, -0.5, &even);
        let reference = |r2: f64| {
            crate::model::surface::conic_sag(0.02, -0.5, r2).unwrap()
                + even[0] * r2
                + even[1] * r2 * r2
                + even[2] * r2 * r2 * r2
        };
        assert!(matches!(shape, SurfaceShape::OddAsphere { .. }));
        for r in [0.5, 2.0, 7.0] {
            let r2: f64 = r * r;
            assert!((shape.sag(r2).unwrap() - reference(r2)).abs() < 1e-12);
        }
        assert!(matches!(
            rotational_shape(0.02, 0.0, &[0.0, 1e-5]),
            SurfaceShape::EvenAsphere { ref coefficients, .. } if coefficients == &[1e-5]
        ));
        assert_eq!(rotational_shape(0.0, 0.0, &[0.0, 0.0]), SurfaceShape::Plane);
    }

    #[test]
    fn unit_scaling() {
        let cm = Units(10.0);
        assert_eq!(cm.len(2.0), 20.0);
        assert_eq!(cm.curv(0.1), 0.01);
        // z = a r⁴ in cm: z_mm = 10·a·(r_mm/10)⁴.
        assert!((cm.coef(1.0, 4) - 1e-3).abs() < 1e-18);
    }

    #[test]
    fn repeated_warnings_are_counted() {
        let mut w = Warnings::default();
        w.once(
            "X",
            ImportWarningKind::UnsupportedRecord,
            Some(3),
            "X dropped",
        );
        w.once(
            "X",
            ImportWarningKind::UnsupportedRecord,
            Some(9),
            "X dropped",
        );
        w.once(
            "Y",
            ImportWarningKind::UnsupportedRecord,
            Some(4),
            "Y dropped",
        );
        let list = w.finish();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].message, "X dropped (2 times)");
        assert_eq!(list[0].line, Some(3));
        assert_eq!(list[1].message, "Y dropped");
    }

    #[test]
    fn formats_from_extensions() {
        assert_eq!(
            PrescriptionFormat::from_extension("ZMX"),
            Some(PrescriptionFormat::Zmx)
        );
        assert_eq!(
            PrescriptionFormat::from_path(Path::new("/a/b/lens.seq")),
            Some(PrescriptionFormat::Seq)
        );
        assert_eq!(
            PrescriptionFormat::from_path(Path::new("lens.mklight")),
            None
        );
        assert_eq!(
            ".zmx".parse::<PrescriptionFormat>().unwrap(),
            PrescriptionFormat::Zmx
        );
    }
}
