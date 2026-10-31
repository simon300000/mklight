//! Layout scene for the viewport: every piece of geometry is computed here,
//! in **global** coordinates (mm), so a front end only projects and draws
//! polylines (it never computes optics).
//!
//! # Contents
//!
//! * **Surface profiles**: the selected local yz or xz section of z(x,y),
//!   `|y| ≤ semi-diameter`, of each optical surface, mapped with
//!   `ResolvedSurface::frame`; with `include_3d`, also the other section
//!   and `rings` concentric rings up to the semi-diameter (the outermost is the
//!   clear-aperture rim). Semi-diameters are the reference tracer's
//!   ([`SequentialTracer::semi_diameters`]: fixed, stop or traced); the stop
//!   uses `max(paraxial, traced)` as when apertures are frozen. Surfaces
//!   without optical effect (dummies, and planes between equal media such as
//!   a pure aperture stop) get no profile. A profile stops at the edge of a
//!   conic's sag domain.
//! * **Lens elements**: a refracting surface followed by glass and then by
//!   another surface forms an element `(j, j+1)` (a cemented doublet is two
//!   elements sharing a surface). Its outline joins the front and back
//!   profiles, sampled at matching heights up to the mechanical
//!   semi-diameter `H = max(mech_j, mech_{j+1})`, where `mech` is
//!   `EdgeSpec::mechanical_semi_diameter` or the clear semi-diameter plus
//!   `edge_margin`. Beyond its clear semi-diameter each profile continues as
//!   a flat at the sag of the clear edge (the edge flats), and the edge
//!   cylinder closes the outline (`element_edge` polylines).
//! * **Stop marker**: short radial strokes outside the stop's clear radius;
//!   **image plane**: the image surface's profile (and rim in 3-D); **axis**:
//!   the polyline through the surface vertices, which follows folds.
//! * **Rays**: for each selected field and wavelength, a tangential fan of
//!   `rays_per_field` rays (and a sagittal fan in 3-D), traced with clipping
//!   and per-surface records; each ray is the polyline of its surface hits
//!   (coordinate breaks skipped), ending at the clipping surface if it is
//!   vignetted. For an object at infinity the ray starts `lead_in` mm before
//!   its first hit.
//!
//! A polyline is `in_section` when all of its points lie in the global
//! section plane (x=0 for yz, y=0 for xz, to 10⁻⁹ of the scene size).
//! Optional polar triangle meshes carry the full Cartesian sag into 3-D.
//!
//! # Compact buffers
//!
//! [`LayoutScene::to_buffers`] flattens the scene into structure-of-arrays
//! form (one record per polyline, vertices as `f32` relative to the scene
//! centre, which keeps f32 precision local, plus the absolute `f64`
//! vertices); [`LayoutBuffers::to_le_bytes`] serialises that into one
//! little-endian blob for the C ABI (format in its docs).

use mklight_core::job::RunTimer;
use mklight_core::materials::ResolvedMaterial;
use mklight_core::math::Frame;
use mklight_core::{Interaction, JobContext, MkResult, RunInfo, SurfaceId, SurfaceShape, Vec3};
use mklight_trace::{SemiDiameterSettings, SemiDiameterSource, SequentialTracer, TraceStatus};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::util::{FieldSelection, WavelengthSelection, check_count, linspace, nf, run_info};

/// Local profile plane; section flags refer to the corresponding global plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutSection {
    #[default]
    Yz,
    Xz,
}
impl LayoutSection {
    fn point(self, h: f64) -> (f64, f64) {
        match self {
            Self::Yz => (0.0, h),
            Self::Xz => (h, 0.0),
        }
    }
    fn normal_axis(self) -> usize {
        match self {
            Self::Yz => 0,
            Self::Xz => 1,
        }
    }
    fn other(self) -> Self {
        match self {
            Self::Yz => Self::Xz,
            Self::Xz => Self::Yz,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayoutSettings {
    /// Rays per fan, evenly spaced across the pupil (`1`: the chief ray).
    pub rays_per_field: usize,
    pub section: LayoutSection,
    /// Radial subdivisions of the optional triangle mesh; zero disables it.
    pub mesh_rings: usize,
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    /// Points per surface profile.
    pub profile_samples: usize,
    /// Also produce out-of-section geometry (sagittal profiles, rings, rims,
    /// sagittal ray fans).
    #[serde(rename = "include_3d")]
    pub include_3d: bool,
    /// Concentric rings per surface in 3-D (the last one is the rim).
    pub rings: usize,
    /// Points per ring.
    pub ring_samples: usize,
    /// Mechanical margin (mm) beyond the clear semi-diameter where a surface
    /// has no `EdgeSpec::mechanical_semi_diameter`.
    pub edge_margin: f64,
    /// Ray length drawn before the first surface for an object at infinity
    /// (mm); a quarter of the vertex track when `None`.
    pub lead_in: Option<f64>,
}

impl Default for LayoutSettings {
    fn default() -> Self {
        LayoutSettings {
            rays_per_field: 7,
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::Primary,
            profile_samples: 41,
            section: LayoutSection::Yz,
            mesh_rings: 0,
            include_3d: true,
            rings: 3,
            ring_samples: 64,
            edge_margin: 0.5,
            lead_in: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolylineKind {
    SurfaceProfile,
    SurfaceRim,
    ElementEdge,
    Stop,
    ImagePlane,
    Ray,
    Axis,
}

impl PolylineKind {
    /// Names of the codes in [`LayoutBuffers::kinds`], indexed by code
    /// (`7` is an element outline).
    pub const CODE_NAMES: [&'static str; 8] = [
        "surface_profile",
        "surface_rim",
        "element_edge",
        "stop",
        "image_plane",
        "ray",
        "axis",
        "element_outline",
    ];

    /// Code in [`LayoutBuffers::kinds`] (`7` is an element outline).
    pub fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Polyline {
    pub kind: PolylineKind,
    /// Global points (mm).
    #[serde(with = "nf::vec3")]
    pub points: Vec<[f64; 3]>,
    pub in_section: bool,
    pub surface: Option<usize>,
    pub field: Option<usize>,
    pub wavelength: Option<usize>,
    /// Rays only: clipped (or failed) before the image.
    pub vignetted: bool,
}

/// A lens element's section: front and back profiles at matching heights
/// (`front[i]` and `back[i]` share a height), and the closed outline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub first_surface: usize,
    pub last_surface: usize,
    #[serde(with = "nf::vec3")]
    pub front: Vec<[f64; 3]>,
    #[serde(with = "nf::vec3")]
    pub back: Vec<[f64; 3]>,
    /// `front` (−H → H), then `back` (H → −H), closed.
    #[serde(with = "nf::vec3")]
    pub outline: Vec<[f64; 3]>,
    /// Mechanical semi-diameter `H` (mm).
    pub mechanical_semi_diameter: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceInfo {
    pub index: usize,
    pub id: SurfaceId,
    pub label: String,
    /// Global vertex position.
    #[serde(with = "nf::arr3")]
    pub vertex: [f64; 3],
    /// Global direction of the local `z` axis.
    pub axis: [f64; 3],
    /// Clear semi-diameter used for drawing (mm).
    pub semi_diameter: f64,
    pub semi_diameter_is_auto: bool,
    pub mechanical_semi_diameter: f64,
    pub is_mirror: bool,
    pub is_stop: bool,
    pub is_diffractive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bounds3 {
    #[serde(with = "nf::arr3")]
    pub min: [f64; 3],
    #[serde(with = "nf::arr3")]
    pub max: [f64; 3],
}

impl Bounds3 {
    fn empty() -> Self {
        Bounds3 {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }
    fn add(&mut self, p: [f64; 3]) {
        for ((lo, hi), v) in self.min.iter_mut().zip(&mut self.max).zip(p) {
            if v.is_finite() {
                *lo = lo.min(v);
                *hi = hi.max(v);
            }
        }
    }
    pub fn center(&self) -> [f64; 3] {
        std::array::from_fn(|k| 0.5 * (self.min[k] + self.max[k]))
    }
    pub fn size(&self) -> f64 {
        (0..3)
            .map(|k| self.max[k] - self.min[k])
            .fold(0.0, f64::max)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayoutScene {
    pub section: LayoutSection,
    pub meshes: Vec<SurfaceMesh>,
    pub polylines: Vec<Polyline>,
    pub elements: Vec<Element>,
    pub surfaces: Vec<SurfaceInfo>,
    pub bounds: Bounds3,
    pub wavelengths_um: Vec<f64>,
    pub field_count: usize,
    pub field_values: Vec<[f64; 2]>,
    pub run_info: RunInfo,
}

/// Absolute global triangle vertices (mm), with no shared-vertex indexing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMesh {
    pub surface: usize,
    pub triangles: Vec<[[f64; 3]; 3]>,
}

fn arr(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// Largest radius inside a shape's sag domain (∞ when unbounded): the
/// conic domain `(1 + k)c²r² ≤ 1` of each axis, and the normalisation
/// radius of normalised polynomials.
fn domain_radius(shape: &SurfaceShape) -> f64 {
    let conic = |c: f64, k: f64| {
        let a = (1.0 + k) * c * c;
        // Stay a hair inside the domain (the sag is singular-sloped there).
        if a > 0.0 {
            a.sqrt().recip() * (1.0 - 1e-12)
        } else {
            f64::INFINITY
        }
    };
    match *shape {
        SurfaceShape::Diffractive { ref base, .. } => domain_radius(base),
        SurfaceShape::QCon { norm_radius, .. }
        | SurfaceShape::QBfs { norm_radius, .. }
        | SurfaceShape::XYPolynomial { norm_radius, .. }
        | SurfaceShape::ZernikeSag { norm_radius, .. } => {
            norm_radius.min(conic(shape.curvature(), shape.conic()))
        }
        SurfaceShape::Toroid {
            curvature_x,
            curvature_y,
            conic_y,
            ..
        } => conic(curvature_x, 0.0).min(conic(curvature_y, conic_y)),
        SurfaceShape::Biconic {
            curvature_x,
            curvature_y,
            conic_x,
            conic_y,
            ..
        } => conic(curvature_x, conic_x).min(conic(curvature_y, conic_y)),
        _ => conic(shape.curvature(), shape.conic()),
    }
}

fn sag(shape: &SurfaceShape, x: f64, y: f64) -> f64 {
    mklight_core::surface_math::evaluate(shape, x, y).map_or(f64::NAN, |v| v.sag)
}

/// Local point at signed coordinate `h` in the section, flat beyond `clear`.
fn section_point(shape: &SurfaceShape, h: f64, clear: f64, section: LayoutSection) -> Vec3 {
    let (x, y) = section.point(h);
    let (sx, sy) = section.point(h.clamp(-clear, clear));
    Vec3::new(x, y, sag(shape, sx, sy))
}

/// Global point at local polar `(r, 2πk/m)`, flat beyond `clear`.
fn polar_point(
    f: &Frame,
    shape: &SurfaceShape,
    r: f64,
    clear: f64,
    k: usize,
    m: usize,
) -> [f64; 3] {
    let (s, c) = (std::f64::consts::TAU * (k % m) as f64 / m as f64).sin_cos();
    let rc = r.min(clear);
    arr(f.to_global_point(Vec3::new(r * c, r * s, sag(shape, rc * c, rc * s))))
}

/// Closed circle of `m` segments at radius `r`, flat beyond `clear`.
fn circle(f: &Frame, shape: &SurfaceShape, r: f64, clear: f64, m: usize) -> Vec<[f64; 3]> {
    (0..=m)
        .map(|k| polar_point(f, shape, r, clear, k, m))
        .collect()
}

fn polyline(kind: PolylineKind, pts: Vec<[f64; 3]>, surface: Option<usize>) -> Polyline {
    Polyline {
        kind,
        points: pts,
        in_section: false,
        surface,
        field: None,
        wavelength: None,
        vignetted: false,
    }
}

fn is_air(m: &ResolvedMaterial) -> bool {
    matches!(m, ResolvedMaterial::Air | ResolvedMaterial::Vacuum { .. })
}

/// Layout scene (module docs).
pub fn layout(tracer: &SequentialTracer, settings: &LayoutSettings) -> MkResult<LayoutScene> {
    layout_job(tracer, settings, &JobContext::new())
}

/// Display radii from the layout aperture solve, including domain clipping.
pub fn resolved_semi_diameters(
    tracer: &SequentialTracer,
    sd: &mklight_trace::semidia::SemiDiameters,
) -> Vec<f64> {
    let sys = tracer.system();
    let max_sd = sd.values.iter().copied().fold(0.0, f64::max);
    let floor = if max_sd > 0.0 { 0.05 * max_sd } else { 1.0 };
    (0..sys.num_surfaces())
        .map(|j| {
            let v = match sd.sources[j] {
                SemiDiameterSource::Stop => sd.values[j].max(sd.traced[j]),
                _ => sd.values[j],
            };
            let v = if v > 0.0 { v } else { floor };
            v.min(domain_radius(&sys.surfaces[j].shape))
        })
        .collect()
}

/// [`layout`] with cancellation and progress (geometry, then rays).
pub fn layout_job(
    tracer: &SequentialTracer,
    settings: &LayoutSettings,
    ctx: &JobContext,
) -> MkResult<LayoutScene> {
    let fields_tracer = crate::util::default_fields(tracer, &settings.fields)?;
    let tracer = fields_tracer.as_ref();
    let timer = RunTimer::start();
    let sys = tracer.system();
    check_count("mesh rings", settings.mesh_rings, 0, 100)?;
    check_count("rays per field", settings.rays_per_field, 1, 10_001)?;
    check_count("profile samples", settings.profile_samples, 3, 100_001)?;
    check_count("rings", settings.rings, 1, 100)?;
    check_count("ring samples", settings.ring_samples, 8, 100_000)?;
    if !(settings.edge_margin >= 0.0 && settings.edge_margin.is_finite()) {
        return Err(mklight_core::MkError::InvalidModel(
            "edge margin must be finite and ≥ 0".into(),
        ));
    }
    let fields = settings.fields.resolve(tracer.fields().len())?;
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let n = sys.num_surfaces();
    let img = sys.image_index();
    let sd = tracer.semi_diameters(&SemiDiameterSettings::default());
    let clear = resolved_semi_diameters(tracer, &sd);
    let mech: Vec<f64> = (0..n)
        .map(|j| {
            sys.system().surfaces[j]
                .edge
                .mechanical_semi_diameter
                .unwrap_or(clear[j] + settings.edge_margin)
                .max(clear[j])
                .min(domain_radius(&sys.surfaces[j].shape).max(clear[j]))
        })
        .collect();
    let frame = |j: usize| -> Frame { sys.surfaces[j].frame };
    let first = if sys.object_at_infinity { 1 } else { 0 };
    let mut polys: Vec<Polyline> = Vec::new();
    let mut elements = Vec::new();
    let mut infos = Vec::new();
    let mut meshes = Vec::new();
    let section = settings.section;
    let np = settings.profile_samples;
    let m = settings.ring_samples;
    let ring = |j: usize, r: f64| circle(&frame(j), &sys.surfaces[j].shape, r, r, m);

    for j in first..=img {
        ctx.check()?;
        let s = &sys.surfaces[j];
        let osurf = &sys.system().surfaces[j];
        if matches!(s.shape, SurfaceShape::CoordinateBreak) {
            continue;
        }
        let f = frame(j);
        infos.push(SurfaceInfo {
            index: j,
            id: s.id,
            label: osurf.label.clone(),
            vertex: arr(f.origin),
            axis: arr(f.to_global_dir(Vec3::Z)),
            semi_diameter: clear[j],
            semi_diameter_is_auto: matches!(
                sd.sources[j],
                SemiDiameterSource::Traced | SemiDiameterSource::Stop
            ) && s.fixed_semi_diameter.is_none()
                && s.aperture.is_none(),
            mechanical_semi_diameter: mech[j],
            is_mirror: s.is_mirror(),
            is_stop: j == sys.stop,
            is_diffractive: matches!(s.shape, SurfaceShape::Diffractive { .. }),
        });
        let same_media = j > 0
            && s.interaction == Interaction::Refract
            && matches!(s.shape, SurfaceShape::Plane)
            && (0..sys.wavelengths_um.len()).all(|w| sys.n_before(j, w) == s.n_after[w]);
        let no_effect = j > 0 && j < img && (s.interaction == Interaction::Dummy || same_media);
        let kind = if j == img {
            PolylineKind::ImagePlane
        } else {
            PolylineKind::SurfaceProfile
        };
        let c = clear[j];
        if !no_effect {
            let pts = linspace(-c, c, np)
                .into_iter()
                .map(|y| arr(f.to_global_point(section_point(&s.shape, y, c, section))))
                .collect();
            polys.push(polyline(kind, pts, Some(j)));
            if settings.include_3d {
                let pts = linspace(-c, c, np)
                    .into_iter()
                    .map(|x| arr(f.to_global_point(section_point(&s.shape, x, c, section.other()))))
                    .collect();
                polys.push(polyline(kind, pts, Some(j)));
                for k in 1..=settings.rings {
                    let r = c * k as f64 / settings.rings as f64;
                    let kk = if j == img {
                        PolylineKind::ImagePlane
                    } else {
                        PolylineKind::SurfaceRim
                    };
                    polys.push(polyline(kk, ring(j, r), Some(j)));
                }
            }
        }
        if settings.include_3d && settings.mesh_rings > 0 && !no_effect {
            let mut triangles = Vec::new();
            let point = |r: usize, k: usize| {
                let r = c * r as f64 / settings.mesh_rings as f64;
                polar_point(&f, &s.shape, r, r, k, m)
            };
            for r in 0..settings.mesh_rings {
                for k in 0..m {
                    let [a, b, c, d] =
                        [(r, k), (r + 1, k), (r + 1, k + 1), (r, k + 1)].map(|(r, k)| point(r, k));
                    // At the centre (r = 0) a = d: one triangle per sector.
                    let quad = [[a, b, c], [a, c, d]];
                    triangles.extend(
                        quad[..if r == 0 { 1 } else { 2 }]
                            .iter()
                            .filter(|t| t.iter().flatten().all(|v| v.is_finite())),
                    );
                }
            }
            meshes.push(SurfaceMesh {
                surface: j,
                triangles,
            });
        }
        if j == sys.stop {
            let ext = c + (0.3 * c).max(settings.edge_margin);
            for sgn in [1.0, -1.0] {
                let pts = vec![
                    arr(f.to_global_point(section_point(&s.shape, sgn * c, c, section))),
                    arr(f.to_global_point(section_point(&s.shape, sgn * ext, c, section))),
                ];
                polys.push(polyline(PolylineKind::Stop, pts, Some(j)));
            }
            if settings.include_3d {
                polys.push(polyline(PolylineKind::Stop, ring(j, c), Some(j)));
            }
        }
        // Lens element (j, j+1).
        let next = j + 1;
        if j >= 1
            && next < img
            && s.interaction == Interaction::Refract
            && !is_air(&s.medium_after)
            && !sys.surfaces[next].is_passive()
        {
            let s2 = &sys.surfaces[next];
            let h = mech[j].max(mech[next]);
            let f2 = frame(next);
            let ys = linspace(-h, h, np);
            let front: Vec<[f64; 3]> = ys
                .iter()
                .map(|&y| arr(f.to_global_point(section_point(&s.shape, y, clear[j], section))))
                .collect();
            let back: Vec<[f64; 3]> = ys
                .iter()
                .map(|&y| {
                    arr(f2.to_global_point(section_point(&s2.shape, y, clear[next], section)))
                })
                .collect();
            let mut outline = front.clone();
            outline.extend(back.iter().rev());
            outline.push(front[0]);
            for sgn in [1.0, -1.0] {
                let p = |fr: &Frame, sh: &SurfaceShape, y: f64, cl: f64| {
                    arr(fr.to_global_point(section_point(sh, y, cl, section)))
                };
                let pts = vec![
                    p(&f, &s.shape, sgn * clear[j], clear[j]),
                    p(&f, &s.shape, sgn * h, clear[j]),
                    p(&f2, &s2.shape, sgn * h, clear[next]),
                    p(&f2, &s2.shape, sgn * clear[next], clear[next]),
                ];
                polys.push(polyline(PolylineKind::ElementEdge, pts, Some(j)));
            }
            if settings.include_3d {
                for (fr, sh, cl) in [(&f, &s.shape, clear[j]), (&f2, &s2.shape, clear[next])] {
                    let pts = circle(fr, sh, h, cl, m);
                    polys.push(polyline(PolylineKind::ElementEdge, pts, Some(j)));
                }
            }
            elements.push(Element {
                first_surface: j,
                last_surface: next,
                front,
                back,
                outline,
                mechanical_semi_diameter: h,
            });
        }
    }
    // Axis through the vertices.
    let axis: Vec<[f64; 3]> = (first..=img)
        .filter(|&j| !matches!(sys.surfaces[j].shape, SurfaceShape::CoordinateBreak))
        .map(|j| arr(frame(j).origin))
        .collect();
    let track = axis
        .windows(2)
        .map(|w| {
            ((w[1][0] - w[0][0]).powi(2)
                + (w[1][1] - w[0][1]).powi(2)
                + (w[1][2] - w[0][2]).powi(2))
            .sqrt()
        })
        .sum::<f64>();
    polys.push(polyline(PolylineKind::Axis, axis, None));

    ctx.check()?;
    ctx.report(0.3, "layout geometry");
    // Rays.
    let lead = settings.lead_in.unwrap_or(0.25 * track).max(1e-3);
    let fan = linspace(-1.0, 1.0, settings.rays_per_field);
    let fan = if settings.rays_per_field == 1 {
        vec![0.0]
    } else {
        fan
    };
    let mut jobs = Vec::new();
    for &f in &fields {
        for &w in &wls {
            for &p in &fan {
                let (px, py) = section.point(p);
                jobs.push((f, w, px, py));
            }
            if settings.include_3d {
                for &p in &fan {
                    if p != 0.0 {
                        let (px, py) = section.other().point(p);
                        jobs.push((f, w, px, py));
                    }
                }
            }
        }
    }
    let opts = tracer.options();
    let rays: Vec<Polyline> = jobs
        .par_iter()
        .map(|&(f, w, px, py)| {
            let mut rec = Vec::with_capacity(n);
            let r = tracer.trace_with(tracer.field(f), w, px, py, &opts, Some(&mut rec));
            let mut pts = Vec::with_capacity(n + 1);
            for (i, sr) in rec.iter().enumerate() {
                if i == 0 {
                    if !sys.object_at_infinity {
                        pts.push(arr(frame(0).to_global_point(sr.point)));
                    }
                    continue;
                }
                if matches!(sys.surfaces[i].shape, SurfaceShape::CoordinateBreak) {
                    continue;
                }
                pts.push(arr(frame(i).to_global_point(sr.point)));
            }
            if !r.status.is_ok()
                && let TraceStatus::Clipped { surface }
                | TraceStatus::TotalInternalReflection { surface } = r.status
            {
                pts.push(arr(frame(surface).to_global_point(r.ray.pos)));
            }
            if sys.object_at_infinity && !rec.is_empty() {
                // Global (entrance-frame) direction of the incoming ray.
                let d = rec[0].dir;
                let start = match pts.first() {
                    Some(p) => Vec3::new(p[0], p[1], p[2]) - d * lead,
                    None => rec[0].point,
                };
                pts.insert(0, arr(start));
                if pts.len() == 1 {
                    pts.push(arr(start + d * (2.0 * lead)));
                }
            }
            Polyline {
                kind: PolylineKind::Ray,
                points: pts,
                in_section: false,
                surface: None,
                field: Some(f),
                wavelength: Some(w),
                vignetted: !r.status.is_ok(),
            }
        })
        .collect();
    ctx.check()?;
    polys.extend(rays);
    polys.retain(|p| p.points.len() >= 2 || p.kind != PolylineKind::Ray);

    let mut bounds = Bounds3::empty();
    for p in &polys {
        p.points.iter().for_each(|&q| bounds.add(q));
    }
    for e in &elements {
        e.outline.iter().for_each(|&q| bounds.add(q));
    }
    for point in meshes.iter().flat_map(|m| &m.triangles).flatten() {
        bounds.add(*point);
    }
    if !bounds.min[0].is_finite() {
        bounds = Bounds3 {
            min: [0.0; 3],
            max: [0.0; 3],
        };
    }
    let tol = 1e-9 * (1.0 + bounds.size());
    for p in &mut polys {
        p.in_section = p
            .points
            .iter()
            .all(|q| q[section.normal_axis()].abs() <= tol);
    }
    ctx.report(1.0, "layout");
    Ok(LayoutScene {
        section,
        meshes,
        polylines: polys,
        elements,
        surfaces: infos,
        bounds,
        wavelengths_um: sys.wavelengths_um.clone(),
        field_count: tracer.fields().len(),
        field_values: (0..tracer.fields().len())
            .map(|i| crate::util::field_value(tracer, i))
            .collect(),
        run_info: run_info(timer, tracer, "layout", settings, None, Vec::new()),
    })
}

// ----- Compact buffers --------------------------------------------------------------------------

/// Structure-of-arrays form of a [`LayoutScene`]: polylines first, then one
/// closed outline per element (kind code 7, `surface` = first surface).
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutBuffers {
    /// Subtracted from every `f32` vertex (the scene's bounding-box centre).
    pub origin: [f64; 3],
    /// `x, y, z` per vertex, relative to `origin` (mm).
    pub vertices: Vec<f32>,
    /// `x, y, z` per vertex, absolute (mm).
    pub vertices_f64: Vec<f64>,
    /// Polyline `i` is vertices `offsets[i]..offsets[i+1]` (vertex units).
    pub offsets: Vec<u32>,
    /// [`PolylineKind::code`], or 7 for an element outline.
    pub kinds: Vec<u8>,
    /// Surface, field and wavelength index, `−1` for none.
    pub surfaces: Vec<i32>,
    pub fields: Vec<i32>,
    pub wavelengths: Vec<i32>,
    /// Bit 0: in selected global section; bit 1: vignetted ray; bit 2: DOE.
    pub flags: Vec<u8>,
}

/// Kind code of an element outline in [`LayoutBuffers::kinds`].
pub const ELEMENT_OUTLINE_KIND: u8 = 7;

impl LayoutScene {
    /// Flatten into [`LayoutBuffers`].
    pub fn to_buffers(&self) -> LayoutBuffers {
        let origin = self.bounds.center();
        let mut b = LayoutBuffers {
            origin,
            vertices: Vec::new(),
            vertices_f64: Vec::new(),
            offsets: vec![0],
            kinds: Vec::new(),
            surfaces: Vec::new(),
            fields: Vec::new(),
            wavelengths: Vec::new(),
            flags: Vec::new(),
        };
        let idx = |v: Option<usize>| v.map_or(-1, |x| x as i32);
        let mut push = |pts: &[[f64; 3]], kind: u8, s: i32, f: i32, w: i32, flags: u8| {
            for p in pts {
                for k in 0..3 {
                    b.vertices.push((p[k] - origin[k]) as f32);
                    b.vertices_f64.push(p[k]);
                }
            }
            b.offsets.push((b.vertices_f64.len() / 3) as u32);
            b.kinds.push(kind);
            b.surfaces.push(s);
            b.fields.push(f);
            b.wavelengths.push(w);
            b.flags.push(flags);
        };
        let doe = |first: usize, last: usize| {
            self.surfaces
                .iter()
                .any(|s| (first..=last).contains(&s.index) && s.is_diffractive)
        };
        for p in &self.polylines {
            let flags = p.in_section as u8
                | (p.vignetted as u8) << 1
                | (p.surface.is_some_and(|j| doe(j, j)) as u8) << 2;
            push(
                &p.points,
                p.kind.code(),
                idx(p.surface),
                idx(p.field),
                idx(p.wavelength),
                flags,
            );
        }
        let tol = 1e-9 * (1.0 + self.bounds.size());
        for e in &self.elements {
            let in_section = e
                .outline
                .iter()
                .all(|q| q[self.section.normal_axis()].abs() <= tol);
            let flags = in_section as u8 | (doe(e.first_surface, e.last_surface) as u8) << 2;
            push(
                &e.outline,
                ELEMENT_OUTLINE_KIND,
                e.first_surface as i32,
                -1,
                -1,
                flags,
            );
        }
        b
    }
}

impl LayoutBuffers {
    pub fn polyline_count(&self) -> usize {
        self.kinds.len()
    }

    /// One little-endian blob:
    ///
    /// | bytes | content |
    /// |---|---|
    /// | 4 | magic `b"MKLL"` |
    /// | 4 | `u32` format version (1) |
    /// | 4 | `u32` polyline count `P` |
    /// | 4 | `u32` vertex count `V` |
    /// | 24 | `f64 × 3` origin |
    /// | 4(P+1) | `u32` offsets |
    /// | 4P each | `i32` surfaces, fields, wavelengths |
    /// | P, P | `u8` kinds, `u8` flags, then zero padding to a multiple of 4 |
    /// | 12V | `f32 × 3` vertices relative to the origin |
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let p = self.polyline_count();
        let v = self.vertices.len() / 3;
        let mut out = Vec::with_capacity(40 + 16 * p + 12 * v + 8);
        out.extend_from_slice(b"MKLL");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(p as u32).to_le_bytes());
        out.extend_from_slice(&(v as u32).to_le_bytes());
        for o in self.origin {
            out.extend_from_slice(&o.to_le_bytes());
        }
        for o in &self.offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        for a in [&self.surfaces, &self.fields, &self.wavelengths] {
            for x in a.iter() {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.kinds);
        out.extend_from_slice(&self.flags);
        while out.len() % 4 != 0 {
            out.push(0);
        }
        for x in &self.vertices {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod code_name_tests {
    use super::PolylineKind::{self, *};

    #[test]
    fn code_names_follow_the_enum() {
        for k in [
            SurfaceProfile,
            SurfaceRim,
            ElementEdge,
            Stop,
            ImagePlane,
            Ray,
            Axis,
        ] {
            let name = serde_json::to_value(k).unwrap();
            assert_eq!(
                PolylineKind::CODE_NAMES[k.code() as usize],
                name.as_str().unwrap()
            );
        }
    }
}
