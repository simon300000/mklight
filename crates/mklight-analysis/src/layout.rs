//! Layout scene for the viewport: every piece of geometry is computed here,
//! in **global** coordinates (mm), so a front end only projects and draws
//! polylines (it never computes optics).
//!
//! # Contents
//!
//! * **Surface profiles**: the meridional section `(0, y, sag(y²))`,
//!   `|y| ≤ semi-diameter`, of each optical surface, mapped with
//!   `ResolvedSurface::frame`; with `include_3d`, also the sagittal section
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
//! `x = 0` plane (to 10⁻⁹ of the scene size), i.e. it can be drawn in the 2-D
//! meridional view.
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayoutSettings {
    /// Rays per fan, evenly spaced across the pupil (`1`: the chief ray).
    pub rays_per_field: usize,
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
    pub polylines: Vec<Polyline>,
    pub elements: Vec<Element>,
    pub surfaces: Vec<SurfaceInfo>,
    pub bounds: Bounds3,
    pub wavelengths_um: Vec<f64>,
    pub field_count: usize,
    pub run_info: RunInfo,
}

fn arr(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// Largest radius inside a shape's sag domain (∞ when unbounded).
fn domain_radius(shape: &SurfaceShape) -> f64 {
    let (c, k) = (shape.curvature(), shape.conic());
    let a = (1.0 + k) * c * c;
    if a > 0.0 {
        // Stay a hair inside the domain (the sag is singular-sloped there).
        (1.0 / a).sqrt() * (1.0 - 1e-12)
    } else {
        f64::INFINITY
    }
}

fn sag(shape: &SurfaceShape, r: f64) -> f64 {
    shape.sag(r * r).unwrap_or(f64::NAN)
}

/// Local point at height `y` in the meridional section, flat beyond `clear`.
fn section_point(shape: &SurfaceShape, y: f64, clear: f64) -> Vec3 {
    let r = y.abs().min(clear);
    Vec3::new(0.0, y, sag(shape, r))
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

/// [`layout`] with cancellation and progress (geometry, then rays).
pub fn layout_job(
    tracer: &SequentialTracer,
    settings: &LayoutSettings,
    ctx: &JobContext,
) -> MkResult<LayoutScene> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    if sys.surfaces.iter().any(|s| !s.shape.is_legacy()) {
        return Err(mklight_core::MkError::Unsupported(
            "extended surface shape in layout".into(),
        ));
    }
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
    let max_sd = sd.values.iter().copied().fold(0.0, f64::max);
    let floor = if max_sd > 0.0 { 0.05 * max_sd } else { 1.0 };
    let clear: Vec<f64> = (0..n)
        .map(|j| {
            let v = match sd.sources[j] {
                SemiDiameterSource::Stop => sd.values[j].max(sd.traced[j]),
                _ => sd.values[j],
            };
            let v = if v > 0.0 { v } else { floor };
            v.min(domain_radius(&sys.surfaces[j].shape))
        })
        .collect();
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
    let np = settings.profile_samples;
    let ring = |j: usize, r: f64| -> Vec<[f64; 3]> {
        let m = settings.ring_samples;
        let z = sag(&sys.surfaces[j].shape, r);
        (0..=m)
            .map(|k| {
                let (s, c) = (2.0 * std::f64::consts::PI * (k % m) as f64 / m as f64).sin_cos();
                arr(frame(j).to_global_point(Vec3::new(r * c, r * s, z)))
            })
            .collect()
    };

    for j in first..=img {
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
                .map(|y| arr(f.to_global_point(Vec3::new(0.0, y, sag(&s.shape, y.abs())))))
                .collect();
            polys.push(polyline(kind, pts, Some(j)));
            if settings.include_3d {
                let pts = linspace(-c, c, np)
                    .into_iter()
                    .map(|x| arr(f.to_global_point(Vec3::new(x, 0.0, sag(&s.shape, x.abs())))))
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
        if j == sys.stop {
            let z = sag(&s.shape, c);
            let ext = c + (0.3 * c).max(settings.edge_margin);
            for sgn in [1.0, -1.0] {
                let pts = vec![
                    arr(f.to_global_point(Vec3::new(0.0, sgn * c, z))),
                    arr(f.to_global_point(Vec3::new(0.0, sgn * ext, z))),
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
                .map(|&y| arr(f.to_global_point(section_point(&s.shape, y, clear[j]))))
                .collect();
            let back: Vec<[f64; 3]> = ys
                .iter()
                .map(|&y| arr(f2.to_global_point(section_point(&s2.shape, y, clear[next]))))
                .collect();
            let mut outline = front.clone();
            outline.extend(back.iter().rev());
            outline.push(front[0]);
            for sgn in [1.0, -1.0] {
                let p = |fr: &Frame, sh: &SurfaceShape, y: f64, cl: f64| {
                    arr(fr.to_global_point(section_point(sh, y, cl)))
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
                    let z = sag(sh, cl);
                    let m = settings.ring_samples;
                    let pts = (0..=m)
                        .map(|k| {
                            let (sn, cs) =
                                (2.0 * std::f64::consts::PI * (k % m) as f64 / m as f64).sin_cos();
                            arr(fr.to_global_point(Vec3::new(h * cs, h * sn, z)))
                        })
                        .collect();
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
                jobs.push((f, w, 0.0, p));
            }
            if settings.include_3d {
                for &p in &fan {
                    if p != 0.0 {
                        jobs.push((f, w, p, 0.0));
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
    if !bounds.min[0].is_finite() {
        bounds = Bounds3 {
            min: [0.0; 3],
            max: [0.0; 3],
        };
    }
    let tol = 1e-9 * (1.0 + bounds.size());
    for p in &mut polys {
        p.in_section = p.points.iter().all(|q| q[0].abs() <= tol);
    }
    ctx.report(1.0, "layout");
    Ok(LayoutScene {
        polylines: polys,
        elements,
        surfaces: infos,
        bounds,
        wavelengths_um: sys.wavelengths_um.clone(),
        field_count: tracer.fields().len(),
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
    /// Bit 0: in section; bit 1: vignetted ray.
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
        for p in &self.polylines {
            let flags = p.in_section as u8 | (p.vignetted as u8) << 1;
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
            let flags = e.outline.iter().all(|q| q[0].abs() <= tol) as u8;
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
