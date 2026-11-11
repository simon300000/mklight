//! Numeric upload boundary of aiming: small first-order field metadata, and chief
//! rows converted from the reference tracer's f64 field preparation (serial,
//! per field, and therefore host work). No pupil ray is traced here.
use super::*;
use bytemuck::{Pod, Zeroable};
use mklight_core::{FieldKind, RayAiming, ResolvedSystem};
use mklight_trace::{FieldPoint, ObjectSource, PreparedField, SequentialTracer};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FieldSpec {
    pub system: u32,
    pub wavelength: u32,
    pub flags: u32,
    pub reserved: u32,
    pub source: [Df32; 2],
    pub target: [Df32; 2],
    /// Entrance pupil z, aim scale (radius or tangent), stop radius.
    pub pupil: [Df32; 3],
    /// Decentre x/y, compression x/y.
    pub vignette: [Df32; 4],
    pub origin: [Df32; 3],
    pub rotation: [Df32; 9],
}
impl FieldSpec {
    pub fn new(
        sys: &ResolvedSystem,
        tracer: &SequentialTracer,
        point: FieldPoint,
        system: u32,
        wavelength: u32,
        aiming: mklight_core::RayAiming,
    ) -> MkResult<Self> {
        let fo = tracer.paraxial().primary();
        let model = sys.system();
        let target = [
            point.hx * model.fields.max_radial(),
            point.hy * model.fields.max_radial(),
        ];
        let finite = !sys.object_at_infinity;
        let tele = fo.object_space_telecentric();
        let mut flags = if finite { MKL_AIM_FINITE } else { 0 }
            | if tele { MKL_AIM_TELECENTRIC } else { 0 }
            | if matches!(aiming, RayAiming::Real { .. }) {
                MKL_AIM_REAL
            } else {
                0
            };
        let source = match model.fields.kind {
            FieldKind::ObjectAngle => {
                flags |= MKL_AIM_ANGLE;
                target.map(|x| x.to_radians().tan())
            }
            FieldKind::ObjectHeight => target,
            FieldKind::ParaxialImageHeight | FieldKind::RealImageHeight => {
                if model.fields.kind == FieldKind::RealImageHeight {
                    flags |= MKL_AIM_IMAGE_HEIGHT;
                }
                let h = if finite {
                    fo.image_height_per_object_height
                } else {
                    fo.image_height_per_tan
                };
                let h = h
                    .filter(|h| h.is_finite() && *h != 0.0)
                    .ok_or_else(|| bad("paraxial chief does not reach the image"))?;
                target.map(|x| x / h)
            }
        };
        let frame = mklight_trace::trace::start_frame(sys);
        let v = point.vignetting;
        let spec = Self {
            system,
            wavelength,
            flags,
            reserved: 0,
            source: source.map(Df32::from_f64),
            target: target.map(Df32::from_f64),
            pupil: [
                if tele {
                    0.0
                } else {
                    fo.entrance_pupil_position
                },
                if tele {
                    fo.marginal_object_slope.abs()
                } else {
                    fo.entrance_pupil_radius()
                },
                tracer.pupil().r_stop,
            ]
            .map(Df32::from_f64),
            vignette: [v.vdx, v.vdy, v.vcx, v.vcy].map(Df32::from_f64),
            origin: [frame.origin.x, frame.origin.y, frame.origin.z].map(Df32::from_f64),
            rotation: std::array::from_fn(|k| Df32::from_f64(frame.rot.m[k / 3][k % 3])),
        };
        if !bytemuck::cast_slice::<_, f32>(&[spec])[4..]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(bad("nonfinite aiming field metadata"));
        }
        Ok(spec)
    }
}

/// Explicit probes, lattice/rim nodes, or a caller's deterministic pupil
/// pattern. Their order and 64-bit IDs survive every operation and failure.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Sample {
    pub pupil: [Df32; 2],
    pub field: u32,
    pub id: [u32; 2],
    pub weight: f32,
}
impl Sample {
    pub fn new(pupil: [f64; 2], field: u32, id: u64) -> Self {
        Self {
            pupil: pupil.map(Df32::from_f64),
            field,
            id: [id as u32, (id >> 32) as u32],
            weight: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ChiefSeed {
    pub source: [Df32; 3],
    pub chief: [Df32; 2],
    pub jacobian: [Df32; 4],
    pub residual: Df32,
    pub status: u32,
    pub iterations: u32,
    pub has_jacobian: u32,
    pub reserved: u32,
}
impl ChiefSeed {
    /// The chief row of system wavelength `w` from the reference tracer's f64
    /// preparation of a field: its object source (the real-image-height solve
    /// for those fields), chief aim point, stop Jacobian and residual, i.e.
    /// what the device chief solve computes in double-float. The tracer must
    /// use the generation's aiming mode. An unusable field or wavelength gives
    /// an unreachable row, as a failed device solve does.
    pub fn host(tracer: &SequentialTracer, field: &PreparedField, w: usize) -> Self {
        let mut seed = ChiefSeed {
            status: MKL_AIM_FIELD_UNREACHABLE,
            ..Zeroable::zeroed()
        };
        let Some(a) = field.aims.get(w).filter(|_| field.is_valid()) else {
            return seed;
        };
        let source = match field.source {
            ObjectSource::Direction { dir } => dir,
            ObjectSource::Point { point } => point,
        };
        seed.source = [source.x, source.y, source.z].map(Df32::from_f64);
        seed.chief = a.chief.map(Df32::from_f64);
        if let Some(j) = a.jacobian {
            seed.jacobian = std::array::from_fn(|k| Df32::from_f64(j[k / 2][k % 2]));
            seed.has_jacobian = 1;
        }
        seed.residual = Df32::from_f64(a.residual * tracer.pupil().r_stop);
        seed.status = if a.converged {
            0
        } else {
            MKL_AIM_FAILED | ((tracer.system().stop as u32) << MKL_RAY_SURFACE_SHIFT)
        };
        seed
    }
}
/// Host chief rows for (field point, system wavelength) pairs. Points the
/// tracer already prepared (its system fields) are reused; the others are
/// solved once each, in parallel. A point that cannot be resolved gives
/// unreachable rows.
pub fn host_seeds(
    tracer: &SequentialTracer,
    rows: &[(FieldPoint, usize)],
) -> (Vec<ChiefSeed>, resident::HostPreparation) {
    use rayon::prelude::*;
    let start = std::time::Instant::now();
    let mut points: Vec<FieldPoint> = Vec::new();
    for (f, _) in rows {
        if !points.contains(f) {
            points.push(*f);
        }
    }
    let prepared: Vec<(PreparedField, bool)> = points
        .par_iter()
        .map(
            |&point| match tracer.fields().iter().find(|pf| pf.point == point) {
                Some(pf) => (pf.clone(), true),
                None => {
                    let pf = tracer
                        .prepare_field(point)
                        .unwrap_or_else(|e| PreparedField {
                            point,
                            source: ObjectSource::Direction {
                                dir: mklight_core::Vec3::Z,
                            },
                            aims: vec![],
                            error: Some(e.to_string()),
                        });
                    (pf, false)
                }
            },
        )
        .collect();
    let seeds = rows
        .iter()
        .map(|(f, w)| {
            let i = points.iter().position(|q| q == f).unwrap();
            ChiefSeed::host(tracer, &prepared[i].0, *w)
        })
        .collect();
    let reused = prepared.iter().filter(|p| p.1).count() as u64;
    (
        seeds,
        resident::HostPreparation {
            seed_rows: rows.len() as u64,
            reused_fields: reused,
            solved_fields: points.len() as u64 - reused,
            seconds: start.elapsed().as_secs_f64(),
        },
    )
}
/// One converged (status 0) or failed seed-table node: the aim parameters of
/// a lattice pupil point.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SeedNode {
    pub aim: [Df32; 2],
    pub status: u32,
    pub reserved: u32,
}
const _: () = assert!(size_of::<SeedNode>() == MKL_AIM_NODE_WORDS as usize * 4);
const _: () = assert!(size_of::<FieldSpec>() == MKL_AIM_FIELD_WORDS as usize * 4);
const _: () = assert!(size_of::<Sample>() == MKL_AIM_SAMPLE_WORDS as usize * 4);
const _: () = assert!(size_of::<ChiefSeed>() == MKL_AIM_CACHE_WORDS as usize * 4);

/// Host f64 seed tables of `side` nodes per axis for (field, wavelength)
/// rows, in row order: node aims solved by the reference tracer's own
/// schedule (chief prediction, then continuation) for the stop target of
/// each node's vignetted pupil coordinate. Nodes beyond pupil radius 1.25,
/// unreachable nodes and rows of unusable fields or fully compressed pupils
/// stay unsolved, as device nodes do. Rows are solved in parallel.
pub fn host_tables(
    tracer: &SequentialTracer,
    rows: &[(&PreparedField, usize)],
    side: u32,
) -> Vec<SeedNode> {
    use rayon::prelude::*;
    let side = side as usize;
    let nodes = side * side;
    let step = 2.25 / (side - 1) as f64;
    let unsolved = SeedNode {
        status: 1,
        ..Zeroable::zeroed()
    };
    let mut out = vec![unsolved; rows.len() * nodes];
    out.par_chunks_mut(nodes)
        .zip(rows)
        .for_each(|(row, &(field, w))| {
            let v = field.point.vignetting;
            if !field.is_valid() || v.vcx >= 1. || v.vcy >= 1. {
                return;
            }
            // One batched solve per row (`generate_many` is bitwise
            // `generate` per ray).
            let mut solved = Vec::with_capacity(nodes);
            let mut requests = Vec::with_capacity(nodes);
            for k in 0..nodes {
                let p = [
                    (k % side) as f64 * step - 1.125,
                    (k / side) as f64 * step - 1.125,
                ];
                if p[0] * p[0] + p[1] * p[1] > 1.5625 {
                    continue;
                }
                // The node lies in vignetted coordinates; `generate` applies
                // the field's vignetting, so hand it the pre-vignetting point.
                let px = (p[0] - v.vdx) / (1. - v.vcx);
                let py = (p[1] - v.vdy) / (1. - v.vcy);
                solved.push(k);
                requests.push((field, w, [px, py]));
            }
            for (k, g) in solved.into_iter().zip(tracer.generate_many(&requests)) {
                if let Ok(g) = g {
                    row[k].aim = g.aim.map(Df32::from_f64);
                    row[k].status = 0;
                }
            }
        });
    out
}
