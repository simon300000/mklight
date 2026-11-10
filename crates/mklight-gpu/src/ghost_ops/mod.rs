//! Ghost sequence geometry and separate s/p transport. Path expansion and
//! coverage belong to the solver; this service consumes validated steps.
mod kernel;
pub use crate::resident::record_desc;
use crate::{Df32, pending::JobControl, radiometry::*, resident::*};
use bytemuck::{Pod, Zeroable};
pub use kernel::trace;
use mklight_core::{MkError, MkResult};

/// The largest integer a leaf word carries exactly: [`Request::Gather`]
/// passes a leaf's map index and its reference bin + 1 to
/// [`Request::PixelKeys`] in one f32 word each.
pub const LEAF_WORD_MAX: u64 = 1 << 24;

/// Whether [`Request::PixelKeys`] bins exactly: every map index below `maps`
/// and every reference bin + 1 up to `bins[0] · bins[1]` fits a leaf word.
/// Larger maps belong to the CPU reference.
pub fn leaf_words_exact(bins: [u64; 2], maps: u64) -> bool {
    maps.saturating_sub(1) <= LEAF_WORD_MAX
        && bins[0]
            .checked_mul(bins[1])
            .is_some_and(|pixels| pixels <= LEAF_WORD_MAX)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Step {
    pub word: u32,
    pub physical: u32,
    pub coating: u32,
    /// 0 transparent, 1 transmission, 2 reflection, 3 sensor transmission,
    /// 4 sensor reflection. Image and passive interactions skip coating queries.
    pub channel: u32,
    pub scale: Df32,
    pub efficiency: Df32,
    pub alpha: Df32,
    /// Ideal-lens power in the incident traversal (inverse on backward visits).
    pub power: Df32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Probe {
    pub position: [Df32; 3],
    pub direction: [Df32; 3],
    pub path: u32,
    pub wavelength: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Value {
    pub position: [Df32; 3],
    pub direction: [Df32; 3],
    pub weight_s: Df32,
    pub weight_p: Df32,
    pub internal: Df32,
    pub status: u32,
    pub done: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LaunchSample {
    pub pupil: [Df32; 2],
    pub lattice: [i32; 2],
    pub field: u32,
    pub path: u32,
    pub wavelength: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Leaf {
    pub row: u32,
    pub destination: u32,
    pub weight: Df32,
    pub batch: u32,
    pub map: u32,
    /// Zero: device binning; otherwise reference bin + 1 (edge ambiguity).
    pub pixel: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FirstOrderStep {
    pub power: Df32,
    pub index: Df32,
    pub distance: Df32,
    pub efficiency: Df32,
    pub absorption: Df32,
    pub coating: u32,
    pub channel: u32,
}
// Byte sizes of the matching MSL structs in mklight_ghost_ops_types.h
// (float2 members, 8-byte aligned).
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<Step>() == 48 && size_of::<Probe>() == 56);
    assert!(size_of::<Value>() == 80 && size_of::<LaunchSample>() == 40);
    assert!(size_of::<Leaf>() == 32 && size_of::<FirstOrderStep>() == 48);
};
#[derive(Clone, Debug)]
pub enum Request {
    MomentTerms {
        hits: PlanView,
        output: BufferId,
    },
    SegmentFlags {
        keys: PlanView,
        bins: usize,
        output: BufferId,
    },
    PixelSegments {
        keys: PlanView,
        prefix: PlanView,
        bins: usize,
        offsets: BufferId,
        pixels: BufferId,
        count: BufferId,
    },
    MapScatter {
        values: PlanView,
        pixels: PlanView,
        output: BufferId,
    },
    FirstOrder {
        steps: PlanView,
        offsets: PlanView,
        initial: PlanView,
        coatings: PlanView,
        sensor_reflectance: Df32,
        output: BufferId,
        throughput: BufferId,
        status: BufferId,
    },
    Gather {
        sources: Vec<PlanView>,
        leaves: PlanView,
        output: BufferId,
    },
    PixelKeys {
        hits: PlanView,
        bins: [u32; 2],
        maps: usize,
        half: [Df32; 2],
        keys: BufferId,
        ids: BufferId,
        weights: BufferId,
    },
    Launch {
        chiefs: PlanView,
        fields: PlanView,
        samples: PlanView,
        output: BufferId,
    },
    Trace {
        systems: crate::SystemHandle,
        probes: PlanView,
        /// Path offsets into steps; steps are wavelength-major, with equal
        /// path offsets in each wavelength plane.
        offsets: PlanView,
        steps: PlanView,
        wavelengths: usize,
        reject_virtual: bool,
        sensor: [Df32; 2],
        newton: crate::NewtonSettings,
        /// Separate s/p powers folded at every completed step; `None` traces
        /// geometry only (unit weights).
        coatings: Option<TraceCoatings>,
        output: BufferId,
    },
}
/// Coating records the ghost trace evaluates at each completed step's
/// incidence with the radiometry coating arithmetic, folding s/p powers, efficiency and
/// absorption in step order. No per-step query or coating value is stored.
#[derive(Clone, Debug)]
pub struct TraceCoatings {
    pub specs: PlanView,
    pub layers: PlanView,
    pub sensor_reflectance: Df32,
}
fn records<T>(v: &PlanView, p: &ResidentPlan) -> MkResult<usize> {
    check_records::<T>(v, p, "ghost")
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    let (probes, offsets, steps, wavelengths, output) = match r {
        Request::MomentTerms { hits, output } => {
            let n = hits.desc.shape.first().copied().unwrap_or(0);
            if hits.desc != BufferDesc::new(ElementType::DoubleFloat, [n, 5])
                || hits.stride != 1
                || p.desc(*output)? != &BufferDesc::new(ElementType::DoubleFloat, [3, n])
            {
                return Err(bad("ghost moment terms"));
            }
            return Ok(());
        }

        Request::SegmentFlags { keys, output, .. } => {
            if keys.desc.element != ElementType::U32
                || keys.desc.shape.len() != 1
                || keys.stride != 1
                || p.desc(*output)? != &keys.desc
            {
                return Err(bad("ghost segment flags"));
            }
            return Ok(());
        }
        Request::PixelSegments {
            keys,
            prefix,
            offsets,
            pixels,
            count,
            ..
        } => {
            let n = keys.desc.len()?;
            if keys.desc != BufferDesc::new(ElementType::U32, [n])
                || prefix.desc != keys.desc
                || keys.stride != 1
                || prefix.stride != 1
                || p.desc(*pixels)? != &keys.desc
                || p.desc(*offsets)? != &BufferDesc::new(ElementType::U32, [n + 1])
                || p.desc(*count)? != &BufferDesc::new(ElementType::U32, [2])
            {
                return Err(bad("ghost pixel segments"));
            }
            return Ok(());
        }
        Request::MapScatter {
            values,
            pixels,
            output,
        } => {
            let n = pixels.desc.len()?;
            if pixels.desc != BufferDesc::new(ElementType::U32, [n])
                || pixels.stride != 1
                || values.stride != 1
                || values.desc != BufferDesc::new(ElementType::DoubleFloat, [1, n])
                || p.desc(*output)?.element != ElementType::DoubleFloat
                || p.desc(*output)?.shape.len() != 1
            {
                return Err(bad("ghost map scatter"));
            }
            return Ok(());
        }

        Request::FirstOrder {
            steps,
            offsets,
            initial,
            coatings,
            sensor_reflectance: _,
            output,
            throughput,
            status,
        } => {
            let ns = records::<FirstOrderStep>(steps, p)?;
            records::<CoatingValue>(coatings, p)?;
            let n = offsets.desc.len()?.saturating_sub(1);
            if offsets.desc != BufferDesc::new(ElementType::U32, [n + 1])
                || offsets.stride != 1
                || initial.desc != BufferDesc::new(ElementType::DoubleFloat, [n, 4])
                || initial.stride != 1
                || p.desc(*output)? != &BufferDesc::new(ElementType::DoubleFloat, [ns, 4])
                || p.desc(*throughput)? != &BufferDesc::new(ElementType::DoubleFloat, [n, 2])
                || p.desc(*status)? != &BufferDesc::new(ElementType::U32, [n])
            {
                return Err(bad("ghost first order layout"));
            }
            return Ok(());
        }
        Request::Gather {
            sources,
            leaves,
            output,
        } => {
            let n = records::<Leaf>(leaves, p)?;
            for source in sources {
                records::<Value>(source, p)?;
            }
            if p.desc(*output)? != &BufferDesc::new(ElementType::DoubleFloat, [n, 5]) {
                return Err(bad("ghost gather output"));
            }
            return Ok(());
        }
        Request::PixelKeys {
            hits,
            bins,
            maps,
            half,
            keys,
            ids,
            weights,
        } => {
            let n = hits.desc.shape.first().copied().unwrap_or(0);
            if hits.desc != BufferDesc::new(ElementType::DoubleFloat, [n, 5])
                || hits.stride != 1
                || bins.contains(&0)
                || *maps == 0
                || !half
                    .iter()
                    .all(|v| v.to_f64().is_finite() && v.to_f64() > 0.)
                || p.desc(*keys)? != &BufferDesc::new(ElementType::U32, [n])
                || p.desc(*ids)? != &BufferDesc::new(ElementType::U32, [n])
                || p.desc(*weights)? != &BufferDesc::new(ElementType::DoubleFloat, [n])
            {
                return Err(bad("ghost pixel records"));
            }
            // A support limit, not a malformed plan: the request runs on
            // the CPU reference.
            if !leaf_words_exact(bins.map(u64::from), *maps as u64) {
                return Err(MkError::Unsupported(format!(
                    "ghost pixel keys beyond exact f32 leaf words \
                     ({LEAF_WORD_MAX} pixels per map, or as many maps)"
                )));
            }
            // Keys are u32: `map · bins + pixel`, and `maps · bins` for a miss.
            let pixels = u64::from(bins[0]) * u64::from(bins[1]);
            if (*maps as u64)
                .checked_mul(pixels)
                .is_none_or(|keys| keys >= u64::from(u32::MAX))
            {
                return Err(bad("ghost pixel keys exceed u32"));
            }
            return Ok(());
        }
        Request::Launch {
            chiefs,
            fields,
            samples,
            output,
        } => {
            let nf = records::<crate::aiming::FieldSpec>(fields, p)?;
            let n = records::<LaunchSample>(samples, p)?;
            if chiefs.desc != BufferDesc::new(ElementType::F32, [15, nf])
                || chiefs.stride != 1
                || p.desc(*output)? != &record_desc::<Probe>(n)
            {
                return Err(bad("ghost launch layout"));
            }
            return Ok(());
        }
        Request::Trace {
            probes,
            offsets,
            steps,
            wavelengths,
            output,
            coatings,
            newton,
            ..
        } => {
            if let Some(c) = coatings {
                records::<CoatingSpec>(&c.specs, p)?;
                records::<CoatingLayer>(&c.layers, p)?;
            }
            if !newton.tolerance_abs.is_finite()
                || newton.tolerance_abs < 0.
                || !newton.tolerance_rel.is_finite()
                || newton.tolerance_rel < 0.
            {
                return Err(bad("ghost trace settings"));
            }
            (probes, offsets, steps, wavelengths, output)
        }
    };
    let n = records::<Probe>(probes, p)?;
    let ns = records::<Step>(steps, p)?;
    if *wavelengths == 0
        || ns % wavelengths != 0
        || offsets.desc.element != ElementType::U32
        || offsets.desc.shape.len() != 1
        || offsets.desc.len()? < 2
        || offsets.stride != 1
        || p.desc(*output)? != &record_desc::<Value>(n)
    {
        return Err(bad("ghost path layout"));
    }
    Ok(())
}
pub fn accesses(r: &Request) -> Access {
    match r {
        Request::MomentTerms { hits, output } => Access {
            reads: vec![hits.clone()],
            writes: vec![*output],
            systems: vec![],
        },
        Request::SegmentFlags { keys, output, .. } => Access {
            reads: vec![keys.clone()],
            writes: vec![*output],
            systems: vec![],
        },
        Request::PixelSegments {
            keys,
            prefix,
            offsets,
            pixels,
            count,
            ..
        } => Access {
            reads: vec![keys.clone(), prefix.clone()],
            writes: vec![*offsets, *pixels, *count],
            systems: vec![],
        },
        Request::MapScatter {
            values,
            pixels,
            output,
        } => Access {
            reads: vec![values.clone(), pixels.clone()],
            writes: vec![*output],
            systems: vec![],
        },
        Request::FirstOrder {
            steps,
            offsets,
            initial,
            coatings,
            sensor_reflectance: _,
            output,
            throughput,
            status,
        } => Access {
            reads: vec![
                steps.clone(),
                offsets.clone(),
                initial.clone(),
                coatings.clone(),
            ],
            writes: vec![*output, *throughput, *status],
            systems: vec![],
        },
        Request::Gather {
            sources,
            leaves,
            output,
        } => {
            let mut reads = sources.clone();
            reads.push(leaves.clone());
            Access {
                reads,
                writes: vec![*output],
                systems: vec![],
            }
        }
        Request::PixelKeys {
            hits,
            keys,
            ids,
            weights,
            ..
        } => Access {
            reads: vec![hits.clone()],
            writes: vec![*keys, *ids, *weights],
            systems: vec![],
        },
        Request::Launch {
            chiefs,
            fields,
            samples,
            output,
        } => Access {
            reads: vec![chiefs.clone(), fields.clone(), samples.clone()],
            writes: vec![*output],
            systems: vec![],
        },
        Request::Trace {
            systems,
            probes,
            offsets,
            steps,
            coatings,
            output,
            ..
        } => {
            let mut reads = vec![probes.clone(), offsets.clone(), steps.clone()];
            if let Some(c) = coatings {
                reads.extend([c.specs.clone(), c.layers.clone()]);
            }
            Access {
                reads,
                writes: vec![*output],
                systems: vec![*systems],
            }
        }
    }
}
pub fn execute(r: &Request, x: &mut cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    if x.flush_to_zero {
        execute_in::<crate::Ftz>(r, x, ctl)
    } else {
        execute_in::<crate::Ieee>(r, x, ctl)
    }
}
fn execute_in<E: crate::FpEnv>(
    r: &Request,
    x: &mut cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    match r {
        Request::MomentTerms { hits, output } => {
            let n = hits.desc.shape[0];
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let v = x.rows::<Df32>(hits);
                let row = [v[5 * i], v[5 * i + 1], v[5 * i + 2]];
                let out = bytemuck::cast_slice_mut::<_, Df32>(x.words_mut(*output));
                for k in 0..3 {
                    out[k * n + i] = row[k];
                }
            }
        }

        Request::SegmentFlags { keys, bins, output } => {
            for i in 0..keys.desc.len()? {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let ks = x.rows::<u32>(keys);
                let flag = ks[i] < *bins as u32 && (i == 0 || ks[i] != ks[i - 1]);
                x.words_mut(*output)[i] = flag as u32;
            }
        }
        Request::PixelSegments {
            keys,
            prefix,
            bins,
            offsets,
            pixels,
            count,
        } => {
            let n = keys.desc.len()?;
            let total = x.rows::<u32>(prefix).last().copied().unwrap_or(0);
            let end = x.rows::<u32>(keys).partition_point(|&k| k < *bins as u32);
            x.words_mut(*count)[0] = total;
            x.words_mut(*count)[1] = 0;
            x.words_mut(*offsets)[total as usize] = end as u32;
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let ks = x.rows::<u32>(keys);
                if ks[i] >= *bins as u32 || (i > 0 && ks[i] == ks[i - 1]) {
                    continue;
                }
                let pixel = ks[i];
                let end = ks.partition_point(|&k| k <= pixel);
                let dest = x.rows::<u32>(prefix)[i] as usize - 1;
                x.words_mut(*offsets)[dest] = i as u32;
                x.words_mut(*pixels)[dest] = pixel;
                x.words_mut(*count)[1] = x.words(*count)[1].max((end - i) as u32);
            }
        }
        Request::MapScatter {
            values,
            pixels,
            output,
        } => {
            x.words_mut(*output).fill(0);
            for i in 0..pixels.desc.len()? {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let pixel = x.rows::<u32>(pixels)[i] as usize;
                let value = x.rows::<Df32>(values)[i];
                bytemuck::cast_slice_mut::<_, Df32>(x.words_mut(*output))[pixel] = value;
            }
        }

        Request::FirstOrder {
            steps,
            offsets,
            initial,
            coatings,
            sensor_reflectance,
            output,
            throughput,
            status,
        } => {
            for i in 0..offsets.desc.len()? - 1 {
                if i % 256 == 0 {
                    ctl.check()?;
                }
                let offsets = x.rows::<u32>(offsets);
                let (a, z) = (offsets[i] as usize, offsets[i + 1] as usize);
                use crate::{
                    consts::*,
                    fx::{D, F, f},
                };
                let (mut reflectance, mut transmission, mut absorption) =
                    (D::<E>::from_f(F::ONE), D::<E>::from_f(F::ONE), D::<E>::ZERO);
                let sensor = D::from_df(*sensor_reflectance);
                let mut reference = false;
                let mut v: [crate::fx::D<E>; 4] = std::array::from_fn(|k| {
                    crate::fx::D::from_df(x.rows::<Df32>(initial)[4 * i + k])
                });
                for k in a..z {
                    let st = x.rows::<FirstOrderStep>(steps)[k];
                    let phi = crate::fx::D::from_df(st.power);
                    let index = crate::fx::D::from_df(st.index);
                    let dz = crate::fx::D::from_df(st.distance);
                    transmission = transmission * D::from_df(st.efficiency);
                    absorption = absorption + D::from_df(st.absorption);
                    match st.channel {
                        MKL_GHOST_AXIAL_SENSOR => reflectance = reflectance * sensor,
                        MKL_GHOST_AXIAL_DETECT => {
                            transmission = transmission * (D::from_f(F::ONE) - sensor)
                        }
                        MKL_GHOST_AXIAL_TRANSMIT
                        | MKL_GHOST_AXIAL_REFLECT
                        | MKL_GHOST_AXIAL_MIRROR => {
                            let c = x.rows::<CoatingValue>(coatings)[st.coating as usize];
                            if c.status != MKL_RAD_OK {
                                reference = true;
                            } else {
                                let r = (D::from_df(c.powers[0]) + D::from_df(c.powers[1])) * 0.5;
                                let t = (D::from_df(c.powers[2]) + D::from_df(c.powers[3])) * 0.5;
                                match st.channel {
                                    MKL_GHOST_AXIAL_REFLECT => reflectance = reflectance * r,
                                    MKL_GHOST_AXIAL_MIRROR => transmission = transmission * r,
                                    _ => transmission = transmission * t,
                                }
                            }
                        }
                        _ => {}
                    }

                    v[1] = v[1] - v[0] * phi;
                    v[3] = v[3] - v[2] * phi;
                    let out = bytemuck::cast_slice_mut::<_, Df32>(x.words_mut(*output));
                    for j in 0..4 {
                        out[4 * k + j] = v[j].0;
                    }
                    v[0] = v[0] + (dz / index) * v[1];
                    v[2] = v[2] + (dz / index) * v[3];
                }
                transmission = transmission
                    * D::from_f(f::<E>(crate::fmath::exp_in::<E>((-absorption).hi().v())));
                reference |= v[1].hi().abs() <= f32::from_bits(MKL_GHOST_FIRSTORDER_SLOPE_BITS);
                reference |= !reflectance.0.is_finite()
                    || !transmission.0.is_finite()
                    || v.iter().any(|v| !v.0.is_finite());
                let out = bytemuck::cast_slice_mut::<_, Df32>(x.words_mut(*throughput));
                out[2 * i] = reflectance.0;
                out[2 * i + 1] = transmission.0;
                x.words_mut(*status)[i] = if reference { MKL_GHOST_REFERENCE } else { 0 };
                x.stats.fallback_count += reference as u64;
            }
        }
        Request::Gather {
            sources,
            leaves,
            output,
        } => {
            for i in 0..leaves.desc.shape[0] {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let l = x.rows::<Leaf>(leaves)[i];
                let v = x.rows::<Value>(&sources[l.batch as usize])[l.row as usize];
                let w = kernel::leaf_weight::<E>(v, l.weight);
                let data = bytemuck::cast_slice_mut::<_, Df32>(x.words_mut(*output));
                data[5 * l.destination as usize..][..5].copy_from_slice(&[
                    v.position[0],
                    v.position[1],
                    w,
                    Df32::from_f32(l.map as f32),
                    Df32::from_f32(l.pixel as f32),
                ]);
            }
        }
        Request::PixelKeys {
            hits,
            bins,
            maps,
            half,
            keys,
            ids,
            weights,
        } => {
            for i in 0..hits.desc.shape[0] {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let h = x.rows::<Df32>(hits);
                let pixel = if h[5 * i + 4].hi > 0. {
                    h[5 * i + 4].hi as u32 - 1
                } else {
                    kernel::pixel::<E>([h[5 * i], h[5 * i + 1]], *bins, *half)
                };
                let key = if pixel == bins[0] * bins[1] {
                    (*maps as u32) * bins[0] * bins[1]
                } else {
                    h[5 * i + 3].hi as u32 * bins[0] * bins[1] + pixel
                };
                let w = h[5 * i + 2];
                x.words_mut(*keys)[i] = key;
                x.words_mut(*ids)[i] = i as u32;
                bytemuck::cast_slice_mut::<_, Df32>(x.words_mut(*weights))[i] = w;
            }
        }
        Request::Launch {
            chiefs,
            fields,
            samples,
            output,
        } => {
            for i in 0..samples.desc.shape[0] {
                if i % 1024 == 0 {
                    ctl.check()?;
                }
                let q = x.rows::<LaunchSample>(samples)[i];
                let fields = x.rows::<crate::aiming::FieldSpec>(fields);
                let field = fields
                    .get(q.field as usize)
                    .ok_or_else(|| bad("ghost launch field"))?;
                let data = x.rows::<f32>(chiefs);
                let n = fields.len();
                let j = q.field as usize;
                let pos =
                    std::array::from_fn(|k| Df32::new(data[k * n + j], data[(9 + k) * n + j]));
                let dir = std::array::from_fn(|k| {
                    Df32::new(data[(3 + k) * n + j], data[(12 + k) * n + j])
                });
                let v = kernel::launch::<E>(field, q, pos, dir);
                bytemuck::cast_slice_mut::<_, Probe>(x.words_mut(*output))[i] = v;
            }
        }
        Request::Trace {
            systems,
            probes,
            offsets,
            steps,
            wavelengths,
            reject_virtual,
            sensor,
            newton,
            coatings,
            output,
        } => {
            let s = x.system(*systems)?;
            let n = probes.desc.shape[0];
            for i in 0..n {
                if i % 256 == 0 {
                    ctl.check()?;
                }
                let q = x.rows::<Probe>(probes)[i];
                let os = x.rows::<u32>(offsets);
                let ss = x.rows::<Step>(steps);
                if q.path as usize + 1 >= os.len() || q.wavelength as usize >= *wavelengths {
                    return Err(bad("ghost probe index"));
                }
                let (a, z) = (
                    os[q.path as usize] as usize,
                    os[q.path as usize + 1] as usize,
                );
                let plane = ss.len() / wavelengths;
                if z < a || z > plane {
                    return Err(bad("ghost path offset"));
                }
                let coat = coatings.as_ref().map(|c| kernel::Coatings {
                    specs: x.rows::<CoatingSpec>(&c.specs),
                    layers: x.rows::<CoatingLayer>(&c.layers),
                    sensor: c.sensor_reflectance,
                });
                let v = kernel::trace::<E>(
                    &s.headers[0],
                    &s.surfaces,
                    &s.coefficients,
                    &ss[q.wavelength as usize * plane + a..q.wavelength as usize * plane + z],
                    q,
                    *reject_virtual,
                    *sensor,
                    newton,
                    coat,
                );
                x.stats.fallback_count += (v.status & crate::consts::MKL_RAY_CODE_MASK
                    == crate::consts::MKL_GHOST_REFERENCE)
                    as u64;
                bytemuck::cast_slice_mut::<_, Value>(x.words_mut(*output))[i] = v;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::{Snapshot, samples};

    fn pixel_keys(bins: [u32; 2], maps: usize) -> MkResult<()> {
        let mut p = ResidentPlan::new(Snapshot::new(samples::singlet()).unwrap());
        let hits = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 5]));
        let r = Request::PixelKeys {
            hits: p.view(hits)?,
            bins,
            maps,
            half: [Df32::from_f32(1.); 2],
            keys: p.scratch(BufferDesc::new(ElementType::U32, [1])),
            ids: p.scratch(BufferDesc::new(ElementType::U32, [1])),
            weights: p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1])),
        };
        validate(&r, &p)
    }

    /// A leaf's map and reference bin + 1 are f32 words, exact up to 2^24:
    /// the plan is accepted up to that count and unsupported above it.
    #[test]
    fn pixel_keys_stay_within_exact_leaf_words() {
        let unsupported = |r: MkResult<()>| matches!(r, Err(MkError::Unsupported(_)));
        assert_eq!(LEAF_WORD_MAX as f32 as u64, LEAF_WORD_MAX);
        assert_eq!((LEAF_WORD_MAX + 1) as f32 as u64, LEAF_WORD_MAX);
        pixel_keys([4096, 4096], 1).unwrap();
        pixel_keys([1 << 24, 1], 255).unwrap();
        assert!(unsupported(pixel_keys([4097, 4096], 1)));
        assert!(unsupported(pixel_keys([(1 << 24) + 1, 1], 1)));
        assert!(unsupported(pixel_keys([u32::MAX, u32::MAX], 1)));
        pixel_keys([1, 1], (1 << 24) + 1).unwrap();
        assert!(unsupported(pixel_keys([1, 1], (1 << 24) + 2)));
        // Exact leaf words whose keys leave u32 are a malformed plan.
        assert!(matches!(
            pixel_keys([4096, 4096], 256),
            Err(MkError::Backend(_))
        ));
    }
}
