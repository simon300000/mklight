//! Ghost sequence geometry and separate s/p transport. Path expansion and
//! coverage belong to the solver; this service consumes validated steps.
mod kernel;
use crate::{Df32, pending::JobControl, radiometry::*, resident::*};
use bytemuck::{Pod, Zeroable};
pub use kernel::trace;
use mklight_core::MkResult;

/// Frozen coverage contract shared with stray; ordered signatures are unchanged.
#[derive(Clone, Debug)]
pub struct Coverage {
    pub aperture_words: PlanView,
    pub path_offsets: PlanView,
    pub path_events: PlanView,
    pub diffraction_visits: PlanView,
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
        systems: SystemSet,
        probes: PlanView,
        /// Path offsets into steps; steps are wavelength-major, with equal
        /// path offsets in each wavelength plane.
        offsets: PlanView,
        steps: PlanView,
        wavelengths: usize,
        reject_virtual: bool,
        sensor: [Df32; 2],
        newton: crate::NewtonSettings,
        /// U32 [count + 1]: each probe's compact step rows, one per path step.
        step_rows: PlanView,
        output: BufferId,
        /// One coating query per step row. Rows after a probe's last completed
        /// step, and spare rows after the final probe, are zeroed.
        queries: BufferId,
        /// [rows, 2] absorption and efficiency per completed step.
        factors: BufferId,
    },
    Weights {
        values: PlanView,
        coatings: PlanView,
        factors: PlanView,
        offsets: PlanView,
        steps: PlanView,
        probes: PlanView,
        wavelengths: usize,
        step_rows: PlanView,
        sensor_reflectance: Df32,
        output: BufferId,
    },
}
pub fn record_desc<T>(n: usize) -> BufferDesc {
    crate::radiometry::record_desc::<T>(n)
}
fn records<T>(v: &PlanView, p: &ResidentPlan) -> MkResult<usize> {
    let n = v.desc.shape.first().copied().unwrap_or(0);
    let base = match &p.buffers[v.buffer.0] {
        PlanBuffer::Import(v) => v.offset,
        _ => 0,
    };
    if v.desc != record_desc::<T>(n)
        || v.stride != 1
        || !(base + v.offset).is_multiple_of(2)
        || v.desc.len()? > u32::MAX as usize
    {
        return Err(bad("ghost record layout"));
    }
    Ok(n)
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    let (probes, offsets, steps, wavelengths, step_rows, output) = match r {
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
            systems,
            probes,
            offsets,
            steps,
            wavelengths,
            step_rows,
            output,
            queries,
            factors,
            newton,
            ..
        } => {
            if !matches!(systems, SystemSet::Uploaded(_)) {
                return unsupported("ghost resident candidate systems");
            }
            let rows = p.desc(*queries)?.shape.first().copied().unwrap_or(0);
            if p.desc(*queries)? != &record_desc::<CoatingQuery>(rows)
                || p.desc(*factors)? != &BufferDesc::new(ElementType::DoubleFloat, [rows, 2])
                || !newton.tolerance_abs.is_finite()
                || newton.tolerance_abs < 0.
                || !newton.tolerance_rel.is_finite()
                || newton.tolerance_rel < 0.
            {
                return Err(bad("ghost trace outputs/settings"));
            }
            (probes, offsets, steps, wavelengths, step_rows, output)
        }
        Request::Weights {
            probes,
            offsets,
            steps,
            wavelengths,
            step_rows,
            output,
            values,
            coatings,
            factors,
            ..
        } => {
            let n = records::<Probe>(probes, p)?;
            let rows = records::<CoatingValue>(coatings, p)?;
            if records::<Value>(values, p)? != n
                || factors.desc != BufferDesc::new(ElementType::DoubleFloat, [rows, 2])
                || factors.stride != 1
            {
                return Err(bad("ghost weight inputs"));
            }
            (probes, offsets, steps, wavelengths, step_rows, output)
        }
    };
    let n = records::<Probe>(probes, p)?;
    let ns = records::<Step>(steps, p)?;
    if *wavelengths == 0
        || ns % wavelengths != 0
        || step_rows.desc != BufferDesc::new(ElementType::U32, [n + 1])
        || step_rows.stride != 1
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
            step_rows,
            output,
            queries,
            factors,
            ..
        } => Access {
            reads: vec![
                probes.clone(),
                offsets.clone(),
                steps.clone(),
                step_rows.clone(),
            ],
            writes: vec![*output, *queries, *factors],
            systems: match systems {
                SystemSet::Uploaded(h) => vec![*h],
                _ => vec![],
            },
        },
        Request::Weights {
            values,
            coatings,
            factors,
            offsets,
            steps,
            probes,
            step_rows,
            output,
            ..
        } => Access {
            reads: vec![
                values.clone(),
                coatings.clone(),
                factors.clone(),
                offsets.clone(),
                steps.clone(),
                probes.clone(),
                step_rows.clone(),
            ],
            writes: vec![*output],
            systems: vec![],
        },
    }
}
fn rows<'a, T: Pod>(x: &'a cpu::Execution, v: &PlanView) -> &'a [T] {
    bytemuck::cast_slice(
        &x.words(v.buffer)[v.offset * v.desc.element.words()
            ..(v.offset + v.desc.len().unwrap()) * v.desc.element.words()],
    )
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
                let v = rows::<Df32>(x, hits);
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
                let ks = rows::<u32>(x, keys);
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
            let total = rows::<u32>(x, prefix).last().copied().unwrap_or(0);
            let end = rows::<u32>(x, keys).partition_point(|&k| k < *bins as u32);
            x.words_mut(*count)[0] = total;
            x.words_mut(*count)[1] = 0;
            x.words_mut(*offsets)[total as usize] = end as u32;
            for i in 0..n {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                let ks = rows::<u32>(x, keys);
                if ks[i] >= *bins as u32 || (i > 0 && ks[i] == ks[i - 1]) {
                    continue;
                }
                let pixel = ks[i];
                let end = ks.partition_point(|&k| k <= pixel);
                let dest = rows::<u32>(x, prefix)[i] as usize - 1;
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
                let pixel = rows::<u32>(x, pixels)[i] as usize;
                let value = rows::<Df32>(x, values)[i];
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
                let offsets = rows::<u32>(x, offsets);
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
                    crate::fx::D::from_df(rows::<Df32>(x, initial)[4 * i + k])
                });
                for k in a..z {
                    let st = rows::<FirstOrderStep>(x, steps)[k];
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
                            let c = rows::<CoatingValue>(x, coatings)[st.coating as usize];
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
                let l = rows::<Leaf>(x, leaves)[i];
                let v = rows::<Value>(x, &sources[l.batch as usize])[l.row as usize];
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
                let h = rows::<Df32>(x, hits);
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
                let q = rows::<LaunchSample>(x, samples)[i];
                let fields = rows::<crate::aiming::FieldSpec>(x, fields);
                let field = fields
                    .get(q.field as usize)
                    .ok_or_else(|| bad("ghost launch field"))?;
                let data = rows::<f32>(x, chiefs);
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
            step_rows,
            output,
            queries,
            factors,
        } => {
            let (headers, surfaces, pool) = match systems {
                SystemSet::Uploaded(h) => {
                    let s = x.system(*h)?;
                    (
                        s.headers.clone(),
                        s.surfaces.clone(),
                        s.coefficients.clone(),
                    )
                }
                SystemSet::Resident { .. } => {
                    return unsupported("ghost resident candidate systems");
                }
            };
            let n = probes.desc.shape[0];
            let spare = rows::<u32>(x, step_rows)[n] as usize;
            if spare > x.words(*queries).len() / 4 {
                return Err(bad("ghost step rows"));
            }
            bytemuck::cast_slice_mut::<_, CoatingQuery>(x.words_mut(*queries))[spare..]
                .fill(CoatingQuery::zeroed());
            bytemuck::cast_slice_mut::<_, [Df32; 2]>(x.words_mut(*factors))[spare..]
                .fill([Df32::ZERO; 2]);
            for i in 0..n {
                if i % 256 == 0 {
                    ctl.check()?;
                }
                let q = rows::<Probe>(x, probes)[i];
                let os = rows::<u32>(x, offsets);
                let ss = rows::<Step>(x, steps);
                if q.path as usize + 1 >= os.len() || q.wavelength as usize >= *wavelengths {
                    return Err(bad("ghost probe index"));
                }
                let (a, z) = (
                    os[q.path as usize] as usize,
                    os[q.path as usize + 1] as usize,
                );
                let plane = ss.len() / wavelengths;
                let rs = rows::<u32>(x, step_rows);
                let (ra, rz) = (rs[i] as usize, rs[i + 1] as usize);
                if z < a || z > plane || rz < ra || rz - ra != z - a || rz > rs[n] as usize {
                    return Err(bad("ghost path offset or step rows"));
                }
                let mut qs = vec![CoatingQuery::zeroed(); z - a];
                let mut fs = vec![[Df32::ZERO; 2]; z - a];
                let v = kernel::trace::<E>(
                    &headers[0],
                    &surfaces,
                    &pool,
                    &ss[q.wavelength as usize * plane + a..q.wavelength as usize * plane + z],
                    q,
                    *reject_virtual,
                    *sensor,
                    newton,
                    |k, q, f| {
                        qs[k] = q;
                        fs[k] = f;
                    },
                );
                x.stats.fallback_count += (v.status & crate::consts::MKL_RAY_CODE_MASK
                    == crate::consts::MKL_GHOST_REFERENCE)
                    as u64;
                bytemuck::cast_slice_mut::<_, Value>(x.words_mut(*output))[i] = v;
                bytemuck::cast_slice_mut::<_, CoatingQuery>(x.words_mut(*queries))[ra..rz]
                    .copy_from_slice(&qs);
                bytemuck::cast_slice_mut::<_, [Df32; 2]>(x.words_mut(*factors))[ra..rz]
                    .copy_from_slice(&fs);
            }
        }
        Request::Weights {
            values,
            coatings,
            factors,
            offsets,
            steps,
            probes,
            wavelengths,
            step_rows,
            sensor_reflectance,
            output,
        } => {
            for i in 0..probes.desc.shape[0] {
                if i % 1024 == 0 {
                    ctl.check()?;
                }
                let q = rows::<Probe>(x, probes)[i];
                let os = rows::<u32>(x, offsets);
                let ss = rows::<Step>(x, steps);
                let plane = ss.len() / wavelengths;
                let a = q.wavelength as usize * plane + os[q.path as usize] as usize;
                let rs = rows::<u32>(x, step_rows);
                let (ra, rz) = (rs[i] as usize, rs[i + 1] as usize);
                let v = kernel::weights::<E>(
                    rows::<Value>(x, values)[i],
                    &ss[a..a + (os[q.path as usize + 1] - os[q.path as usize]) as usize],
                    &rows::<CoatingValue>(x, coatings)[ra..rz],
                    &rows::<[Df32; 2]>(x, factors)[ra..rz],
                    *sensor_reflectance,
                );
                bytemuck::cast_slice_mut::<_, Value>(x.words_mut(*output))[i] = v;
            }
        }
    }
    Ok(())
}
