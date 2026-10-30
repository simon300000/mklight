//! Fixed-grid image-quality operands. Pupil caches live for one candidate,
//! so a changed prescription can never reuse another candidate's OPD.

use std::sync::Arc;

use mklight_analysis::energy::{GeometricMtfSettings, geometric_mtf_job};
use mklight_analysis::util::{FieldSelection, WavelengthSelection};
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};
use mklight_trace::{FieldPoint, PreparedField, SequentialTracer};
use mklight_wave::mtf::{MtfOrientation as Orientation, Otf, otf_fft};
use mklight_wave::traced::{PupilSettings, TracedPupil, trace_pupil};

use super::operand::{Capability, MtfMethod, MtfOrientation, OperandKind};
use super::{
    FieldSelector, Operand, OperandError, OperandEvaluator, OperandValue, TraceEvaluator,
    WavelengthSelector,
};
use crate::deriv::FdScheme;

/// Relative FD steps: h = eta * max(abs(parameter), variable scale).
pub const MTF_FD_FORWARD: f64 = 1e-6;
pub const MTF_FD_CENTRAL: f64 = 1e-5;

#[derive(Clone)]
pub struct MtfEvaluator {
    trace: TraceEvaluator,
    pub pupil: PupilSettings,
}

impl MtfEvaluator {
    pub const CAPABILITY: Capability = Capability::Mtf;

    pub fn new(library: Arc<MaterialLibrary>) -> Self {
        Self {
            trace: TraceEvaluator::new(library),
            pupil: Self::pupil_settings(),
        }
    }

    /// Fixed entrance and exit grids, with no adaptive rim search. Keep
    /// these settings identical in the merit and in its analysis cross-check.
    pub fn pupil_settings() -> PupilSettings {
        PupilSettings {
            rim: false,
            ..Default::default()
        }
    }

    pub fn builtin() -> Self {
        Self::new(Arc::new(MaterialLibrary::with_builtin()))
    }

    fn candidate(
        &self,
        s: &OpticalSystem,
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<OperandValue>> {
        let tr = match self.trace.tracer(s) {
            Ok(t) => t,
            Err(e) => return Ok(vec![Err(e); ops.len()]),
        };
        let mut cache: Vec<(FieldSelector, usize, TracedPupil)> = Vec::new();
        let mut otfs: Vec<(FieldSelector, usize, f64, Otf)> = Vec::new();
        let mut out = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            ctx.check()?;
            let sub = ctx.subrange(
                i as f64 / ops.len() as f64,
                (i + 1) as f64 / ops.len() as f64,
            );
            let value = (|| {
                let OperandKind::Mtf {
                    frequency,
                    orientation,
                    method,
                    defocus_mm,
                    detector,
                    wavelengths: subset,
                } = &op.kind
                else {
                    return Err(MkError::InvalidParam("expected MTF operand".into()));
                };
                if !frequency.is_finite() || *frequency < 0.0 || !defocus_mm.is_finite() {
                    return Err(MkError::InvalidParam(
                        "MTF needs finite frequency >= 0 and defocus".into(),
                    ));
                }
                if let Some(d) = detector {
                    d.validate()?;
                }
                let wls = wavelengths(&tr, op.wavelength, subset)?;
                let mut sum = 0.0;
                let mut weight = 0.0;
                let fs = fields(s, op.field);
                for (j, (field, fw)) in fs.iter().enumerate() {
                    let job =
                        sub.subrange(j as f64 / fs.len() as f64, (j + 1) as f64 / fs.len() as f64);
                    let prepared = prepare(&tr, *field)?;
                    let o = Orientation::from_field(prepared.point.hx, prepared.point.hy);
                    let (mut t, mut sg) = match method {
                        MtfMethod::Diffraction => {
                            for (k, &(wl, _)) in wls.iter().enumerate() {
                                if !cache.iter().any(|(f, w, _)| f == field && *w == wl) {
                                    let p = trace_pupil(
                                        &tr,
                                        &prepared,
                                        wl,
                                        &self.pupil,
                                        &job.subrange(
                                            0.6 * k as f64 / wls.len() as f64,
                                            0.6 * (k + 1) as f64 / wls.len() as f64,
                                        ),
                                    )?;
                                    cache.push((*field, wl, p));
                                }
                            }
                            // Match the wave analysis' common chief-ray reference,
                            // retaining lateral colour in the complex OTF sum.
                            let chief = tr.chief(&prepared, tr.system().primary_wavelength);
                            for &(wl, _) in &wls {
                                if !otfs
                                    .iter()
                                    .any(|(f, w, d, _)| f == field && *w == wl && *d == *defocus_mm)
                                {
                                    let tp = &cache
                                        .iter()
                                        .find(|(f, wi, _)| f == field && *wi == wl)
                                        .unwrap()
                                        .2;
                                    let r = tp.info.reference_point_mm;
                                    let mut p = tp.pupil.with_focus_shift(
                                        *defocus_mm / tp.pupil.geometry().chief_ray[2],
                                    );
                                    p.set_image_offset_mm([
                                        r[0] - chief.ray.pos.x,
                                        r[1] - chief.ray.pos.y,
                                    ]);
                                    otfs.push((
                                        *field,
                                        wl,
                                        *defocus_mm,
                                        otf_fft(&p, &job.subrange(0.6, 1.0))?,
                                    ));
                                }
                            }
                            let spectral_sum: f64 = wls.iter().map(|(_, w)| w).sum();
                            let modulation = |axis: [f64; 2]| {
                                let (mut re, mut im) = (0.0_f64, 0.0_f64);
                                for &(wl, w) in &wls {
                                    let otf = &otfs
                                        .iter()
                                        .find(|(f, wi, d, _)| {
                                            f == field && *wi == wl && *d == *defocus_mm
                                        })
                                        .unwrap()
                                        .3;
                                    let z = otf.at(frequency * axis[0], frequency * axis[1]);
                                    re += z.re * (w / spectral_sum);
                                    im += z.im * (w / spectral_sum);
                                }
                                re.hypot(im)
                            };
                            (modulation(o.tangential), modulation(o.sagittal()))
                        }
                        MtfMethod::Geometric => {
                            let (local, index) = match field {
                                FieldSelector::Index { index } => (tr.clone(), *index),
                                FieldSelector::Normalized { hx, hy } => (
                                    tr.with_additional_fields(&[FieldPoint::normalized(*hx, *hy)])?,
                                    tr.fields().len(),
                                ),
                                FieldSelector::All => unreachable!(),
                            };
                            let mut settings = GeometricMtfSettings {
                                defocus: *defocus_mm,
                                max_frequency: Some(frequency.max(f64::MIN_POSITIVE)),
                                frequency_samples: 2,
                                ..Default::default()
                            };
                            settings.sampling.fields = FieldSelection::Index { index };
                            settings.sampling.wavelengths = WavelengthSelection::List {
                                indices: wls.iter().map(|x| x.0).collect(),
                            };
                            let c = geometric_mtf_job(&local, &settings, &job)?;
                            (c.curves[0].modulation[1], c.curves[1].modulation[1])
                        }
                    };
                    if let Some(d) = detector {
                        let dc = d.curve(&[*frequency], &o);
                        t *= dc.tangential[0];
                        sg *= dc.sagittal[0];
                    }
                    sum += fw
                        * match orientation {
                            MtfOrientation::Tangential => t,
                            MtfOrientation::Sagittal => sg,
                            MtfOrientation::Average => 0.5 * (t + sg),
                        };
                    weight += fw;
                }
                if weight <= 0.0 {
                    return Err(MkError::InvalidParam(
                        "no field with positive weight".into(),
                    ));
                }
                Ok(sum / weight)
            })();
            out.push(operand_result(value)?);
            sub.report(1.0, "MTF merit");
        }
        Ok(out)
    }
}

impl OperandEvaluator for MtfEvaluator {
    fn supports(&self, kind: &OperandKind) -> bool {
        kind.capability() == Self::CAPABILITY
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        systems
            .iter()
            .enumerate()
            .map(|(i, s)| {
                self.candidate(
                    s,
                    ops,
                    &ctx.subrange(
                        i as f64 / systems.len() as f64,
                        (i + 1) as f64 / systems.len() as f64,
                    ),
                )
            })
            .collect()
    }
    fn fd_relative_step(&self, _: &OperandKind, scheme: FdScheme) -> Option<f64> {
        Some(match scheme {
            FdScheme::Forward => MTF_FD_FORWARD,
            FdScheme::Central => MTF_FD_CENTRAL,
        })
    }
}

pub(super) fn fields(s: &OpticalSystem, sel: FieldSelector) -> Vec<(FieldSelector, f64)> {
    match sel {
        FieldSelector::All => s
            .fields
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| f.weight > 0.0)
            .map(|(index, f)| (FieldSelector::Index { index }, f.weight))
            .collect(),
        f => vec![(f, 1.0)],
    }
}

pub(super) fn prepare(tr: &SequentialTracer, field: FieldSelector) -> MkResult<PreparedField> {
    match field {
        FieldSelector::Index { index } => tr
            .fields()
            .get(index)
            .cloned()
            .ok_or_else(|| MkError::InvalidParam(format!("field {index} does not exist"))),
        FieldSelector::Normalized { hx, hy } => tr.prepare_field(FieldPoint::normalized(hx, hy)),
        FieldSelector::All => unreachable!(),
    }
}

pub(super) fn wavelengths(
    tr: &SequentialTracer,
    sel: WavelengthSelector,
    subset: &[usize],
) -> MkResult<Vec<(usize, f64)>> {
    let sys = tr.system();
    let sel = if subset.is_empty() {
        match sel {
            WavelengthSelector::Primary => WavelengthSelection::Primary,
            WavelengthSelector::All => WavelengthSelection::All,
            WavelengthSelector::Index { index } => WavelengthSelection::Index { index },
        }
    } else {
        WavelengthSelection::List {
            indices: subset.to_vec(),
        }
    };
    let indices = sel.resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let sum: f64 = indices.iter().map(|&i| sys.wavelength_weights[i]).sum();
    if indices.len() > 1 && sum <= 0.0 {
        return Err(MkError::InvalidParam("no positive spectral weights".into()));
    }
    Ok(indices
        .iter()
        .map(|&i| {
            (
                i,
                if indices.len() == 1 {
                    1.0
                } else {
                    sys.wavelength_weights[i] / sum
                },
            )
        })
        .collect())
}

pub(super) fn operand_result(value: MkResult<f64>) -> MkResult<OperandValue> {
    match value {
        Err(MkError::Cancelled) => Err(MkError::Cancelled),
        Err(e) => Ok(Err(OperandError::RayFailure(e.to_string()))),
        Ok(v) => Ok(Ok(v)),
    }
}
