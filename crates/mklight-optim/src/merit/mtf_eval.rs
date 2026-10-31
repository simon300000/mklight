//! Fixed-grid MTF operands. Pupil and OTF caches live for one candidate,
//! so a changed prescription never reuses another candidate's OPD.

use std::sync::Arc;

use mklight_analysis::energy::{GeometricMtfSettings, geometric_mtf_job};
use mklight_analysis::util::{FieldSelection, WavelengthSelection};
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};
use mklight_trace::{FieldPoint, PreparedField, SequentialTracer};
use mklight_wave::mtf::{MtfOrientation as Orientation, Otf, otf_fft};
use mklight_wave::traced::{PupilSettings, TracedPupil, trace_pupil};
use rayon::prelude::*;

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
    /// Fixed entrance and exit grids, with no adaptive rim search, so the
    /// sampling never changes between candidates.
    pub pupil: PupilSettings,
}

/// One candidate's traced pupils by (field, wavelength) and OTFs by
/// (field, wavelength, defocus), shared by all its MTF operands.
struct Candidate<'a> {
    tr: &'a SequentialTracer,
    settings: &'a PupilSettings,
    pupils: Vec<(FieldSelector, usize, TracedPupil)>,
    otfs: Vec<(FieldSelector, usize, f64, Otf)>,
}

impl Candidate<'_> {
    fn otf(
        &mut self,
        field: FieldSelector,
        prepared: &PreparedField,
        wl: usize,
        defocus_mm: f64,
        ctx: &JobContext,
    ) -> MkResult<&Otf> {
        let hit = |&(f, w, d, _): &(FieldSelector, usize, f64, Otf)| {
            f == field && w == wl && d == defocus_mm
        };
        if !self.otfs.iter().any(hit) {
            if !self.pupils.iter().any(|(f, w, _)| *f == field && *w == wl) {
                let p = trace_pupil(self.tr, prepared, wl, self.settings, ctx)?;
                self.pupils.push((field, wl, p));
            }
            let tp = &self.pupils.iter().find(|(f, w, _)| *f == field && *w == wl);
            let tp = &tp.unwrap().2;
            // The wave analysis' common reference: the primary chief ray,
            // which keeps lateral colour in the polychromatic OTF sum.
            let r = tp.info.reference_point_mm;
            let chief = self.tr.chief(prepared, self.tr.system().primary_wavelength);
            let c = if chief.status.is_ok() {
                [chief.ray.pos.x, chief.ray.pos.y]
            } else {
                [r[0], r[1]]
            };
            let mut p = tp
                .pupil
                .with_focus_shift(defocus_mm / tp.pupil.geometry().chief_ray[2]);
            p.set_image_offset_mm([r[0] - c[0], r[1] - c[1]]);
            let otf = otf_fft(&p, ctx)?;
            self.otfs.push((field, wl, defocus_mm, otf));
        }
        Ok(&self.otfs.iter().find(|e| hit(e)).unwrap().3)
    }
}

impl MtfEvaluator {
    pub const CAPABILITY: Capability = Capability::Mtf;

    pub fn new(library: Arc<MaterialLibrary>) -> Self {
        Self {
            trace: TraceEvaluator::new(library),
            pupil: PupilSettings {
                grid: 64,
                rim: false,
                ..Default::default()
            },
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
        let mut cand = Candidate {
            tr: &tr,
            settings: &self.pupil,
            pupils: Vec::new(),
            otfs: Vec::new(),
        };
        let n = ops.len() as f64;
        ops.iter()
            .enumerate()
            .map(|(i, op)| {
                ctx.check()?;
                let sub = ctx.subrange(i as f64 / n, (i + 1) as f64 / n);
                let v = value(&mut cand, s, op, &sub);
                sub.report(1.0, "MTF merit");
                operand_result(v)
            })
            .collect()
    }
}

/// The operand's field-weighted MTF.
fn value(cand: &mut Candidate, s: &OpticalSystem, op: &Operand, ctx: &JobContext) -> MkResult<f64> {
    let OperandKind::Mtf {
        frequency,
        orientation,
        method,
        defocus_mm,
        detector,
        wavelengths: subset,
    } = &op.kind
    else {
        unreachable!("routed by capability")
    };
    let (f, dz) = (*frequency, *defocus_mm);
    if let Some(d) = detector {
        d.validate()?;
    }
    let wls = wavelengths(cand.tr, op.wavelength, subset)?;
    let (mut sum, mut total) = (0.0, 0.0);
    for (field, fw) in fields(s, op.field) {
        let prepared = prepare(cand.tr, field)?;
        let o = Orientation::from_field(prepared.point.hx, prepared.point.hy);
        let (mut t, mut sg) = match method {
            MtfMethod::Diffraction => {
                // Complex OTFs summed with the spectral weights.
                let dirs = [o.tangential, o.sagittal()];
                let mut z = [(0.0_f64, 0.0_f64); 2];
                for &(wl, w) in &wls {
                    let otf = cand.otf(field, &prepared, wl, dz, ctx)?;
                    for (z, d) in z.iter_mut().zip(dirs) {
                        let v = otf.at(f * d[0], f * d[1]);
                        z.0 += w * v.re;
                        z.1 += w * v.im;
                    }
                }
                (z[0].0.hypot(z[0].1), z[1].0.hypot(z[1].1))
            }
            MtfMethod::Geometric => geometric(cand.tr, field, &wls, f, dz, ctx)?,
        };
        if let Some(d) = detector {
            let dc = d.curve(&[f], &o);
            t *= dc.tangential[0];
            sg *= dc.sagittal[0];
        }
        sum += fw
            * match orientation {
                MtfOrientation::Tangential => t,
                MtfOrientation::Sagittal => sg,
                MtfOrientation::Average => 0.5 * (t + sg),
            };
        total += fw;
    }
    Ok(sum / total)
}

/// Tangential and sagittal geometric MTF at `frequency`.
fn geometric(
    tr: &SequentialTracer,
    field: FieldSelector,
    wls: &[(usize, f64)],
    frequency: f64,
    defocus: f64,
    ctx: &JobContext,
) -> MkResult<(f64, f64)> {
    let (local, index) = match field {
        FieldSelector::Index { index } => (tr.clone(), index),
        FieldSelector::Normalized { hx, hy } => (
            tr.with_additional_fields(&[FieldPoint::normalized(hx, hy)])?,
            tr.fields().len(),
        ),
        FieldSelector::All => unreachable!("expanded by `fields`"),
    };
    let mut settings = GeometricMtfSettings {
        defocus,
        max_frequency: Some(frequency.max(f64::MIN_POSITIVE)),
        frequency_samples: 2,
        ..Default::default()
    };
    settings.sampling.fields = FieldSelection::Index { index };
    settings.sampling.wavelengths = WavelengthSelection::List {
        indices: wls.iter().map(|x| x.0).collect(),
    };
    let c = geometric_mtf_job(&local, &settings, ctx)?;
    Ok((c.curves[0].modulation[1], c.curves[1].modulation[1]))
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
            .par_iter()
            .map(|s| self.candidate(s, ops, ctx))
            .collect()
    }
    fn fd_relative_step(&self, _: &OperandKind, scheme: FdScheme) -> Option<f64> {
        Some(match scheme {
            FdScheme::Forward => MTF_FD_FORWARD,
            FdScheme::Central => MTF_FD_CENTRAL,
        })
    }
}

/// Positive-weight system fields for `all`, else the single selection.
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

fn prepare(tr: &SequentialTracer, field: FieldSelector) -> MkResult<PreparedField> {
    match field {
        FieldSelector::Index { index } => tr
            .fields()
            .get(index)
            .cloned()
            .ok_or_else(|| MkError::InvalidParam(format!("field {index} does not exist"))),
        FieldSelector::Normalized { hx, hy } => tr.prepare_field(FieldPoint::normalized(hx, hy)),
        FieldSelector::All => unreachable!("expanded by `fields`"),
    }
}

/// Selected wavelengths with normalised spectral weights; an explicit
/// `subset` overrides the selector.
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
    if let [i] = indices[..] {
        return Ok(vec![(i, 1.0)]);
    }
    let sum: f64 = indices.iter().map(|&i| sys.wavelength_weights[i]).sum();
    if !(sum > 0.0) {
        return Err(MkError::InvalidParam("no positive spectral weights".into()));
    }
    Ok(indices
        .iter()
        .map(|&i| (i, sys.wavelength_weights[i] / sum))
        .collect())
}

/// A solver error becomes this operand's failure; cancellation aborts.
pub(super) fn operand_result(value: MkResult<f64>) -> MkResult<OperandValue> {
    match value {
        Err(MkError::Cancelled) => Err(MkError::Cancelled),
        Err(e) => Ok(Err(OperandError::RayFailure(e.to_string()))),
        Ok(v) => Ok(Ok(v)),
    }
}
