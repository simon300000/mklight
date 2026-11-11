//! Host/device divisions of the chief-ray maps: `chief_ray_angle`,
//! `pupil_aberration` and `grid_distortion` ([`Route::chief`]). Each
//! field point needs one aimed chief ray (plus, for the pupil aberration,
//! four un-aimed probes); the per-field chief solve is small, serial and
//! branchy, the trace regular.
//!
//! * `host-chief` (default): the reference tracer solves every field's
//!   chief in f64 on all CPU cores ([`Geometry::append_grid`]), the GPU
//!   traces the rays.
//! * `device-chief`: the GPU solves the chiefs too (the aiming kernels'
//!   own chief solve; no host seeds), so the whole map runs on the device.
use super::*;

/// The chief-map designs (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChiefDesign {
    #[default]
    HostChief,
    DeviceChief,
}
impl ChiefDesign {
    /// The design's name in the run evidence (`fieldmaps_variant`).
    pub fn name(self) -> &'static str {
        match self {
            Self::HostChief => "host-chief",
            Self::DeviceChief => "device-chief",
        }
    }
}

impl Geometry<'_> {
    /// [`Self::append_grid`] with the chief rows solved on the device
    /// (`device_chiefs`, no host seeds) or by the host (the same as
    /// `append_grid`).
    #[allow(clippy::too_many_arguments)]
    pub fn append_chief_grid(
        &self,
        p: &mut ResidentPlan,
        fields: &[FieldPoint],
        wl: usize,
        pupil: &[[f64; 2]],
        aiming: mklight_core::RayAiming,
        last: usize,
        records: bool,
        device_chiefs: bool,
        ctx: &JobContext,
    ) -> MkResult<ResidentHits> {
        if !device_chiefs {
            return Ok(self
                .append_grid(p, fields, &[wl], pupil, aiming, last, false, records, ctx)?
                .hits);
        }
        ctx.check()?;
        // The primary wavelength's rows too (real-image-height fields).
        let mut cw = vec![wl, self.tracer.system().primary_wavelength];
        cw.sort_unstable();
        cw.dedup();
        let nf = fields.len() * cw.len();
        let metadata = self.records::<FieldSpec>(p, nf, |i| {
            let (si, wi) = self.chunk(cw[i % cw.len()])?;
            FieldSpec::new(
                self.tracer.system(),
                self.tracer,
                fields[i / cw.len()],
                si,
                wi,
                aiming,
            )
        })?;
        let steps = self.import::<u32>(p, [last], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i + 1) as u32;
            }
            Ok(())
        })?;
        let row = cw.iter().position(|&v| v == wl).unwrap();
        let count = fields.len() * pupil.len();
        let samples = self.records::<Sample>(p, count, |i| {
            let index = (i / pupil.len()) * cw.len() + row;
            let mut s = Sample::new(pupil[i % pupil.len()], index as u32, i as u64);
            s.weight = self.tracer.system().wavelength_weights[wl] as f32;
            Ok(s)
        })?;
        let tolerance = self.device_tolerance(aiming)?;
        let generated = rays(p, count);
        p.operations
            .push(ResidentOp::Aiming(Request::GenerateAim(GenerateAim {
                systems: self.systems,
                samples: p.view(samples)?,
                field_wavelength: p.view(metadata)?,
                field_key: None,
                seeds: None,
                tolerance,
                max_iterations: 60,
                output: generated.clone(),
                options: Default::default(),
            })));
        let out = hits(p, count, if records { last + 1 } else { 0 });
        p.operations
            .push(ResidentOp::Aiming(Request::TraceResident(TraceResident {
                systems: self.systems,
                input: generated,
                steps: p.view(steps)?,
                start_surface: 0,
                apply_apertures: false,
                radiometry: 0,
                newton: mklight_gpu::NewtonSettings::default(),
                output: out.clone(),
            })));
        Ok(out)
    }
}
