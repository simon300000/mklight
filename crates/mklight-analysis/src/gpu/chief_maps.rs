//! Host/device divisions of the chief-ray maps: `chief_ray_angle`,
//! `pupil_aberration` and `grid_distortion` (route key `fieldmaps`). Each
//! field point needs one aimed chief ray (plus, for the pupil aberration,
//! four un-aimed probes); the per-field chief solve is small, serial and
//! branchy, the trace regular.
//!
//! * `host-chief` (default): the reference tracer solves every field's
//!   chief in f64 on all CPU cores (`aiming::host_seeds`), the GPU traces
//!   the rays.
//! * `device-chief`: the GPU solves the chiefs too (the aiming kernels'
//!   own chief solve; no host seeds), so the whole map runs on the device.
use super::*;

/// The chief-map designs (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChiefDesign {
    HostChief,
    DeviceChief,
}
impl ChiefDesign {
    pub const DEFAULT: Self = Self::HostChief;
    pub fn name(self) -> &'static str {
        match self {
            Self::HostChief => "host-chief",
            Self::DeviceChief => "device-chief",
        }
    }
    /// The selected design and its evidence name; an unknown name is an
    /// error, as for the other route keys.
    pub fn selected() -> MkResult<(Self, String)> {
        let all = [Self::HostChief, Self::DeviceChief];
        match mklight_core::split::variant(super::SPLIT_KEY) {
            None => Ok((Self::DEFAULT, format!("default:{}", Self::DEFAULT.name()))),
            Some(v) => match all.into_iter().find(|d| d.name() == v) {
                Some(d) => Ok((d, v)),
                None => Err(mklight_core::MkError::InvalidParam(format!(
                    "unknown fieldmaps design '{v}' (host-chief, device-chief)"
                ))),
            },
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
                self.tracer.paraxial(),
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
        let tolerance = match aiming {
            mklight_core::RayAiming::Real { .. } if aiming == self.tracer.aiming() => {
                self.tracer.device_aim_tolerance() as f32
            }
            mklight_core::RayAiming::Real { .. } => {
                self.tracer_for(aiming)?.device_aim_tolerance() as f32
            }
            _ => 1e-10,
        };
        let generated = rays(p, count);
        p.operations
            .push(ResidentOp::Aiming(Request::GenerateAim(GenerateAim {
                systems: SystemSet::Uploaded(self.systems),
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
                systems: SystemSet::Uploaded(self.systems),
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
