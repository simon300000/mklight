//! Resident scalar-wave adapter. Prepared pupils keep amplitude, coverage and
//! unwrapped OPD on the backend; results carry the actual plan's RunInfo.
//! Engine routing and trace qualification belong to the engine integration.
use crate::{
    HuygensSettings, ImageSampling, Psf, PsfNormalization, PsfSettings, PsfTransform,
    PupilFunction, PupilGeometry, WaveWarning,
};
use mklight_core::{JobContext, MkError, MkResult, RunInfo, Snapshot};
use mklight_gpu::{ComputeBackend, Df32, ReduceMode, resident::*, transforms as tr, wave_ops as w};

#[derive(Clone, Debug)]
pub struct WaveResult<T> {
    pub value: T,
    pub run_info: RunInfo,
    pub stats: PlanStats,
}
/// Reusable resident pupil. The geometry and sampling are the same as the f64
/// PupilFunction; no pupil disk or aperture approximation is introduced here.
#[derive(Clone, Debug)]
pub struct ResidentPupil {
    pub amplitude: BufferHandle,
    pub coverage: BufferHandle,
    pub opd: BufferHandle,
    pub n: usize,
    pub wavelength_um: f64,
    pub geometry: PupilGeometry,
    pub image_offset_mm: [f64; 2],
    pub warnings: Vec<WaveWarning>,
}
impl ResidentPupil {
    /// Convert an existing validated pupil directly into shared allocations.
    /// This is an explicit host-pupil boundary, not a GPU tracing claim.
    pub fn from_pupil(
        backend: &dyn ComputeBackend,
        p: &PupilFunction,
        ctx: &JobContext,
    ) -> MkResult<Self> {
        let n = p.n();
        let fill = |src: &[f64]| {
            backend.fill_buffer(BufferDesc::new(ElementType::F32, [n, n]), &mut |mut b| {
                for (i, (a, &v)) in b.as_mut_slice::<f32>()?.iter_mut().zip(src).enumerate() {
                    if i % 4096 == 0 {
                        ctx.check()?;
                    }
                    *a = v as f32;
                }
                Ok(())
            })
        };
        let amplitude = fill(p.amplitude())?;
        let coverage = fill(p.coverage())?;
        let opd = backend.fill_buffer(
            BufferDesc::new(ElementType::DoubleFloat, [n, n]),
            &mut |mut b| {
                for (i, (a, &v)) in b
                    .as_mut_slice::<Df32>()?
                    .iter_mut()
                    .zip(p.opd_waves())
                    .enumerate()
                {
                    if i % 4096 == 0 {
                        ctx.check()?;
                    }
                    *a = Df32::from_f64(v);
                }
                Ok(())
            },
        )?;
        Ok(Self {
            amplitude,
            coverage,
            opd,
            n,
            wavelength_um: p.wavelength_um(),
            geometry: *p.geometry(),
            image_offset_mm: p.image_offset_mm(),
            warnings: p.sampling_warnings(),
        })
    }
    pub fn import(&self, p: &mut ResidentPlan) -> w::Pupil {
        let a = p.import(self.amplitude.view());
        let c = p.import(self.coverage.view());
        let o = p.import(self.opd.view());
        w::Pupil {
            amplitude: p.view(a).unwrap(),
            coverage: p.view(c).unwrap(),
            opd: p.view(o).unwrap(),
        }
    }
    pub fn wave_geometry(&self) -> w::Geometry {
        w::Geometry {
            sin_max: Df32::from_f64(self.geometry.sin_max()),
            wavelength: Df32::from_f64(self.wavelength_um * 1e-3 / self.geometry.image_index),
            radius: Df32::from_f64(if self.geometry.is_telecentric() {
                0.
            } else {
                self.geometry.reference_radius_mm
            }),
        }
    }
    fn lambda_f(&self) -> f64 {
        self.geometry.lambda_f_mm(self.wavelength_um)
    }
    fn checks(&self, spacing: f64, size: usize) -> Vec<WaveWarning> {
        let mut warnings = self.warnings.clone();
        let lf = self.lambda_f();
        let l = self.n as f64 * lf;
        if spacing > 0.5 * lf * (1. + 1e-9) {
            warnings.push(WaveWarning::ImageUndersampled {
                spacing_mm: spacing,
                nyquist_mm: 0.5 * lf,
            });
        }
        if size as f64 * spacing > l * (1. + 1e-9) {
            warnings.push(WaveWarning::FieldWraps {
                grid_mm: size as f64 * spacing,
                alias_free_mm: l,
            });
        }
        warnings
    }
    fn prepare(
        &self,
        p: &mut ResidentPlan,
        v: w::Pupil,
        side: usize,
        normal: [f64; 3],
    ) -> MkResult<(BufferId, BufferId)> {
        let field = p.scratch(BufferDesc::new(ElementType::Complex32, [side, side]));
        let moments = p.scratch(BufferDesc::new(ElementType::F32, [3, self.n * self.n]));
        let sums = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3]));
        p.operations.push(ResidentOp::Wave(w::Request::Prepare {
            pupil: v,
            geometry: self.wave_geometry(),
            normal: normal.map(Df32::from_f64),
            field,
            moments,
        }));
        p.operations.push(ResidentOp::Sum(Sum {
            input: p.view(moments)?,
            mode: ReduceMode::DoubleFloat,
            output: sums,
        }));
        Ok((field, sums))
    }
    /// Append a monochromatic Fourier PSF to a larger resident plan. Returned
    /// constants are three reduced pupil moments; no host optical readback.
    pub fn append_psf(
        &self,
        p: &mut ResidentPlan,
        settings: &PsfSettings,
        origin: [f64; 2],
    ) -> MkResult<PsfOutput> {
        let size = settings.grid_size;
        let requested = match settings.sampling {
            ImageSampling::Padding { factor } => self.lambda_f() / factor,
            ImageSampling::Spacing { mm } => mm,
        };
        if size == 0
            || size > 8192
            || !requested.is_finite()
            || requested <= 0.
            || origin.iter().any(|v| !v.is_finite())
        {
            return Err(MkError::InvalidParam("PSF grid/spacing/origin".into()));
        }
        let l = self.n as f64 * self.lambda_f();
        let ratio = l / requested;
        let m = (ratio >= self.n as f64 * (1. - 1e-12) && ratio <= crate::psf::MAX_FFT_LEN as f64)
            .then(|| {
                crate::fft::next_fast_len(((ratio * (1. - 1e-12)).ceil() as usize).max(self.n))
            });
        let fft = settings.transform == PsfTransform::Fft
            && self.geometry.is_on_axis()
            && origin == [0.; 2]
            && m.is_some_and(|m| m <= crate::psf::MAX_FFT_LEN);
        let side = if fft { m.unwrap() } else { self.n };
        let spacing = if fft { l / side as f64 } else { requested };
        let v = self.import(p);
        let (field, sums) = self.prepare(p, v, side, [0., 0., 1.])?;
        let transformed_side = if fft { side } else { size };
        let transformed = p.scratch(BufferDesc::new(
            ElementType::Complex32,
            [transformed_side, transformed_side],
        ));
        let b = self.geometry.surface_to_normal();
        let dnu = 1. / l;
        let c = (size / 2) as f64;
        let affine = [
            dnu * b[0][0] * spacing,
            dnu * b[0][1] * spacing,
            dnu * (b[0][0] * (origin[0] - c * spacing) + b[0][1] * (origin[1] - c * spacing)),
            dnu * b[1][0] * spacing,
            dnu * b[1][1] * spacing,
            dnu * (b[1][0] * (origin[0] - c * spacing) + b[1][1] * (origin[1] - c * spacing)),
        ]
        .map(Df32::from_f64);
        let kind = if fft {
            tr::TransformKind::Fft
        } else if b[0][1].abs() < 1e-15 && b[1][0].abs() < 1e-15 {
            tr::TransformKind::ChirpZ
        } else {
            tr::TransformKind::AffineMft
        };
        p.operations
            .push(ResidentOp::Transforms(tr::Request::Transform(
                tr::TransformPlan {
                    kind,
                    input: p.view(field)?,
                    output: transformed,
                    dimensions: vec![side, side],
                    output_dimensions: vec![transformed_side, transformed_side],
                    sign: 1,
                    scale: 1.,
                    affine: if fft { [Df32::ZERO; 6] } else { affine },
                },
            )));
        let image = p.scratch(BufferDesc::new(ElementType::F32, [size, size]));
        p.operations.push(ResidentOp::Wave(w::Request::Intensity {
            field: p.view(transformed)?,
            sums: p.view(sums)?,
            geometry: self.wave_geometry(),
            pupil_side: self.n,
            size,
            shifted: fft,
            unit_energy: settings.normalization == PsfNormalization::UnitEnergy,
            projection: self.geometry.projection_jacobian() as f32,
            output: image,
        }));
        let mut warnings = self.checks(spacing, size);
        if fft && (spacing / requested - 1.).abs() > 1e-6 {
            warnings.push(WaveWarning::SpacingAdjusted {
                requested_mm: requested,
                actual_mm: spacing,
            });
        }
        Ok(PsfOutput {
            image,
            sums,
            size,
            spacing,
            center: self.image_offset_mm,
            normalization: settings.normalization,
            plane_energy: false,
            warnings,
        })
    }
    pub fn append_huygens(
        &self,
        p: &mut ResidentPlan,
        settings: &HuygensSettings,
    ) -> MkResult<PsfOutput> {
        settings.plane.validate()?;
        let v = self.import(p);
        let normal = settings.plane.normal();
        let (_, sums) = self.prepare(p, v.clone(), self.n, [normal.x, normal.y, normal.z])?;
        let image = p.scratch(BufferDesc::new(
            ElementType::F32,
            [settings.grid_size, settings.grid_size],
        ));
        let xyz = |v: mklight_core::Vec3| [v.x, v.y, v.z].map(Df32::from_f64);
        p.operations.push(ResidentOp::Wave(w::Request::Huygens {
            pupil: v,
            geometry: self.wave_geometry(),
            sums: p.view(sums)?,
            size: settings.grid_size,
            spacing: Df32::from_f64(settings.spacing_mm),
            origin: xyz(settings.plane.origin_mm),
            x_axis: xyz(settings.plane.x_axis),
            y_axis: xyz(settings.plane.y_axis),
            unit_energy: settings.normalization == PsfNormalization::UnitEnergy,
            output: image,
        }));
        Ok(PsfOutput {
            image,
            sums,
            size: settings.grid_size,
            spacing: settings.spacing_mm,
            center: [
                self.image_offset_mm[0] + settings.plane.origin_mm.x,
                self.image_offset_mm[1] + settings.plane.origin_mm.y,
            ],
            normalization: settings.normalization,
            plane_energy: true,
            warnings: self.checks(settings.spacing_mm, settings.grid_size),
        })
    }
    pub fn psf(
        &self,
        backend: &dyn ComputeBackend,
        snapshot: Snapshot,
        settings: &PsfSettings,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<Psf>> {
        let mut p = ResidentPlan::new(snapshot);
        p.settings = serde_json::json!({"p6":"psf","settings":settings,"pupil_grid":self.n});
        let output = self.append_psf(&mut p, settings, [0.; 2])?;
        p.outputs = vec![output.image, output.sums];
        let r = backend.submit_plan(p, ctx)?.wait()?;
        let value = output.finish(self, backend, &r.outputs[0], &r.outputs[1])?;
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
    pub fn huygens(
        &self,
        backend: &dyn ComputeBackend,
        snapshot: Snapshot,
        settings: &HuygensSettings,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<Psf>> {
        let mut p = ResidentPlan::new(snapshot);
        p.settings = serde_json::json!({"p6":"huygens","settings":settings,"pupil_grid":self.n});
        let output = self.append_huygens(&mut p, settings)?;
        p.outputs = vec![output.image, output.sums];
        let r = backend.submit_plan(p, ctx)?.wait()?;
        let value = output.finish(self, backend, &r.outputs[0], &r.outputs[1])?;
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
    pub fn append_correlation(
        &self,
        p: &mut ResidentPlan,
        frequencies: PlanView,
    ) -> MkResult<BufferId> {
        let pupil = self.import(p);
        let output = p.scratch(BufferDesc::new(
            ElementType::Complex32,
            [frequencies.desc.shape[0]],
        ));
        let frame = self
            .geometry
            .chief_frame()
            .map(|v| [v.x, v.y, v.z].map(Df32::from_f64));
        p.operations
            .push(ResidentOp::Transforms(tr::Request::OpticalCorrelation(
                tr::OpticalCorrelation {
                    stencil: None,
                    amplitude: pupil.amplitude,
                    opd: pupil.opd,
                    coverage: pupil.coverage,
                    frequencies,
                    frame,
                    sin_max: self.wave_geometry().sin_max,
                    wavelength_over_index: self.wave_geometry().wavelength,
                    output,
                },
            )));
        Ok(output)
    }
    /// Complex sensor OTF at arbitrary frequencies. Lateral-colour placement is
    /// kept as phase, so callers can combine wavelengths before taking magnitude.
    pub fn otf(
        &self,
        backend: &dyn ComputeBackend,
        snapshot: Snapshot,
        frequencies: &[[f64; 2]],
        ctx: &JobContext,
    ) -> MkResult<WaveResult<Vec<[f64; 2]>>> {
        let fs = frequency_buffer(backend, frequencies, ctx)?;
        let mut p = ResidentPlan::new(snapshot);
        p.settings =
            serde_json::json!({"p6":"direct_otf","frequencies":frequencies,"pupil_grid":self.n});
        let id = p.import(fs.view());
        let output = self.append_correlation(&mut p, p_view(id, fs.desc()))?;
        p.outputs = vec![output];
        let r = backend.submit_plan(p, ctx)?.wait()?;
        let guard = backend.read_view(r.outputs[0].view())?;
        let value = guard
            .as_slice::<[f32; 2]>()?
            .iter()
            .zip(frequencies)
            .map(|(z, f)| {
                let t = -std::f64::consts::TAU
                    * (f[0] * self.image_offset_mm[0] + f[1] * self.image_offset_mm[1]);
                let (s, c) = t.sin_cos();
                [
                    z[0] as f64 * c - z[1] as f64 * s,
                    z[0] as f64 * s + z[1] as f64 * c,
                ]
            })
            .collect();
        Ok(WaveResult {
            value,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
}
fn p_view(id: BufferId, desc: &BufferDesc) -> PlanView {
    PlanView {
        buffer: id,
        offset: 0,
        stride: 1,
        desc: desc.clone(),
    }
}
pub fn frequency_buffer(
    backend: &dyn ComputeBackend,
    fs: &[[f64; 2]],
    ctx: &JobContext,
) -> MkResult<BufferHandle> {
    backend.fill_buffer(
        BufferDesc::new(ElementType::F32, [fs.len(), 2]),
        &mut |mut b| {
            let a = b.as_mut_slice::<f32>()?;
            for (i, v) in fs.iter().enumerate() {
                if i % 4096 == 0 {
                    ctx.check()?;
                }
                for j in 0..2 {
                    if !v[j].is_finite() || !(v[j] as f32).is_finite() {
                        return Err(MkError::InvalidParam("non-finite frequency".into()));
                    }
                    a[2 * i + j] = v[j] as f32;
                }
            }
            Ok(())
        },
    )
}
/// Handles and small image metadata for a PSF appended to a larger plan.
pub struct PsfOutput {
    pub image: BufferId,
    pub sums: BufferId,
    pub size: usize,
    pub spacing: f64,
    pub center: [f64; 2],
    pub normalization: PsfNormalization,
    pub plane_energy: bool,
    pub warnings: Vec<WaveWarning>,
}
impl PsfOutput {
    pub fn finish(
        &self,
        p: &ResidentPupil,
        b: &dyn ComputeBackend,
        image: &BufferHandle,
        sums: &BufferHandle,
    ) -> MkResult<Psf> {
        let s = b.read_view(sums.view())?;
        let s = s.as_slice::<Df32>()?;
        let peak = s[0].to_f64().powi(2);
        let l = p.n as f64 * p.lambda_f();
        let energy = l
            * l
            * if self.plane_energy {
                s[2].to_f64()
            } else {
                s[1].to_f64() / p.geometry.projection_jacobian()
            };
        if !energy.is_finite() || energy <= 0. || !peak.is_finite() || peak <= 0. {
            return Err(MkError::Numerical(
                "wave pupil/plane has no finite normalization".into(),
            ));
        }
        let data = b
            .read_view(image.view())?
            .as_slice::<f32>()?
            .iter()
            .map(|&v| v as f64)
            .collect();
        Ok(Psf {
            size: self.size,
            spacing_mm: self.spacing,
            center_mm: self.center,
            data,
            normalization: self.normalization,
            total_energy: if self.normalization == PsfNormalization::UnitEnergy {
                1.
            } else {
                energy / peak
            },
            diffraction_peak: if self.normalization == PsfNormalization::UnitEnergy {
                peak / energy
            } else {
                1.
            },
            wavelength_um: Some(p.wavelength_um),
            warnings: self.warnings.clone(),
        })
    }
}
