//! Resident transforms. Row-major grids use [rows, columns] (or [length]).
//!
//! FFT bins are unshifted; sign -1 is forward and +1 inverse. Scaling is
//! explicit, including the inverse's 1/N. MFT evaluates exactly
//! `scale * sum f[y,x] exp(sign*2πi * (x*(a0*k+a1*l+a2)
//!                                  +y*(a3*k+a4*l+a5)))`.
//! Coefficients are turns per lattice index: callers fold physical grid
//! spacings/origins into them. A source-origin output phase can be composed
//! with `Phase`. ChirpZ requires a diagonal map; AffineMft a symmetric map.
use crate::{Df32, pending::JobControl, resident::*};
use mklight_core::MkResult;
pub mod correlation;
mod cpu;
pub(crate) use cpu::cis;
pub mod recipe;

#[derive(Clone, Copy, Debug)]
pub enum TransformKind {
    Fft,
    ChirpZ,
    AffineMft,
}
#[derive(Clone, Debug)]
pub struct TransformPlan {
    pub kind: TransformKind,
    pub input: PlanView,
    pub output: BufferId,
    pub dimensions: Vec<usize>,
    pub output_dimensions: Vec<usize>,
    pub sign: i32,
    pub scale: f32,
    pub affine: [Df32; 6],
}
/// Bilinear complex autocorrelation on a resident angular-spectrum grid.
/// `mapping` converts sensor frequencies (cycles/mm) to lattice shifts by
/// the same affine convention as TransformPlan. Coverage weights each sample;
/// fractional-cell zero-lag energy is restored before normalization.
/// For the nonlinear off-axis optical map use `OpticalCorrelation`.
#[derive(Clone, Debug)]
pub struct CorrelationStencil {
    pub pupil: PlanView,
    pub frequencies: PlanView,
    pub mapping: [Df32; 6],
    pub coverage: Option<PlanView>,
    pub output: BufferId,
}
/// Optical correlation keeps amplitude and unwrapped OPD separate, matching
/// wave's coverage-weighted interpolation even across whole phase turns.
/// The frame columns are eu, ev, ew in sensor coordinates; sin_max is the
/// pupil angular scale and wavelength_over_index is λ(mm)/n'.
#[derive(Clone, Debug)]
pub struct OpticalCorrelation {
    /// Optional reusable F32 [frequency_count, n*n, 5] geometry stencil.
    pub stencil: Option<PlanView>,
    pub amplitude: PlanView,
    pub opd: PlanView,
    pub coverage: PlanView,
    pub frequencies: PlanView,
    pub frame: [[Df32; 3]; 3],
    pub sin_max: Df32,
    pub wavelength_over_index: Df32,
    pub output: BufferId,
}
#[derive(Clone, Debug)]
pub enum Request {
    Transform(TransformPlan),
    Correlation(CorrelationStencil),
    OpticalCorrelation(OpticalCorrelation),
    /// Produce reusable sensor geometry, independent of amplitude/OPD/focus.
    PrepareOpticalStencil(OpticalCorrelation),
    Transpose {
        input: PlanView,
        output: BufferId,
    },
    /// Circular convolution on the exact common grid, with no implicit padding.
    Convolve {
        left: PlanView,
        right: PlanView,
        output: BufferId,
    },
    /// Multiply by exp(2πi*(a0*x²+a1*x*y+a2*y²+a3*x+a4*y+a5)).
    Phase {
        input: PlanView,
        coefficients: [Df32; 6],
        output: BufferId,
    },
}
fn grid(v: &PlanView, element: ElementType) -> MkResult<()> {
    if v.desc.element != element
        || v.stride != 1
        || !(1..=2).contains(&v.desc.shape.len())
        || v.desc.shape.iter().any(|&n| n == 0 || n > 8192)
    {
        return Err(bad(
            "transform requires a contiguous positive 1-D/2-D grid, side <=8192",
        ));
    }
    Ok(())
}
fn finite(a: &[Df32]) -> MkResult<()> {
    if a.iter().any(|v| !v.is_finite() || v.to_f64().abs() > 1e8) {
        return Err(bad(
            "transform phase/map coefficients must be finite and <=1e8 turns",
        ));
    }
    Ok(())
}
fn out(p: &ResidentPlan, id: BufferId, shape: &[usize]) -> MkResult<()> {
    if p.desc(id)? != &BufferDesc::new(ElementType::Complex32, shape.to_vec()) {
        return Err(bad("transform output layout"));
    }
    Ok(())
}
fn frequencies(v: &PlanView) -> MkResult<usize> {
    if v.desc.element != ElementType::F32
        || v.stride != 1
        || v.desc.shape.len() != 2
        || v.desc.shape[1] != 2
    {
        return Err(bad("frequencies require f32 [count,2]"));
    }
    Ok(v.desc.shape[0])
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    match r {
        Request::Transform(t) => {
            if t.input.buffer == t.output {
                return unsupported("in-place transform; use a distinct resident destination");
            }
            grid(&t.input, ElementType::Complex32)?;
            if t.input.desc.shape != t.dimensions
                || t.output_dimensions.len() != t.dimensions.len()
                || t.output_dimensions.iter().any(|&n| n == 0 || n > 8192)
                || ![-1, 1].contains(&t.sign)
                || !t.scale.is_finite()
            {
                return Err(bad("transform dimensions/sign/scale"));
            }
            finite(&t.affine)?;
            match t.kind {
                TransformKind::Fft if t.dimensions != t.output_dimensions => {
                    return Err(bad("FFT preserves dimensions"));
                }
                TransformKind::ChirpZ if t.affine[1] != Df32::ZERO || t.affine[3] != Df32::ZERO => {
                    return Err(bad("ChirpZ requires a diagonal map"));
                }
                TransformKind::AffineMft if t.affine[1] != t.affine[3] => {
                    return Err(bad("affine MFT requires a symmetric map"));
                }
                _ => {}
            }
            out(p, t.output, &t.output_dimensions)?;
            // Bound the phase argument before quadratic products lose the T budget.
            let extent = t
                .dimensions
                .iter()
                .chain(&t.output_dimensions)
                .copied()
                .max()
                .unwrap() as f64;
            if t.affine
                .iter()
                .any(|a| a.to_f64().abs() * extent * extent > 1e9)
            {
                return Err(bad("MFT phase exceeds qualified double-float range"));
            }
        }
        Request::Transpose { input, output } => {
            grid(input, ElementType::Complex32)?;
            if input.desc.shape.len() != 2 {
                return Err(bad("transpose requires a 2-D grid"));
            }
            out(p, *output, &[input.desc.shape[1], input.desc.shape[0]])?;
        }
        Request::Convolve {
            left,
            right,
            output,
        } => {
            grid(left, ElementType::Complex32)?;
            grid(right, ElementType::Complex32)?;
            if left.desc != right.desc {
                return Err(bad("spectral inputs must have identical grids"));
            }
            out(p, *output, &left.desc.shape)?;
        }
        Request::Phase {
            input,
            coefficients,
            output,
        } => {
            grid(input, ElementType::Complex32)?;
            finite(coefficients)?;
            let n = *input.desc.shape.iter().max().unwrap() as f64;
            if coefficients.iter().any(|a| a.to_f64().abs() * n * n > 1e9) {
                return Err(bad("phase exceeds qualified double-float range"));
            }
            out(p, *output, &input.desc.shape)?;
        }
        Request::Correlation(c) => {
            grid(&c.pupil, ElementType::Complex32)?;
            finite(&c.mapping)?;
            if let Some(v) = &c.coverage {
                grid(v, ElementType::F32)?;
                if v.desc.shape != c.pupil.desc.shape {
                    return Err(bad("coverage shape"));
                }
            }
            out(p, c.output, &[frequencies(&c.frequencies)?])?;
        }
        Request::OpticalCorrelation(c) | Request::PrepareOpticalStencil(c) => {
            grid(&c.amplitude, ElementType::F32)?;
            grid(&c.coverage, ElementType::F32)?;
            grid(&c.opd, ElementType::DoubleFloat)?;
            let s = &c.amplitude.desc.shape;
            if s.len() != 2 || s[0] != s[1] || c.coverage.desc.shape != *s || c.opd.desc.shape != *s
            {
                return Err(bad("optical correlation requires matching square grids"));
            }
            for v in &c.frame {
                finite(v)?;
            }
            if !(0.0..1.0).contains(&c.sin_max.to_f64())
                || c.sin_max.to_f64() == 0.0
                || !c.wavelength_over_index.is_finite()
                || c.wavelength_over_index.to_f64() <= 0.0
            {
                return Err(bad("optical correlation angular scale/wavelength"));
            }
            let frame = c.frame.map(|v| v.map(Df32::to_f64));
            for i in 0..3 {
                for j in 0..3 {
                    let dot = frame[i]
                        .iter()
                        .zip(frame[j])
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    if (dot - if i == j { 1.0 } else { 0.0 }).abs() > 1e-6 {
                        return Err(bad("optical sensor frame must be orthonormal"));
                    }
                }
            }
            let det = frame[0][0] * (frame[1][1] * frame[2][2] - frame[1][2] * frame[2][1])
                - frame[0][1] * (frame[1][0] * frame[2][2] - frame[1][2] * frame[2][0])
                + frame[0][2] * (frame[1][0] * frame[2][1] - frame[1][1] * frame[2][0]);
            if det < 0.0 {
                return Err(bad("optical sensor frame must be right-handed"));
            }
            if c.sin_max.to_f64() >= frame[2][2] {
                return unsupported("optical angular spectrum crosses the sensor horizon");
            }
            let count = frequencies(&c.frequencies)?;
            let stencil_desc = BufferDesc::new(ElementType::F32, vec![count, s[0] * s[1], 5]);
            if let Some(v) = &c.stencil
                && (v.desc != stencil_desc || v.stride != 1)
            {
                return Err(bad("optical stencil layout"));
            }
            if matches!(r, Request::PrepareOpticalStencil(_)) {
                if p.desc(c.output)? != &stencil_desc
                    || stencil_desc.byte_len()? > 256 * 1024 * 1024
                {
                    return Err(bad(
                        "optical stencil layout/256 MiB limit; tile frequencies",
                    ));
                }
            } else {
                out(p, c.output, &[count])?;
            }
        }
    }
    if !matches!(
        r,
        Request::Correlation(_)
            | Request::OpticalCorrelation(_)
            | Request::PrepareOpticalStencil(_)
    ) {
        let recipe = recipe::request(r)?;
        if recipe
            .steps
            .iter()
            .any(|s| s.p.count as u64 * 32 > 1024 * 1024 * 1024)
        {
            return unsupported("transform working set exceeds 1 GiB; tile the batch");
        }
    }
    Ok(())
}
pub fn accesses(r: &Request) -> Access {
    let (reads, output) = match r {
        Request::Transform(t) => (vec![t.input.clone()], t.output),
        Request::Correlation(c) => {
            let mut v = vec![c.pupil.clone(), c.frequencies.clone()];
            v.extend(c.coverage.clone());
            (v, c.output)
        }
        Request::OpticalCorrelation(c) => {
            let mut v = vec![
                c.amplitude.clone(),
                c.opd.clone(),
                c.coverage.clone(),
                c.frequencies.clone(),
            ];
            v.extend(c.stencil.clone());
            (v, c.output)
        }
        Request::PrepareOpticalStencil(c) => (vec![c.frequencies.clone()], c.output),
        Request::Transpose { input, output } | Request::Phase { input, output, .. } => {
            (vec![input.clone()], *output)
        }
        Request::Convolve {
            left,
            right,
            output,
        } => (vec![left.clone(), right.clone()], *output),
    };
    Access {
        reads,
        writes: vec![output],
        systems: vec![],
    }
}
pub fn execute(
    r: &Request,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    cpu::execute(r, x, ctl)
}
