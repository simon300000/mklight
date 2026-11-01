//! Resident scalar-wave operations. Phase and distance differences use double-float;
//! complex fields use f32. All images have low-y row zero.
use crate::{Df32, pending::JobControl, resident::*};
use mklight_core::MkResult;
mod cpu;
pub mod resample;
pub mod spec;

/// Resident cell-centred pupil: amplitude and fractional coverage, with unwrapped
/// OPD in waves. These are separate channels so interpolation never wraps phase.
#[derive(Clone, Debug)]
pub struct Pupil {
    pub amplitude: PlanView,
    pub coverage: PlanView,
    pub opd: PlanView,
}
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub sin_max: Df32,
    /// Wavelength in mm divided by image-space index.
    pub wavelength: Df32,
    /// Zero denotes the telecentric limit; otherwise positive mm.
    pub radius: Df32,
}
#[derive(Clone, Debug)]
pub enum Request {
    /// Weighted Zernike design matrix for P3 QR. Samples are DF [rows,4]
    /// (u,v,OPD,sqrt(weight)); zero weight masks a row. Terms are U32 [cols,2]
    /// (n,m as signed bits), validated by the basis producer.
    Zernike {
        samples: PlanView,
        terms: PlanView,
        standard: bool,
        matrix: BufferId,
        rhs: BufferId,
    },
    Resample(resample::Resample),
    /// Many focus planes and frequencies in one tiled operation. Geometry is
    /// P2's reusable [frequencies,n*n,5] optical correlation stencil.
    FocusCorrelation {
        on_axis: bool,
        pupil: Pupil,
        geometry: Geometry,
        stencil: PlanView,
        shifts: Option<PlanView>,
        output: BufferId,
    },
    /// Debye field padded at the low corner. The three scalar planes are
    /// c A/gamma, c A²/gamma², c A²/(gamma |k.normal|), for fixed-tree sums.
    Prepare {
        pupil: Pupil,
        geometry: Geometry,
        normal: [Df32; 3],
        field: BufferId,
        moments: BufferId,
    },
    /// Normalize |field|² using Prepare's reduced moments. FFT bins optionally
    /// wrap about floor(size/2); MFT output is already in image order.
    Intensity {
        field: PlanView,
        sums: PlanView,
        geometry: Geometry,
        pupil_side: usize,
        size: usize,
        shifted: bool,
        unit_energy: bool,
        projection: f32,
        output: BufferId,
    },
    /// Exact scalar propagation, tiled in both samples and pixels. Plane vectors
    /// are in the chief frame. The same reduced moments normalize this plane.
    Huygens {
        pupil: Pupil,
        geometry: Geometry,
        sums: PlanView,
        size: usize,
        spacing: Df32,
        origin: [Df32; 3],
        x_axis: [Df32; 3],
        y_axis: [Df32; 3],
        unit_energy: bool,
        output: BufferId,
    },
    /// Exact refocus OPD for every requested chief-axis shift, [planes,n,n].
    Focus {
        pupil: Pupil,
        geometry: Geometry,
        shifts: PlanView,
        output: BufferId,
    },
    /// Pixel-aperture and Gaussian diffusion MTF at arbitrary sensor frequencies.
    Detector {
        frequencies: PlanView,
        aperture: [f32; 2],
        sigma: f32,
        output: BufferId,
    },
    /// Integrate PSF rows and columns, yielding [2,size], x first.
    LineSpread {
        image: PlanView,
        spacing: f32,
        output: BufferId,
    },
}
fn view(v: &PlanView, ty: ElementType, shape: &[usize]) -> MkResult<()> {
    if v.stride != 1 || v.desc != BufferDesc::new(ty, shape) {
        return Err(bad("wave input layout"));
    }
    Ok(())
}
fn out(p: &ResidentPlan, id: BufferId, ty: ElementType, shape: &[usize]) -> MkResult<()> {
    if p.desc(id)? != &BufferDesc::new(ty, shape) {
        return Err(bad("wave output layout"));
    }
    Ok(())
}
fn square(v: &PlanView, ty: ElementType) -> MkResult<usize> {
    let s = &v.desc.shape;
    if s.len() != 2 || s[0] != s[1] || !(1..=8192).contains(&s[0]) {
        return Err(bad("wave square grid"));
    }
    view(v, ty, s)?;
    Ok(s[0])
}
fn pupil(v: &Pupil, g: Geometry) -> MkResult<usize> {
    let n = square(&v.amplitude, ElementType::F32)?;
    view(&v.coverage, ElementType::F32, &[n, n])?;
    view(&v.opd, ElementType::DoubleFloat, &[n, n])?;
    if !g.sin_max.is_finite()
        || !(0.0..1.0).contains(&g.sin_max.to_f64())
        || g.sin_max.hi == 0.0
        || !g.wavelength.is_finite()
        || g.wavelength.hi <= 0.0
        || !g.radius.is_finite()
        || g.radius.hi < 0.0
    {
        return Err(bad("wave geometry"));
    }
    Ok(n)
}
fn sums(v: &PlanView) -> MkResult<()> {
    view(v, ElementType::DoubleFloat, &[3])
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        Request::Zernike {
            samples,
            terms,
            matrix,
            rhs,
            ..
        } => {
            let a = &samples.desc.shape;
            let t = &terms.desc.shape;
            if a.len() != 2 || a[1] != 4 || t.len() != 2 || t[1] != 2 || t[0] == 0 || a[0] < t[0] {
                return Err(bad("Zernike fit layout"));
            }
            view(samples, DoubleFloat, a)?;
            view(terms, U32, t)?;
            out(p, *matrix, DoubleFloat, &[1, a[0], t[0]])?;
            out(p, *rhs, DoubleFloat, &[1, a[0]])?;
        }

        Request::Resample(r) => resample::validate(r, p)?,
        Request::FocusCorrelation {
            pupil: v,
            geometry,
            stencil,
            shifts,
            output,
            ..
        } => {
            let n = pupil(v, *geometry)?;
            let s = &stencil.desc.shape;
            if s.len() != 3
                || s[1] != n * n
                || s[2] != 5
                || shifts.as_ref().is_some_and(|v| v.desc.shape.len() != 1)
            {
                return Err(bad("focus correlation layout"));
            }
            view(stencil, F32, s)?;
            if let Some(shifts) = shifts {
                view(shifts, DoubleFloat, &shifts.desc.shape)?;
            }
            let planes = shifts.as_ref().map_or(1, |s| s.desc.shape[0]);
            out(p, *output, Complex32, &[planes, s[0]])?;
        }
        Request::Prepare {
            pupil: v,
            geometry,
            normal,
            field,
            moments,
        } => {
            let n = pupil(v, *geometry)?;
            let d = p.desc(*field)?;
            if d.element != Complex32
                || d.shape.len() != 2
                || d.shape[0] != d.shape[1]
                || d.shape[0] < n
                || d.shape[0] > 8192
            {
                return Err(bad("padded wave field"));
            }
            if normal.iter().any(|v| !v.is_finite())
                || (normal.iter().map(|v| v.to_f64().powi(2)).sum::<f64>() - 1.0).abs() > 1e-6
            {
                return Err(bad("wave plane normal"));
            }
            out(p, *moments, F32, &[3, n * n])?;
        }
        Request::Intensity {
            field,
            sums: s,
            geometry,
            pupil_side,
            size,
            projection,
            output,
            ..
        } => {
            square(field, Complex32)?;
            sums(s)?;
            if *pupil_side == 0
                || *size == 0
                || *size > 8192
                || !projection.is_finite()
                || *projection <= 0.
                || !geometry.wavelength.is_finite()
                || geometry.wavelength.hi <= 0.
                || !geometry.sin_max.is_finite()
                || geometry.sin_max.hi <= 0.
            {
                return Err(bad("wave intensity geometry"));
            }
            if let Request::Intensity { shifted: false, .. } = r {
                view(field, Complex32, &[*size, *size])?;
            }
            out(p, *output, F32, &[*size, *size])?;
        }
        Request::Huygens {
            pupil: v,
            geometry,
            sums: s,
            size,
            spacing,
            origin,
            x_axis,
            y_axis,
            output,
            ..
        } => {
            pupil(v, *geometry)?;
            sums(s)?;
            if *size == 0 || *size > 1024 || !spacing.is_finite() || spacing.hi <= 0. {
                return Err(bad("Huygens grid/work/spacing limit"));
            }
            if origin
                .iter()
                .chain(x_axis)
                .chain(y_axis)
                .any(|v| !v.is_finite())
            {
                return Err(bad("Huygens plane"));
            }
            for (a, b, want) in [
                (x_axis, x_axis, 1.),
                (y_axis, y_axis, 1.),
                (x_axis, y_axis, 0.),
            ] {
                if (a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| a.to_f64() * b.to_f64())
                    .sum::<f64>()
                    - want)
                    .abs()
                    > 1e-6
                {
                    return Err(bad("Huygens orthonormal axes"));
                }
            }
            out(p, *output, F32, &[*size, *size])?;
        }
        Request::Focus {
            pupil: v,
            geometry,
            shifts,
            output,
        } => {
            let n = pupil(v, *geometry)?;
            if shifts.desc.shape.len() != 1 {
                return Err(bad("focus shifts layout"));
            }
            view(shifts, DoubleFloat, &shifts.desc.shape)?;
            out(p, *output, DoubleFloat, &[shifts.desc.shape[0], n, n])?;
        }
        Request::Detector {
            frequencies,
            aperture,
            sigma,
            output,
        } => {
            let s = &frequencies.desc.shape;
            if s.len() != 2
                || s[1] != 2
                || aperture.iter().any(|a| !a.is_finite() || *a <= 0.)
                || !sigma.is_finite()
                || *sigma < 0.
            {
                return Err(bad("detector frequencies/aperture"));
            }
            view(frequencies, F32, s)?;
            out(p, *output, F32, &[s[0]])?;
        }
        Request::LineSpread {
            image,
            spacing,
            output,
        } => {
            let n = square(image, F32)?;
            if !spacing.is_finite() || *spacing <= 0. {
                return Err(bad("line spread spacing"));
            }
            out(p, *output, F32, &[2, n])?;
        }
    }
    Ok(())
}
pub fn accesses(r: &Request) -> Access {
    let pv = |v: &Pupil| vec![v.amplitude.clone(), v.coverage.clone(), v.opd.clone()];
    let (reads, writes) = match r {
        Request::Zernike {
            samples,
            terms,
            matrix,
            rhs,
            ..
        } => (vec![samples.clone(), terms.clone()], vec![*matrix, *rhs]),
        Request::Resample(r) => return resample::access(r),
        Request::FocusCorrelation {
            pupil,
            stencil,
            shifts,
            output,
            ..
        } => {
            let mut v = pv(pupil);
            v.push(stencil.clone());
            v.extend(shifts.clone());
            (v, vec![*output])
        }
        Request::Prepare {
            pupil,
            field,
            moments,
            ..
        } => (pv(pupil), vec![*field, *moments]),
        Request::Intensity {
            field,
            sums,
            output,
            ..
        } => (vec![field.clone(), sums.clone()], vec![*output]),
        Request::Huygens {
            pupil,
            sums,
            output,
            ..
        } => {
            let mut v = pv(pupil);
            v.push(sums.clone());
            (v, vec![*output])
        }
        Request::Focus {
            pupil,
            shifts,
            output,
            ..
        } => {
            let mut v = pv(pupil);
            v.push(shifts.clone());
            (v, vec![*output])
        }
        Request::Detector {
            frequencies,
            output,
            ..
        } => (vec![frequencies.clone()], vec![*output]),
        Request::LineSpread { image, output, .. } => (vec![image.clone()], vec![*output]),
    };
    Access {
        reads,
        writes,
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
