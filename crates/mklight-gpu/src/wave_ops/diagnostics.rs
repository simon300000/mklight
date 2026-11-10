//! Compact diagnostics; fixed 256-cell blocks preserve first-index gap ties.
use super::*;
#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub coverage: PlanView,
    pub opd: PlanView,
    pub extrapolation: PlanView,
    /// DF [10]: covered, unresolved, phase step, gap, pitch, min x/y, max x/y, coverage sum.
    pub output: BufferId,
}
pub fn validate(r: &Diagnostics, p: &ResidentPlan) -> MkResult<()> {
    let n = square(&r.coverage, ElementType::F32)?;
    view(&r.opd, ElementType::DoubleFloat, &[n, n])?;
    view(&r.extrapolation, ElementType::F32, &[3, n * n])?;
    out(p, r.output, ElementType::DoubleFloat, &[10])
}
pub fn access(r: &Diagnostics) -> Access {
    Access {
        reads: vec![r.coverage.clone(), r.opd.clone(), r.extrapolation.clone()],
        writes: vec![r.output],
        systems: vec![],
    }
}
pub fn cpu(
    r: &Diagnostics,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    let c: &[f32] = x.rows(&r.coverage);
    let w: &[Df32] = x.rows(&r.opd);
    let d: &[f32] = x.rows(&r.extrapolation);
    let n = r.coverage.desc.shape[0];
    let nn = n * n;
    let mut a = [0., 0., 0., 0., 1., n as f32, n as f32, 0., 0.];
    for k in 0..nn {
        if k % 4096 == 0 {
            ctl.check()?;
        }
        a[1] += f32::from(d[2 * nn + k] > 0.);
        if d[k] / d[nn + k] > a[3] / a[4] {
            a[3] = d[k];
            a[4] = d[nn + k];
        }
        if c[k] <= 0. {
            continue;
        }
        a[0] += 1.;
        let (ix, iy) = (k % n, k / n);
        a[5] = a[5].min(ix as f32);
        a[6] = a[6].min(iy as f32);
        a[7] = a[7].max(ix as f32);
        a[8] = a[8].max(iy as f32);
        for j in [
            if ix + 1 < n { k + 1 } else { k },
            if iy + 1 < n { k + n } else { k },
        ] {
            if c[j] > 0. {
                a[2] = a[2].max(spec::scalar(w[j] - w[k]).abs());
            }
        }
    }
    let mut area = Df32::ZERO;
    for block in c[..nn].chunks(spec::SUM_BLOCK) {
        let sum = block.iter().fold(Df32::ZERO, |sum, v| sum.add_f32(*v));
        area = area.add_f32(spec::scalar(sum));
    }
    let mut result = [Df32::ZERO; 10];
    result[..9].copy_from_slice(&a.map(Df32::from_f32));
    result[9] = area;
    x.words_mut(r.output)
        .copy_from_slice(bytemuck::cast_slice(&result));
    Ok(())
}
