//! Resident optimisation algebra. All numeric channels retain double-float
//! residuals; candidate/stencil order is supplied by the solver, never scheduled
//! by a kernel. No operation reads another row for a bad index: a
//! `Difference` column with a row index out of range publishes status 1 (and
//! NaN), and a `FitRows` row with a value index out of range is zero, like a
//! masked one.
use crate::{Df32, resident::*};
use mklight_core::MkResult;
mod cpu;
pub const VERSION: u32 = crate::consts::MKL_OPTIM_VERSION;

#[derive(Clone, Debug)]
pub enum Request {
    /// Gather weighted pupil-fit rows. Values/mask [K], indices [B,M],
    /// basis [B,M,N] with sqrt(weight) in column zero; masked rows are zero.
    FitRows {
        values: PlanView,
        mask: PlanView,
        indices: PlanView,
        basis: PlanView,
        matrix: BufferId,
        rhs: BufferId,
    },
    /// Squared weighted fit residual and weight, channel-major [2,B*M].
    FitError {
        matrix: PlanView,
        rhs: PlanView,
        coefficients: PlanView,
        output: BufferId,
    },
    /// Values [B,O], targets [O], comparison U32 [O]: equal=0, <=1, >=2,
    /// identity=3 (constraints). Nonfinite values remain nonfinite.
    Residuals {
        values: PlanView,
        targets: PlanView,
        comparison: PlanView,
        output: BufferId,
    },
    /// Values [B,O], rows U32 [V,3] (base, first, second); steps DF [V,2]
    /// (first multiplier, second multiplier). Computes
    /// a*(first-base)+b*(second-base), avoiding cancellation of large offsets.
    /// Output [O,V]; a pinned column has both multipliers zero.
    Difference {
        values: PlanView,
        rows: PlanView,
        steps: PlanView,
        output: BufferId,
        status: BufferId,
    },
    /// Batched matrix product A [B,M,K] × C [B,K,N], optionally transpose A.
    /// Fixed ascending inner order, double-float products and accumulation.
    Product {
        left: PlanView,
        right: PlanView,
        transpose_left: bool,
        output: BufferId,
    },
    /// Candidate/trial update [B,V]; nominal may be [1,V] for broadcast.
    Perturb {
        nominal: PlanView,
        delta: PlanView,
        output: BufferId,
    },
    /// Weighted damped least squares without normal equations: [sqrt(w) J;
    /// sqrt(lambda) diag(d)] and [-sqrt(w) r;0], ready for the collectives'
    /// batched QR.
    LmRows {
        jacobian: PlanView,
        residuals: PlanView,
        weights: PlanView,
        diagonal: PlanView,
        damping: Df32,
        matrix: BufferId,
        rhs: BufferId,
    },
}
fn input(v: &PlanView, ty: ElementType, dims: usize) -> MkResult<()> {
    if v.desc.element != ty
        || v.desc.shape.len() != dims
        || v.stride != 1
        || v.desc.len()? > u32::MAX as usize / ty.words()
    {
        return Err(bad("optim input layout"));
    }
    Ok(())
}
fn output(
    p: &ResidentPlan,
    id: BufferId,
    ty: ElementType,
    shape: impl Into<Vec<usize>>,
) -> MkResult<()> {
    let d = BufferDesc::new(ty, shape);
    if p.desc(id)? != &d || d.len()? > u32::MAX as usize / ty.words() {
        return Err(bad("optim output layout"));
    }
    Ok(())
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::{DoubleFloat as D, U32};
    match r {
        Request::FitRows {
            values,
            mask,
            indices,
            basis,
            matrix,
            rhs,
        } => {
            input(values, D, 1)?;
            input(mask, U32, 1)?;
            input(indices, U32, 2)?;
            input(basis, D, 3)?;
            if values.desc.shape != mask.desc.shape
                || indices.desc.shape != basis.desc.shape[..2]
                || basis.desc.shape[2] == 0
            {
                return Err(bad("optim pupil fit dimensions"));
            }
            output(p, *matrix, D, basis.desc.shape.clone())?;
            output(p, *rhs, D, indices.desc.shape.clone())?;
        }
        Request::FitError {
            matrix,
            rhs,
            coefficients,
            output: o,
        } => {
            input(matrix, D, 3)?;
            input(rhs, D, 2)?;
            input(coefficients, D, 2)?;
            let [b, m, n] = matrix.desc.shape[..] else {
                unreachable!()
            };
            if n == 0 || rhs.desc.shape != [b, m] || coefficients.desc.shape != [b, n] {
                return Err(bad("optim fit residual dimensions"));
            }
            output(p, *o, D, [2, b * m])?;
        }
        Request::Residuals {
            values,
            targets,
            comparison,
            output: o,
        } => {
            input(values, D, 2)?;
            input(targets, D, 1)?;
            input(comparison, U32, 1)?;
            if targets.desc.shape != [values.desc.shape[1]]
                || comparison.desc.shape != targets.desc.shape
            {
                return Err(bad("optim operands"));
            }
            output(p, *o, D, values.desc.shape.clone())?;
        }
        Request::Difference {
            values,
            rows,
            steps,
            output: o,
            status,
        } => {
            input(values, D, 2)?;
            input(rows, U32, 2)?;
            input(steps, D, 2)?;
            if values.desc.shape[1] == 0
                || rows.desc.shape[1] != 3
                || steps.desc.shape != [rows.desc.shape[0], 2]
            {
                return Err(bad("optim stencils"));
            }
            output(p, *o, D, [values.desc.shape[1], rows.desc.shape[0]])?;
            output(p, *status, U32, [rows.desc.shape[0]])?;
        }
        Request::Product {
            left,
            right,
            transpose_left,
            output: o,
        } => {
            input(left, D, 3)?;
            input(right, D, 3)?;
            let (m, k) = if *transpose_left {
                (left.desc.shape[2], left.desc.shape[1])
            } else {
                (left.desc.shape[1], left.desc.shape[2])
            };
            if left.desc.shape[0] != right.desc.shape[0] || k != right.desc.shape[1] {
                return Err(bad("optim product dimensions"));
            }
            output(p, *o, D, [left.desc.shape[0], m, right.desc.shape[2]])?;
        }
        Request::Perturb {
            nominal,
            delta,
            output: o,
        } => {
            input(nominal, D, 2)?;
            input(delta, D, 2)?;
            if nominal.desc.shape[1] != delta.desc.shape[1]
                || (nominal.desc.shape[0] != 1 && nominal.desc.shape[0] != delta.desc.shape[0])
            {
                return Err(bad("optim perturb dimensions"));
            }
            output(p, *o, D, delta.desc.shape.clone())?;
        }
        Request::LmRows {
            jacobian,
            residuals,
            weights,
            diagonal,
            damping,
            matrix,
            rhs,
        } => {
            input(jacobian, D, 3)?;
            input(residuals, D, 2)?;
            input(weights, D, 1)?;
            input(diagonal, D, 2)?;
            let [b, m, n] = jacobian.desc.shape[..] else {
                unreachable!()
            };
            if n == 0
                || residuals.desc.shape != [b, m]
                || weights.desc.shape != [m]
                || diagonal.desc.shape != [b, n]
                || !damping.is_finite()
                || damping.to_f64() < 0.0
            {
                return Err(bad("optim LM dimensions or damping"));
            }
            output(p, *matrix, D, [b, m + n, n])?;
            output(p, *rhs, D, [b, m + n])?;
        }
    }
    Ok(())
}
pub fn accesses(r: &Request) -> Access {
    let (reads, writes) = match r {
        Request::FitRows {
            values,
            mask,
            indices,
            basis,
            matrix,
            rhs,
        } => (
            vec![values.clone(), mask.clone(), indices.clone(), basis.clone()],
            vec![*matrix, *rhs],
        ),
        Request::FitError {
            matrix,
            rhs,
            coefficients,
            output,
        } => (
            vec![matrix.clone(), rhs.clone(), coefficients.clone()],
            vec![*output],
        ),
        Request::Residuals {
            values,
            targets,
            comparison,
            output,
        } => (
            vec![values.clone(), targets.clone(), comparison.clone()],
            vec![*output],
        ),
        Request::Difference {
            values,
            rows,
            steps,
            output,
            status,
        } => (
            vec![values.clone(), rows.clone(), steps.clone()],
            vec![*output, *status],
        ),
        Request::Product {
            left,
            right,
            output,
            ..
        } => (vec![left.clone(), right.clone()], vec![*output]),
        Request::Perturb {
            nominal,
            delta,
            output,
        } => (vec![nominal.clone(), delta.clone()], vec![*output]),
        Request::LmRows {
            jacobian,
            residuals,
            weights,
            diagonal,
            matrix,
            rhs,
            ..
        } => (
            vec![
                jacobian.clone(),
                residuals.clone(),
                weights.clone(),
                diagonal.clone(),
            ],
            vec![*matrix, *rhs],
        ),
    };
    Access {
        reads,
        writes,
        ..Default::default()
    }
}
/// Shared scalar dispatch layout, also used by the emulation twin.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub kind: u32,
    pub count: u32,
    pub offset: u32,
    pub b: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub flag: u32,
    pub damping: Df32,
}
pub fn params(r: &Request) -> Params {
    let mut p = Params {
        kind: 0,
        count: 0,
        offset: 0,
        b: 0,
        m: 0,
        n: 0,
        k: 0,
        flag: 0,
        damping: Df32::ZERO,
    };
    match r {
        Request::FitRows { values, basis, .. } => {
            p.kind = 5;
            p.b = basis.desc.shape[0] as u32;
            p.m = basis.desc.shape[1] as u32;
            p.n = basis.desc.shape[2] as u32;
            p.k = values.desc.shape[0] as u32;
            p.count = p.b * p.m * p.n;
        }
        Request::FitError { matrix, .. } => {
            p.kind = 6;
            p.b = matrix.desc.shape[0] as u32;
            p.m = matrix.desc.shape[1] as u32;
            p.n = matrix.desc.shape[2] as u32;
            p.count = 2 * p.b * p.m;
        }
        Request::Residuals { values, .. } => {
            p.m = values.desc.shape[1] as u32;
            p.count = values.desc.len().unwrap() as u32;
        }
        Request::Difference { values, rows, .. } => {
            p.kind = 1;
            p.b = values.desc.shape[0] as u32;
            p.m = values.desc.shape[1] as u32;
            p.n = rows.desc.shape[0] as u32;
            p.count = p.m * p.n;
        }
        Request::Product {
            left,
            right,
            transpose_left,
            ..
        } => {
            p.kind = 2;
            p.b = left.desc.shape[0] as u32;
            p.flag = u32::from(*transpose_left);
            p.m = left.desc.shape[if *transpose_left { 2 } else { 1 }] as u32;
            p.k = right.desc.shape[1] as u32;
            p.n = right.desc.shape[2] as u32;
            p.count = p.b * p.m * p.n;
        }
        Request::Perturb { nominal, delta, .. } => {
            p.kind = 3;
            p.m = delta.desc.shape[1] as u32;
            p.flag = u32::from(nominal.desc.shape[0] == 1);
            p.count = delta.desc.len().unwrap() as u32;
        }
        Request::LmRows {
            jacobian, damping, ..
        } => {
            p.kind = 4;
            p.b = jacobian.desc.shape[0] as u32;
            p.m = jacobian.desc.shape[1] as u32;
            p.n = jacobian.desc.shape[2] as u32;
            p.count = p.b * (p.m + p.n) * p.n;
            p.damping = *damping;
        }
    }
    p
}
pub fn execute(
    r: &Request,
    x: &mut crate::resident::cpu::Execution,
    ctl: &crate::pending::JobControl,
) -> MkResult<()> {
    cpu::execute(r, x, ctl)
}
