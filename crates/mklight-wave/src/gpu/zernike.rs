use super::*;
use crate::{RaySample, ZernikeBasis, ZernikeOrdering, ZernikeSet};
use mklight_gpu::collectives::{QrFit, Request};
/// P3 reports the infinity-norm condition estimate of the equilibrated R,
/// distinct from the f64 reference fit's spectral condition number.
#[derive(Clone, Debug)]
pub struct ZernikeSolution {
    pub set: ZernikeSet,
    pub rank: usize,
    pub condition_inf: f64,
    pub residual_norm: f64,
}
/// Fit the original traced lattice, weighted by amplitude², using device
/// basis evaluation and double-float pivoted/reorthogonalized QR (P3).
pub fn fit_zernike(
    backend: &dyn ComputeBackend,
    snapshot: Snapshot,
    samples: &[RaySample],
    basis: &ZernikeBasis,
    ctx: &JobContext,
) -> MkResult<WaveResult<ZernikeSolution>> {
    if samples.len() < basis.len() || basis.is_empty() || basis.terms().iter().any(|t| t.n > 64) {
        return Err(MkError::InvalidParam("Zernike sample/term count".into()));
    }
    let rows = samples.len();
    let cols = basis.len();
    let a = backend.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [rows, 4]),
        &mut |mut b| {
            for (i, (row, s)) in b
                .as_mut_slice::<Df32>()?
                .chunks_mut(4)
                .zip(samples)
                .enumerate()
            {
                if i % 1024 == 0 {
                    ctx.check()?;
                }
                if [s.u, s.v, s.opd_waves, s.amplitude]
                    .iter()
                    .any(|v| !v.is_finite())
                    || s.amplitude < 0.
                {
                    return Err(MkError::InvalidParam("Zernike sample".into()));
                }
                row.copy_from_slice(&[s.u, s.v, s.opd_waves, s.amplitude].map(Df32::from_f64));
            }
            Ok(())
        },
    )?;
    let terms = backend.fill_buffer(
        BufferDesc::new(ElementType::U32, [cols, 2]),
        &mut |mut b| {
            for (v, t) in b.as_mut_slice::<u32>()?.chunks_mut(2).zip(basis.terms()) {
                v[0] = t.n;
                v[1] = t.m as u32;
            }
            Ok(())
        },
    )?;
    let mut p = ResidentPlan::new(snapshot);
    p.settings = serde_json::json!({"p6":"zernike","ordering":basis.ordering(),"terms":cols,"rows":rows,"condition":"equilibrated_R_infinity"});
    let a = p.import(a.view());
    let t = p.import(terms.view());
    let matrix = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, rows, cols]));
    let rhs = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, rows]));
    p.operations.push(ResidentOp::Wave(w::Request::Zernike {
        samples: p.view(a)?,
        terms: p.view(t)?,
        standard: basis.ordering() == ZernikeOrdering::Standard,
        matrix,
        rhs,
    }));
    let factors = p.scratch(BufferDesc::new(
        ElementType::DoubleFloat,
        [1, rows + cols + 2, cols],
    ));
    let solution = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, cols]));
    let rank_condition = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 4]));
    p.operations
        .push(ResidentOp::Collectives(Request::QrFit(QrFit {
            matrix: p.view(matrix)?,
            rhs: p.view(rhs)?,
            rank_tolerance: 1e-12,
            factors,
            solution,
            rank_condition,
        })));
    p.outputs = vec![solution, rank_condition];
    let r = backend.submit_plan(p, ctx)?.wait()?;
    let guard = backend.read_view(r.outputs[1].view())?;
    let info = guard.as_slice::<Df32>()?;
    if info[3].hi != 0. {
        return Err(MkError::Numerical(format!(
            "Zernike QR status {}, rank {} of {cols}",
            info[3].hi, info[0].hi
        )));
    }
    let coefficients = backend
        .read_view(r.outputs[0].view())?
        .as_slice::<Df32>()?
        .iter()
        .map(|v| v.to_f64())
        .collect();
    Ok(WaveResult {
        value: ZernikeSolution {
            set: ZernikeSet::new(basis.ordering(), coefficients)?,
            rank: info[0].hi as usize,
            condition_inf: info[1].to_f64(),
            residual_norm: info[2].to_f64(),
        },
        run_info: r.run_info,
        stats: r.stats,
    })
}
