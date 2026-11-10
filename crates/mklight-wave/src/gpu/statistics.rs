use super::*;
use mklight_gpu::{
    collectives::{QrFit, Request as Collective},
    wave_ops::statistics::Statistics,
};
#[derive(Clone, Debug)]
pub struct WavefrontStatistics {
    pub rms: f64,
    pub pv: f64,
    pub rms_tilt_removed: f64,
    pub pv_tilt_removed: f64,
    pub strehl: f64,
}
impl ResidentPupil {
    /// Debye-weighted RMS/PV and exact reference-point Strehl. The three-column
    /// tilt fit uses the same pivoted QR as the lattice Zernike fit.
    pub fn statistics(
        &self,
        backend: &dyn ComputeBackend,
        snapshot: Snapshot,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<WavefrontStatistics>> {
        let mut p = ResidentPlan::new(snapshot);
        p.settings = serde_json::json!({"operation":"wavefront_statistics","pupil_grid":self.n});
        let rows = self.n * self.n;
        let pupil = self.import(&mut p);
        let samples = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [rows, 4]));
        let sums = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [4]));
        p.operations.push(ResidentOp::Wave(w::Request::Statistics(
            Statistics::Prepare {
                pupil,
                geometry: self.wave_geometry(),
                samples,
                sums,
            },
        )));
        let terms =
            backend.fill_buffer(BufferDesc::new(ElementType::U32, [3, 2]), &mut |mut f| {
                f.as_mut_slice::<u32>()?
                    .copy_from_slice(&[0, 0, 1, 1, 1, (-1i32) as u32]);
                Ok(())
            })?;
        let t = p.import(terms.view());
        let matrix = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, rows, 3]));
        let rhs = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, rows]));
        p.operations.push(ResidentOp::Wave(w::Request::Zernike {
            samples: p.view(samples)?,
            terms: p.view(t)?,
            matrix,
            rhs,
        }));
        let factors = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, rows + 5, 3]));
        let solution = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 3]));
        let rank_condition = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 4]));
        p.operations
            .push(ResidentOp::Collectives(Collective::QrFit(QrFit {
                matrix: p.view(matrix)?,
                rhs: p.view(rhs)?,
                rank_tolerance: 1e-12,
                factors,
                solution,
                rank_condition,
            })));
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [5]));
        p.operations.push(ResidentOp::Wave(w::Request::Statistics(
            Statistics::Residual {
                samples: p.view(samples)?,
                sums: p.view(sums)?,
                solution: p.view(solution)?,
                output,
            },
        )));
        p.outputs = vec![output, rank_condition];
        let r = backend.submit_plan(p, ctx)?.wait()?;
        let info = backend.read_view(r.outputs[1].view())?;
        let info = info.as_slice::<Df32>()?;
        if info[3].hi != 0. {
            return Err(MkError::Numerical(format!(
                "wavefront tilt QR status {}, rank {}",
                info[3].hi, info[0].hi
            )));
        }
        let a = backend.read_view(r.outputs[0].view())?;
        let a = a.as_slice::<Df32>()?;
        Ok(WaveResult {
            value: WavefrontStatistics {
                rms: a[0].to_f64(),
                pv: a[1].to_f64(),
                rms_tilt_removed: a[2].to_f64(),
                pv_tilt_removed: a[3].to_f64(),
                strehl: a[4].to_f64(),
            },
            run_info: r.run_info,
            stats: r.stats,
        })
    }
}
