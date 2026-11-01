use mklight_core::{JobContext, MkResult, OpticalSystem, samples};
use mklight_optim::{gpu::scheduler, merit::*};
use std::sync::Mutex;
#[derive(Default)]
struct BatchSizes(Mutex<Vec<usize>>);
impl OperandEvaluator for BatchSizes {
    fn supports(&self, _: &OperandKind) -> bool {
        true
    }
    fn evaluate_batch(
        &self,
        s: &[OpticalSystem],
        ops: &[&Operand],
        _: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        self.0.lock().unwrap().push(s.len());
        Ok(s.iter()
            .map(|s| vec![Ok(s.surfaces[1].thickness); ops.len()])
            .collect())
    }
}
#[test]
fn independent_workers_share_stable_rounds() {
    let e = BatchSizes::default();
    let ctx = JobContext::new();
    let out = scheduler::run(&e, 4, &ctx, |id, w| {
        let mut s = samples::singlet();
        s.surfaces[1].thickness = id as f64;
        let op = Operand::new(OperandKind::Efl);
        let mut values = Vec::new();
        for _ in 0..2 + id % 2 {
            values.push(
                w.evaluate_batch(std::slice::from_ref(&s), &[&op], &ctx)?[0][0]
                    .clone()
                    .unwrap(),
            );
        }
        Ok(values)
    })
    .unwrap();
    assert_eq!(*e.0.lock().unwrap(), vec![4, 4, 2]);
    assert_eq!(
        out,
        vec![vec![0.; 2], vec![1.; 3], vec![2.; 2], vec![3.; 3]]
    );
}
#[test]
fn cancelled_workers_finish_without_replacement() {
    let e = BatchSizes::default();
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    let result = scheduler::run(&e, 4, &ctx, |_, w| {
        w.evaluate_batch(&[samples::singlet()], &[], &ctx)
    });
    assert!(matches!(result, Err(mklight_core::MkError::Cancelled)));
    assert!(e.0.lock().unwrap().is_empty());
}
