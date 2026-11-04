//! Deterministic rendezvous batching for independent refinements. Each worker
//! runs the unchanged serial solver; the coordinator combines its next optical
//! evaluation with all other active workers. No timer, pruning or rayon worker
//! blocking changes the candidate order or the solver's stopping semantics.
use crate::deriv::FdScheme;
use crate::merit::{Operand, OperandEvaluator, OperandKind, OperandValue};
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};
use std::sync::{Mutex, mpsc};

enum Message {
    Evaluate(usize, Vec<OpticalSystem>, Vec<Operand>),
    Done(usize),
}
type Values = MkResult<Vec<Vec<OperandValue>>>;
pub struct Worker<'a, E: ?Sized> {
    evaluator: &'a E,
    id: usize,
    send: mpsc::Sender<Message>,
    receive: Mutex<mpsc::Receiver<Values>>,
}
impl<E: OperandEvaluator + ?Sized> OperandEvaluator for Worker<'_, E> {
    fn supports(&self, k: &OperandKind) -> bool {
        self.evaluator.supports(k)
    }
    fn fd_relative_step(&self, k: &OperandKind, s: FdScheme) -> Option<f64> {
        self.evaluator.fd_relative_step(k, s)
    }
    fn anchored_differences(&self, base: &OpticalSystem) -> bool {
        self.evaluator.anchored_differences(base)
    }
    fn evaluate_batch(&self, s: &[OpticalSystem], o: &[&Operand], ctx: &JobContext) -> Values {
        ctx.check()?;
        self.send
            .send(Message::Evaluate(
                self.id,
                s.to_vec(),
                o.iter().map(|v| (*v).clone()).collect(),
            ))
            .map_err(|_| MkError::Cancelled)?;
        self.receive
            .lock()
            .unwrap()
            .recv()
            .map_err(|_| MkError::Cancelled)?
    }
}
struct Done {
    id: usize,
    send: mpsc::Sender<Message>,
}
impl Drop for Done {
    fn drop(&mut self) {
        let _ = self.send.send(Message::Done(self.id));
    }
}
/// Run bounded independent jobs in stable rounds. The caller supplies each
/// job's existing solve; reference scoring remains with that solve's caller.
pub fn run<E, T, F>(evaluator: &E, count: usize, ctx: &JobContext, work: F) -> MkResult<Vec<T>>
where
    E: OperandEvaluator + ?Sized,
    T: Send,
    F: Fn(usize, &Worker<'_, E>) -> MkResult<T> + Sync,
{
    std::thread::scope(|scope| {
        let (send, receive) = mpsc::channel();
        let mut replies = Vec::new();
        let mut threads = Vec::new();
        for id in 0..count {
            let (tx, rx) = mpsc::channel();
            replies.push(tx);
            let worker = Worker {
                evaluator,
                id,
                send: send.clone(),
                receive: Mutex::new(rx),
            };
            let f = &work;
            threads.push(scope.spawn(move || {
                let _done = Done {
                    id,
                    send: worker.send.clone(),
                };
                f(id, &worker)
            }));
        }
        drop(send);
        let mut active = vec![true; count];
        let mut pending: Vec<Option<(Vec<OpticalSystem>, Vec<Operand>)>> =
            (0..count).map(|_| None).collect();
        while active.iter().any(|&a| a) {
            while active
                .iter()
                .enumerate()
                .any(|(i, &a)| a && pending[i].is_none())
            {
                match receive.recv() {
                    Ok(Message::Evaluate(id, s, o)) => pending[id] = Some((s, o)),
                    Ok(Message::Done(id)) => active[id] = false,
                    Err(_) => break,
                }
            }
            // Different configurations/merit specs form separate stable groups.
            while let Some(first) = pending.iter().position(Option::is_some) {
                let operands = pending[first].as_ref().unwrap().1.clone();
                let ids: Vec<_> = (first..count)
                    .filter(|&i| pending[i].as_ref().is_some_and(|p| p.1 == operands))
                    .collect();
                let mut systems = Vec::new();
                let mut lengths = Vec::new();
                for &i in &ids {
                    let (s, _) = pending[i].take().unwrap();
                    lengths.push(s.len());
                    systems.extend(s);
                }
                let refs: Vec<_> = operands.iter().collect();
                let result = ctx
                    .check()
                    .and_then(|_| evaluator.evaluate_batch(&systems, &refs, ctx));
                match result {
                    Ok(v) => {
                        let mut rows = v.into_iter();
                        for (&id, n) in ids.iter().zip(lengths) {
                            let _ = replies[id].send(Ok(rows.by_ref().take(n).collect()));
                        }
                    }
                    Err(e) => {
                        for &id in &ids {
                            let _ = replies[id].send(Err(if matches!(e, MkError::Cancelled) {
                                MkError::Cancelled
                            } else {
                                MkError::Numerical(e.to_string())
                            }));
                        }
                    }
                }
            }
        }
        threads.into_iter().map(|t| t.join().unwrap()).collect()
    })
}
