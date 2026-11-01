use super::*;
use crate::{
    fenv::{FpEnv, Ftz, Ieee},
    pending::JobControl,
    resident::cpu::Execution,
};
fn words<'a>(x: &'a Execution, v: &PlanView) -> &'a [u32] {
    let w = v.desc.element.words();
    &x.words(v.buffer)[v.offset * w..(v.offset + v.desc.len().unwrap()) * w]
}
fn d(w: &[u32], i: usize) -> Df32 {
    Df32::new(f32::from_bits(w[2 * i]), f32::from_bits(w[2 * i + 1]))
}
pub fn execute(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    if x.flush_to_zero {
        run::<Ftz>(r, x, ctl)
    } else {
        run::<Ieee>(r, x, ctl)
    }
}
fn run<E: FpEnv>(r: &Request, x: &mut Execution, ctl: &JobControl) -> MkResult<()> {
    let a = accesses(r);
    let p = params(r);
    let ins: Vec<_> = a.reads.iter().map(|v| words(x, v)).collect();
    let mut out = vec![Df32::ZERO; p.count as usize];
    let mut status = vec![];
    let mut rhs = vec![];
    if p.kind == 1 {
        status.resize(p.n as usize, 0);
    }
    if p.kind == 4 {
        rhs.resize((p.b * (p.m + p.n)) as usize, Df32::ZERO);
    }
    let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
    for (i, v) in out.iter_mut().enumerate() {
        if i % 1024 == 0 {
            ctl.check()?;
        }
        *v = match p.kind {
            0 => {
                let col = i % m;
                let c = ins[2][col];
                let raw = d(ins[0], i);
                let z = raw.add_in::<E>(-d(ins[1], col));
                if c == 3 {
                    raw
                } else if (c == 1 && z.to_f64() < 0.0) || (c == 2 && z.to_f64() > 0.0) {
                    Df32::ZERO
                } else {
                    z
                }
            }
            1 => {
                let col = i % n;
                let row = i / n;
                let ids = &ins[1][3 * col..3 * col + 3];
                if ids.iter().any(|&q| q >= p.b) {
                    status[col] = 1;
                    Df32::from_f32(f32::NAN)
                } else {
                    let base = d(ins[0], ids[0] as usize * m + row);
                    let x = d(ins[0], ids[1] as usize * m + row).add_in::<E>(-base);
                    let y = d(ins[0], ids[2] as usize * m + row).add_in::<E>(-base);
                    x.mul_in::<E>(d(ins[2], 2 * col))
                        .add_in::<E>(y.mul_in::<E>(d(ins[2], 2 * col + 1)))
                }
            }
            2 => {
                let batch = i / (m * n);
                let row = i / n % m;
                let col = i % n;
                let mut s = Df32::ZERO;
                for j in 0..k {
                    let ai = batch * m * k
                        + if p.flag == 1 {
                            j * m + row
                        } else {
                            row * k + j
                        };
                    s = s.add_in::<E>(
                        d(ins[0], ai).mul_in::<E>(d(ins[1], (batch * k + j) * n + col)),
                    );
                }
                s
            }
            3 => d(ins[0], if p.flag == 1 { i % m } else { i }).add_in::<E>(d(ins[1], i)),
            4 => {
                let batch = i / ((m + n) * n);
                let row = i / n % (m + n);
                let col = i % n;
                if row < m {
                    let w = d(ins[2], row).sqrt_in::<E>();
                    if col == 0 {
                        rhs[batch * (m + n) + row] = -w.mul_in::<E>(d(ins[1], batch * m + row));
                    }
                    w.mul_in::<E>(d(ins[0], (batch * m + row) * n + col))
                } else if row - m == col {
                    p.damping
                        .sqrt_in::<E>()
                        .mul_in::<E>(d(ins[3], batch * n + col))
                } else {
                    Df32::ZERO
                }
            }
            _ => unreachable!(),
        };
    }
    x.words_mut(a.writes[0])
        .copy_from_slice(bytemuck::cast_slice(&out));
    if p.kind == 1 {
        x.words_mut(a.writes[1]).copy_from_slice(&status);
    }
    if p.kind == 4 {
        x.words_mut(a.writes[1])
            .copy_from_slice(bytemuck::cast_slice(&rhs));
    }
    ctl.check()
}
