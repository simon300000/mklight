//! Phase-slope image offsets for the existing automatic sampling rules.
use super::*;
#[derive(Clone, Debug)]
pub struct Blur {
    pub pupil: Pupil,
    pub step: usize,
    pub scale: Df32,
    pub inverse: [[Df32; 2]; 2],
    pub offset: [Df32; 2],
    pub points: BufferId,
    pub mask: BufferId,
}
pub fn validate(r: &Blur, p: &ResidentPlan) -> MkResult<()> {
    let n = square(&r.pupil.amplitude, ElementType::F32)?;
    view(&r.pupil.coverage, ElementType::F32, &[n, n])?;
    view(&r.pupil.opd, ElementType::DoubleFloat, &[n, n])?;
    if n < 2
        || r.step == 0
        || !r.scale.is_finite()
        || r.inverse
            .iter()
            .flatten()
            .chain(&r.offset)
            .any(|x| !x.is_finite())
    {
        return Err(bad("blur sampling"));
    }
    let count = (n - 1).div_ceil(r.step).pow(2);
    out(p, r.points, ElementType::DoubleFloat, &[5, count])?;
    out(p, r.mask, ElementType::U32, &[count])
}
pub fn access(r: &Blur) -> Access {
    Access {
        reads: vec![
            r.pupil.amplitude.clone(),
            r.pupil.coverage.clone(),
            r.pupil.opd.clone(),
        ],
        writes: vec![r.points, r.mask],
        systems: vec![],
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub n: u32,
    pub step: u32,
    pub side: u32,
    pub count: u32,
    pub scale: Df32,
    pub offset: [Df32; 2],
    pub inverse: [[Df32; 2]; 2],
}
pub fn params(r: &Blur) -> Params {
    let n = r.pupil.amplitude.desc.shape[0];
    let side = (n - 1).div_ceil(r.step);
    Params {
        n: n as u32,
        step: r.step as u32,
        side: side as u32,
        count: (side * side) as u32,
        scale: r.scale,
        offset: r.offset,
        inverse: r.inverse,
    }
}
pub fn cpu(r: &Blur, x: &mut crate::resident::cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    fn vals<'a, T: bytemuck::Pod>(x: &'a crate::resident::cpu::Execution, v: &PlanView) -> &'a [T] {
        let a = bytemuck::cast_slice(x.words(v.buffer));
        &a[v.offset..v.offset + v.desc.len().unwrap()]
    }
    let a = vals::<f32>(x, &r.pupil.amplitude);
    let c = vals::<f32>(x, &r.pupil.coverage);
    let w = vals::<Df32>(x, &r.pupil.opd);
    let p = params(r);
    let count = p.count as usize;
    let n = p.n as usize;
    let mut out = vec![Df32::ZERO; 5 * count];
    let mut mask = vec![0; count];
    for k in 0..count {
        if k % 4096 == 0 {
            ctl.check()?;
        }
        let i = k % p.side as usize * r.step;
        let j = k / p.side as usize * r.step;
        let q = j * n + i;
        if c[q] <= 0. || c[q + 1] <= 0. || c[q + n] <= 0. {
            continue;
        }
        mask[k] = 1;
        let x = (w[q + 1] - w[q]) * r.scale;
        let y = (w[q + n] - w[q]) * r.scale;
        for axis in 0..2 {
            let v = r.inverse[axis][0] * x + r.inverse[axis][1] * y + r.offset[axis];
            out[axis * count + k] = v;
            out[(axis + 2) * count + k] = if v.hi < 0. { -v } else { v };
        }
        out[4 * count + k] = Df32::from_f32(c[q]) * Df32::from_f32(a[q]) * Df32::from_f32(a[q]);
    }
    x.words_mut(r.points)
        .copy_from_slice(bytemuck::cast_slice(&out));
    x.words_mut(r.mask).copy_from_slice(&mask);
    Ok(())
}
