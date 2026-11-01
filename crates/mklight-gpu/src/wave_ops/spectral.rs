//! Incoherent wavelength accumulation on a common image/frequency grid.
use super::*;
#[derive(Clone, Debug)]
pub struct Spectral {
    /// Equally shaped images (F32) or OTF curves (Complex32), at most 16 inputs.
    pub inputs: Vec<PlanView>,
    pub weights: Vec<Df32>,
    /// For complex curves, apply the placement phase before summing.
    pub frequencies: Option<PlanView>,
    pub offsets: Vec<[Df32; 2]>,
    pub output: BufferId,
}
pub fn validate(r: &Spectral, p: &ResidentPlan) -> MkResult<()> {
    let n = r.inputs.len();
    if n == 0 || n > 16 || r.weights.len() != n || r.offsets.len() != n {
        return Err(bad("spectral input count"));
    }
    let desc = &r.inputs[0].desc;
    if !matches!(desc.element, ElementType::F32 | ElementType::Complex32)
        || r.inputs.iter().any(|v| v.stride != 1 || &v.desc != desc)
        || r.weights
            .iter()
            .chain(r.offsets.iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err(bad("spectral layout/weights"));
    }
    if let Some(v) = &r.frequencies {
        view(v, ElementType::F32, &[desc.len()?, 2])?;
        if desc.element != ElementType::Complex32 {
            return Err(bad("spectral phase needs complex inputs"));
        }
    }
    out(p, r.output, desc.element, &desc.shape)
}
pub fn access(r: &Spectral) -> Access {
    let mut reads = r.inputs.clone();
    reads.extend(r.frequencies.clone());
    Access {
        reads,
        writes: vec![r.output],
        systems: vec![],
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub count: u32,
    pub inputs: u32,
    pub complex: u32,
    pub phase: u32,
    pub weights: [Df32; 16],
    pub offsets: [[Df32; 2]; 16],
}
pub fn params(r: &Spectral) -> Params {
    use bytemuck::Zeroable;
    let mut p = Params::zeroed();
    p.count = r.inputs[0].desc.len().unwrap() as u32;
    p.inputs = r.inputs.len() as u32;
    p.complex = u32::from(r.inputs[0].desc.element == ElementType::Complex32);
    p.phase = u32::from(r.frequencies.is_some());
    p.weights[..r.weights.len()].copy_from_slice(&r.weights);
    p.offsets[..r.offsets.len()].copy_from_slice(&r.offsets);
    p
}
pub fn cpu(
    r: &Spectral,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    let p = params(r);
    let complex = p.complex != 0;
    let width = if complex { 2 } else { 1 };
    let values: Vec<&[f32]> = r
        .inputs
        .iter()
        .map(|v| {
            let a: &[f32] = bytemuck::cast_slice(x.words(v.buffer));
            &a[v.offset * width..]
        })
        .collect();
    let frequencies = r.frequencies.as_ref().map(|v| {
        let a: &[f32] = bytemuck::cast_slice(x.words(v.buffer));
        &a[v.offset..]
    });
    let mut output = vec![0f32; p.count as usize * width];
    for k in 0..p.count as usize {
        if k % 4096 == 0 {
            ctl.check()?;
        }
        let mut sum = [Df32::ZERO; 2];
        for (j, a) in values.iter().enumerate() {
            let mut z = [a[k * width], if complex { a[k * width + 1] } else { 0. }];
            if let Some(f) = frequencies {
                let phase =
                    -(r.offsets[j][0].mul_f32(f[k * 2]) + r.offsets[j][1].mul_f32(f[k * 2 + 1]));
                let q = spec::cis(phase);
                z = [z[0] * q[0] - z[1] * q[1], z[0] * q[1] + z[1] * q[0]];
            }
            for i in 0..width {
                sum[i] = sum[i] + r.weights[j].mul_f32(z[i]);
            }
        }
        for i in 0..width {
            output[k * width + i] = spec::scalar(sum[i]);
        }
    }
    x.words_mut(r.output)
        .copy_from_slice(bytemuck::cast_slice(&output));
    Ok(())
}
