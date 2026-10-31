//! Backend-neutral Stockham/chirp schedule. No optical data is inspected here.
use super::*;
use crate::consts::*;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    pub mode: u32,
    pub count: u32,
    pub width: u32,
    pub height: u32,
    pub nx: u32,
    pub ny: u32,
    pub ox: u32,
    pub oy: u32,
    pub radix: u32,
    pub span: u32,
    pub sign_bits: u32,
    pub scale_bits: u32,
    pub start: u32,
    pub conjugate: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub affine: [[f32; 2]; 6],
}
impl Params {
    pub fn new(mode: u32, w: usize, h: usize) -> Self {
        Self {
            mode,
            count: (w * h) as u32,
            width: w as u32,
            height: h as u32,
            scale_bits: 1f32.to_bits(),
            ..Self::zeroed()
        }
    }
}
#[derive(Clone, Debug)]
pub struct Step {
    pub a: usize,
    pub b: usize,
    pub p: Params,
    pub cache: bool,
}
/// Slots 0 and 1 are external inputs; step i produces slot i+2.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub steps: Vec<Step>,
    pub output: usize,
}
pub fn shape(d: &[usize]) -> (usize, usize) {
    (*d.last().unwrap(), if d.len() == 2 { d[0] } else { 1 })
}
pub fn smooth(mut n: usize) -> bool {
    for r in [2, 3, 5, 7] {
        while n.is_multiple_of(r) {
            n /= r;
        }
    }
    n == 1
}
pub fn fast(mut n: usize) -> usize {
    while !smooth(n) {
        n += 1;
    }
    n
}
impl Recipe {
    pub fn new() -> Self {
        Self {
            steps: vec![],
            output: 0,
        }
    }
    /// Content keys for immutable subgraphs; no data-dependent node is cached.
    pub fn cache_keys(&self) -> Vec<String> {
        let mut hashes = vec![blake3::hash(b"input0"), blake3::hash(b"input1")];
        let mut keys = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let mut h = blake3::Hasher::new();
            h.update(bytemuck::bytes_of(&step.p));
            for d in dependencies(step) {
                h.update(hashes[d].as_bytes());
            }
            let hash = h.finalize();
            keys.push(format!("p2-v1:{}", hash.to_hex()));
            hashes.push(hash);
        }
        keys
    }
    pub fn push(&mut self, a: usize, b: usize, p: Params, cache: bool) -> usize {
        let id = self.steps.len() + 2;
        self.steps.push(Step { a, b, p, cache });
        self.output = id;
        id
    }
    fn rows(&mut self, mut a: usize, w: usize, h: usize, sign: i32) -> usize {
        let mut p = Params::new(MKL_TRANSFORM_TWIDDLE, w, 1);
        p.sign_bits = sign as u32;
        let tw = self.push(0, 0, p, true);
        if w <= 4096 {
            return self.push(a, tw, Params::new(MKL_TRANSFORM_LOCAL_FFT, w, h), false);
        }
        let mut span = 1;
        let mut rem = w;
        for r in [2, 3, 5, 7] {
            while rem.is_multiple_of(r) {
                let mut p = Params::new(MKL_TRANSFORM_FFT, w, h);
                p.radix = r as u32;
                p.span = span as u32;
                a = self.push(a, tw, p, false);
                span *= r;
                rem /= r;
            }
        }
        a
    }
    fn fft(&mut self, a: usize, w: usize, h: usize, sign: i32) -> usize {
        let a = self.rows(a, w, h, sign);
        if h == 1 {
            return a;
        }
        let a = self.push(a, 0, Params::new(MKL_TRANSFORM_TRANSPOSE, w, h), false);
        let a = self.rows(a, h, w, sign);
        self.push(a, 0, Params::new(MKL_TRANSFORM_TRANSPOSE, h, w), false)
    }
    fn chirp_rows(
        &mut self,
        input: usize,
        w: usize,
        rows: usize,
        ow: usize,
        sign: i32,
        scale: f32,
        a: Df32,
        b: Df32,
    ) -> usize {
        let pw = fast(w + ow - 1);
        let mut p = Params::new(MKL_TRANSFORM_KERNEL, pw, 1);
        p.nx = w as u32;
        p.ny = 1;
        p.ox = ow as u32;
        p.oy = 1;
        p.sign_bits = sign as u32;
        p.affine = [a, Df32::ZERO, b, Df32::ZERO, Df32::ZERO, Df32::ZERO].map(|v| [v.hi, v.lo]);
        let kernel = self.push(0, 0, p, false);
        let kernel = self.rows(kernel, pw, 1, -1);
        self.steps[kernel - 2].cache = true;
        p.mode = MKL_TRANSFORM_PRE;
        p.height = rows as u32;
        p.ny = rows as u32;
        p.oy = rows as u32;
        p.count = (pw * rows) as u32;
        p.reserved0 = 1;
        let src = self.push(input, 0, p, false);
        let src = self.rows(src, pw, rows, -1);
        let mut multiply = Params::new(MKL_TRANSFORM_MULTIPLY, pw, rows);
        multiply.reserved0 = 1;
        let src = self.push(src, kernel, multiply, false);
        let src = self.rows(src, pw, rows, 1);
        p.mode = MKL_TRANSFORM_POST;
        p.count = (ow * rows) as u32;
        p.scale_bits = (scale / pw as f32).to_bits();
        self.push(src, 0, p, false)
    }
    fn chirp(
        &mut self,
        input: usize,
        w: usize,
        h: usize,
        ow: usize,
        oh: usize,
        sign: i32,
        scale: f32,
        affine: [Df32; 6],
    ) -> usize {
        let (pw, ph) = (
            fast(w + ow - 1),
            if h == 1 && oh == 1 {
                1
            } else {
                fast(h + oh - 1)
            },
        );
        let mut p = Params::new(MKL_TRANSFORM_KERNEL, pw, ph);
        p.nx = w as u32;
        p.ny = h as u32;
        p.ox = ow as u32;
        p.oy = oh as u32;
        p.sign_bits = sign as u32;
        p.affine = affine.map(|v| [v.hi, v.lo]);
        let kernel = self.push(0, 0, p, false);
        let kernel = self.fft(kernel, pw, ph, -1);
        self.steps[kernel - 2].cache = true;
        p.mode = MKL_TRANSFORM_PRE;
        let a = self.push(input, 0, p, false);
        let a = self.fft(a, pw, ph, -1);
        let a = self.push(
            a,
            kernel,
            Params::new(MKL_TRANSFORM_MULTIPLY, pw, ph),
            false,
        );
        let a = self.fft(a, pw, ph, 1);
        p.mode = MKL_TRANSFORM_POST;
        p.count = (ow * oh) as u32;
        p.scale_bits = (scale / (pw * ph) as f32).to_bits();
        self.push(a, 0, p, false)
    }
}
impl Default for Recipe {
    fn default() -> Self {
        Self::new()
    }
}
pub fn build(t: &TransformPlan) -> MkResult<Recipe> {
    let mut r = Recipe::new();
    let (w, h) = shape(&t.dimensions);
    let (ow, oh) = shape(&t.output_dimensions);
    match t.kind {
        TransformKind::Fft if smooth(w) && smooth(h) => {
            let a = r.fft(0, w, h, t.sign);
            let mut p = Params::new(MKL_TRANSFORM_COPY, w, h);
            p.scale_bits = t.scale.to_bits();
            r.push(a, 0, p, false);
        }
        _ => {
            let affine = if matches!(t.kind, TransformKind::Fft) {
                [
                    Df32::from_f64(1.0 / w as f64),
                    Df32::ZERO,
                    Df32::ZERO,
                    Df32::ZERO,
                    Df32::from_f64(1.0 / h as f64),
                    Df32::ZERO,
                ]
            } else {
                t.affine
            };
            if affine[1] == Df32::ZERO && affine[3] == Df32::ZERO {
                let a = r.chirp_rows(
                    0,
                    w,
                    h,
                    ow,
                    t.sign,
                    if h == 1 && oh == 1 { t.scale } else { 1.0 },
                    affine[0],
                    affine[2],
                );
                if h != 1 || oh != 1 {
                    let a = r.push(a, 0, Params::new(MKL_TRANSFORM_TRANSPOSE, ow, h), false);
                    let a = r.chirp_rows(a, h, ow, oh, t.sign, t.scale, affine[4], affine[5]);
                    r.push(a, 0, Params::new(MKL_TRANSFORM_TRANSPOSE, oh, ow), false);
                }
            } else {
                r.chirp(0, w, h, ow, oh, t.sign, t.scale, affine);
            }
        }
    }
    Ok(r)
}
pub fn request(r: &Request) -> MkResult<Recipe> {
    if let Request::Transform(t) = r {
        return build(t);
    }
    let a = accesses(r);
    let (w, h) = shape(&a.reads[0].desc.shape);
    let mut out = Recipe::new();
    match r {
        Request::Transpose { .. } => {
            out.push(0, 0, Params::new(MKL_TRANSFORM_TRANSPOSE, w, h), false);
        }
        Request::Multiply {
            scale, conjugate, ..
        } => {
            let mut p = Params::new(MKL_TRANSFORM_MULTIPLY, w, h);
            p.scale_bits = scale.to_bits();
            p.conjugate = *conjugate as u32;
            out.push(0, 1, p, false);
        }
        Request::Phase { coefficients, .. } => {
            let mut p = Params::new(MKL_TRANSFORM_PHASE, w, h);
            p.affine = coefficients.map(|v| [v.hi, v.lo]);
            out.push(0, 0, p, false);
        }
        Request::Convolve { .. } => {
            // Prime lengths use the same Bluestein recipe as an ordinary FFT.
            let transform = |input: usize, out: &mut Recipe, sign: i32| -> MkResult<usize> {
                if smooth(w) && smooth(h) {
                    Ok(out.fft(input, w, h, sign))
                } else {
                    Ok(out.chirp(
                        input,
                        w,
                        h,
                        w,
                        h,
                        sign,
                        1.0,
                        [
                            Df32::from_f64(1.0 / w as f64),
                            Df32::ZERO,
                            Df32::ZERO,
                            Df32::ZERO,
                            Df32::from_f64(1.0 / h as f64),
                            Df32::ZERO,
                        ],
                    ))
                }
            };
            let l = transform(0, &mut out, -1)?;
            let r = transform(1, &mut out, -1)?;
            let a = out.push(l, r, Params::new(MKL_TRANSFORM_MULTIPLY, w, h), false);
            let a = transform(a, &mut out, 1)?;
            let mut p = Params::new(MKL_TRANSFORM_COPY, w, h);
            p.scale_bits = (1.0 / (w * h) as f32).to_bits();
            out.push(a, 0, p, false);
        }
        _ => return Err(bad("correlation has its own tiled schedule")),
    }
    Ok(out)
}
/// Dependencies exclude dummy bindings, allowing immutable kernel reuse.
pub fn dependencies(s: &Step) -> Vec<usize> {
    match s.p.mode {
        MKL_TRANSFORM_TWIDDLE | MKL_TRANSFORM_KERNEL => vec![],
        MKL_TRANSFORM_FFT | MKL_TRANSFORM_LOCAL_FFT | MKL_TRANSFORM_MULTIPLY => vec![s.a, s.b],
        _ => vec![s.a],
    }
}
