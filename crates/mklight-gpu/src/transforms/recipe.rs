//! Backend-neutral transform schedule, shared by the CPU-f32 specification
//! and Metal. Slots 0 and 1 are the external inputs; step i writes slot i+2.
use super::*;
use crate::consts::*;
use bytemuck::{Pod, Zeroable};

/// Longest row a backend transforms in one shared-memory pass; longer FFT
/// rows run as global Stockham stages and longer chirps are not fused.
pub const LOCAL_MAX: usize = 4096;

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
    /// FFT stage radices, four bits per stage, first stage lowest.
    pub radices: [u32; 2],
    pub stages: u32,
    pub span: u32,
    pub sign_bits: u32,
    pub scale_bits: u32,
    /// Bit 1 ([`BROADCAST`]): broadcast the second operand over rows. Bit 0
    /// is reserved.
    pub flags: u32,
    pub reserved: u32,
    pub affine: [[f32; 2]; 6],
}
pub const BROADCAST: u32 = 2;
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
    /// `h` independent length-`w` FFT rows.
    pub fn fft(w: usize, h: usize, sign: i32) -> Self {
        let mut p = Self::new(MKL_TRANSFORM_FFT_ROWS, w, h);
        p.sign_bits = sign as u32;
        let r = radices(w);
        p.stages = r.len() as u32;
        for (i, r) in r.into_iter().enumerate() {
            p.radices[i / 8] |= r << (4 * (i % 8));
        }
        p
    }
    pub fn radix(&self, stage: usize) -> usize {
        (self.radices[stage / 8] >> (4 * (stage % 8)) & 15) as usize
    }
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
    pub fn sign(&self) -> f32 {
        self.sign_bits as i32 as f32
    }
}
/// Stockham radices of a 2/3/5/7-smooth length: eights, then a four or two
/// (1024 = 8·8·4·4), then 3, 5 and 7.
pub fn radices(mut n: usize) -> Vec<u32> {
    let mut twos = 0;
    while n.is_multiple_of(2) {
        n /= 2;
        twos += 1;
    }
    let mut r = vec![8; twos / 3];
    match twos % 3 {
        1 if !r.is_empty() => {
            r.pop();
            r.extend([4, 4]);
        }
        1 => r.push(2),
        2 => r.push(4),
        _ => {}
    }
    for f in [3, 5, 7] {
        while n.is_multiple_of(f) {
            n /= f;
            r.push(f as u32);
        }
    }
    debug_assert_eq!(n, 1);
    r
}
/// Forward Stockham twiddles from f64: for the stage with span p and radix
/// R, entry `p-1 + k(R-1) + t-1` is exp(-2πi·t·k/(pR)). Inverse
/// transforms conjugate them.
pub fn twiddles(n: usize) -> Vec<[f32; 2]> {
    let mut out = vec![[0.; 2]; n - 1];
    fill_twiddles(n, &mut out);
    out
}
/// Write the same table directly into shared storage (n-1 complex elements).
pub fn fill_twiddles(n: usize, out: &mut [[f32; 2]]) {
    let mut index = 0;
    let mut p = 1;
    for r in radices(n).into_iter().map(|r| r as usize) {
        for k in 0..p {
            for t in 1..r {
                let (s, c) = (std::f64::consts::TAU * (t * k) as f64 / (p * r) as f64).sin_cos();
                out[index] = [c as f32, -s as f32];
                index += 1;
            }
        }
        p *= r;
    }
}
/// Fused chirp-row factors from f64: nx pre-chirps exp(s·2πi·(a·x²/2 + b·x))
/// then ox post-chirps exp(s·2πi·a·x²/2). Each double-float coefficient part
/// times an integer power of x is exact in f64 and is reduced modulo a turn.
pub fn chirp_factors(p: &Params) -> Vec<[f32; 2]> {
    let mut out = vec![[0.; 2]; (p.nx + p.ox) as usize];
    fill_chirp_factors(p, &mut out);
    out
}
/// Write chirp factors into the caller's shared allocation.
pub fn fill_chirp_factors(p: &Params, out: &mut [[f32; 2]]) {
    let turns = |c: [f32; 2], m: f64| {
        c.iter()
            .map(|&v| v as f64 * m - (v as f64 * m).round())
            .sum::<f64>()
    };
    let quadratic = |x: usize| turns(p.affine[0], (x * x) as f64 / 2.0);
    let cis = |t: f64| {
        let (s, c) = (std::f64::consts::TAU * p.sign() as f64 * t).sin_cos();
        [c as f32, s as f32]
    };
    let pre = (0..p.nx as usize).map(|x| cis(quadratic(x) + turns(p.affine[2], x as f64)));
    for (out, v) in out
        .iter_mut()
        .zip(pre.chain((0..p.ox as usize).map(|x| cis(quadratic(x)))))
    {
        *out = v;
    }
}
#[derive(Clone, Debug)]
pub struct Step {
    pub a: usize,
    pub b: usize,
    pub p: Params,
    /// Immutable result (a chirp spectrum) kept in the operation cache.
    pub cache: bool,
}
#[derive(Clone, Debug, Default)]
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
fn fast(mut n: usize) -> usize {
    while !smooth(n) {
        n += 1;
    }
    n
}
fn pack(a: [Df32; 6]) -> [[f32; 2]; 6] {
    a.map(|v| [v.hi, v.lo])
}
impl Recipe {
    /// Content keys of immutable subgraphs; data-dependent nodes are never cached.
    pub fn cache_keys(&self) -> Vec<String> {
        let mut hashes = vec![blake3::hash(b"input0"), blake3::hash(b"input1")];
        for step in &self.steps {
            let mut h = blake3::Hasher::new();
            h.update(bytemuck::bytes_of(&step.p));
            for d in dependencies(step) {
                h.update(hashes[d].as_bytes());
            }
            hashes.push(h.finalize());
        }
        hashes[2..]
            .iter()
            .map(|h| format!("p2-v2:{}", h.to_hex()))
            .collect()
    }
    fn push(&mut self, a: usize, b: usize, p: Params) -> usize {
        self.steps.push(Step {
            a,
            b,
            p,
            cache: false,
        });
        self.output = self.steps.len() + 1;
        self.output
    }
    /// Multiply the output step's scale.
    fn scale(&mut self, s: f32) {
        let p = &mut self.steps.last_mut().unwrap().p;
        p.scale_bits = (p.scale() * s).to_bits();
    }
    fn transpose(&mut self, a: usize, w: usize, h: usize) -> usize {
        self.push(a, a, Params::new(MKL_TRANSFORM_TRANSPOSE, w, h))
    }
    fn rows(&mut self, a: usize, w: usize, h: usize, sign: i32) -> usize {
        self.push(a, a, Params::fft(w, h, sign))
    }
    /// 2-D FFT ending in transposed [w][h] layout (a single row FFT if h=1).
    fn fft_t(&mut self, a: usize, w: usize, h: usize, sign: i32) -> usize {
        let a = self.rows(a, w, h, sign);
        if h == 1 {
            return a;
        }
        let a = self.transpose(a, w, h);
        self.rows(a, h, w, sign)
    }
    /// Inverse of [`Self::fft_t`]'s layout: transposed input, natural output.
    fn fft_from_t(&mut self, a: usize, w: usize, h: usize, sign: i32) -> usize {
        if h == 1 {
            return self.rows(a, w, 1, sign);
        }
        let a = self.rows(a, h, w, sign);
        let a = self.transpose(a, h, w);
        self.rows(a, w, h, sign)
    }
    fn fft(&mut self, a: usize, w: usize, h: usize, sign: i32) -> usize {
        let a = self.fft_t(a, w, h, sign);
        if h == 1 { a } else { self.transpose(a, h, w) }
    }
    /// Cached spectrum of the chirp kernel `exp(-sign·πi·q(lag))`, transposed if 2-D.
    fn spectrum(&mut self, mut p: Params, pw: usize, ph: usize) -> usize {
        p.mode = MKL_TRANSFORM_KERNEL;
        p.count = (pw * ph) as u32;
        let k = self.push(0, 0, p);
        let k = self.fft_t(k, pw, ph, -1);
        self.steps[k - 2].cache = true;
        k
    }
    /// Exact 1-D chirp-z along `rows` rows: w inputs → ow outputs per row.
    fn chirp_rows(
        &mut self,
        input: usize,
        (w, rows): (usize, usize),
        ow: usize,
        sign: i32,
        [a, b]: [Df32; 2],
    ) -> usize {
        let pw = fast(w + ow - 1);
        let mut p = Params::new(MKL_TRANSFORM_PRE, pw, rows);
        (p.nx, p.ny, p.ox, p.oy) = (w as u32, rows as u32, ow as u32, rows as u32);
        p.sign_bits = sign as u32;
        p.affine = pack([a, Df32::ZERO, b, Df32::ZERO, Df32::ZERO, Df32::ZERO]);
        let mut k = p;
        k.height = 1;
        (k.ny, k.oy) = (1, 1);
        let kernel = self.spectrum(k, pw, 1);
        if pw <= LOCAL_MAX {
            p.mode = MKL_TRANSFORM_CHIRP_ROWS;
            p.count = (ow * rows) as u32;
            p.scale_bits = (1.0 / pw as f32).to_bits();
            let f = Params::fft(pw, 1, 1);
            (p.radices, p.stages) = (f.radices, f.stages);
            return self.push(input, kernel, p);
        }
        let src = self.push(input, 0, p);
        let src = self.push(src, src, Params::fft(pw, rows, -1));
        let mut m = Params::new(MKL_TRANSFORM_MULTIPLY, pw, rows);
        m.flags = BROADCAST;
        let src = self.push(src, kernel, m);
        let src = self.push(src, src, Params::fft(pw, rows, 1));
        p.mode = MKL_TRANSFORM_POST;
        p.count = (ow * rows) as u32;
        p.scale_bits = (1.0 / pw as f32).to_bits();
        self.push(src, 0, p)
    }
    /// Exact 2-D chirp convolution for a symmetric (possibly skew) map.
    fn chirp(
        &mut self,
        input: usize,
        (w, h): (usize, usize),
        (ow, oh): (usize, usize),
        sign: i32,
        affine: [Df32; 6],
    ) -> usize {
        let pw = fast(w + ow - 1);
        let ph = if h == 1 && oh == 1 {
            1
        } else {
            fast(h + oh - 1)
        };
        let mut p = Params::new(MKL_TRANSFORM_PRE, pw, ph);
        (p.nx, p.ny, p.ox, p.oy) = (w as u32, h as u32, ow as u32, oh as u32);
        p.sign_bits = sign as u32;
        p.affine = pack(affine);
        let kernel = self.spectrum(p, pw, ph);
        let a = self.push(input, 0, p);
        let a = self.fft_t(a, pw, ph, -1);
        let a = self.push(a, kernel, Params::new(MKL_TRANSFORM_MULTIPLY, ph, pw));
        let a = self.fft_from_t(a, pw, ph, 1);
        p.mode = MKL_TRANSFORM_POST;
        p.count = (ow * oh) as u32;
        p.scale_bits = (1.0 / (pw * ph) as f32).to_bits();
        self.push(a, 0, p)
    }
    /// Any-length transform: Stockham FFT if smooth, otherwise exact chirp.
    fn transform(
        &mut self,
        input: usize,
        (w, h): (usize, usize),
        (ow, oh): (usize, usize),
        sign: i32,
        kind: TransformKind,
        affine: [Df32; 6],
    ) -> usize {
        if matches!(kind, TransformKind::Fft) && smooth(w) && smooth(h) {
            return self.fft(input, w, h, sign);
        }
        let affine = if matches!(kind, TransformKind::Fft) {
            let (x, y) = (
                Df32::from_f64(1.0 / w as f64),
                Df32::from_f64(1.0 / h as f64),
            );
            [x, Df32::ZERO, Df32::ZERO, Df32::ZERO, y, Df32::ZERO]
        } else {
            affine
        };
        if affine[1] != Df32::ZERO {
            return self.chirp(input, (w, h), (ow, oh), sign, affine);
        }
        let a = self.chirp_rows(input, (w, h), ow, sign, [affine[0], affine[2]]);
        if h == 1 && oh == 1 {
            return a;
        }
        let a = self.transpose(a, ow, h);
        let a = self.chirp_rows(a, (h, ow), oh, sign, [affine[4], affine[5]]);
        self.transpose(a, oh, ow)
    }
}
pub fn request(r: &Request) -> MkResult<Recipe> {
    let mut out = Recipe::default();
    if let Request::Transform(t) = r {
        let (w, h) = shape(&t.dimensions);
        let (ow, oh) = shape(&t.output_dimensions);
        out.transform(0, (w, h), (ow, oh), t.sign, t.kind, t.affine);
        out.scale(t.scale);
        return Ok(out);
    }
    let (w, h) = shape(&accesses(r).reads[0].desc.shape);
    match r {
        Request::Transpose { .. } => {
            out.transpose(0, w, h);
        }
        Request::Phase { coefficients, .. } => {
            let mut p = Params::new(MKL_TRANSFORM_PHASE, w, h);
            p.affine = pack(*coefficients);
            out.push(0, 0, p);
        }
        Request::Convolve { .. } if smooth(w) && smooth(h) => {
            // Both spectra stay transposed; only the inverse restores the layout.
            let l = out.fft_t(0, w, h, -1);
            let r = out.fft_t(1, w, h, -1);
            let a = out.push(l, r, Params::new(MKL_TRANSFORM_MULTIPLY, h, w));
            out.fft_from_t(a, w, h, 1);
            out.scale(1.0 / (w * h) as f32);
        }
        Request::Convolve { .. } => {
            let z = [Df32::ZERO; 6];
            let l = out.transform(0, (w, h), (w, h), -1, TransformKind::Fft, z);
            let r = out.transform(1, (w, h), (w, h), -1, TransformKind::Fft, z);
            let a = out.push(l, r, Params::new(MKL_TRANSFORM_MULTIPLY, w, h));
            out.transform(a, (w, h), (w, h), 1, TransformKind::Fft, z);
            out.scale(1.0 / (w * h) as f32);
        }
        _ => return Err(bad("correlation has its own schedule")),
    }
    Ok(out)
}
/// Slots a step reads; unused bindings are excluded so kernels can be reused.
pub fn dependencies(s: &Step) -> Vec<usize> {
    match s.p.mode {
        MKL_TRANSFORM_KERNEL => vec![],
        MKL_TRANSFORM_MULTIPLY | MKL_TRANSFORM_CHIRP_ROWS => vec![s.a, s.b],
        _ => vec![s.a],
    }
}
/// The steps a backend runs for a recipe and how many of them read each slot.
#[derive(Clone, Debug)]
pub struct Liveness {
    /// Step i runs: the output needs it and its result is not held already.
    pub active: Vec<bool>,
    /// Active steps reading each slot (inputs 0 and 1, then step i at i+2).
    pub uses: Vec<usize>,
}
impl Recipe {
    /// Walk back from the output, stopping at immutable (`cache`) steps whose
    /// result `held(i)` reports as already available (a cached chirp
    /// spectrum); `held` is asked only for needed cacheable steps.
    pub fn liveness(&self, mut held: impl FnMut(usize) -> MkResult<bool>) -> MkResult<Liveness> {
        let n = self.steps.len();
        let mut needed = vec![false; n + 2];
        let mut have = vec![false; n];
        needed[self.output] = true;
        for i in (0..n).rev() {
            if !needed[i + 2] {
                continue;
            }
            if self.steps[i].cache && held(i)? {
                have[i] = true;
                continue;
            }
            for d in dependencies(&self.steps[i]) {
                needed[d] = true;
            }
        }
        let active: Vec<bool> = (0..n).map(|i| needed[i + 2] && !have[i]).collect();
        let mut uses = vec![0; n + 2];
        for (s, _) in self.steps.iter().zip(&active).filter(|(_, a)| **a) {
            for d in dependencies(s) {
                uses[d] += 1;
            }
        }
        Ok(Liveness { active, uses })
    }
}
/// Geometry of the shared-memory row kernels for rows of `n` points in a
/// group of `shared` complex elements: (elements per thread E, threads per
/// row T = ⌈n/E⌉, rows per group G = shared/(E·T)).
pub fn row_geometry(shared: usize, n: usize) -> (usize, usize, usize) {
    let per = if shared > 2048 { 16 } else { 8 };
    let t = n.div_ceil(per);
    (per, t, shared / (per * t))
}
/// Threads per group of the global Stockham stage kernel; a stage of radix
/// R over `rows` rows of n launches ⌈(n/R)/256⌉ × rows groups.
pub const STAGE_THREADS: usize = 256;
/// The global Stockham stages of an `FFT_ROWS` step whose rows are too long
/// for one shared-memory pass: each stage's parameters (its radix in
/// `radices[0]`, its `span`, the step's scale on the last stage only) and
/// whether it writes the step's destination (else a temporary of the same
/// size), alternating so that the last stage writes the destination.
pub fn global_stages(p: &Params) -> Vec<(Params, bool)> {
    let stages = p.stages as usize;
    let mut span = 1;
    (0..stages)
        .map(|k| {
            let r = p.radix(k);
            let mut q = *p;
            q.radices = [r as u32, 0];
            q.span = span as u32;
            if k + 1 < stages {
                q.scale_bits = 1f32.to_bits();
            }
            span *= r;
            (q, (stages - 1 - k).is_multiple_of(2))
        })
        .collect()
}
