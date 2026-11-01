//! Linear-light image stages. Mapping uses double-float; noise uses the shared
//! Philox raw stream addressed by the final output sample, never tile order.
use super::*;
#[derive(Clone, Debug)]
pub enum Simulation {
    Pattern {
        size: [usize; 2],
        kind: u32,
        output: BufferId,
    },
    Warp {
        source: PlanView,
        mapping: PlanView,
        size: [usize; 2],
        channels: usize,
        origin: [Df32; 2],
        pitch: [Df32; 2],
        output: BufferId,
        uv: BufferId,
    },
    Kernel {
        psf: PlanView,
        spacing: Df32,
        normalization: Df32,
        pitch: [Df32; 2],
        aperture: Option<[Df32; 2]>,
        radius: usize,
        canvas: usize,
        output: BufferId,
    },
    Active {
        uv: PlanView,
        grid: usize,
        tile: [usize; 4],
        radius: usize,
        output: BufferId,
    },
    Weighted {
        image: PlanView,
        uv: PlanView,
        grid: usize,
        node: usize,
        channel: usize,
        tile: [usize; 4],
        radius: usize,
        canvas: usize,
        output: BufferId,
    },
    Crop {
        planes: Vec<PlanView>,
        previous: Option<PlanView>,
        size: [usize; 2],
        radius: usize,
        output: BufferId,
    },
    Assemble {
        tiles: Vec<PlanView>,
        size: [usize; 2],
        channels: usize,
        tile_size: usize,
        output: BufferId,
    },
    Noise {
        input: PlanView,
        seed: u64,
        well: Option<f32>,
        sigma: f32,
        output: BufferId,
    },
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub mode: u32,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub source_channels: u32,
    pub grid: u32,
    pub radius: u32,
    pub canvas: u32,
    pub node: u32,
    pub channel: u32,
    pub tile: [u32; 4],
    pub pattern: u32,
    pub quadrature: u32,
    pub previous: u32,
    pub first: u32,
    pub count: u32,
    pub seed: [u32; 2],
    pub reserved: u32,
    pub origin: [Df32; 2],
    pub pitch: [Df32; 2],
    pub aperture: [Df32; 2],
    pub spacing: Df32,
    pub normalization: Df32,
    pub well: f32,
    pub sigma: f32,
}
fn raster(v: &PlanView) -> MkResult<(usize, usize, usize)> {
    let s = &v.desc.shape;
    if s.len() != 3 || s[0] == 0 || s[1] == 0 || ![1, 3].contains(&s[2]) {
        return Err(bad("image raster"));
    }
    view(v, ElementType::F32, s)?;
    Ok((s[1], s[0], s[2]))
}
pub fn validate(r: &Simulation, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        Simulation::Pattern { size, kind, output } => {
            if size.iter().any(|n| *n == 0 || *n > 4096) || *kind > 3 {
                return Err(bad("image pattern"));
            }
            out(p, *output, F32, &[size[1], size[0], 1])
        }
        Simulation::Warp {
            source,
            mapping,
            size,
            channels,
            origin,
            pitch,
            output,
            uv,
        } => {
            raster(source)?;
            let n = mapping.desc.shape[0].isqrt();
            view(mapping, DoubleFloat, &[n * n, 5])?;
            if n < 2
                || ![1, 3].contains(channels)
                || size.iter().any(|n| *n == 0 || *n > 4096)
                || origin.iter().any(|v| !v.is_finite())
                || pitch.iter().any(|v| !v.is_finite() || v.hi <= 0.)
            {
                return Err(bad("image mapping"));
            }
            out(p, *output, F32, &[size[1], size[0], *channels])?;
            out(p, *uv, DoubleFloat, &[size[1], size[0], 2])
        }
        Simulation::Kernel {
            psf,
            spacing,
            normalization,
            pitch,
            aperture,
            radius,
            canvas,
            output,
        } => {
            square(psf, F32)?;
            if *canvas < 2 * radius + 1
                || *canvas > 8192
                || !spacing.is_finite()
                || spacing.hi <= 0.
                || !normalization.is_finite()
                || normalization.hi <= 0.
                || pitch.iter().any(|v| !v.is_finite() || v.hi <= 0.)
                || aperture.is_some_and(|a| a.iter().any(|v| !v.is_finite() || v.hi <= 0.))
            {
                return Err(bad("image kernel geometry"));
            }
            out(p, *output, Complex32, &[*canvas, *canvas])
        }
        Simulation::Active {
            uv,
            grid,
            tile,
            radius,
            output,
        } => {
            if uv.desc.shape.len() != 3
                || uv.desc.shape[2] != 2
                || *grid < 2
                || tile[2] == 0
                || tile[3] == 0
                || *radius > 1025
            {
                return Err(bad("image active tile"));
            }
            view(uv, DoubleFloat, &uv.desc.shape)?;
            out(p, *output, U32, &[grid * grid])
        }
        Simulation::Weighted {
            image,
            uv,
            grid,
            node,
            channel,
            tile,
            radius,
            canvas,
            output,
        } => {
            let (w, h, c) = raster(image)?;
            view(uv, DoubleFloat, &[h, w, 2])?;
            if *grid < 2
                || *node >= grid * grid
                || *channel >= c
                || *canvas < tile[2].max(tile[3]) + 2 * radius
                || *canvas > 8192
            {
                return Err(bad("weighted image tile"));
            }
            out(p, *output, Complex32, &[*canvas, *canvas])
        }
        Simulation::Crop {
            planes,
            previous,
            size,
            radius,
            output,
        } => {
            if ![1, 3].contains(&planes.len()) {
                return Err(bad("convolution channels"));
            }
            let n = square(&planes[0], Complex32)?;
            for v in planes {
                view(v, Complex32, &[n, n])?;
            }
            if size[0] + 2 * radius > n || size[1] + 2 * radius > n {
                return Err(bad("convolution crop"));
            }
            let shape = [size[1], size[0], planes.len()];
            if let Some(v) = previous {
                view(v, F32, &shape)?;
            }
            out(p, *output, F32, &shape)
        }
        Simulation::Assemble {
            tiles,
            size,
            channels,
            tile_size,
            output,
        } => {
            if *tile_size == 0
                || size.iter().any(|v| *v == 0)
                || ![1, 3].contains(channels)
                || tiles.len() != size[0].div_ceil(*tile_size) * size[1].div_ceil(*tile_size)
            {
                return Err(bad("image tile assembly"));
            }
            let nx = size[0].div_ceil(*tile_size);
            for (i, v) in tiles.iter().enumerate() {
                let w = (*tile_size).min(size[0] - i % nx * tile_size);
                let h = (*tile_size).min(size[1] - i / nx * tile_size);
                view(v, F32, &[h, w, *channels])?;
            }
            out(p, *output, F32, &[size[1], size[0], *channels])
        }
        Simulation::Noise {
            input,
            well,
            sigma,
            output,
            ..
        } => {
            raster(input)?;
            if well.is_some_and(|v| !v.is_finite() || v <= 0.) || !sigma.is_finite() || *sigma < 0.
            {
                return Err(bad("image noise"));
            }
            if well.is_some_and(|v| v < f32::MIN_POSITIVE) {
                return Err(mklight_core::MkError::Unsupported(
                    "shot-noise full well is below the normal f32 range on an FTZ device".into(),
                ));
            }
            out(p, *output, F32, &input.desc.shape)
        }
    }
}
pub fn access(r: &Simulation) -> Access {
    let (reads, writes) = match r {
        Simulation::Pattern { output, .. } => (vec![], vec![*output]),
        Simulation::Warp {
            source,
            mapping,
            output,
            uv,
            ..
        } => (vec![source.clone(), mapping.clone()], vec![*output, *uv]),
        Simulation::Kernel { psf, output, .. } => (vec![psf.clone()], vec![*output]),
        Simulation::Active { uv, output, .. } => (vec![uv.clone()], vec![*output]),
        Simulation::Weighted {
            image, uv, output, ..
        } => (vec![image.clone(), uv.clone()], vec![*output]),
        Simulation::Crop {
            planes,
            previous,
            output,
            ..
        } => {
            let mut v = planes.clone();
            v.extend(previous.clone());
            (v, vec![*output])
        }
        Simulation::Assemble { tiles, output, .. } => (tiles.clone(), vec![*output]),
        Simulation::Noise { input, output, .. } => (vec![input.clone()], vec![*output]),
    };
    Access {
        reads,
        writes,
        systems: vec![],
    }
}
pub fn params(r: &Simulation) -> Params {
    use bytemuck::Zeroable;
    let mut p = Params::zeroed();
    match r {
        Simulation::Pattern { size, kind, .. } => {
            p.width = size[0] as u32;
            p.height = size[1] as u32;
            p.channels = 1;
            p.pattern = *kind;
        }
        Simulation::Warp {
            source,
            mapping,
            size,
            channels,
            origin,
            pitch,
            ..
        } => {
            p.mode = 1;
            p.width = size[0] as u32;
            p.height = size[1] as u32;
            p.channels = *channels as u32;
            p.source_width = source.desc.shape[1] as u32;
            p.source_height = source.desc.shape[0] as u32;
            p.source_channels = source.desc.shape[2] as u32;
            p.grid = mapping.desc.shape[0].isqrt() as u32;
            p.origin = *origin;
            p.pitch = *pitch;
        }
        Simulation::Kernel {
            psf,
            spacing,
            normalization,
            pitch,
            aperture,
            radius,
            canvas,
            ..
        } => {
            p.mode = 2;
            p.source_width = psf.desc.shape[0] as u32;
            p.width = *canvas as u32;
            p.height = *canvas as u32;
            p.channels = 1;
            p.canvas = *canvas as u32;
            p.radius = *radius as u32;
            p.spacing = *spacing;
            p.normalization = *normalization;
            p.pitch = *pitch;
            p.aperture = aperture.unwrap_or([Df32::ZERO; 2]);
            p.quadrature = if aperture.is_some() { 4 } else { 1 };
        }
        Simulation::Active {
            uv,
            grid,
            tile,
            radius,
            ..
        } => {
            p.mode = 3;
            p.width = uv.desc.shape[1] as u32;
            p.height = uv.desc.shape[0] as u32;
            p.grid = *grid as u32;
            p.tile = tile.map(|v| v as u32);
            p.radius = *radius as u32;
        }
        Simulation::Weighted {
            image,
            grid,
            node,
            channel,
            tile,
            radius,
            canvas,
            ..
        } => {
            p.mode = 4;
            p.width = image.desc.shape[1] as u32;
            p.height = image.desc.shape[0] as u32;
            p.channels = image.desc.shape[2] as u32;
            p.grid = *grid as u32;
            p.node = *node as u32;
            p.channel = *channel as u32;
            p.tile = tile.map(|v| v as u32);
            p.radius = *radius as u32;
            p.canvas = *canvas as u32;
        }
        Simulation::Crop {
            planes,
            previous,
            size,
            radius,
            ..
        } => {
            p.mode = 5;
            p.width = size[0] as u32;
            p.height = size[1] as u32;
            p.channels = planes.len() as u32;
            p.canvas = planes[0].desc.shape[0] as u32;
            p.radius = *radius as u32;
            p.previous = u32::from(previous.is_some());
        }
        Simulation::Assemble {
            size,
            channels,
            tile_size,
            ..
        } => {
            p.mode = 6;
            p.width = size[0] as u32;
            p.height = size[1] as u32;
            p.channels = *channels as u32;
            p.canvas = *tile_size as u32;
        }
        Simulation::Noise {
            input,
            seed,
            well,
            sigma,
            ..
        } => {
            p.mode = 7;
            p.width = input.desc.shape[1] as u32;
            p.height = input.desc.shape[0] as u32;
            p.channels = input.desc.shape[2] as u32;
            p.seed = [*seed as u32, (*seed >> 32) as u32];
            p.well = well.unwrap_or(0.);
            p.sigma = *sigma;
        }
    }
    p
}
fn d(v: f32) -> Df32 {
    Df32::from_f32(v)
}
fn f(v: Df32) -> f32 {
    spec::scalar(v)
}
fn clamp(v: Df32) -> Df32 {
    if v.to_f64() < 0. {
        Df32::ZERO
    } else if v.to_f64() > 1. {
        d(1.)
    } else {
        v
    }
}
pub fn pattern(p: &Params, k: usize) -> f32 {
    let x = (k % p.width as usize) as f32;
    let y = (k / p.width as usize) as f32;
    let w = d(p.width as f32);
    let h = d(p.height as f32);
    match p.pattern {
        0 => {
            let white = |v: Df32| {
                let a = v.mul_f32(8.);
                let floor = a.hi.floor();
                let r = a - d(floor) - d(0.1);
                (d(0.9).mul_f32(floor) + if r.hi > 0. { r } else { Df32::ZERO }).mul_f32(0.125)
            };
            f((white(d(x + 1.) / w) - white(d(x) / w))
                * w
                * (white(d(y + 1.) / h) - white(d(y) / h))
                * h)
        }
        1 => {
            let mut sum = 0.;
            for j in 0..16 {
                for i in 0..16 {
                    let u = (x + (i as f32 + 0.5) / 16.) / p.width as f32;
                    let v = (y + (j as f32 + 0.5) / 16.) / p.height as f32;
                    if ((v - 0.5).atan2(u - 0.5) * 24.).cos() > 0. {
                        sum += 1.;
                    }
                }
            }
            sum / 256.
        }
        2 => {
            let lo = d(x) / w;
            let hi = d(x + 1.) / w;
            let primitive = |v: Df32| {
                let edge = d(0.45) + v * d(0.1);
                let a = hi - edge;
                let b = lo - edge;
                let a = if a.hi > 0. { a } else { Df32::ZERO };
                let b = if b.hi > 0. { b } else { Df32::ZERO };
                (a * a - b * b) / d(-0.2)
            };
            f((primitive(d(y + 1.) / h) - primitive(d(y) / h)) * w * h)
        }
        _ => {
            let y = p.height as usize - 1 - k / p.width as usize;
            let scale = [4, 2, 1][(y * 3 / p.height as usize).min(2)];
            let gx = k % p.width as usize / scale;
            let gy = y / scale;
            let font = [
                [17, 27, 21, 17, 17, 17, 17],
                [17, 18, 20, 24, 20, 18, 17],
                [16, 16, 16, 16, 16, 16, 31],
            ];
            if gy % 9 < 7 && gx % 6 < 5 && font[(gx / 6) % 3][gy % 9] & (1 << (4 - gx % 6)) != 0 {
                0.
            } else {
                1.
            }
        }
    }
}
pub fn weights(grid: usize, uv: [Df32; 2]) -> [(usize, Df32); 4] {
    let x = clamp(uv[0]).mul_f32((grid - 1) as f32);
    let y = clamp(uv[1]).mul_f32((grid - 1) as f32);
    let ix = (x.hi.floor() as usize).min(grid - 2);
    let iy = (y.hi.floor() as usize).min(grid - 2);
    let a = x - d(ix as f32);
    let b = y - d(iy as f32);
    let k = iy * grid + ix;
    [
        (k, (d(1.) - a) * (d(1.) - b)),
        (k + 1, a * (d(1.) - b)),
        (k + grid, (d(1.) - a) * b),
        (k + grid + 1, a * b),
    ]
}
fn map(nodes: &[Df32], grid: usize, uv: [Df32; 2], offset: usize) -> [Df32; 2] {
    let mut p = [Df32::ZERO; 2];
    for (k, w) in weights(grid, uv) {
        for j in 0..2 {
            p[j] = p[j] + nodes[k * 5 + offset + j] * w;
        }
    }
    p
}
pub fn inverse(nodes: &[Df32], grid: usize, target: [Df32; 2]) -> Option<[Df32; 2]> {
    let mut uv = std::array::from_fn(|j| {
        clamp((target[j] - nodes[j]) / (nodes[(grid * grid - 1) * 5 + j] - nodes[j]))
    });
    for _ in 0..16 {
        let p = map(nodes, grid, uv, 2);
        let r = [p[0] - target[0], p[1] - target[1]];
        if (r[0] * r[0] + r[1] * r[1]).sqrt().to_f64() < 1e-9 {
            return Some(uv);
        }
        let pairs = weights(grid, uv);
        let ids = pairs.map(|(k, _)| k);
        let xx = clamp(uv[0]).mul_f32((grid - 1) as f32);
        let yy = clamp(uv[1]).mul_f32((grid - 1) as f32);
        let a = xx - d((ids[0] % grid) as f32);
        let b = yy - d((ids[0] / grid) as f32);
        let q = ids.map(|k| [nodes[5 * k + 2], nodes[5 * k + 3]]);
        let du: [Df32; 2] = std::array::from_fn(|j| {
            ((d(1.) - b) * (q[1][j] - q[0][j]) + b * (q[3][j] - q[2][j])).mul_f32((grid - 1) as f32)
        });
        let dv: [Df32; 2] = std::array::from_fn(|j| {
            ((d(1.) - a) * (q[2][j] - q[0][j]) + a * (q[3][j] - q[1][j])).mul_f32((grid - 1) as f32)
        });
        let det = du[0] * dv[1] - dv[0] * du[1];
        if det.to_f64().abs() < 1e-20 {
            return None;
        }
        uv[0] = uv[0] - (dv[1] * r[0] - dv[0] * r[1]) / det;
        uv[1] = uv[1] - (du[0] * r[1] - du[1] * r[0]) / det;
        if uv
            .iter()
            .any(|v| v.to_f64() < -1e-9 || v.to_f64() > 1. + 1e-9)
        {
            return None;
        }
        uv = uv.map(clamp);
    }
    None
}
fn raster_sample(p: &Params, r: &[f32], u: Df32, v: Df32, ch: usize) -> f32 {
    if [u, v].iter().any(|v| v.to_f64() < 0. || v.to_f64() > 1.) {
        return 0.;
    }
    let w = p.source_width as usize;
    let h = p.source_height as usize;
    let x = f(u.mul_f32(w as f32).add_f32(-0.5)).clamp(0., (w - 1) as f32);
    let y = f(v.mul_f32(h as f32).add_f32(-0.5)).clamp(0., (h - 1) as f32);
    let ix = x.floor() as usize;
    let iy = y.floor() as usize;
    let (a, b) = (x - ix as f32, y - iy as f32);
    let at = |x, y| {
        let k = (y * w + x) * p.source_channels as usize;
        if p.source_channels == 1 {
            r[k]
        } else if p.channels == 1 {
            0.2126 * r[k] + 0.7152 * r[k + 1] + 0.0722 * r[k + 2]
        } else {
            r[k + ch]
        }
    };
    (1. - b) * ((1. - a) * at(ix, iy) + a * at((ix + 1).min(w - 1), iy))
        + b * ((1. - a) * at(ix, (iy + 1).min(h - 1))
            + a * at((ix + 1).min(w - 1), (iy + 1).min(h - 1)))
}
pub fn warp(p: &Params, k: usize, source: &[f32], nodes: &[Df32]) -> ([f32; 3], [Df32; 2]) {
    let x = k % p.width as usize;
    let y = k / p.width as usize;
    let target = [
        p.origin[0] + p.pitch[0].mul_f32(x as f32 + 0.5),
        p.origin[1] + p.pitch[1].mul_f32(y as f32 + 0.5),
    ];
    let Some(uv) = inverse(nodes, p.grid as usize, target) else {
        return ([0.; 3], [d(-1.); 2]);
    };
    let ideal = map(nodes, p.grid as usize, uv, 0);
    let coords = [
        (ideal[0] - p.origin[0]) / p.pitch[0].mul_f32(p.width as f32),
        (ideal[1] - p.origin[1]) / p.pitch[1].mul_f32(p.height as f32),
    ];
    let ri = weights(p.grid as usize, uv)
        .iter()
        .fold(Df32::ZERO, |a, (k, w)| a + nodes[k * 5 + 4] * *w);
    let mut out = [0.; 3];
    for ch in 0..p.channels as usize {
        out[ch] = raster_sample(p, source, coords[0], coords[1], ch) * f(ri);
    }
    (out, uv)
}
/// Input-pixel support and the sum of pixel-aperture quadrature weights.
pub fn kernel_bounds(p: &Params, o: usize, axis: usize) -> (usize, usize) {
    let m = p.source_width as usize;
    let lo = ((d(o as f32 - p.radius as f32 - 1.) * p.pitch[axis] - p.aperture[axis].mul_f32(0.5))
        / p.spacing)
        .add_f32((m / 2) as f32);
    let hi = ((d(o as f32 - p.radius as f32 + 1.) * p.pitch[axis] + p.aperture[axis].mul_f32(0.5))
        / p.spacing)
        .add_f32((m / 2) as f32);
    (
        (lo.to_f64().floor() as i64).max(0) as usize,
        (hi.to_f64().ceil() as i64).clamp(0, m as i64 - 1) as usize,
    )
}
pub fn kernel_weight(p: &Params, out: usize, src: usize, axis: usize) -> Df32 {
    let mut weight = Df32::ZERO;
    for q in 0..p.quadrature {
        let at = (p.spacing.mul_f32(src as f32 - (p.source_width / 2) as f32)
            + p.aperture[axis].mul_f32((q as f32 + 0.5) / p.quadrature as f32 - 0.5))
            / p.pitch[axis]
            + d(p.radius as f32);
        let delta = at - d(out as f32);
        let v = d(1.) - if delta.hi < 0. { -delta } else { delta };
        if v.hi > 0. {
            weight = weight + v;
        }
    }
    weight
}
pub fn tile_source(p: &Params, k: usize) -> Option<usize> {
    let hw = p.tile[2] + 2 * p.radius;
    let hh = p.tile[3] + 2 * p.radius;
    let x = k % (if p.mode == 4 { p.canvas } else { hw }) as usize;
    let y = k / (if p.mode == 4 { p.canvas } else { hw }) as usize;
    if x >= hw as usize || y >= hh as usize {
        return None;
    }
    let sx = p.tile[0] as i64 + x as i64 - p.radius as i64;
    let sy = p.tile[1] as i64 + y as i64 - p.radius as i64;
    (sx >= 0 && sy >= 0 && sx < p.width as i64 && sy < p.height as i64)
        .then_some((sy * p.width as i64 + sx) as usize)
}
/// Fixed-series logarithm with power-of-two reduction. It avoids library-log
/// branch differences in rejection decisions; all inputs here are positive normal.
fn logarithm(x: Df32) -> Df32 {
    let exponent = ((x.hi.to_bits() >> 23) & 255) as i32 - 127;
    let scale = f32::from_bits(((127 - exponent) as u32) << 23);
    let m = x.mul_f32(scale);
    let y = (m - d(1.)) / (m + d(1.));
    let yy = y * y;
    let mut term = y;
    let mut sum = y;
    for k in 1..=16 {
        term = term * yy;
        sum = sum + term / d((2 * k + 1) as f32);
    }
    sum.mul_f32(2.)
        + Df32::new(std::f32::consts::LN_2, -1.904654323148236e-9).mul_f32(exponent as f32)
}
/// Cancellation-free Stirling log probability around the mode.
fn log_probability(mean: f32, k: f32) -> Df32 {
    if k < 10. {
        let mut factorial = Df32::ZERO;
        for i in 2..=k as usize {
            factorial = factorial + logarithm(d(i as f32));
        }
        return -d(mean) + d(k) * logarithm(d(mean)) - factorial;
    }
    let t = (d(mean) - d(k)) / d(k);
    let deviance = if t.to_f64().abs() < 0.125 {
        let mut term = t * t;
        let mut sum = -term / d(2.);
        for n in 3..=16 {
            term = term * t;
            sum = if n % 2 == 0 {
                sum - term / d(n as f32)
            } else {
                sum + term / d(n as f32)
            };
        }
        d(k) * sum
    } else {
        d(k) * (logarithm(d(1.) + t) - t)
    };
    let inv = d(1.) / d(k);
    deviance - logarithm(d(k)).mul_f32(0.5) - Df32::from_f64(0.9189385332046727) - inv / d(12.)
        + inv * inv * inv / d(360.)
}
pub fn poisson(mean: f32, next: &mut impl FnMut() -> f32) -> f32 {
    if mean < 30. {
        let limit = crate::fmath::exp(-mean);
        let mut product = 1.;
        let mut n = 0.;
        loop {
            product *= next();
            if product <= limit {
                return n;
            }
            n += 1.;
        }
    }
    let b = 0.931 + 2.53 * mean.sqrt();
    let a = -0.059 + 0.02483 * b;
    let inv = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.);
    loop {
        let u = next() - 0.5;
        let v = next();
        let us = 0.5 - u.abs();
        let k = ((2. * a / us + b) * u + mean + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k;
        }
        if k < 0. || (us < 0.013 && v > us) {
            continue;
        }
        let lhs = logarithm(d(v * inv / (a / (us * us) + b)));
        if (lhs - log_probability(mean, k)).hi <= 0. {
            return k;
        }
    }
}
pub fn noise(p: &Params, k: usize, value: f32) -> f32 {
    if !value.is_finite()
        || (p.well > 0. && (!(value.max(0.) * p.well).is_finite() || value.max(0.) * p.well > 1e12))
    {
        return f32::NAN;
    }
    let mut rng =
        mklight_core::rng::Philox::new(p.seed[0] as u64 | ((p.seed[1] as u64) << 32), k as u64);
    let mut next = || mklight_core::rng::u32_to_unit_f32(rng.next_u32());
    let mut v = value.max(0.);
    if p.well > 0. {
        v = poisson(v * p.well, &mut next) / p.well;
    }
    if p.sigma > 0. {
        v += p.sigma * (-2. * next().ln()).sqrt() * (std::f32::consts::TAU * next()).cos();
    }
    v
}
pub fn cpu(
    r: &Simulation,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    fn vals<'a, T: bytemuck::Pod>(x: &'a crate::resident::cpu::Execution, v: &PlanView) -> &'a [T] {
        let a = bytemuck::cast_slice(x.words(v.buffer));
        &a[v.offset..v.offset + v.desc.len().unwrap()]
    }
    let p = params(r);
    match r {
        Simulation::Pattern { output, .. } => {
            let mut a = vec![0.; (p.width * p.height) as usize];
            for (k, v) in a.iter_mut().enumerate() {
                if k % 256 == 0 {
                    ctl.check()?;
                }
                *v = pattern(&p, k);
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&a));
        }
        Simulation::Warp {
            source,
            mapping,
            output,
            uv,
            ..
        } => {
            let source = vals(x, source);
            let mapping = vals(x, mapping);
            let mut a = vec![0.; (p.width * p.height * p.channels) as usize];
            let mut coords = vec![[Df32::ZERO; 2]; (p.width * p.height) as usize];
            for (k, v) in coords.iter_mut().enumerate() {
                if k % 256 == 0 {
                    ctl.check()?;
                }
                let (z, u) = warp(&p, k, source, mapping);
                *v = u;
                a[k * p.channels as usize..(k + 1) * p.channels as usize]
                    .copy_from_slice(&z[..p.channels as usize]);
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&a));
            x.words_mut(*uv)
                .copy_from_slice(bytemuck::cast_slice(&coords));
        }
        Simulation::Kernel { psf, output, .. } => {
            let psf = vals::<f32>(x, psf);
            let side = (2 * p.radius + 1) as usize;
            let m = p.source_width as usize;
            let mut horizontal = vec![Df32::ZERO; m * side];
            for (k, v) in horizontal.iter_mut().enumerate() {
                if k % 256 == 0 {
                    ctl.check()?;
                }
                let ox = k % side;
                let y = k / side;
                let (lo, hi) = kernel_bounds(&p, ox, 0);
                let mut sum = Df32::ZERO;
                for ix in lo..=hi {
                    sum = sum + d(psf[y * m + ix]) * kernel_weight(&p, ox, ix, 0);
                }
                *v = sum / p.normalization / d(p.quadrature as f32);
            }
            let mut out = vec![[0.; 2]; (p.canvas * p.canvas) as usize];
            for y in 0..side {
                ctl.check()?;
                let (lo, hi) = kernel_bounds(&p, y, 1);
                for ox in 0..side {
                    let mut sum = Df32::ZERO;
                    for iy in lo..=hi {
                        sum = sum + horizontal[iy * side + ox] * kernel_weight(&p, y, iy, 1);
                    }
                    out[y * p.canvas as usize + ox][0] = f(sum / d(p.quadrature as f32));
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
        Simulation::Active { uv, output, .. } => {
            let uv = vals::<Df32>(x, uv).as_chunks::<2>().0;
            let mut active = vec![0; (p.grid * p.grid) as usize];
            for k in 0..((p.tile[2] + 2 * p.radius) * (p.tile[3] + 2 * p.radius)) as usize {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                if let Some(i) = tile_source(&p, k) {
                    if uv[i][0].hi >= 0. {
                        for (node, w) in weights(p.grid as usize, uv[i]) {
                            if w.hi > 0. {
                                active[node] = 1;
                            }
                        }
                    }
                }
            }
            x.words_mut(*output).copy_from_slice(&active);
        }
        Simulation::Weighted {
            image, uv, output, ..
        } => {
            let image = vals::<f32>(x, image);
            let uv = vals::<Df32>(x, uv).as_chunks::<2>().0;
            let mut a = vec![[0.; 2]; (p.canvas * p.canvas) as usize];
            for (k, v) in a.iter_mut().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                if let Some(i) = tile_source(&p, k) {
                    if uv[i][0].hi >= 0. {
                        let weight = weights(p.grid as usize, uv[i])
                            .iter()
                            .filter(|(n, _)| *n == p.node as usize)
                            .fold(Df32::ZERO, |a, (_, w)| a + *w);
                        v[0] =
                            f(weight.mul_f32(image[i * p.channels as usize + p.channel as usize]));
                    }
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&a));
        }
        Simulation::Crop {
            planes,
            previous,
            output,
            ..
        } => {
            let planes: Vec<&[[f32; 2]]> = planes.iter().map(|v| vals(x, v)).collect();
            let prev = previous.as_ref().map(|v| vals::<f32>(x, v));
            let mut a = vec![0.; (p.width * p.height * p.channels) as usize];
            for (k, v) in a.iter_mut().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let ch = k % p.channels as usize;
                let pixel = k / p.channels as usize;
                let x = pixel % p.width as usize;
                let y = pixel / p.width as usize;
                *v = prev.map_or(0., |a| a[k])
                    + planes[ch][(y + 2 * p.radius as usize) * p.canvas as usize
                        + x
                        + 2 * p.radius as usize][0];
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&a));
        }
        Simulation::Assemble {
            tiles,
            tile_size,
            output,
            ..
        } => {
            let nx = (p.width as usize).div_ceil(*tile_size);
            let mut a = vec![0.; (p.width * p.height * p.channels) as usize];
            for (i, v) in tiles.iter().enumerate() {
                ctl.check()?;
                let input = vals::<f32>(x, v);
                let w = v.desc.shape[1];
                let h = v.desc.shape[0];
                let c = p.channels as usize;
                let x0 = i % nx * tile_size;
                let y0 = i / nx * tile_size;
                for y in 0..h {
                    a[((y0 + y) * p.width as usize + x0) * c
                        ..((y0 + y) * p.width as usize + x0 + w) * c]
                        .copy_from_slice(&input[y * w * c..(y + 1) * w * c]);
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&a));
        }
        Simulation::Noise { input, output, .. } => {
            let input = vals::<f32>(x, input);
            let mut a = vec![0.; input.len()];
            for (k, (v, a)) in a.iter_mut().zip(input).enumerate() {
                if k % 256 == 0 {
                    ctl.check()?;
                }
                *v = noise(&p, k, *a);
                if !v.is_finite() {
                    return Err(mklight_core::MkError::Unsupported("image noise exceeds its qualified finite f32 domain (Poisson mean <= 1e12)".into()));
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&a));
        }
    }
    Ok(())
}
