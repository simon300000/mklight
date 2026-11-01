//! Portable scalar-wave algebra; trigonometry is tolerance-qualified, not bitwise.
use super::*;
use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    pub n: u32,
    pub size: u32,
    pub field_side: u32,
    pub mode: u32,
    pub first: u32,
    pub count: u32,
    pub sample_first: u32,
    pub sample_count: u32,
    pub unit_energy: u32,
    pub shifted: u32,
    pub reserved: [u32; 2],
    pub sin_max: [f32; 2],
    pub wavelength: [f32; 2],
    pub radius: [f32; 2],
    pub spacing: [f32; 2],
    pub origin: [[f32; 2]; 3],
    pub x_axis: [[f32; 2]; 3],
    pub y_axis: [[f32; 2]; 3],
    pub normal: [[f32; 2]; 3],
    pub aperture: [f32; 2],
    pub sigma: f32,
    pub projection: f32,
}
pub fn pair(v: Df32) -> [f32; 2] {
    [v.hi, v.lo]
}
pub fn df(v: [f32; 2]) -> Df32 {
    Df32::new(v[0], v[1])
}
pub fn scalar(v: Df32) -> f32 {
    v.hi + v.lo
}
pub fn params(r: &Request) -> Params {
    let mut p = Params::zeroed();
    let g = match r {
        Request::Zernike {
            samples,
            terms,
            standard,
            ..
        } => {
            p.mode = 7;
            p.n = samples.desc.shape[0] as u32;
            p.field_side = terms.desc.shape[0] as u32;
            p.size = *standard as u32;
            None
        }
        Request::Resample(_) => unreachable!("resample has separate parameters"),
        Request::FocusCorrelation {
            on_axis,
            pupil,
            geometry,
            stencil,
            shifts,
            ..
        } => {
            p.shifted = *on_axis as u32;
            p.mode = 6;
            p.n = pupil.amplitude.desc.shape[0] as u32;
            p.field_side = stencil.desc.shape[0] as u32;
            p.size = shifts.as_ref().map_or(1, |s| s.desc.shape[0]) as u32;
            p.unit_energy = shifts.is_some() as u32;
            Some(geometry)
        }
        Request::Prepare {
            pupil,
            geometry,
            normal,
            field: _,
            ..
        } => {
            p.mode = 0;
            p.n = pupil.amplitude.desc.shape[0] as u32;
            p.normal = normal.map(pair);
            Some(geometry)
        }
        Request::Intensity {
            field,
            geometry,
            pupil_side,
            size,
            shifted,
            unit_energy,
            projection,
            ..
        } => {
            p.mode = 1;
            p.n = *pupil_side as u32;
            p.size = *size as u32;
            p.field_side = field.desc.shape[0] as u32;
            p.shifted = *shifted as u32;
            p.unit_energy = *unit_energy as u32;
            p.projection = *projection;
            Some(geometry)
        }
        Request::Huygens {
            pupil,
            geometry,
            size,
            spacing,
            origin,
            x_axis,
            y_axis,
            unit_energy,
            ..
        } => {
            p.mode = 2;
            p.n = pupil.amplitude.desc.shape[0] as u32;
            p.size = *size as u32;
            p.spacing = pair(*spacing);
            p.origin = origin.map(pair);
            p.x_axis = x_axis.map(pair);
            p.y_axis = y_axis.map(pair);
            p.unit_energy = *unit_energy as u32;
            Some(geometry)
        }
        Request::Focus {
            pupil,
            geometry,
            shifts,
            ..
        } => {
            p.mode = 3;
            p.n = pupil.amplitude.desc.shape[0] as u32;
            p.size = shifts.desc.shape[0] as u32;
            Some(geometry)
        }
        Request::Detector {
            frequencies,
            aperture,
            sigma,
            ..
        } => {
            p.mode = 4;
            p.size = frequencies.desc.shape[0] as u32;
            p.aperture = *aperture;
            p.sigma = *sigma;
            None
        }
        Request::LineSpread { image, spacing, .. } => {
            p.mode = 5;
            p.n = image.desc.shape[0] as u32;
            p.spacing = pair(Df32::from_f32(*spacing));
            None
        }
    };
    if let Some(g) = g {
        p.sin_max = pair(g.sin_max);
        p.wavelength = pair(g.wavelength);
        p.radius = pair(g.radius);
    }
    p
}
pub fn cis(mut phase: Df32) -> [f32; 2] {
    phase = phase.add_f32(-phase.hi.round_ties_even());
    phase = phase.add_f32(-phase.hi.round_ties_even());
    let t = scalar(phase) * std::f32::consts::TAU;
    [t.cos(), t.sin()]
}
pub fn direction(p: &Params, k: usize) -> [Df32; 3] {
    let n = p.n as usize;
    let uv = |i| Df32::from_f32((2 * i + 1) as f32) / Df32::from_f32(n as f32) - Df32::from_f32(1.);
    let a = df(p.sin_max) * uv(k % n);
    let b = df(p.sin_max) * uv(k / n);
    let g = (Df32::from_f32(1.) - a * a - b * b).sqrt();
    [a, b, -g]
}
pub fn dot(a: [Df32; 3], b: [Df32; 3]) -> Df32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn prepare(p: &Params, k: usize, a: f32, c: f32, w: Df32) -> ([f32; 2], [f32; 3]) {
    if c <= 0. {
        return ([0.; 2], [0.; 3]);
    }
    let q = direction(p, k);
    let gamma = -scalar(q[2]);
    let z = scalar(dot(q, p.normal.map(df))).abs();
    let weight = c * a / gamma;
    let e = c * (a / gamma) * (a / gamma);
    let phase = cis(w);
    (
        [weight * phase[0], weight * phase[1]],
        [weight, e, if z > 1e-9 { e * gamma / z } else { f32::NAN }],
    )
}
pub fn intensity(p: &Params, z: [f32; 2], s: &[Df32]) -> f32 {
    let l = Df32::from_f32(p.n as f32) * df(p.wavelength) / df(p.sin_max).mul_f32(2.);
    let div = if p.unit_energy != 0 {
        l * l * s[1] / Df32::from_f32(p.projection)
    } else {
        s[0] * s[0]
    };
    scalar(Df32::from_f32(z[0] * z[0] + z[1] * z[1]) / div)
}
pub fn focus(p: &Params, k: usize, dz: Df32) -> Df32 {
    let q = direction(p, k);
    let gamma = -q[2];
    let r = df(p.radius);
    let dr = if r.hi == 0. {
        dz * (q[0] * q[0] + q[1] * q[1]) / (Df32::from_f32(1.) + gamma)
    } else {
        let delta = (r * dz).mul_f32(2.) * gamma + dz * dz;
        let path = (r * r + delta).sqrt();
        // |R+dz|-path on either side of the sphere, rationalized again.
        let axial = r + dz;
        let axial = if axial.hi < 0. { -axial } else { axial };
        if q[0].hi == 0. && q[1].hi == 0. {
            return Df32::ZERO;
        }
        (r * dz).mul_f32(2.) * (q[0] * q[0] + q[1] * q[1])
            / ((Df32::from_f32(1.) + gamma) * (axial + path))
    };
    dr / df(p.wavelength)
}
pub fn huygens_term(p: &Params, pixel: usize, k: usize, a: f32, c: f32, w: Df32) -> [f32; 2] {
    if c <= 0. {
        return [0.; 2];
    }
    let x = Df32::from_f32((pixel % p.size as usize) as f32 - (p.size / 2) as f32) * df(p.spacing);
    let y = Df32::from_f32((pixel / p.size as usize) as f32 - (p.size / 2) as f32) * df(p.spacing);
    let point =
        std::array::from_fn(|i| df(p.origin[i]) + df(p.x_axis[i]) * x + df(p.y_axis[i]) * y);
    let q = direction(p, k);
    let r = df(p.radius);
    let qx = dot(q, point);
    let (phase, factor) = if r.hi == 0. {
        (qx / df(p.wavelength), Df32::from_f32(1.))
    } else {
        let delta = dot(point, point) - (r * qx).mul_f32(2.);
        let dist = (r * r + delta).sqrt();
        (
            -(delta / (dist + r)) / df(p.wavelength),
            (r * r - r * qx) / (dist * dist),
        )
    };
    let z = cis(w + phase);
    let weight = c * a / (-scalar(q[2])) * scalar(factor);
    [weight * z[0], weight * z[1]]
}
pub fn huygens_intensity(p: &Params, z: [Df32; 2], s: &[Df32]) -> f32 {
    let l = Df32::from_f32(p.n as f32) * df(p.wavelength) / df(p.sin_max).mul_f32(2.);
    let div = if p.unit_energy != 0 {
        l * l * s[2]
    } else {
        s[0] * s[0]
    };
    scalar((z[0] * z[0] + z[1] * z[1]) / div)
}
pub fn detector(p: &Params, nu: [f32; 2]) -> f32 {
    let sinc = |x: f32| {
        if x == 0. {
            1.
        } else {
            let t = std::f32::consts::PI * x;
            t.sin() / t
        }
    };
    (sinc(p.aperture[0] * nu[0]) * sinc(p.aperture[1] * nu[1])).abs()
        * (-2. * std::f32::consts::PI.powi(2) * p.sigma * p.sigma * (nu[0] * nu[0] + nu[1] * nu[1]))
            .exp()
}

/// One optical-correlation contribution at a focus shift. Matches P2's
/// unwrapped phase interpolation and fractional-cell zero-lag restoration.
pub fn correlation_term(
    p: &Params,
    k: usize,
    a: &[f32],
    c: &[f32],
    w: &[Df32],
    g: &[f32],
    dz: Df32,
) -> [f32; 3] {
    if c[k] <= 0. {
        return [0.; 3];
    }
    let (ix, iy, qz, gamma, srcz) = (g[0], g[1], g[2], g[3], g[4]);
    let energy = c[k] * a[k] * a[k] / (gamma * srcz);
    if !ix.is_finite() || !iy.is_finite() {
        return [0., 0., energy];
    }
    let n = p.n as usize;
    let (fx, fy) = (ix.floor(), iy.floor());
    let (tx, ty) = (ix - fx, iy - fy);
    let (mut ar, mut cov, mut phase) = (0., 0., Df32::ZERO);
    for (dx, dy, wt) in [
        (0, 0, (1. - tx) * (1. - ty)),
        (1, 0, tx * (1. - ty)),
        (0, 1, (1. - tx) * ty),
        (1, 1, tx * ty),
    ] {
        let (x, y) = (fx as i64 + dx, fy as i64 + dy);
        if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
            continue;
        }
        let j = y as usize * n + x as usize;
        let weight = wt * c[j];
        if weight == 0. {
            continue;
        }
        ar += weight * a[j]
            / if p.shifted != 0 {
                -scalar(direction(p, j)[2])
            } else {
                1.
            };
        cov += weight;
        phase = phase + (w[j] + focus(p, j, dz)).mul_f32(weight);
    }
    let deficit = (1. - c[k])
        * energy
        * (1. - (ix - (k % n) as f32).abs()).max(0.)
        * (1. - (iy - (k / n) as f32).abs()).max(0.);
    if cov <= 0. {
        return [deficit, 0., energy];
    }
    let z = cis(phase / Df32::from_f32(cov) - (w[k] + focus(p, k, dz)));
    let m = c[k] * a[k] / gamma * ar / if p.shifted != 0 { 1. } else { qz };
    [m * z[0] + deficit, m * z[1], energy]
}

/// Cartesian angular powers and Jacobi recurrence avoid polar-coordinate
/// singularities at the origin and factorial cancellation at high order.
pub fn zernike(x: Df32, y: Df32, n: u32, m: i32, standard: bool) -> Df32 {
    let am = m.unsigned_abs();
    let degree = (n - am) / 2;
    let one = Df32::from_f32(1.);
    let t = (x * x + y * y).mul_f32(2.) - one;
    let b = am as f32;
    let (mut prev, mut cur) = (one, one + (t - one).mul_f32((b + 2.) * 0.5));
    for k in 1..degree {
        let f = k as f32;
        let s = 2. * f + b;
        let a1 = 2. * (f + 1.) * (f + b + 1.) * s;
        let a2 = (t.mul_f32((s + 2.) * s) - Df32::from_f32(b * b)).mul_f32(s + 1.);
        let a3 = 2. * f * (f + b) * (s + 2.);
        let next = (a2 * cur - prev.mul_f32(a3)) / Df32::from_f32(a1);
        prev = cur;
        cur = next;
    }
    let (mut re, mut im) = (one, Df32::ZERO);
    for _ in 0..am {
        (re, im) = (re * x - im * y, re * y + im * x);
    }
    let z = (if m >= 0 { re } else { im }) * if degree == 0 { one } else { cur };
    if standard {
        z * Df32::from_f32((n + 1) as f32 * if m == 0 { 1. } else { 2. }).sqrt()
    } else {
        z
    }
}
