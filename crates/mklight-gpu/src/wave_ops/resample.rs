//! Structured triangulation and exact Chebyshev nearest-raster extension.
//! Ties choose the lowest row-major source index, matching the reference BFS.
use super::*;
use bytemuck::{Pod, Zeroable};
#[derive(Clone, Copy, Debug)]
pub enum Mask {
    Triangles,
    Circular,
    Annular(f32),
    Rectangle([f32; 2]),
    Polygons,
}
#[derive(Clone, Debug)]
pub struct Resample {
    /// DF [ny,nx,4]: u,v,unwrapped OPD,amplitude. Invalid rows are not loaded.
    pub nodes: PlanView,
    pub valid: PlanView,
    /// DF [edges,4]: x0,y0,x1,y1. Loops have already been chained by the host.
    pub edges: PlanView,
    pub mask: Mask,
    pub size: usize,
    pub supersample: usize,
    pub amplitude: BufferId,
    pub coverage: BufferId,
    pub opd: BufferId,
    /// F32 [3,size*size]: largest extrapolation distance and its local lattice
    /// pitch, then unresolved covered subpoints (zero for a usable grid).
    pub diagnostics: BufferId,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    pub nx: u32,
    pub ny: u32,
    pub n: u32,
    pub supersample: u32,
    pub mask: u32,
    pub edges: u32,
    pub first: u32,
    pub count: u32,
    pub aperture: [f32; 2],
    pub reserved: [u32; 2],
}
pub fn params(r: &Resample) -> Params {
    Params {
        nx: r.nodes.desc.shape[1] as u32,
        ny: r.nodes.desc.shape[0] as u32,
        n: r.size as u32,
        supersample: r.supersample as u32,
        mask: match r.mask {
            Mask::Triangles => 0,
            Mask::Circular => 1,
            Mask::Annular(_) => 2,
            Mask::Rectangle(_) => 3,
            Mask::Polygons => 4,
        },
        edges: r.edges.desc.shape[0] as u32,
        first: 0,
        count: 0,
        aperture: match r.mask {
            Mask::Annular(a) => [a, 0.],
            Mask::Rectangle(a) => a,
            _ => [0.; 2],
        },
        reserved: [0; 2],
    }
}
pub fn validate(r: &Resample, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    let s = &r.nodes.desc.shape;
    if s.len() != 3
        || s[2] != 4
        || s[0] < 2
        || s[1] < 2
        || s[0] > 2050
        || s[1] > 2050
        || !(2..=4096).contains(&r.size)
        || !(1..=32).contains(&r.supersample)
        || r.size * r.supersample > 8192
    {
        return Err(bad("wave resample dimensions/working set"));
    }
    view(&r.nodes, DoubleFloat, s)?;
    view(&r.valid, U32, &s[..2])?;
    let e = &r.edges.desc.shape;
    if e.len() != 2 || e[1] != 4 {
        return Err(bad("wave mask edges"));
    }
    view(&r.edges, DoubleFloat, e)?;
    match r.mask {
        Mask::Annular(a) if !a.is_finite() || !(0.0..1.0).contains(&a) => {
            return Err(bad("annular mask"));
        }
        Mask::Rectangle(a) if a.iter().any(|v| !v.is_finite() || *v <= 0.) => {
            return Err(bad("rectangular mask"));
        }
        Mask::Polygons if e[0] < 3 => return Err(bad("polygon mask")),
        _ => {}
    }
    out(p, r.amplitude, F32, &[r.size, r.size])?;
    out(p, r.coverage, F32, &[r.size, r.size])?;
    out(p, r.opd, DoubleFloat, &[r.size, r.size])?;
    out(p, r.diagnostics, F32, &[3, r.size * r.size])
}
pub fn access(r: &Resample) -> Access {
    Access {
        reads: vec![r.nodes.clone(), r.valid.clone(), r.edges.clone()],
        writes: vec![r.amplitude, r.coverage, r.opd, r.diagnostics],
        systems: vec![],
    }
}
fn coord(i: usize, ns: usize) -> Df32 {
    Df32::from_f32((2 * i + 1) as f32) / Df32::from_f32(ns as f32) - Df32::from_f32(1.)
}
fn norm2(a: [Df32; 4], b: [Df32; 4]) -> Df32 {
    let x = a[0] - b[0];
    let y = a[1] - b[1];
    x * x + y * y
}
fn bary(v: [[Df32; 4]; 3], u: Df32, w: Df32) -> Option<[Df32; 3]> {
    let a = [v[1][0] - v[0][0], v[1][1] - v[0][1]];
    let b = [v[2][0] - v[0][0], v[2][1] - v[0][1]];
    let det = a[0] * b[1] - a[1] * b[0];
    if det.hi == 0. {
        return None;
    }
    let x = u - v[0][0];
    let y = w - v[0][1];
    let l1 = (x * b[1] - y * b[0]) / det;
    let l2 = (a[0] * y - a[1] * x) / det;
    Some([Df32::from_f32(1.) - l1 - l2, l1, l2])
}
fn inside(p: &Params, edges: &[[Df32; 4]], u: Df32, v: Df32) -> bool {
    let scalar = super::spec::scalar;
    let r = scalar(u * u + v * v);
    match p.mask {
        0 => true,
        1 => r <= 1.,
        2 => r <= 1. && r >= p.aperture[0] * p.aperture[0],
        3 => scalar(u).abs() <= p.aperture[0] && scalar(v).abs() <= p.aperture[1],
        _ => {
            let mut b = false;
            for e in edges {
                if (e[1].to_f64() > v.to_f64()) != (e[3].to_f64() > v.to_f64())
                    && (u - (e[0] + (v - e[1]) * (e[2] - e[0]) / (e[3] - e[1]))).hi < 0.
                {
                    b = !b;
                }
            }
            b
        }
    }
}
pub fn cpu(
    r: &Resample,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    let p = params(r);
    let (nx, ny, n, ss) = (p.nx as usize, p.ny as usize, r.size, r.supersample);
    let ns = n * ss;
    let nodes = x.rows::<Df32>(&r.nodes).as_chunks::<4>().0;
    let valid = x.rows::<u32>(&r.valid);
    let edges = x.rows::<Df32>(&r.edges).as_chunks::<4>().0;
    let mut tris = vec![None; 2 * (nx - 1) * (ny - 1)];
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let k = j * nx + i;
            let ids = [k, k + 1, k + nx + 1, k + nx];
            let vs: Vec<_> = ids.into_iter().filter(|&k| valid[k] != 0).collect();
            let at = 2 * (j * (nx - 1) + i);
            if vs.len() == 4 {
                let d02 = norm2(nodes[vs[0]], nodes[vs[2]]);
                let d13 = norm2(nodes[vs[1]], nodes[vs[3]]);
                let pair = if (d02 - d13).hi <= 0. {
                    [[vs[0], vs[1], vs[2]], [vs[0], vs[2], vs[3]]]
                } else {
                    [[vs[0], vs[1], vs[3]], [vs[1], vs[2], vs[3]]]
                };
                for z in 0..2 {
                    tris[at + z] = Some(pair[z].map(|k| nodes[k]));
                }
            } else if vs.len() == 3 {
                tris[at] = Some([nodes[vs[0]], nodes[vs[1]], nodes[vs[2]]]);
            }
        }
    }
    let mut owner = vec![u32::MAX; ns * ns];
    for (t, v) in tris.iter().enumerate() {
        ctl.check()?;
        let Some(v) = v else {
            continue;
        };
        let bounds = |axis: usize| {
            let a = v
                .iter()
                .map(|x| x[axis].to_f64())
                .fold(f64::INFINITY, f64::min);
            let b = v
                .iter()
                .map(|x| x[axis].to_f64())
                .fold(f64::NEG_INFINITY, f64::max);
            (
                ((a + 1.) * ns as f64 * 0.5 - 0.5).ceil().max(0.) as usize,
                ((b + 1.) * ns as f64 * 0.5 - 0.5)
                    .floor()
                    .min(ns as f64 - 1.) as isize,
            )
        };
        let (x0, x1) = bounds(0);
        let (y0, y1) = bounds(1);
        if x1 < 0 || y1 < 0 {
            continue;
        }
        for y in y0..=y1 as usize {
            for xx in x0..=x1 as usize {
                if owner[y * ns + xx] != u32::MAX {
                    continue;
                }
                if let Some(l) = bary(*v, coord(xx, ns), coord(y, ns))
                    && l.iter().all(|l| l.to_f64() >= -1e-12)
                {
                    owner[y * ns + xx] = t as u32;
                }
            }
        }
    }
    let mut amp = vec![0.; n * n];
    let mut cov = amp.clone();
    let mut opd = vec![Df32::ZERO; n * n];
    let mut diag = vec![0.; 3 * n * n];
    for k in 0..n * n {
        if k % 4096 == 0 {
            ctl.check()?;
            ctl.report(k as f64 / (n * n) as f64, "pupil resampling");
        }
        let (mut a, mut w, mut count) = (Df32::ZERO, Df32::ZERO, 0usize);
        let (mut gap, mut pitch) = (0f32, 1f32);
        for sy in 0..ss {
            for sx in 0..ss {
                let xx = k % n * ss + sx;
                let y = k / n * ss + sy;
                let u = coord(xx, ns);
                let v = coord(y, ns);
                if p.mask != 0 && !inside(&p, edges, u, v) {
                    continue;
                }
                let mut q = y * ns + xx;
                let missing = owner[q] == u32::MAX;
                if missing {
                    if p.mask == 0 {
                        continue;
                    }
                    let mut found = false;
                    'radius: for d in
                        1..ns.min(crate::consts::MKL_WAVE_MAX_EXTRAPOLATION as usize + 1)
                    {
                        for yy in y.saturating_sub(d)..=(y + d).min(ns - 1) {
                            for xxx in xx.saturating_sub(d)..=(xx + d).min(ns - 1) {
                                if xx.abs_diff(xxx).max(y.abs_diff(yy)) == d
                                    && owner[yy * ns + xxx] != u32::MAX
                                {
                                    q = yy * ns + xxx;
                                    found = true;
                                    break 'radius;
                                }
                            }
                        }
                    }
                    if !found {
                        diag[2 * n * n + k] += 1.;
                        continue;
                    }
                }
                let t = tris[owner[q] as usize].unwrap();
                let Some(l) = bary(t, u, v) else {
                    continue;
                };
                let ww = l[0] * t[0][2] + l[1] * t[1][2] + l[2] * t[2][2];
                let aa = l[0] * t[0][3] + l[1] * t[1][3] + l[2] * t[2][3];
                w = w + ww;
                a = a + if aa.hi > 0. { aa } else { Df32::ZERO };
                count += 1;
                if missing {
                    let dx = xx as f32 - (q % ns) as f32;
                    let dy = y as f32 - (q / ns) as f32;
                    let g = (dx * dx + dy * dy).sqrt() * 2. / ns as f32;
                    let mut e = [
                        norm2(t[0], t[1]).sqrt().hi,
                        norm2(t[1], t[2]).sqrt().hi,
                        norm2(t[2], t[0]).sqrt().hi,
                    ];
                    e.sort_by(f32::total_cmp);
                    if g / e[1] > gap / pitch {
                        gap = g;
                        pitch = e[1];
                    }
                }
            }
        }
        if count > 0 {
            cov[k] = count as f32 / (ss * ss) as f32;
            amp[k] = super::spec::scalar(a / Df32::from_f32(count as f32));
            opd[k] = w / Df32::from_f32(count as f32);
        }
        diag[k] = gap;
        diag[n * n + k] = pitch;
    }
    x.words_mut(r.amplitude)
        .copy_from_slice(bytemuck::cast_slice(&amp));
    x.words_mut(r.coverage)
        .copy_from_slice(bytemuck::cast_slice(&cov));
    x.words_mut(r.opd)
        .copy_from_slice(bytemuck::cast_slice(&opd));
    x.words_mut(r.diagnostics)
        .copy_from_slice(bytemuck::cast_slice(&diag));
    Ok(())
}
