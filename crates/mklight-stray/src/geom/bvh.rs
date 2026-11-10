//! Placed primitives and closest-hit queries.
//!
//! [`RayCaster`] is the only interface the path tracer uses for spatial
//! search. [`Bvh`] is the CPU implementation. It is a binary bounding-volume
//! hierarchy over the primitives' global AABBs, built by median split on the
//! longest centroid axis with at most two primitives per leaf, and traversed
//! front-to-back with a fixed stack, pruning by the best hit so far. Scenes
//! hold tens to a few hundred analytic primitives, so a simple build is
//! enough. [`BruteForce`] tests every primitive and serves as the reference in
//! tests.
//!
//! A hardware ray-tracing backend (Metal acceleration structures with custom
//! intersection functions, or similar) implements `RayCaster` over the same
//! primitive list: [`Primitive::aabb`] is the bounding box handed to the
//! acceleration-structure build, and [`Primitive::intersect`] is the custom
//! intersection function's reference implementation.

use mklight_core::math::{Frame, Mat3, Vec3};

use super::{Aabb, LocalHit, Ray, Shape};

/// A shape placed in the global frame.
#[derive(Clone, Debug)]
pub struct Primitive {
    pub frame: Frame,
    pub shape: Shape,
    inv_rot: Mat3,
    aabb: Aabb,
}

impl Primitive {
    pub fn new(frame: Frame, shape: Shape) -> Primitive {
        let aabb = shape.local_bounds().transformed(&frame);
        Primitive {
            inv_rot: frame.rot.transpose(),
            frame,
            shape,
            aabb,
        }
    }
    /// Global bounding box (padded).
    pub fn aabb(&self) -> Aabb {
        self.aabb
    }
    /// Local-frame intersection of a global ray.
    #[inline]
    pub fn intersect(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<LocalHit> {
        let o = self.inv_rot.mul_vec(ray.origin - self.frame.origin);
        let d = self.inv_rot.mul_vec(ray.dir);
        self.shape.intersect(o, d, t_min, t_max)
    }
}

/// Closest intersection of a global ray with the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub prim: u32,
    pub local: LocalHit,
}

/// Spatial search used by the path tracer.
pub trait RayCaster: Sync {
    /// Nearest hit with `0 < t ≤ t_max`.
    fn closest_hit(&self, ray: &Ray, t_max: f64) -> Option<Hit>;
}

/// Reference caster that tests every primitive.
pub struct BruteForce<'a>(pub &'a [Primitive]);

impl RayCaster for BruteForce<'_> {
    fn closest_hit(&self, ray: &Ray, t_max: f64) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut t_best = t_max;
        for (i, p) in self.0.iter().enumerate() {
            if let Some(h) = p.intersect(ray, 0.0, t_best)
                && (best.is_none() || h.t < t_best)
            {
                t_best = h.t;
                best = Some(Hit {
                    prim: i as u32,
                    local: h,
                });
            }
        }
        best
    }
}

#[derive(Clone, Copy, Debug)]
struct Node {
    aabb: Aabb,
    /// Leaf: first primitive index in `order`. Inner: index of the left child
    /// (the right child is `left + 1`).
    first: u32,
    /// Number of primitives (0 for an inner node).
    count: u32,
}

/// Bounding-volume hierarchy over a primitive list it owns.
#[derive(Clone, Debug)]
pub struct Bvh {
    prims: Vec<Primitive>,
    nodes: Vec<Node>,
    order: Vec<u32>,
}

const STACK: usize = 64;

impl Bvh {
    pub fn build(prims: Vec<Primitive>) -> Bvh {
        let mut bvh = Bvh {
            nodes: Vec::new(),
            order: (0..prims.len() as u32).collect(),
            prims,
        };
        if !bvh.prims.is_empty() {
            bvh.nodes.push(Node {
                aabb: Aabb::EMPTY,
                first: 0,
                count: 0,
            });
            let n = bvh.prims.len();
            bvh.build_node(0, 0, n, 0);
        }
        bvh
    }

    pub fn primitives(&self) -> &[Primitive] {
        &self.prims
    }

    fn build_node(&mut self, node: usize, lo: usize, hi: usize, depth: usize) {
        let mut bb = Aabb::EMPTY;
        let mut cb = Aabb::EMPTY;
        for &i in &self.order[lo..hi] {
            let a = self.prims[i as usize].aabb;
            bb = bb.union(&a);
            cb.grow(a.centroid());
        }
        self.nodes[node].aabb = bb;
        // The depth cap keeps the traversal stack bounded for any input.
        if hi - lo <= 2 || depth >= STACK / 2 - 2 {
            self.nodes[node].first = lo as u32;
            self.nodes[node].count = (hi - lo) as u32;
            return;
        }
        let e = cb.extent();
        let axis = if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        };
        let key = |p: &Primitive| {
            let c = p.aabb.centroid();
            [c.x, c.y, c.z][axis]
        };
        let prims = &self.prims;
        self.order[lo..hi].sort_by(|a, b| {
            key(&prims[*a as usize])
                .total_cmp(&key(&prims[*b as usize]))
                .then(a.cmp(b))
        });
        let mid = (lo + hi) / 2;
        let left = self.nodes.len();
        let empty = Node {
            aabb: Aabb::EMPTY,
            first: 0,
            count: 0,
        };
        self.nodes.push(empty);
        self.nodes.push(empty);
        self.nodes[node].first = left as u32;
        self.nodes[node].count = 0;
        self.build_node(left, lo, mid, depth + 1);
        self.build_node(left + 1, mid, hi, depth + 1);
    }
}

impl RayCaster for Bvh {
    fn closest_hit(&self, ray: &Ray, t_max: f64) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = Vec3::new(1.0 / ray.dir.x, 1.0 / ray.dir.y, 1.0 / ray.dir.z);
        let mut best: Option<Hit> = None;
        let mut t_best = t_max;
        let mut stack = [0u32; STACK];
        let mut sp = 0;
        self.nodes[0].aabb.hit(ray.origin, inv, t_best)?;
        stack[sp] = 0;
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];
            if node.aabb.hit(ray.origin, inv, t_best).is_none() {
                continue;
            }
            if node.count > 0 {
                for k in node.first..node.first + node.count {
                    let pi = self.order[k as usize];
                    if let Some(h) = self.prims[pi as usize].intersect(ray, 0.0, t_best) {
                        // A first hit at `t_max` counts; ties go to the lower
                        // primitive index, as in BruteForce.
                        if h.t < t_best || best.is_none_or(|b| h.t == t_best && pi < b.prim) {
                            t_best = h.t;
                            best = Some(Hit { prim: pi, local: h });
                        }
                    }
                }
                continue;
            }
            let (l, r) = (node.first, node.first + 1);
            let tl = self.nodes[l as usize].aabb.hit(ray.origin, inv, t_best);
            let tr = self.nodes[r as usize].aabb.hit(ray.origin, inv, t_best);
            // Push the farther child first so the nearer one is popped first.
            match (tl, tr) {
                (Some(a), Some(b)) => {
                    let (near, far) = if a <= b { (l, r) } else { (r, l) };
                    stack[sp] = far;
                    stack[sp + 1] = near;
                    sp += 2;
                }
                (Some(_), None) => {
                    stack[sp] = l;
                    sp += 1;
                }
                (None, Some(_)) => {
                    stack[sp] = r;
                    sp += 1;
                }
                (None, None) => {}
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Cylinder, SagPatch};
    use mklight_core::model::SurfaceShape;
    use mklight_core::rng::Philox;

    #[test]
    fn bvh_matches_brute_force() {
        let mut prims = Vec::new();
        let mut rng = Philox::new(3, 0);
        for i in 0..60 {
            let frame = Frame {
                origin: Vec3::new(
                    40.0 * rng.uniform() - 20.0,
                    40.0 * rng.uniform() - 20.0,
                    40.0 * rng.uniform() - 20.0,
                ),
                rot: Mat3::rot_x(rng.uniform() * 3.0).mul_mat(&Mat3::rot_y(rng.uniform() * 3.0)),
            };
            let shape = if i % 3 == 0 {
                Shape::Cylinder(Cylinder::new(1.0 + 3.0 * rng.uniform(), 0.0, 5.0).unwrap())
            } else {
                Shape::Sag(
                    SagPatch::new(
                        SurfaceShape::sphere(if i % 2 == 0 { 15.0 } else { -12.0 }),
                        0.0,
                        2.0 + 4.0 * rng.uniform(),
                        None,
                    )
                    .unwrap(),
                )
            };
            prims.push(Primitive::new(frame, shape));
        }
        let bvh = Bvh::build(prims.clone());
        let brute = BruteForce(&prims);
        let mut n_hit = 0;
        for _ in 0..3000 {
            let o = Vec3::new(
                60.0 * rng.uniform() - 30.0,
                60.0 * rng.uniform() - 30.0,
                60.0 * rng.uniform() - 30.0,
            );
            let d = Vec3::new(
                rng.uniform() - 0.5,
                rng.uniform() - 0.5,
                rng.uniform() - 0.5,
            );
            let ray = Ray::new(o, d);
            let a = bvh.closest_hit(&ray, f64::INFINITY);
            let b = brute.closest_hit(&ray, f64::INFINITY);
            assert_eq!(a, b);
            n_hit += a.is_some() as usize;
        }
        assert!(n_hit > 300, "{n_hit}");
    }
}
