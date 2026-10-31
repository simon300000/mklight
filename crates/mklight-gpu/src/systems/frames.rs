//! The core's frame composition, compiled into candidate arithmetic.
use super::{algebra::E, compile::Compiler};
use mklight_core::{
    MkResult,
    model::{Placement, SurfaceId, TransformOrder},
    param::{ParamPath, PlacementComponent},
};
type V = [E; 3];
type M = [[E; 3]; 3];
#[derive(Clone)]
pub(super) struct Frame {
    pub o: V,
    pub r: M,
}
fn add(a: &V, b: &V) -> V {
    std::array::from_fn(|i| a[i].clone() + b[i].clone())
}
fn sub(a: &V, b: &V) -> V {
    std::array::from_fn(|i| a[i].clone() - b[i].clone())
}
pub(super) fn mv(a: &M, b: &V) -> V {
    std::array::from_fn(|i| (0..3).fold(b[0].c(0.0), |s, j| s + a[i][j].clone() * b[j].clone()))
}
fn mm(a: &M, b: &M) -> M {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (0..3).fold(a[0][0].c(0.0), |s, k| s + a[i][k].clone() * b[k][j].clone())
        })
    })
}
pub(super) fn transpose(a: &M) -> M {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i].clone()))
}
impl Frame {
    fn identity(e: &E) -> Self {
        Self {
            o: std::array::from_fn(|_| e.c(0.0)),
            r: std::array::from_fn(|i| {
                std::array::from_fn(|j| e.c(if i == j { 1.0 } else { 0.0 }))
            }),
        }
    }
    fn compose(&self, b: &Self) -> Self {
        Self {
            o: add(&self.o, &mv(&self.r, &b.o)),
            r: mm(&self.r, &b.r),
        }
    }
    pub fn relative_to(&self, b: &Self) -> Self {
        let inv = transpose(&b.r);
        Self {
            o: mv(&inv, &sub(&self.o, &b.o)),
            r: mm(&inv, &self.r),
        }
    }
}
impl Compiler {
    fn placement(
        &mut self,
        id: SurfaceId,
        p: &Placement,
        range: Option<(SurfaceId, Option<SurfaceId>)>,
    ) -> MkResult<Frame> {
        let mut get = |component: PlacementComponent| {
            let path = if let Some((last, pivot)) = range {
                ParamPath::ElementPlacement {
                    surface: id,
                    last,
                    pivot,
                    component,
                }
            } else {
                match component {
                    PlacementComponent::DecenterX => ParamPath::DecenterX { surface: id },
                    PlacementComponent::DecenterY => ParamPath::DecenterY { surface: id },
                    PlacementComponent::TiltX => ParamPath::TiltX { surface: id },
                    PlacementComponent::TiltY => ParamPath::TiltY { surface: id },
                    PlacementComponent::TiltZ => ParamPath::TiltZ { surface: id },
                }
            };
            self.param(&path)
        };
        let dx = get(PlacementComponent::DecenterX)?;
        let dy = get(PlacementComponent::DecenterY)?;
        let x = get(PlacementComponent::TiltX)?;
        let y = get(PlacementComponent::TiltY)?;
        let z = get(PlacementComponent::TiltZ)?;
        for a in [&x, &y, &z] {
            self.positive(a.c(360.000001) - a.clone().abs());
        }
        let (sx, cx) = (x * (std::f64::consts::PI / 180.0)).sin_cos();
        let (sy, cy) = (y * (std::f64::consts::PI / 180.0)).sin_cos();
        let (sz, cz) = (z * (std::f64::consts::PI / 180.0)).sin_cos();
        let zero = dx.c(0.0);
        let one = dx.c(1.0);
        let rx = [
            [one.clone(), zero.clone(), zero.clone()],
            [zero.clone(), cx.clone(), -sx.clone()],
            [zero.clone(), sx, cx],
        ];
        let ry = [
            [cy.clone(), zero.clone(), sy.clone()],
            [zero.clone(), one.clone(), zero.clone()],
            [-sy, zero.clone(), cy],
        ];
        let rz = [
            [cz.clone(), -sz.clone(), zero.clone()],
            [sz, cz, zero.clone()],
            [zero.clone(), zero.clone(), one],
        ];
        let r = mm(&mm(&rx, &ry), &rz);
        let dec = [dx, dy, zero];
        let o = if p.order == TransformOrder::TiltThenDecenter {
            mv(&r, &dec)
        } else {
            dec
        };
        Ok(Frame { o, r })
    }
    pub(super) fn frames(&mut self, thickness: &[E]) -> MkResult<Vec<Frame>> {
        let sys = self.model.clone();
        let n = sys.surfaces.len();
        let identity = Frame::identity(&thickness[1]);
        let mut frames = vec![identity.clone(); n];
        let mut running = identity.clone();
        for i in 1..n {
            let s = &sys.surfaces[i];
            let placed = if let Some(p) = &s.placement {
                running.compose(&self.placement(s.id, p, None)?)
            } else {
                running.clone()
            };
            frames[i] = placed.clone();
            if !s.placement.as_ref().is_some_and(|p| p.restore_after) {
                running = placed;
            }
            let mut shift = identity.clone();
            shift.o[2] = thickness[i].clone();
            running = running.compose(&shift);
        }
        if !sys.object_at_infinity() {
            frames[0].o[2] = -thickness[0].clone();
            if let Some(p) = &sys.surfaces[0].placement {
                frames[0] = frames[0].compose(&self.placement(sys.surfaces[0].id, p, None)?);
            }
        }
        for (i, s) in sys.surfaces.iter().enumerate() {
            if let Some(e) = &s.element_placement {
                let end = sys.surface_index(e.last).unwrap();
                let pivot = frames[e.pivot.and_then(|id| sys.surface_index(id)).unwrap_or(i)]
                    .o
                    .clone();
                let anchor = frames[i].r.clone();
                let motion = self.placement(s.id, &e.motion, Some((e.last, e.pivot)))?;
                let rotation = mm(&mm(&anchor, &motion.r), &transpose(&anchor));
                let shift = mv(&anchor, &motion.o);
                for f in &mut frames[i..=end] {
                    f.o = add(&add(&pivot, &mv(&rotation, &sub(&f.o, &pivot))), &shift);
                    f.r = mm(&rotation, &f.r);
                }
            }
        }
        Ok(frames)
    }
}
