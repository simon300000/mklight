//! Solves and pickups: parameters whose values follow from other parameters
//! or from a paraxial ray condition (`docs/SOLVES.md`).
//!
//! A [`Solve`] controls exactly one [`ParamPath`] (its `target`). This module
//! holds the data model and its structural checks: target/kind compatibility,
//! duplicate targets, and the dependency graph between solves (pickup cycles
//! are rejected here). The numerical solver is `mklight_trace::solves`, which
//! returns a *new* system with every solved parameter set; snapshots stay
//! immutable.
//!
//! Paraxial conditions use the primary wavelength and the y–nu rays of
//! `mklight_trace::paraxial` (marginal ray scaled by the aperture
//! specification, chief ray scaled to the maximum field). Slopes are the
//! paraxial `u = dy/dz` along the local axis, dimensionless (not degrees).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::error::{MkError, MkResult};
use crate::model::{
    ApertureSpec, Interaction, MaterialRef, OpticalSystem, SurfaceId, SurfaceShape,
};
use crate::param::ParamPath;

/// One solve: `target` is set by the rule `kind`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Solve {
    /// The parameter this solve controls. It must not be an optimisation
    /// variable (see `OpticalSystem::solved_params`).
    pub target: ParamPath,
    pub kind: SolveKind,
}

fn one() -> f64 {
    1.0
}

/// The rule that sets a solve's target (`docs/SOLVES.md` gives the exact
/// definitions and sign conventions).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SolveKind {
    /// On `Thickness { surface: j }`: the paraxial marginal ray's height on
    /// the vertex of surface `j+1` is `height` (mm). `height = 0` on the last
    /// thickness puts the image surface at the paraxial focus.
    MarginalRayHeight { height: f64 },
    /// On `Curvature` (or `ParaxialFocalLength`) of surface `k`: the paraxial
    /// marginal-ray slope after surface `k` is `slope` (`u = dy/dz`,
    /// dimensionless: not an angle in degrees, CONVENTIONS §8).
    MarginalRayAngle { slope: f64 },
    /// As [`SolveKind::MarginalRayHeight`] for the paraxial chief ray.
    ChiefRayHeight { height: f64 },
    /// As [`SolveKind::MarginalRayAngle`] for the paraxial chief ray.
    ChiefRayAngle { slope: f64 },
    /// On `Curvature` (or `ParaxialFocalLength`) of a powered surface,
    /// usually the last one: the paraxial image-space F/# `f'/EPD` equals
    /// `value`. Not allowed with an aperture specified as an F/#.
    FNumber { value: f64 },
    /// Any parameter: `target = scale · source + offset`.
    Pickup {
        source: ParamPath,
        #[serde(default = "one")]
        scale: f64,
        #[serde(default)]
        offset: f64,
    },
    /// On a thickness between `from` and `to`: the axial distance from the
    /// vertex of `from` to the vertex of `to` (the signed sum of the
    /// thicknesses in between) is `distance` (mm). With `from` = surface 1
    /// and `to` = the image this keeps the total track constant.
    Position {
        from: SurfaceId,
        to: SurfaceId,
        distance: f64,
    },
    /// On `Curvature` of surface `i`: the centre of curvature lies on the
    /// vertex of `surface` (axial distance along the unfolded axis).
    CenterOfCurvature { surface: SurfaceId },
    /// On `Thickness { surface: j }`: the signed axial distance between the
    /// points of surfaces `j` and `j+1` at `radius` is `value` (mm):
    /// `t_j + sag_{j+1}(r) − sag_j(r) = value`. `radius: None` uses the
    /// larger of the two surfaces' semi-diameters (`docs/SOLVES.md`).
    EdgeThickness {
        value: f64,
        #[serde(default)]
        radius: Option<f64>,
    },
}

impl SolveKind {
    /// Short snake-case name, as in JSON.
    pub fn name(&self) -> &'static str {
        match self {
            SolveKind::MarginalRayHeight { .. } => "marginal_ray_height",
            SolveKind::MarginalRayAngle { .. } => "marginal_ray_angle",
            SolveKind::ChiefRayHeight { .. } => "chief_ray_height",
            SolveKind::ChiefRayAngle { .. } => "chief_ray_angle",
            SolveKind::FNumber { .. } => "f_number",
            SolveKind::Pickup { .. } => "pickup",
            SolveKind::Position { .. } => "position",
            SolveKind::CenterOfCurvature { .. } => "center_of_curvature",
            SolveKind::EdgeThickness { .. } => "edge_thickness",
        }
    }
    /// Does evaluating this solve need a paraxial ray trace?
    pub fn is_paraxial(&self) -> bool {
        matches!(
            self,
            SolveKind::MarginalRayHeight { .. }
                | SolveKind::MarginalRayAngle { .. }
                | SolveKind::ChiefRayHeight { .. }
                | SolveKind::ChiefRayAngle { .. }
                | SolveKind::FNumber { .. }
        )
    }
    fn numbers(&self) -> Vec<f64> {
        match *self {
            SolveKind::MarginalRayHeight { height } | SolveKind::ChiefRayHeight { height } => {
                vec![height]
            }
            SolveKind::MarginalRayAngle { slope } | SolveKind::ChiefRayAngle { slope } => {
                vec![slope]
            }
            SolveKind::FNumber { value } => vec![value],
            SolveKind::Pickup { scale, offset, .. } => vec![scale, offset],
            SolveKind::Position { distance, .. } => vec![distance],
            SolveKind::CenterOfCurvature { .. } => vec![],
            SolveKind::EdgeThickness { value, radius } => {
                let mut v = vec![value];
                v.extend(radius);
                v
            }
        }
    }
}

impl Solve {
    pub fn new(target: ParamPath, kind: SolveKind) -> Self {
        Solve { target, kind }
    }
    /// Marginal-ray height `height` on the vertex after `surface`'s thickness.
    pub fn marginal_ray_height(surface: SurfaceId, height: f64) -> Self {
        Solve::new(
            ParamPath::Thickness { surface },
            SolveKind::MarginalRayHeight { height },
        )
    }
    /// Image-space F/# `value` by `surface`'s curvature.
    pub fn f_number(surface: SurfaceId, value: f64) -> Self {
        Solve::new(
            ParamPath::Curvature { surface },
            SolveKind::FNumber { value },
        )
    }
    /// `target = scale · source + offset`.
    pub fn pickup(target: ParamPath, source: ParamPath, scale: f64, offset: f64) -> Self {
        Solve::new(
            target,
            SolveKind::Pickup {
                source,
                scale,
                offset,
            },
        )
    }

    /// Does this solve *read* parameter `p` directly (an explicit dependency)?
    /// Paraxial ray solves read every upstream parameter; that implicit
    /// dependency only orders solves and never makes a cycle an error.
    pub fn reads(&self, sys: &OpticalSystem, p: &ParamPath) -> bool {
        let index = |id: SurfaceId| sys.surface_index(id);
        let thickness_in = |p: &ParamPath, lo: usize, hi: usize| match p {
            ParamPath::Thickness { surface } => index(*surface).is_some_and(|i| lo <= i && i < hi),
            _ => false,
        };
        match &self.kind {
            SolveKind::Pickup { source, .. } => source == p,
            SolveKind::Position { from, to, .. } => match (index(*from), index(*to)) {
                (Some(a), Some(b)) => *p != self.target && thickness_in(p, a, b),
                _ => false,
            },
            SolveKind::CenterOfCurvature { surface } => {
                match (self.target.surface().and_then(index), index(*surface)) {
                    (Some(a), Some(b)) => thickness_in(p, a.min(b), a.max(b)),
                    _ => false,
                }
            }
            SolveKind::EdgeThickness { .. } => {
                let Some(j) = self.target.surface().and_then(index) else {
                    return false;
                };
                let shape_param = matches!(
                    p,
                    ParamPath::Curvature { .. }
                        | ParamPath::Conic { .. }
                        | ParamPath::Asphere { .. }
                        | ParamPath::SemiDiameter { .. }
                );
                shape_param
                    && p.surface()
                        .and_then(index)
                        .is_some_and(|i| i == j || i == j + 1)
            }
            _ => false,
        }
    }

    /// Soft ordering key: upstream parameters first, so paraxial ray solves
    /// see solved upstream values. Global parameters (aperture, fields,
    /// wavelengths, environment) come first; within a surface, its shape and
    /// material before its thickness before its semi-diameter.
    fn order_key(&self, sys: &OpticalSystem) -> (usize, u8) {
        let Some(id) = self.target.surface() else {
            return (0, 0);
        };
        let i = sys.surface_index(id).unwrap_or(usize::MAX);
        let rank = match self.target {
            ParamPath::Thickness { .. } => 2,
            ParamPath::SemiDiameter { .. } => 3,
            _ => 1,
        };
        (i, rank)
    }
}

impl OpticalSystem {
    /// Every parameter controlled by a solve, sorted and deduplicated. The
    /// optimiser must exclude these from its variables.
    pub fn solved_params(&self) -> Vec<ParamPath> {
        let mut v: Vec<ParamPath> = self.solves.iter().map(|s| s.target.clone()).collect();
        v.sort();
        v.dedup();
        v
    }

    /// The order in which solves are evaluated: a topological order of the
    /// explicit dependencies ([`Solve::reads`]) in which, among the solves
    /// that are ready, the most upstream target goes first (ties by
    /// declaration order). Deterministic. A cycle of explicit dependencies
    /// (e.g. two pickups of each other) is an error.
    pub fn solve_order(&self) -> MkResult<Vec<usize>> {
        let n = self.solves.len();
        // deps[a] = solves whose target `a` reads.
        let mut indegree = vec![0usize; n];
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (a, sa) in self.solves.iter().enumerate() {
            for (b, sb) in self.solves.iter().enumerate() {
                if a != b && sa.reads(self, &sb.target) {
                    indegree[a] += 1;
                    users[b].push(a);
                }
            }
        }
        let key = |k: usize| {
            let (i, r) = self.solves[k].order_key(self);
            Reverse((i, r, k))
        };
        let mut ready: BinaryHeap<_> = (0..n).filter(|&k| indegree[k] == 0).map(key).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(Reverse((_, _, k))) = ready.pop() {
            order.push(k);
            for &u in &users[k] {
                indegree[u] -= 1;
                if indegree[u] == 0 {
                    ready.push(key(u));
                }
            }
        }
        if order.len() < n {
            let stuck: Vec<String> = (0..n)
                .filter(|&k| indegree[k] > 0)
                .map(|k| format!("{k} ({})", self.solves[k].kind.name()))
                .collect();
            return Err(MkError::InvalidModel(format!(
                "cyclic solve dependencies among solves {}",
                stuck.join(", ")
            )));
        }
        Ok(order)
    }
}

/// Can `p` be assigned in `sys`? Like `ParamPath::get`, except that a
/// curvature solve on a plane is allowed (it turns the plane into a sphere)
/// and an automatic semi-diameter can be the target (it becomes fixed).
fn check_settable(sys: &OpticalSystem, p: &ParamPath) -> MkResult<()> {
    match p {
        ParamPath::SemiDiameter { surface } => sys
            .surface_index(*surface)
            .map(|_| ())
            .ok_or_else(|| MkError::InvalidParam(format!("no surface with id {}", surface.0))),
        _ => p.get(sys).map(|_| ()),
    }
}

/// The medium after surface `m` as the resolver sees it: mirrors, dummies and
/// coordinate breaks keep the medium before them.
fn medium_after(sys: &OpticalSystem, mut m: usize) -> &MaterialRef {
    while m > 0 {
        let s = &sys.surfaces[m];
        if s.interaction == Interaction::Refract && !s.is_coordinate_break() {
            break;
        }
        m -= 1;
    }
    &sys.surfaces[m].material
}

/// Can surface `m` have paraxial power (structurally, without resolving
/// materials)? Conservative: any curved or ideal-lens surface that refracts
/// or reflects, and any surface whose curvature or focal length a solve
/// sets.
fn may_have_power(sys: &OpticalSystem, m: usize) -> bool {
    let s = &sys.surfaces[m];
    if s.interaction == Interaction::Dummy || s.is_coordinate_break() {
        return false;
    }
    let solved = sys.solves.iter().any(|o| {
        matches!(
            o.target,
            ParamPath::Curvature { surface } | ParamPath::ParaxialFocalLength { surface }
                if surface == s.id
        )
    });
    solved || matches!(s.shape, SurfaceShape::Paraxial { .. }) || s.shape.curvature() != 0.0
}

/// Structural checks of `sys.solves` (called by `OpticalSystem::validate`).
pub(crate) fn validate_solves(sys: &OpticalSystem) -> MkResult<()> {
    let n = sys.surfaces.len();
    let mut targets = std::collections::BTreeSet::new();
    for (k, s) in sys.solves.iter().enumerate() {
        let err = |m: String| MkError::InvalidModel(format!("solve {k} ({}): {m}", s.kind.name()));
        if s.kind.numbers().iter().any(|v| !v.is_finite()) {
            return Err(err("values must be finite".into()));
        }
        check_settable(sys, &s.target).map_err(|e| err(e.to_string()))?;
        if !targets.insert(&s.target) {
            return Err(err(format!(
                "{:?} is already controlled by another solve",
                s.target
            )));
        }
        let index = |id: SurfaceId| {
            sys.surface_index(id)
                .ok_or_else(|| err(format!("no surface with id {}", id.0)))
        };
        let target_index = s.target.surface().map(index).transpose()?;
        // Thickness target strictly inside the lens (not object, not image).
        let inner_thickness = || match (&s.target, target_index) {
            (ParamPath::Thickness { .. }, Some(j)) if j >= 1 && j + 1 < n => Ok(j),
            (ParamPath::Thickness { .. }, _) => Err(err(
                "the thickness must belong to a surface between the object and the image".into(),
            )),
            _ => Err(err("the target must be a thickness".into())),
        };
        // Power-type target on a surface that refracts or reflects.
        let powered = || {
            let j = match (&s.target, target_index) {
                (ParamPath::Curvature { .. } | ParamPath::ParaxialFocalLength { .. }, Some(j)) => j,
                _ => {
                    return Err(err(
                        "the target must be a curvature or an ideal-lens focal length".into(),
                    ));
                }
            };
            let surf = &sys.surfaces[j];
            if j == 0 || j + 1 == n {
                return Err(err("the object and image surfaces have no power".into()));
            }
            if surf.interaction == Interaction::Dummy || surf.is_coordinate_break() {
                return Err(err(format!(
                    "surface {j} is a dummy or coordinate break and has no power"
                )));
            }
            // A refracting curvature between equal media has no power
            // (`φ = c·(N' − N)`); an ideal lens and a mirror always do.
            if matches!(s.target, ParamPath::Curvature { .. })
                && surf.interaction == Interaction::Refract
                && medium_after(sys, j - 1) == &surf.material
            {
                return Err(err(format!(
                    "surface {j} separates equal media ({:?}): its curvature has no paraxial power",
                    surf.material
                )));
            }
            Ok(j)
        };
        match &s.kind {
            SolveKind::MarginalRayHeight { .. } | SolveKind::ChiefRayHeight { .. } => {
                inner_thickness()?;
            }
            SolveKind::ChiefRayAngle { .. } => {
                powered()?;
            }
            SolveKind::MarginalRayAngle { .. } => {
                let j = powered()?;
                // A working F/#, or an image-space F/# at infinity, fixes the
                // image-space reduced slope `ω' = ∓1/(2F)`; with no power
                // behind surface `j`, `ω_j = ω'` cannot change.
                let fixes_slope = match sys.aperture {
                    ApertureSpec::ParaxialWorkingFNumber { .. } => true,
                    ApertureSpec::ImageSpaceFNumber { .. } => sys.object_at_infinity(),
                    _ => false,
                };
                if fixes_slope && !(j + 1..n - 1).any(|m| may_have_power(sys, m)) {
                    return Err(err(format!(
                        "the aperture is specified as an F/#, which fixes the image-space \
                         marginal slope, and no surface after surface {j} has power"
                    )));
                }
            }
            SolveKind::FNumber { value } => {
                powered()?;
                if *value <= 0.0 {
                    return Err(err("the F/# must be > 0".into()));
                }
                if matches!(
                    sys.aperture,
                    ApertureSpec::ImageSpaceFNumber { .. }
                        | ApertureSpec::ParaxialWorkingFNumber { .. }
                ) {
                    return Err(err(
                        "the aperture is specified as an F/#, which fixes the F/# already".into(),
                    ));
                }
            }
            SolveKind::Pickup { source, .. } => {
                if *source == s.target {
                    return Err(err("a parameter cannot pick up itself".into()));
                }
                let is_target = sys.solves.iter().any(|o| o.target == *source);
                check_settable(sys, source).map_err(|e| err(format!("source: {e}")))?;
                if !is_target {
                    let v = source.get(sys).map_err(|e| err(format!("source: {e}")))?;
                    // E.g. the thickness of an object at infinity.
                    if !v.is_finite() {
                        return Err(err(format!("source: {source:?} is not finite ({v})")));
                    }
                }
            }
            SolveKind::Position { from, to, .. } => {
                let (a, b) = (index(*from)?, index(*to)?);
                if a >= b {
                    return Err(err("`from` must come before `to`".into()));
                }
                let j = match (&s.target, target_index) {
                    (ParamPath::Thickness { .. }, Some(j)) => j,
                    _ => return Err(err("the target must be a thickness".into())),
                };
                if !(a <= j && j < b) {
                    return Err(err(
                        "the target thickness must lie between `from` and `to`".into()
                    ));
                }
                if a == 0 && sys.object_at_infinity() {
                    return Err(err(
                        "a distance from an object at infinity is undefined".into()
                    ));
                }
            }
            SolveKind::CenterOfCurvature { surface } => {
                let i = match (&s.target, target_index) {
                    (ParamPath::Curvature { .. }, Some(i)) => i,
                    _ => return Err(err("the target must be a curvature".into())),
                };
                let r = index(*surface)?;
                if r == i {
                    return Err(err(
                        "the centre of curvature cannot lie on the surface's own vertex".into(),
                    ));
                }
                if i.min(r) == 0 && sys.object_at_infinity() {
                    return Err(err(
                        "distances to an object at infinity are undefined".into()
                    ));
                }
            }
            SolveKind::EdgeThickness { radius, .. } => {
                let j = inner_thickness()?;
                if radius.is_some_and(|r| r <= 0.0) {
                    return Err(err("the radius must be > 0".into()));
                }
                for i in [j, j + 1] {
                    if matches!(sys.surfaces[i].shape, SurfaceShape::CoordinateBreak) {
                        return Err(err(format!(
                            "surface {i} is a coordinate break and has no edge"
                        )));
                    }
                }
            }
        }
    }
    sys.solve_order().map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Surface;

    fn lens() -> OpticalSystem {
        let mut s = OpticalSystem::new_empty();
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            Surface::new(
                1,
                SurfaceShape::sphere(50.0),
                5.0,
                MaterialRef::glass("N-BK7"),
            ),
            Surface::new(2, SurfaceShape::sphere(-50.0), 45.0, MaterialRef::Air),
            Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s
    }
    fn c(id: u64) -> ParamPath {
        ParamPath::Curvature {
            surface: SurfaceId(id),
        }
    }
    fn t(id: u64) -> ParamPath {
        ParamPath::Thickness {
            surface: SurfaceId(id),
        }
    }

    #[test]
    fn order_follows_pickup_chains_and_upstream_targets() {
        let mut s = lens();
        // Declared in the "wrong" order: thickness 2 reads c2, which reads c1.
        s.solves = vec![
            Solve::marginal_ray_height(SurfaceId(2), 0.0),
            Solve::pickup(c(2), c(1), -1.0, 0.0),
            Solve::pickup(t(1), c(1), 100.0, 0.0),
        ];
        s.validate().unwrap();
        // Pickup c2 (surface 2, shape) before t1? t1 is at surface 1 → first.
        assert_eq!(s.solve_order().unwrap(), vec![2, 1, 0]);
        assert_eq!(s.solved_params(), vec![c(2), t(1), t(2)]);
    }

    #[test]
    fn cycles_and_duplicates_are_rejected() {
        let mut s = lens();
        s.solves = vec![
            Solve::pickup(c(1), c(2), 1.0, 0.0),
            Solve::pickup(c(2), c(1), 1.0, 0.0),
        ];
        let e = s.validate().unwrap_err().to_string();
        assert!(e.contains("cyclic"), "{e}");
        s.solves = vec![
            Solve::pickup(c(1), t(1), 1.0, 0.0),
            Solve::f_number(SurfaceId(1), 4.0),
        ];
        assert!(s.validate().unwrap_err().to_string().contains("already"));
        s.solves = vec![Solve::pickup(c(1), c(1), 1.0, 0.0)];
        assert!(s.validate().is_err());
    }

    #[test]
    fn solves_that_can_never_succeed_are_rejected() {
        let check = |s: &OpticalSystem, needle: &str| {
            let e = s.validate().expect_err(needle).to_string();
            assert!(e.contains(needle), "expected {needle:?} in {e:?}");
        };
        // (a) An air–air plane stop has no power, whatever its curvature.
        let mut s = lens();
        s.surfaces.insert(
            1,
            Surface::new(9, SurfaceShape::Plane, 10.0, MaterialRef::Air),
        );
        s.solves = vec![Solve::f_number(SurfaceId(9), 4.0)];
        check(&s, "equal media");
        s.solves = vec![Solve::new(c(9), SolveKind::ChiefRayAngle { slope: 0.0 })];
        check(&s, "equal media");
        // A mirror keeps the medium before it (air), so the glass-to-air
        // surface after a mirrored surface 1 is air–air too.
        let mut m = s.clone();
        m.surfaces[2].interaction = Interaction::Reflect;
        m.solves = vec![Solve::f_number(SurfaceId(2), 4.0)];
        check(&m, "surface 3 separates equal media");
        // A mirror itself has power, and so has an ideal lens.
        s.surfaces[1].interaction = Interaction::Reflect;
        s.validate().unwrap();
        s.surfaces[1].interaction = Interaction::Refract;
        s.surfaces[1].shape = SurfaceShape::Paraxial { focal_length: 50.0 };
        s.solves = vec![Solve::new(
            ParamPath::ParaxialFocalLength {
                surface: SurfaceId(9),
            },
            SolveKind::FNumber { value: 4.0 },
        )];
        s.validate().unwrap();
        // The glass surfaces of the lens are fine.
        let mut s = lens();
        s.solves = vec![Solve::f_number(SurfaceId(2), 4.0)];
        s.validate().unwrap();

        // (b) A pickup from the infinite object distance.
        let mut s = lens();
        s.solves = vec![Solve::pickup(t(1), t(0), 1.0, 0.0)];
        check(&s, "not finite");
        // A finite object distance is a valid source.
        s.surfaces[0].thickness = 200.0;
        s.validate().unwrap();

        // (c) An F/# aperture fixes the image-space marginal slope: a
        // marginal-angle solve on the last powered surface cannot change it.
        let angle = |id: u64| Solve::new(c(id), SolveKind::MarginalRayAngle { slope: -0.1 });
        for aperture in [
            ApertureSpec::ImageSpaceFNumber { value: 4.0 },
            ApertureSpec::ParaxialWorkingFNumber { value: 4.0 },
        ] {
            let mut s = lens();
            s.aperture = aperture;
            s.solves = vec![angle(2)];
            check(&s, "fixes the image-space marginal slope");
            // On surface 1 the power of surface 2 still bends the ray.
            s.solves = vec![angle(1)];
            s.validate().unwrap();
            // A plane last surface has no power, so surface 1 is the last
            // powered one...
            s.surfaces[2].shape = SurfaceShape::Plane;
            check(&s, "no surface after surface 1 has power");
            // ...unless a solve gives it a curvature.
            s.solves.push(Solve::pickup(c(2), c(1), -1.0, 0.0));
            s.validate().unwrap();
        }
        // With a finite object an image-space F/# does not fix the slope.
        let mut s = lens();
        s.surfaces[0].thickness = 200.0;
        s.aperture = ApertureSpec::ImageSpaceFNumber { value: 4.0 };
        s.solves = vec![angle(2)];
        s.validate().unwrap();
    }

    #[test]
    fn json_form() {
        let s = Solve::marginal_ray_height(SurfaceId(2), 0.0);
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(
            j,
            r#"{"target":{"param":"thickness","surface":2},"kind":{"type":"marginal_ray_height","height":0.0}}"#
        );
        let p: Solve = serde_json::from_str(
            r#"{"target":{"param":"curvature","surface":2},
                "kind":{"type":"pickup","source":{"param":"curvature","surface":1},"scale":-1}}"#,
        )
        .unwrap();
        assert_eq!(p, Solve::pickup(c(2), c(1), -1.0, 0.0));
    }
}
