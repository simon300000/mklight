use super::{
    algebra::{Algebra, Builder, E},
    frames::{mv, transpose},
    *,
};
use crate::{
    PackedSystemSet,
    consts::*,
    layout::{PackedHeader, PackedSurface},
};
use mklight_core::{
    OpticalSystem,
    model::{Interaction, SolveKind, SurfaceShape},
    param::{CoefficientSet, ParamPath, ShapeParameter},
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

fn canonical(path: &ParamPath) -> ParamPath {
    match path {
        ParamPath::ShapeParam {
            surface,
            parameter: ShapeParameter::CurvatureY,
        } => ParamPath::Curvature { surface: *surface },
        ParamPath::ShapeParam {
            surface,
            parameter: ShapeParameter::ConicY,
        } => ParamPath::Conic { surface: *surface },
        _ => path.clone(),
    }
}

pub(super) struct Compiler {
    pub model: OpticalSystem,
    pub library: Arc<MaterialLibrary>,
    b: Builder,
    pub(super) choice_active: bool,
    pub(super) params: BTreeMap<ParamPath, E>,
    guards: Vec<u32>,
    patches: Vec<[u32; 4]>,
}
impl Compiler {
    pub(super) fn c(&self, x: f64) -> E {
        E::constant(&self.b, x)
    }
    pub fn param(&mut self, p: &ParamPath) -> MkResult<E> {
        let p = canonical(p);
        if let Some(e) = self.params.get(&p) {
            return Ok(e.clone());
        }
        let e = self.c(p.get(&self.model)?);
        self.params.insert(p.clone(), e.clone());
        Ok(e)
    }
    pub fn positive(&mut self, e: E) {
        self.guards.push(e.1);
    }
    pub fn nonnegative(&mut self, e: E) {
        self.positive((-e).select(self.c(1.0), self.c(-1.0)));
    }
    fn patch(&mut self, destination: u32, hi: usize, lo: Option<usize>, e: E) {
        self.patches.push([
            destination,
            hi as u32,
            lo.map_or(u32::MAX, |x| x as u32),
            e.1,
        ]);
    }
    pub(super) fn surf(&mut self, row: usize, hi: usize, lo: usize, e: E) {
        self.patch(
            1,
            row * size_of::<PackedSurface>() / 4 + hi / 4,
            Some(row * size_of::<PackedSurface>() / 4 + lo / 4),
            e,
        );
    }
    pub(super) fn coefficient(&mut self, i: usize, e: E) {
        self.patch(2, i * 2, Some(i * 2 + 1), e);
    }

    fn solves(&mut self) -> MkResult<()> {
        let sys = self.model.clone();
        for i in sys.solve_order()? {
            let solve = &sys.solves[i];
            let e = match &solve.kind {
                SolveKind::Pickup {
                    source,
                    scale,
                    offset,
                } => self.param(source)? * *scale + *offset,
                SolveKind::Position { from, to, distance } => {
                    let a = sys.surface_index(*from).unwrap();
                    let b = sys.surface_index(*to).unwrap();
                    let mut value = self.c(*distance);
                    for s in &sys.surfaces[a..b] {
                        let path = ParamPath::Thickness { surface: s.id };
                        if path != solve.target {
                            value = value - self.param(&path)?;
                        }
                    }
                    value
                }
                SolveKind::CenterOfCurvature { surface } => {
                    let j = sys.surface_index(solve.target.surface().unwrap()).unwrap();
                    let k = sys.surface_index(*surface).unwrap();
                    let mut d = self.c(0.0);
                    for s in &sys.surfaces[j.min(k)..j.max(k)] {
                        d = d + self.param(&ParamPath::Thickness { surface: s.id })?;
                    }
                    if k < j {
                        d = -d;
                    }
                    self.positive(d.clone().abs());
                    self.c(1.0) / d
                }
                _ => {
                    return Err(MkError::Unsupported(format!(
                        "P4: reference preparation required for {:?}",
                        solve.kind
                    )));
                }
            };
            self.params.insert(canonical(&solve.target), e);
        }
        Ok(())
    }
}

pub(super) fn compile(
    snapshot: Snapshot,
    library: Arc<MaterialLibrary>,
    variables: Vec<Parameter>,
    material_slots: Vec<MaterialSlot>,
    configuration: Option<usize>,
    ctx: &JobContext,
) -> MkResult<CompiledSystem> {
    ctx.check()?;
    let mut model = if let Some(k) = configuration {
        snapshot.system().with_configuration(k)?
    } else {
        snapshot.system().clone()
    };
    let mut paths = std::collections::BTreeSet::new();
    for v in &variables {
        if !paths.insert(canonical(&v.path))
            || !v.lower.is_finite()
            || !v.upper.is_finite()
            || v.lower > v.upper
        {
            return Err(MkError::InvalidParam(
                "P4 requires unique parameters and finite ordered bounds".into(),
            ));
        }
        // Establish optional placement/range topology and coefficient capacity once.
        let nominal = match v.path.get(&model) {
            Ok(v) => v,
            Err(_) if matches!(v.path, ParamPath::SemiDiameter { .. }) => (v.lower + v.upper) * 0.5,
            Err(e) => return Err(e),
        };
        v.path.set(&mut model, nominal)?;
        if model
            .solved_params()
            .iter()
            .any(|p| canonical(p) == canonical(&v.path))
        {
            return Err(MkError::InvalidParam(
                "P4 variable is solve-controlled".into(),
            ));
        }
        match &v.path {
            ParamPath::Thickness { surface }
                if *surface == model.surfaces[0].id && model.object_at_infinity() =>
            {
                return Err(MkError::Unsupported(
                    "P4: finite/infinite conjugate topology change".into(),
                ));
            }
            ParamPath::Irregularity { .. }
            | ParamPath::ShapeParam {
                parameter:
                    ShapeParameter::DesignOrder
                    | ShapeParameter::TraceOrder
                    | ShapeParameter::DesignWavelengthUm
                    | ShapeParameter::AstigmatismAngleDeg,
                ..
            } => {
                return Err(MkError::Unsupported(format!(
                    "P4: reference preparation for {:?}",
                    v.path
                )));
            }
            _ => {}
        }
    }
    model.validate()?;
    let b = Rc::new(RefCell::new(Algebra::default()));
    let mut material_surfaces = std::collections::BTreeSet::new();
    for slot in &material_slots {
        let i = model
            .surface_index(slot.surface)
            .ok_or_else(|| MkError::InvalidParam("material choice surface".into()))?;
        if slot.choices.is_empty()
            || !material_surfaces.insert(slot.surface)
            || model.surfaces[i].interaction != Interaction::Refract
        {
            return Err(MkError::InvalidParam("invalid material choice slot".into()));
        }
        let is_material = |p: &ParamPath| matches!(p,ParamPath::ModelNd{surface}|ParamPath::ModelVd{surface}|ParamPath::ModelDpgf{surface}|ParamPath::MaterialDnD{surface}|ParamPath::MaterialDvD{surface} if *surface==slot.surface);
        if variables.iter().any(|v| is_material(&v.path))
            || model.solves.iter().any(|s| is_material(&s.target))
            || configuration.is_some_and(|k| {
                snapshot.system().configurations[k]
                    .overrides
                    .iter()
                    .any(|(p, _)| is_material(p))
            })
        {
            return Err(MkError::Unsupported(
                "P4: material choices combined with continuous material edits".into(),
            ));
        }
        for choice in &slot.choices {
            let mut variant = model.clone();
            variant.surfaces[i].material = choice.clone();
            variant.validate()?;
            library.resolve(choice, &model.environment)?;
        }
    }
    let mut c = Compiler {
        choice_active: false,
        model: model.clone(),
        library: library.clone(),
        b,
        params: BTreeMap::new(),
        guards: vec![],
        patches: vec![],
    };
    for (i, v) in variables.iter().enumerate() {
        let overridden = configuration.is_some_and(|k| {
            snapshot.system().configurations[k]
                .overrides
                .iter()
                .any(|(p, _)| canonical(p) == canonical(&v.path))
        });
        if !overridden {
            c.params.insert(canonical(&v.path), E::parameter(&c.b, i));
        }
    }
    c.solves()?;
    // Seed variable coefficient capacity, without changing the published model.
    let mut packing = model.clone();
    for p in c.params.keys() {
        if matches!(
            p,
            ParamPath::Asphere { .. } | ParamPath::ShapeCoefficient { .. }
        ) && p.get(&packing)? == 0.0
        {
            p.set(&mut packing, 1e-12)?;
        }
    }
    // A variable curvature can turn a plane into a conic.
    for p in c.params.keys() {
        if let ParamPath::Curvature { surface } = p {
            let i = packing.surface_index(*surface).unwrap();
            if matches!(packing.surfaces[i].shape, SurfaceShape::Plane) {
                packing.surfaces[i].shape = SurfaceShape::Conic {
                    curvature: 0.0,
                    conic: 0.0,
                };
            }
        }
    }
    let resolved = ResolvedSystem::new(&Snapshot::new(packing)?, library.clone())?;
    let packed = PackedSystem::pack_chunks(&resolved, None)?;
    let mut set = PackedSystemSet::new(&packed)?;
    // A sentinel keeps the typed pool bindable even for all-legacy systems.
    if set.coefficients.is_empty() {
        set.coefficients.push(Df32::ZERO);
    }
    let n = model.surfaces.len();
    let nw = model.wavelengths.wavelengths.len();
    let thickness: Vec<E> = (0..n)
        .map(|i| {
            if (i == 0 && model.object_at_infinity()) || i == n - 1 {
                Ok(c.c(0.0))
            } else {
                c.param(&ParamPath::Thickness {
                    surface: model.surfaces[i].id,
                })
            }
        })
        .collect::<MkResult<_>>()?;
    let frames = c.frames(&thickness)?;
    let mut ns: Vec<Vec<E>> = vec![];
    let mut alphas: Vec<Vec<E>> = vec![];
    let mut wavelengths = vec![];
    for w in 0..nw {
        wavelengths.push(c.param(&ParamPath::WavelengthUm { index: w })?);
        c.positive(wavelengths[w].clone());
    }
    for (i, s) in model.surfaces.iter().enumerate() {
        ctx.check()?;
        ctx.report(0.3 * i as f64 / n as f64, "compile materials");
        if i > 0 && (s.interaction != Interaction::Refract || s.is_coordinate_break()) {
            ns.push(ns[i - 1].clone());
            alphas.push(alphas[i - 1].clone());
            continue;
        }
        let mut ni = vec![];
        let mut ai = vec![];
        for l in &wavelengths {
            let (index, alpha) = if let Some((slot_index, slot)) = material_slots
                .iter()
                .enumerate()
                .find(|(_, slot)| slot.surface == s.id)
            {
                let selector = E::parameter(&c.b, variables.len() + slot_index);
                let mut index = c.c(0.0);
                let mut alpha = c.c(0.0);
                let mut valid = c.c(0.0);
                c.choice_active = true;
                for (j, material) in slot.choices.iter().enumerate() {
                    ctx.check()?;
                    let (nj, aj) = c.material(s.id, material, l.clone())?;
                    let match_choice = (selector.clone() - j as f64).abs();
                    index = match_choice.clone().select(nj, index);
                    alpha = match_choice.clone().select(aj, alpha);
                    valid = valid + match_choice.select(c.c(1.0), c.c(0.0));
                }
                c.choice_active = false;
                c.positive(valid);
                (index, alpha)
            } else {
                c.material(s.id, &s.material, l.clone())?
            };
            ni.push(index);
            ai.push(alpha);
        }
        ns.push(ni);
        alphas.push(ai);
    }
    let mut curvatures = vec![];
    for (i, s) in model.surfaces.iter().enumerate() {
        curvatures.push(c.shape_curvatures(i, &s.shape)?);
    }
    let mut fo = vec![];
    for (w, l) in wavelengths.iter().enumerate() {
        ctx.check()?;
        ctx.report(0.3 + 0.4 * w as f64 / nw as f64, "compile first order");
        let mut signed = Vec::new();
        let mut power_y = Vec::new();
        for axis in 0..2 {
            let mut nn = vec![ns[0][w].clone()];
            let mut powers = vec![c.c(0.0)];
            let mut sign = 1.0;
            for j in 1..n {
                let s = &model.surfaces[j];
                let passive =
                    j == n - 1 || s.is_coordinate_break() || s.interaction == Interaction::Dummy;
                if !passive && s.interaction == Interaction::Reflect {
                    sign = -sign;
                }
                nn.push(ns[j][w].clone() * sign);
                let phi = if passive {
                    c.c(0.0)
                } else if let SurfaceShape::Paraxial { .. } = s.shape {
                    let f = c.param(&ParamPath::ParaxialFocalLength { surface: s.id })?;
                    c.positive(f.clone().abs());
                    ns[j][w].clone() / f
                } else {
                    let mut phi = curvatures[j][axis].clone() * (nn[j].clone() - nn[j - 1].clone());
                    if let SurfaceShape::Diffractive { phase, .. } = &s.shape {
                        let a = c.param(&ParamPath::ShapeCoefficient {
                            surface: s.id,
                            set: CoefficientSet::Phase,
                            index: 0,
                        })?;
                        let r = c.param(&ParamPath::ShapeParam {
                            surface: s.id,
                            parameter: ShapeParameter::NormRadius,
                        })?;
                        phi = phi
                            - a * l.clone()
                                * (phase.trace_order as f64 * 1e-3 / std::f64::consts::PI)
                                / (r.clone() * r);
                    }
                    phi
                };
                powers.push(phi);
            }
            let (mut ya, mut wa, mut yb, mut wb) = (c.c(1.0), c.c(0.0), c.c(0.0), ns[0][w].clone());
            let mut table = vec![];
            let mut last = (ya.clone(), wa.clone(), yb.clone(), wb.clone());
            for j in 0..n {
                if j > 1 {
                    ya = ya + thickness[j - 1].clone() * wa.clone() / nn[j - 1].clone();
                    yb = yb + thickness[j - 1].clone() * wb.clone() / nn[j - 1].clone();
                }
                if j > 0 {
                    wa = wa - ya.clone() * powers[j].clone();
                    wb = wb - yb.clone() * powers[j].clone();
                }
                table.extend([
                    nn[j].clone(),
                    powers[j].clone(),
                    thickness[j].clone(),
                    ya.clone(),
                    wa.clone(),
                    yb.clone(),
                    wb.clone(),
                ]);
                if j == n - 2 {
                    last = (ya.clone(), wa.clone(), yb.clone(), wb.clone());
                }
            }
            let (a, cc, b, d) = last;
            let efl = -c.c(1.0) / cc.clone();
            let bfl = -a.clone() * nn[n - 2].clone() / cc.clone();
            let ffl = d.clone() / cc.clone();
            let track = thickness[1..n - 1]
                .iter()
                .cloned()
                .fold(c.c(0.0), |a, b| a + b);
            fo.extend([efl, bfl, ffl, track, a, b, cc, d]);
            fo.extend(table);
            if axis == 1 {
                signed = nn;
                power_y = powers;
            }
        }
        // Ideal-lens phase uses the candidate's incoming object vergence.
        let (mut y, mut slope) = if model.object_at_infinity() {
            (c.c(1.0), c.c(0.0))
        } else {
            (thickness[0].clone(), ns[0][w].clone())
        };
        for j in 1..n {
            if matches!(model.surfaces[j].shape, SurfaceShape::Paraxial { .. }) {
                let v = slope.clone() / (y.clone() * signed[j - 1].clone().abs());
                let row = (w / crate::layout::MAX_WAVELENGTHS) * n + j;
                let k = w % crate::layout::MAX_WAVELENGTHS;
                c.surf(
                    row,
                    std::mem::offset_of!(PackedSurface, vergence) + k * 4,
                    std::mem::offset_of!(PackedSurface, vergence_lo) + k * 4,
                    v,
                );
            }
            slope = slope - y.clone() * power_y[j].clone();
            y = y + thickness[j].clone() * slope.clone() / signed[j].clone();
        }
    }
    for (chunk, pack) in packed.iter().enumerate() {
        ctx.check()?;
        ctx.report(
            0.7 + 0.25 * chunk as f64 / packed.len() as f64,
            "compile packed outputs",
        );
        for (k, &w) in pack.wavelengths.iter().enumerate() {
            c.patch(
                0,
                chunk * size_of::<PackedHeader>() / 4
                    + std::mem::offset_of!(PackedHeader, wavelength_um) / 4
                    + k,
                None,
                wavelengths[w].clone(),
            );
        }
        for i in 0..n {
            let row = chunk * n + i;
            c.pack_shape(row, i, &mut set)?;
            if i < n - 1 {
                let f = if i == 0 && model.object_at_infinity() {
                    frames[1].clone()
                } else {
                    frames[i + 1].relative_to(&frames[i])
                };
                let rot = transpose(&f.r);
                let shift = mv(&rot, &f.o);
                // General frame path also works at an axial candidate. Never retain
                // a nominal axial shortcut after a variable tilt/decentre.
                set.surfaces[row].flags &= !MKL_SURF_AXIAL;
                set.headers[chunk].flags |= MKL_SYS_HAS_TILTS;
                for k in 0..3 {
                    c.surf(
                        row,
                        std::mem::offset_of!(PackedSurface, to_next_origin) + k * 4,
                        std::mem::offset_of!(PackedSurface, to_next_origin_lo) + k * 4,
                        f.o[k].clone(),
                    );
                    c.surf(
                        row,
                        std::mem::offset_of!(PackedSurface, to_next_shift) + k * 4,
                        std::mem::offset_of!(PackedSurface, to_next_shift_lo) + k * 4,
                        shift[k].clone(),
                    );
                    for j in 0..3 {
                        let off = (k * 3 + j) * 4;
                        c.surf(
                            row,
                            std::mem::offset_of!(PackedSurface, to_next_rot) + off,
                            std::mem::offset_of!(PackedSurface, to_next_rot_lo) + off,
                            rot[k][j].clone(),
                        );
                    }
                }
            }
            let sf = &model.surfaces[i];
            if sf.aperture.is_none()
                && (matches!(
                    sf.semi_diameter,
                    mklight_core::model::SemiDiameter::Fixed { .. }
                ) || c
                    .params
                    .contains_key(&ParamPath::SemiDiameter { surface: sf.id }))
            {
                let r = c.param(&ParamPath::SemiDiameter { surface: sf.id })?;
                c.positive(r.clone());
                set.surfaces[row].aperture = MKL_APERTURE_CIRCULAR;
                set.surfaces[row].flags |= MKL_SURF_APERTURE_FROM_SD;
                c.surf(
                    row,
                    std::mem::offset_of!(PackedSurface, aperture_params),
                    std::mem::offset_of!(PackedSurface, aperture_params_lo),
                    r * (1.0 + mklight_core::model::APERTURE_EDGE_TOLERANCE),
                );
            }
            for (k, &w) in pack.wavelengths.iter().enumerate() {
                c.surf(
                    row,
                    std::mem::offset_of!(PackedSurface, n_after) + k * 4,
                    std::mem::offset_of!(PackedSurface, n_after_lo) + k * 4,
                    ns[i][w].clone(),
                );
                c.patch(
                    1,
                    row * size_of::<PackedSurface>() / 4
                        + std::mem::offset_of!(PackedSurface, alpha_after) / 4
                        + k,
                    None,
                    alphas[i][w].clone(),
                );
                if let SurfaceShape::Diffractive { phase, .. } = &sf.shape {
                    let off = set.surfaces[row].coefficient_offset as usize
                        + MKL_COEFF_PREFIX as usize
                        + k;
                    c.coefficient(
                        off,
                        wavelengths[w].clone()
                            * (phase.trace_order as f64 * 1e-3 / std::f64::consts::TAU),
                    );
                }
            }
        }
    }
    for (i, e) in fo.iter().enumerate() {
        c.patch(3, i * 2, Some(i * 2 + 1), e.clone());
    }
    // Numeric model/spectrum table consumed by aim, analysis and merit clients.
    let aperture = c
        .param(&ParamPath::ApertureValue)
        .unwrap_or_else(|_| c.c(0.0));
    if !matches!(model.aperture, mklight_core::ApertureSpec::FloatByStopSize) {
        c.positive(aperture.clone());
    }
    let temperature = c.param(&ParamPath::TemperatureC)?;
    let pressure = c.param(&ParamPath::PressureAtm)?;
    c.positive(temperature.clone() + 273.15);
    c.nonnegative(pressure.clone());
    let object = if model.object_at_infinity() {
        c.c(f64::INFINITY)
    } else {
        thickness[0].clone()
    };
    let mut numeric = vec![aperture, temperature, pressure, object];
    for (i, f) in model.fields.fields.iter().enumerate() {
        let x = c.param(&ParamPath::FieldX { index: i })?;
        let y = c.param(&ParamPath::FieldY { index: i })?;
        if model.object_at_infinity() && model.fields.kind == mklight_core::FieldKind::ObjectAngle {
            c.positive(c.c(8100.0) - x.clone() * x.clone() - y.clone() * y.clone());
        }
        let weight = c.param(&ParamPath::FieldWeight { index: i })?;
        c.nonnegative(weight.clone());
        numeric.extend([
            x,
            y,
            weight,
            c.c(f.vignetting.vdx),
            c.c(f.vignetting.vdy),
            c.c(f.vignetting.vcx),
            c.c(f.vignetting.vcy),
        ]);
    }
    let mut total = c.c(0.0);
    for (i, w) in wavelengths.iter().enumerate() {
        let weight = c.param(&ParamPath::WavelengthWeight { index: i })?;
        c.nonnegative(weight.clone());
        total = total + weight.clone();
        numeric.extend([w.clone(), weight]);
    }
    c.positive(total);
    for variable in &variables {
        numeric.push(c.param(&variable.path)?);
    }
    for solve in &model.solves {
        numeric.push(c.param(&solve.target)?);
    }
    for (i, e) in numeric.iter().enumerate() {
        c.patch(4, i * 2, Some(i * 2 + 1), e.clone());
    }
    let code = c.b.borrow().code.clone();
    let program = Program {
        code,
        patches: c.patches,
        guards: c.guards,
        headers: bytemuck::cast_slice(&set.headers).to_vec(),
        surfaces: bytemuck::cast_slice(&set.surfaces).to_vec(),
        coefficients: bytemuck::cast_slice(&set.coefficients).to_vec(),
        first_order_len: fo.len(),
        model_len: numeric.len(),
        surface_count: n,
        wavelength_count: nw,
        chunk_count: packed.len(),
    };
    let mut hasher = blake3::Hasher::new();
    for data in [
        &program.headers[..],
        &program.surfaces[..],
        &program.coefficients[..],
        bytemuck::cast_slice(&program.code),
        bytemuck::cast_slice(&program.patches),
        &program.guards,
    ] {
        hasher.update(bytemuck::cast_slice(data));
    }
    let fingerprint = hasher.finalize().to_hex().to_string();
    ctx.check()?;
    ctx.report(1.0, "compile system batch");
    Ok(CompiledSystem {
        snapshot,
        library,
        variables,
        material_slots,
        configuration,
        program: Arc::new(program),
        fingerprint,
        nominal: packed,
    })
}
