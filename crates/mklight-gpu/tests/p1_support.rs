#![allow(dead_code)]
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{aiming::*, resident::*, *};
use mklight_trace::SequentialTracer;
pub fn upload<T: bytemuck::Pod>(
    b: &dyn ComputeBackend,
    p: &mut ResidentPlan,
    rows: usize,
    values: &[T],
) -> PlanView {
    let width = size_of::<T>() / 4;
    let buf = b
        .fill_buffer(
            BufferDesc::new(ElementType::U32, [rows, width]),
            &mut |mut fill| {
                bytemuck::cast_slice_mut::<u32, T>(fill.as_mut_slice::<u32>()?)
                    .copy_from_slice(values);
                Ok(())
            },
        )
        .unwrap();
    let id = p.import(buf.view());
    p.view(id).unwrap()
}
pub fn df(b: &dyn ComputeBackend, p: &mut ResidentPlan, v: &[Df32]) -> PlanView {
    let buf = b
        .fill_buffer(
            BufferDesc::new(ElementType::DoubleFloat, [v.len()]),
            &mut |mut fill| {
                fill.as_mut_slice::<Df32>()?.copy_from_slice(v);
                Ok(())
            },
        )
        .unwrap();
    let id = p.import(buf.view());
    p.view(id).unwrap()
}
#[allow(clippy::too_many_arguments)]
pub fn make_plan(
    b: &dyn QualificationBackend,
    t: &SequentialTracer,
    field: usize,
    wl: usize,
    points: &[[f64; 2]],
    records: bool,
    cap: u32,
    reference: bool,
) -> (ResidentPlan, SystemHandle) {
    let pack = PackedSystem::pack_all(t.system()).unwrap();
    let handle = b.upload_system(&pack).unwrap();
    let snap = Snapshot::new(t.system().system().clone()).unwrap();
    let mut p = ResidentPlan::new(snap);
    p.seed = Some(42);
    p.settings = serde_json::json!({"package":"P1","field":field,"wavelength":wl,"aim":t.aiming(),"iterations":cap,"tolerance":1e-10,"seed":42,"records":records});
    let spec = FieldSpec::new(
        t.system(),
        t,
        t.field(field).point,
        0,
        wl as u32,
        t.aiming(),
    )
    .unwrap();
    let fw = upload(b, &mut p, 1, &[spec]);
    let sample_buffer = b
        .fill_buffer(
            BufferDesc::new(ElementType::U32, [points.len(), size_of::<Sample>() / 4]),
            &mut |mut fill| {
                let out: &mut [Sample] = bytemuck::cast_slice_mut(fill.as_mut_slice::<u32>()?);
                for (i, (&pt, sample)) in points.iter().zip(out).enumerate() {
                    *sample = Sample::new(pt, 0, 0x100000000 + i as u64);
                }
                Ok(())
            },
        )
        .unwrap();
    let si = p.import(sample_buffer.view());
    let sv = p.view(si).unwrap();
    let generated = rays(&mut p, points.len());
    p.operations
        .push(ResidentOp::Aiming(Request::GenerateAim(GenerateAim {
            systems: handle,
            samples: sv,
            field_wavelength: fw,
            field_key: Some(format!("{field}:{wl}:{:?}", bytemuck::bytes_of(&spec))),
            seeds: None,
            tolerance: 1e-10,
            max_iterations: cap,
            output: generated.clone(),
            options: Default::default(),
        })));
    let steps = t.system().image_index();
    let buf = b
        .fill_buffer(
            BufferDesc::new(ElementType::U32, [steps]),
            &mut |mut fill| {
                for (i, word) in fill.as_mut_slice::<u32>()?.iter_mut().enumerate() {
                    *word = (i + 1) as u32;
                }
                Ok(())
            },
        )
        .unwrap();
    let si = p.import(buf.view());
    let hit = hits(&mut p, points.len(), if records { steps + 1 } else { 0 });
    p.operations
        .push(ResidentOp::Aiming(Request::TraceResident(TraceResident {
            systems: handle,
            input: generated.clone(),
            steps: p.view(si).unwrap(),
            start_surface: 0,
            apply_apertures: true,
            radiometry: 0,
            newton: Default::default(),
            output: hit.clone(),
        })));
    p.outputs = vec![
        generated.floats,
        generated.report,
        generated.logical_ids,
        hit.rays.floats,
        hit.status,
    ];
    if records {
        p.outputs
            .extend([hit.records.unwrap(), hit.record_low.unwrap()]);
    }
    if reference {
        let r = t.opd_reference(t.field(field), wl).unwrap();
        let cv = [
            r.center.x,
            r.center.y,
            r.center.z,
            r.chief_dir.x,
            r.chief_dir.y,
            r.chief_dir.z,
            r.chief_opl.value(),
        ]
        .map(Df32::from_f64);
        let chief = df(b, &mut p, &cv);
        let centre = df(b, &mut p, &cv[..3]);
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [4, points.len()]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [points.len()]));
        p.operations
            .push(ResidentOp::Aiming(Request::ProjectPupil(ProjectPupil {
                hits: hit,
                reference: PupilReference {
                    chief,
                    centre,
                    radius: Df32::from_f64(if r.s_exit_pupil.is_finite() {
                        r.s_exit_pupil.abs()
                    } else {
                        0.0
                    }),
                    image_index: Df32::from_f64(r.n_image),
                    wavelength_um: Df32::from_f64(r.wavelength_um),
                    sphere_sheet: if r.s_exit_pupil < 0.0 { -1 } else { 1 },
                    telecentric: !r.s_exit_pupil.is_finite(),
                },
                output,
                status,
            })));
        p.outputs.extend([output, status]);
    }
    (p, handle)
}
/// The plan with its device chief solve replaced by the reference tracer's
/// f64 chief row (`ChiefSeed::host`) for `field` and `wl`.
pub fn host_seeded(
    b: &dyn ComputeBackend,
    mut p: ResidentPlan,
    t: &SequentialTracer,
    field: usize,
    wl: usize,
) -> ResidentPlan {
    let seeds = upload(b, &mut p, 1, &[ChiefSeed::host(t, t.field(field), wl)]);
    for op in &mut p.operations {
        if let ResidentOp::Aiming(Request::GenerateAim(g)) = op {
            g.seeds = Some(seeds.clone());
        }
    }
    p
}
/// Point every operation of `p` at another system handle.
pub fn rebind(mut p: ResidentPlan, h: SystemHandle) -> ResidentPlan {
    for op in &mut p.operations {
        match op {
            ResidentOp::Aiming(Request::GenerateAim(g)) => g.systems = h,
            ResidentOp::Aiming(Request::TraceResident(t)) => t.systems = h,
            _ => {}
        }
    }
    p
}
/// The plan without cross-plan aiming reuse (the uncached reference).
pub fn uncached(mut p: ResidentPlan) -> ResidentPlan {
    for op in &mut p.operations {
        if let ResidentOp::Aiming(Request::GenerateAim(g)) = op {
            g.field_key = None;
        }
    }
    p
}
pub fn read_all(b: &dyn QualificationBackend, result: &PlanResult) -> Vec<BufferData> {
    result
        .outputs
        .iter()
        .map(|x| b.readback(x.view(), &JobContext::new()).unwrap().data)
        .collect()
}
pub fn floats(d: &BufferData) -> &[f32] {
    match d {
        BufferData::F32(v) => v,
        _ => panic!(),
    }
}
pub fn uints(d: &BufferData) -> &[u32] {
    match d {
        BufferData::U32(v) => v,
        _ => panic!(),
    }
}
pub fn doubles(d: &BufferData) -> &[Df32] {
    match d {
        BufferData::DoubleFloat(v) => v,
        _ => panic!(),
    }
}
pub fn bits(a: &[BufferData], b: &[BufferData]) {
    assert_eq!(a.len(), b.len());
    for (k, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.words().len(), b.words().len());
        for (i, (&a, &b)) in a.words().iter().zip(b.words()).enumerate() {
            assert!(
                a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan()),
                "output {k} word {i}: {a:08x} != {b:08x}"
            );
        }
    }
}
#[derive(Default, Debug)]
pub struct Errors {
    pub position: f64,
    pub direction: f64,
    pub opd: f64,
    pub stop: f64,
    pub stop_reference: f64,
    pub record_opl: f64,
    pub normal: f64,
    pub fallback: usize,
    pub survived: usize,
}
pub fn check_f64(
    t: &SequentialTracer,
    field: usize,
    wl: usize,
    points: &[[f64; 2]],
    data: &[BufferData],
    record: bool,
    reference: bool,
) -> Errors {
    let mut e = Errors::default();
    let n = points.len();
    let f = floats(&data[0]);
    let report = uints(&data[1]);
    let ids = uints(&data[2]);
    let hit = floats(&data[3]);
    let status = uints(&data[4]);
    let mut rec = vec![];
    for (i, &pt) in points.iter().enumerate() {
        assert_eq!(ids[i], i as u32);
        assert_eq!(ids[n + i], 1);
        if report[i] != 0 {
            e.fallback += 1;
            continue;
        }
        let g = t.generate(t.field(field), wl, pt[0], pt[1]).unwrap();
        let gp = [g.ray.pos.x, g.ray.pos.y, g.ray.pos.z];
        let gd = [g.ray.dir.x, g.ray.dir.y, g.ray.dir.z];
        for k in 0..3 {
            e.position = e
                .position
                .max(((f[k * n + i] as f64 + f[(9 + k) * n + i] as f64) - gp[k]).abs());
            e.direction = e
                .direction
                .max(((f[(3 + k) * n + i] as f64 + f[(12 + k) * n + i] as f64) - gd[k]).abs());
        }
        if matches!(t.aiming(), mklight_core::RayAiming::Real { .. }) {
            let pos = mklight_core::Vec3::new(
                f[i] as f64 + f[9 * n + i] as f64,
                f[n + i] as f64 + f[10 * n + i] as f64,
                f[2 * n + i] as f64 + f[11 * n + i] as f64,
            );
            let dir = mklight_core::Vec3::new(
                f[3 * n + i] as f64 + f[12 * n + i] as f64,
                f[4 * n + i] as f64 + f[13 * n + i] as f64,
                f[5 * n + i] as f64 + f[14 * n + i] as f64,
            );
            let ray = mklight_trace::trace::Ray::new(pos, dir);
            let stop = mklight_trace::trace_ray(
                t.system(),
                &ray,
                wl,
                &mklight_trace::TraceOptions::unclipped().to_surface(t.system().stop),
                None,
            );
            assert!(stop.status.is_ok());
            let (px, py) = t.field(field).point.vignetting.apply(pt[0], pt[1]);
            let r = t.pupil().r_stop;
            let residual = (stop.ray.pos.x - px * r).hypot(stop.ray.pos.y - py * r) / r;
            e.stop_reference = e.stop_reference.max(residual);
            assert!(
                residual * r <= 1e-10 * r + 1e-7,
                "independent stop residual {residual}"
            );
        }
        let traced = t.trace_with(
            t.field(field),
            wl,
            pt[0],
            pt[1],
            &t.options(),
            Some(&mut rec),
        );
        assert_eq!(
            status[i] == 0,
            traced.status.is_ok(),
            "field {field} wl {wl} pupil {pt:?} GPU {} CPU {:?}",
            status[i],
            traced.status
        );
        if status[i] == 0 {
            e.survived += 1;
            let hp = [traced.ray.pos.x, traced.ray.pos.y, traced.ray.pos.z];
            let hd = [traced.ray.dir.x, traced.ray.dir.y, traced.ray.dir.z];
            for k in 0..3 {
                e.position = e
                    .position
                    .max(((hit[k * n + i] as f64 + hit[(9 + k) * n + i] as f64) - hp[k]).abs());
                e.direction = e.direction.max(
                    ((hit[(3 + k) * n + i] as f64 + hit[(12 + k) * n + i] as f64) - hd[k]).abs(),
                );
            }
            if reference {
                let oi = if record { 7 } else { 5 };
                assert_eq!(uints(&data[oi + 1])[i], 0);
                let cpu = t
                    .opd_reference(t.field(field), wl)
                    .unwrap()
                    .opd_waves(&traced)
                    .unwrap();
                e.opd = e.opd.max((doubles(&data[oi])[i].to_f64() - cpu).abs());
            }
        }
        if record {
            let hi = floats(&data[5]);
            let lo = floats(&data[6]);
            for (k, r) in rec.iter().enumerate() {
                let a = (k * n + i) * 13;
                e.record_opl = e
                    .record_opl
                    .max((hi[a + 6] as f64 + lo[a + 6] as f64 - r.opl).abs());
                if k > 0 {
                    let normal = std::array::from_fn::<_, 3, _>(|j| {
                        hi[a + 8 + j] as f64 + lo[a + 8 + j] as f64
                    });
                    let length = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let expected = [r.normal.x, r.normal.y, r.normal.z];
                    for j in 0..3 {
                        e.normal = e.normal.max((normal[j] / length - expected[j]).abs());
                    }
                }
                let pos = [r.point.x, r.point.y, r.point.z];
                let dir = [r.dir.x, r.dir.y, r.dir.z];
                for j in 0..3 {
                    e.position = e
                        .position
                        .max((hi[a + j] as f64 + lo[a + j] as f64 - pos[j]).abs());
                    e.direction = e
                        .direction
                        .max((hi[a + j + 3] as f64 + lo[a + j + 3] as f64 - dir[j]).abs());
                }
            }
        }
        e.stop = e.stop.max(
            f32::from_bits(report[2 * n + i]) as f64 + f32::from_bits(report[3 * n + i]) as f64,
        );
    }
    assert!(e.normal <= 2e-6, "record normal {:?}", e);
    assert!(e.record_opl <= 1e-7, "record OPL {:?}", e);
    assert!(e.position <= 2e-5, "position {:?}", e);
    assert!(e.direction <= 2e-6, "direction {:?}", e);
    assert!(e.opd <= 5e-3, "OPD {:?}", e);
    assert!(e.stop <= 1.000001e-10, "aim {:?}", e);
    e
}

pub fn special_systems() -> Vec<(String, mklight_core::OpticalSystem)> {
    use mklight_core::*;
    let mut out: Vec<_> = testutil::xval_systems().into_iter().skip(32).collect();
    let mut tele = OpticalSystem::new_empty();
    tele.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, 100.0, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::Paraxial { focal_length: 50.0 },
            50.0,
            MaterialRef::Air,
        ),
        Surface::new(2, SurfaceShape::Plane, 50.0, MaterialRef::Air),
        Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    tele.stop = 2;
    tele.aperture = ApertureSpec::ObjectSpaceNa { value: 0.05 };
    tele.fields = FieldSet {
        kind: FieldKind::ObjectHeight,
        fields: vec![Field::new(0., 0.), Field::new(2., 5.)],
    };
    out.push(("object telecentric".into(), tele));
    let mut image = OpticalSystem::new_empty();
    image.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        Surface::new(1, SurfaceShape::Plane, 50.0, MaterialRef::Air),
        Surface::new(
            2,
            SurfaceShape::Paraxial { focal_length: 50.0 },
            50.5,
            MaterialRef::Air,
        ),
        Surface::new(3, SurfaceShape::Plane, 0., MaterialRef::Air),
    ];
    image.stop = 1;
    image.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
    image.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0., 0.), Field::new(0., 5.)],
    };
    out.push(("image telecentric defocus".into(), image));
    let mut finite = samples::finite_relay();
    finite.fields.kind = FieldKind::RealImageHeight;
    finite.fields.fields = vec![Field::new(0., 0.), Field::new(3., 4.)];
    out.push(("finite real image".into(), finite));
    let mut vignette = samples::cooke_triplet();
    vignette.fields.kind = FieldKind::ParaxialImageHeight;
    vignette.fields.fields[1].vignetting = Vignetting {
        vdx: 0.1,
        vdy: -0.05,
        vcx: 0.2,
        vcy: 0.15,
    };
    out.push(("paraxial image vignette".into(), vignette));
    let mut curved = samples::finite_relay();
    curved.surfaces[0].shape = samples::biconic_element().surfaces[1].shape.clone();
    curved.fields = mklight_core::FieldSet {
        kind: mklight_core::FieldKind::ObjectHeight,
        fields: vec![
            mklight_core::Field::new(0., 0.),
            mklight_core::Field::new(1.2, 0.7),
        ],
    };
    out.push(("finite extended object".into(), curved));
    for order in [-2, 0, 3] {
        let mut model = samples::hybrid_diffractive_achromat();
        for surface in &mut model.surfaces {
            if let mklight_core::SurfaceShape::Diffractive { phase, .. } = &mut surface.shape {
                phase.trace_order = order;
            }
        }
        out.push((format!("DOE order {order}"), model));
    }
    out
}
