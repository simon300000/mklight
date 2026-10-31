#![allow(dead_code)]
use mklight_core::{JobContext, MkResult, ResolvedSystem, Snapshot};
use mklight_gpu::{resident::*, systems::*, *};
use mklight_trace::{
    SequentialTracer,
    paraxial::{ParaxialAxis, first_order_axis},
};

pub fn run(
    b: &dyn ComputeBackend,
    c: &CompiledSystem,
    rows: &[Candidate],
    tile: usize,
) -> (Vec<BufferData>, PlanStats) {
    let u = c.upload(b, &JobContext::new()).unwrap();
    let out = run_uploaded(b, c, &u, rows, tile);
    u.release(b).unwrap();
    out
}
pub fn run_uploaded(
    b: &dyn ComputeBackend,
    c: &CompiledSystem,
    u: &UploadedSystem,
    rows: &[Candidate],
    tile: usize,
) -> (Vec<BufferData>, PlanStats) {
    let mut plan = ResidentPlan::new(c.snapshot().clone());
    let request = u.append(b, &mut plan, rows, &JobContext::new()).unwrap();
    let ResidentOp::Systems(Request::Prepare(r)) = plan.operations.last_mut().unwrap() else {
        unreachable!()
    };
    r.tile_candidates = tile;
    plan.outputs = request.outputs();
    let result = b
        .submit_plan(plan, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(result.stats.readback_bytes, 0);
    assert!(
        result
            .run_info
            .settings
            .to_string()
            .contains("required_per_candidate")
    );
    let data = result
        .outputs
        .iter()
        .map(|h| b.readback(h.view(), &JobContext::new()).unwrap().data)
        .collect();
    (data, result.stats)
}
pub fn decode(data: &[BufferData], row: usize, rs: &ResolvedSystem) -> Vec<PackedSystem> {
    let headers: &[PackedHeader] = bytemuck::cast_slice(data[0].words());
    let surfaces: &[PackedSurface] = bytemuck::cast_slice(data[1].words());
    let coefficients: &[Df32] = bytemuck::cast_slice(data[2].words());
    let chunks = rs.wavelengths_um.len().div_ceil(16);
    let start = row * chunks;
    (start..start + chunks)
        .map(|k| {
            let mut header = headers[k];
            let offset = header.surface_offset as usize;
            let mut ss = surfaces[offset..offset + header.num_surfaces as usize].to_vec();
            let c0 = ss[0].coefficient_offset as usize;
            let c1 = if k + 1 < headers.len() {
                surfaces[headers[k + 1].surface_offset as usize].coefficient_offset as usize
            } else {
                coefficients.len()
            };
            for s in &mut ss {
                s.coefficient_offset -= c0 as u32;
            }
            header.surface_offset = 0;
            let wavelengths = header.wavelength_source[..header.num_wavelengths as usize]
                .iter()
                .map(|&i| i as usize)
                .collect();
            let out = PackedSystem {
                header,
                surfaces: ss,
                coefficients: coefficients[c0..c1].to_vec(),
                snapshot_hash: rs.snapshot.hash(),
                wavelengths,
            };
            out.validate().unwrap();
            out
        })
        .collect()
}
pub fn compare(data: &[BufferData], row: usize, rs: &ResolvedSystem) -> (f64, f64) {
    let got = decode(data, row, rs);
    let mut index_error = 0f64;
    let mut first_error = 0f64;
    assert_eq!(data[6].words()[row * 2], READY, "status row {row}");
    assert_eq!(data[6].words()[row * 2 + 1], DIRTY_ALL);
    for packed in got {
        for (i, s) in packed.surfaces.iter().enumerate() {
            for (j, &w) in packed.wavelengths.iter().enumerate() {
                let n = s.n_after[j] as f64 + s.n_after_lo[j] as f64;
                let error = (n - rs.surfaces[i].n_after[w]).abs();
                index_error = index_error.max(error);
                assert!(error <= 1e-10, "index {i}/{w}: {error}");
            }
        }
        let exact = PackedSystem::pack(
            rs,
            &PackOptions {
                wavelengths: Some(packed.wavelengths.clone()),
                radius_hints: None,
            },
        )
        .unwrap();
        for (s, r) in packed.surfaces.iter().zip(exact.surfaces) {
            for (a, b) in s.to_next_rot.iter().zip(r.to_next_rot) {
                assert!((*a as f64 - b as f64).abs() < 2e-7);
            }
            for k in 0..3 {
                let a = s.to_next_origin[k] as f64 + s.to_next_origin_lo[k] as f64;
                let b = r.to_next_origin[k] as f64 + r.to_next_origin_lo[k] as f64;
                assert!((a - b).abs() < 2e-8, "origin {a}/{b}");
            }
        }
    }
    let fo: &[Df32] = bytemuck::cast_slice(data[3].words());
    let stride = 8 + 7 * rs.surfaces.len();
    let nw = rs.wavelengths_um.len();
    for w in 0..nw {
        for (axis, a) in [ParaxialAxis::X, ParaxialAxis::Y].into_iter().enumerate() {
            let f = first_order_axis(rs, w, a).unwrap();
            let reference = [f.efl, f.bfl, f.ffl, f.total_track];
            for (i, &r) in reference.iter().enumerate() {
                let v = fo[((row * nw + w) * 2 + axis) * stride + i].to_f64();
                let error = (v - r).abs();
                if r.is_finite() {
                    assert!(
                        error <= 2e-5f64.max(r.abs() * 1e-6),
                        "first order {w}/{axis}/{i}: {v} vs {r}"
                    );
                    first_error = first_error.max(error);
                } else {
                    assert_eq!(v.is_finite(), r.is_finite());
                }
            }
        }
    }
    (index_error, first_error)
}
// Use the existing independent f64 ray/OPD/status checker, replacing only its
// upload with the actual prepared candidate. No reference repacking is traced.
pub struct PreparedBackend<'a> {
    pub backend: &'a dyn ComputeBackend,
    pub packed: Vec<PackedSystem>,
}
impl ComputeBackend for PreparedBackend<'_> {
    fn info(&self) -> &DeviceInfo {
        self.backend.info()
    }
    fn precision(&self) -> &PrecisionDescriptor {
        self.backend.precision()
    }
    fn supports(&self, k: JobKind) -> bool {
        self.backend.supports(k)
    }
    fn upload_systems(&self, s: &[PackedSystem]) -> MkResult<SystemHandle> {
        let actual: Vec<_> = s
            .iter()
            .map(|x| {
                self.packed
                    .iter()
                    .find(|p| p.wavelengths == x.wavelengths)
                    .unwrap()
                    .clone()
            })
            .collect();
        self.backend.upload_systems(&actual)
    }
    fn release_systems(&self, h: SystemHandle) -> MkResult<()> {
        self.backend.release_systems(h)
    }
    fn submit(&self, j: Job, c: &JobContext) -> MkResult<Pending<JobResult>> {
        self.backend.submit(j, c)
    }
}
pub fn trace_check(
    b: &dyn ComputeBackend,
    data: &[BufferData],
    row: usize,
    rs: ResolvedSystem,
    n: usize,
) -> xval::XvalReport {
    let packed = decode(data, row, &rs);
    let tr = SequentialTracer::new(rs.into()).unwrap();
    let checker = PreparedBackend { backend: b, packed };
    let settings = xval::XvalSettings {
        pupil: mklight_trace::PupilSampling::Grid { n },
        ..Default::default()
    };
    let report = xval::cross_validate(&checker, &tr, &settings, &JobContext::new()).unwrap();
    report.check(&ErrorBudget::CONVENTIONS).unwrap();
    report
}
pub fn basic_variables(s: &mklight_core::OpticalSystem) -> Vec<Parameter> {
    use mklight_core::param::ParamPath;
    let surface = s.surfaces[1].id;
    let t = s.surfaces[1].thickness;
    vec![
        Parameter {
            path: ParamPath::Thickness { surface },
            lower: t - 0.5,
            upper: t + 0.5,
        },
        Parameter {
            path: ParamPath::Curvature { surface },
            lower: s.surfaces[1].shape.curvature() - 0.002,
            upper: s.surfaces[1].shape.curvature() + 0.002,
        },
    ]
}
pub fn basic_rows(s: &mklight_core::OpticalSystem) -> Vec<Candidate> {
    let t = s.surfaces[1].thickness;
    let c = s.surfaces[1].shape.curvature();
    [(-0.2, -0.0003), (0.0, 0.0), (0.2, 0.0003)]
        .into_iter()
        .enumerate()
        .map(|(i, (dt, dc))| Candidate {
            id: 0x123400000000 + i as u64,
            values: vec![t + dt, c + dc],
        })
        .collect()
}

pub fn extended_cases() -> Vec<(String, CompiledSystem, Vec<Candidate>)> {
    use mklight_core::{materials::*, model::*, param::*, samples};
    use std::sync::Arc;
    let mut out = vec![];
    let mut add = |name: &str,
                   s: OpticalSystem,
                   lib: Arc<MaterialLibrary>,
                   variables: Vec<Parameter>,
                   values: Vec<Vec<f64>>,
                   configuration: Option<usize>| {
        let c = CompiledSystem::compile(
            Snapshot::new(s).unwrap(),
            lib,
            variables,
            configuration,
            &JobContext::new(),
        )
        .unwrap();
        out.push((
            name.into(),
            c,
            values
                .into_iter()
                .enumerate()
                .map(|(i, values)| Candidate {
                    id: 900 + i as u64,
                    values,
                })
                .collect(),
        ));
    };
    let var = |path, lower, upper| Parameter { path, lower, upper };
    let mut s = samples::finite_relay();
    let id = s.surfaces[1].id;
    let last = s.surfaces[2].id;
    let paths = vec![
        var(ParamPath::DecenterX { surface: id }, -0.5, 0.5),
        var(ParamPath::TiltX { surface: id }, -3.0, 3.0),
        var(ParamPath::TiltY { surface: id }, -3.0, 3.0),
        var(ParamPath::TiltZ { surface: id }, -3.0, 3.0),
        var(
            ParamPath::ElementPlacement {
                surface: id,
                last,
                pivot: Some(last),
                component: PlacementComponent::TiltY,
            },
            -2.0,
            2.0,
        ),
    ];
    add(
        "rigid-finite",
        s.clone(),
        testutil::library(),
        paths,
        vec![
            vec![0.2, 1.1, -0.4, 0.7, 1.5],
            vec![-0.2, -1.1, 0.4, -0.7, -1.5],
        ],
        None,
    );
    s = samples::singlet();
    let a = s.surfaces[1].id;
    let b = s.surfaces[2].id;
    s.solves = vec![
        Solve {
            target: ParamPath::Curvature { surface: b },
            kind: SolveKind::Pickup {
                source: ParamPath::Curvature { surface: a },
                scale: -0.3,
                offset: 0.001,
            },
        },
        Solve {
            target: ParamPath::Thickness { surface: b },
            kind: SolveKind::Position {
                from: a,
                to: s.surfaces[3].id,
                distance: 105.0,
            },
        },
    ];
    s.configurations = vec![
        Configuration {
            name: "base".into(),
            overrides: vec![],
        },
        Configuration {
            name: "focus".into(),
            overrides: vec![(ParamPath::Thickness { surface: a }, 8.0)],
        },
    ];
    for cfg in [0, 1] {
        add(
            &format!("configuration-{cfg}"),
            s.clone(),
            testutil::library(),
            vec![
                var(ParamPath::Curvature { surface: a }, 0.01, 0.03),
                var(ParamPath::Thickness { surface: a }, 4.0, 8.0),
            ],
            vec![vec![0.015, 4.5], vec![0.02, 6.0]],
            Some(cfg),
        );
    }
    let mut lib = (*testutil::library()).clone();
    let mut cat = GlassCatalog {
        name: "P4-THERMAL".into(),
        ..Default::default()
    };
    let mut glass = lib.catalogs()[0]
        .glasses
        .values()
        .find(|g| g.name == "N-BK7")
        .unwrap()
        .clone();
    glass.name = "thermal".into();
    glass.thermal = Some(ThermalCoefficients {
        d0: 1.2e-6,
        d1: 2e-8,
        d2: -3e-11,
        e0: 4e-7,
        e1: 1e-9,
        lambda_tk_um: 0.12,
        t_ref_c: 22.0,
    });
    glass.transmittance = vec![
        TransmittancePoint {
            um: 0.4,
            tau: 0.98,
            thickness_mm: 10.0,
        },
        TransmittancePoint {
            um: 0.55,
            tau: 0.995,
            thickness_mm: 10.0,
        },
        TransmittancePoint {
            um: 0.8,
            tau: 0.999,
            thickness_mm: 10.0,
        },
    ];
    cat.glasses.insert("thermal".into(), glass);
    lib.add_catalog(Arc::new(cat));
    s = samples::singlet();
    s.surfaces[1].material = MaterialRef::glass("thermal");
    s.wavelengths.wavelengths = (0..20)
        .map(|i| Wavelength {
            um: 0.43 + i as f64 * 0.014,
            weight: (i + 1) as f64,
        })
        .collect();
    add(
        "thermal-offset-20wl",
        s.clone(),
        Arc::new(lib),
        vec![
            var(ParamPath::TemperatureC, -30.0, 80.0),
            var(ParamPath::PressureAtm, 0.7, 1.3),
            var(ParamPath::MaterialDnD { surface: a }, -0.01, 0.01),
            var(ParamPath::MaterialDvD { surface: a }, -3.0, 3.0),
            var(ParamPath::WavelengthUm { index: 2 }, 0.45, 0.5),
        ],
        vec![
            vec![-20.0, 0.8, -0.005, 2.0, 0.46],
            vec![70.0, 1.2, 0.005, -2.0, 0.49],
        ],
        None,
    );
    s = samples::singlet();
    s.surfaces[1].material = MaterialRef::Model {
        nd: 1.52,
        vd: 60.0,
        dpgf: 0.0,
    };
    add(
        "model-glass",
        s,
        testutil::library(),
        vec![
            var(ParamPath::ModelNd { surface: a }, 1.4, 1.8),
            var(ParamPath::ModelVd { surface: a }, 30.0, 80.0),
            var(ParamPath::ModelDpgf { surface: a }, -0.01, 0.01),
        ],
        vec![vec![1.48, 68.0, 0.005], vec![1.7, 35.0, -0.004]],
        None,
    );
    for (name, s) in samples::all() {
        let mut variables = vec![];
        let mut values = vec![];
        for sf in &s.surfaces {
            let set = match sf.shape {
                SurfaceShape::OddAsphere { .. } => Some(CoefficientSet::Odd),
                SurfaceShape::QCon { .. } | SurfaceShape::QBfs { .. } => Some(CoefficientSet::Q),
                SurfaceShape::XYPolynomial { .. } => Some(CoefficientSet::Xy),
                SurfaceShape::ZernikeSag { .. } => Some(CoefficientSet::Zernike),
                SurfaceShape::Diffractive { .. } => Some(CoefficientSet::Phase),
                SurfaceShape::Toroid { .. } | SurfaceShape::Biconic { .. } => {
                    Some(CoefficientSet::YAsphere)
                }
                _ => None,
            };
            if let Some(set) = set {
                let index = if set == CoefficientSet::Odd { 1 } else { 0 };
                let path = ParamPath::ShapeCoefficient {
                    surface: sf.id,
                    set,
                    index,
                };
                let initial = path.get(&s).unwrap();
                let delta = if set == CoefficientSet::Phase {
                    0.5
                } else if set == CoefficientSet::YAsphere {
                    1e-7
                } else {
                    1e-4
                };
                variables.push(var(path, initial - delta * 2.0, initial + delta * 2.0));
                values.push(initial + delta);
            }
        }
        if !variables.is_empty() {
            add(
                &format!("coefficients-{name}"),
                s,
                testutil::library(),
                variables,
                vec![values],
                None,
            );
        }
    }
    drop(add);
    s = samples::singlet();
    let snapshot = Snapshot::new(s).unwrap();
    let c = CompiledSystem::compile_with_materials(
        snapshot,
        testutil::library(),
        vec![var(ParamPath::TemperatureC, 0.0, 80.0)],
        vec![MaterialSlot {
            surface: a,
            choices: vec![
                MaterialRef::glass("N-BK7"),
                MaterialRef::glass("F2"),
                MaterialRef::Model {
                    nd: 1.6,
                    vd: 40.0,
                    dpgf: 0.002,
                },
            ],
        }],
        None,
        &JobContext::new(),
    )
    .unwrap();
    out.push((
        "material-choices".into(),
        c,
        vec![
            Candidate {
                id: 11,
                values: vec![45.0, 2.0],
            },
            Candidate {
                id: 9,
                values: vec![55.0, 0.0],
            },
            Candidate {
                id: 4,
                values: vec![10.0, 1.0],
            },
        ],
    ));
    out
}
