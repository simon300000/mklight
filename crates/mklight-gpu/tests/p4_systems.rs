mod p4_common;
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{systems::*, *};
use p4_common::*;
#[test]
fn all_samples_prepare_against_independent_reference() {
    let backend = CpuF32Backend::new();
    for (name, system) in mklight_core::samples::all() {
        let variables = basic_variables(&system);
        let rows = basic_rows(&system);
        let c = CompiledSystem::compile(
            Snapshot::new(system).unwrap(),
            testutil::library(),
            variables,
            None,
            &JobContext::new(),
        )
        .unwrap();
        let (out, _) = run(&backend, &c, &rows, 64);
        for (i, row) in rows.iter().enumerate() {
            let rs = c.reference(row, &JobContext::new()).unwrap();
            let (ni, fo) = compare(&out, i, &rs);
            let trace = trace_check(&backend, &out, i, rs, 9);
            eprintln!("{name} row={i} n={ni:e} first={fo:e} trace={trace:?}");
        }
    }
}
#[test]
fn extended_preparation_reference_and_reordering() {
    let backend = CpuF32Backend::new();
    for (name, c, rows) in extended_cases() {
        let (out, _) = run(&backend, &c, &rows, 64);
        for (i, row) in rows.iter().enumerate() {
            let rs = c.reference(row, &JobContext::new()).unwrap();
            let (ni, fo) = compare(&out, i, &rs);
            let trace = trace_check(&backend, &out, i, rs, 9);
            eprintln!("{name}: n={ni:e} first={fo:e} trace={trace:?}");
        }
        let reversed: Vec<_> = rows.iter().rev().cloned().collect();
        let (rev, _) = run(&backend, &c, &reversed, 1);
        for (i, row) in reversed.iter().enumerate() {
            let rs = c.reference(row, &JobContext::new()).unwrap();
            let got = decode(&rev, i, &rs);
            let want = decode(&out, rows.len() - 1 - i, &rs);
            for (a, b) in got.iter().zip(&want) {
                assert_eq!(
                    bytemuck::cast_slice::<_, u32>(&a.surfaces),
                    bytemuck::cast_slice::<_, u32>(&b.surfaces)
                );
                assert_eq!(a.coefficients, b.coefficients);
            }
        }
    }
}

#[test]
fn invalid_slots_fallback_identity_and_thick_lens_derivative() {
    use mklight_core::{
        MkError,
        materials::MaterialLibrary,
        model::{MaterialRef, Solve, SolveKind},
        param::ParamPath,
    };
    use mklight_gpu::resident::*;
    use std::sync::Arc;
    let backend = CpuF32Backend::new();
    let mut s = mklight_core::samples::singlet();
    s.surfaces[1].material = MaterialRef::Constant { n: 1.5 };
    let id = s.surfaces[1].id;
    let variable = Parameter {
        path: ParamPath::Curvature { surface: id },
        lower: 0.0,
        upper: 0.03,
    };
    let nominal = s.surfaces[1].shape.curvature();
    let t = s.surfaces[1].thickness;
    let c2 = s.surfaces[2].shape.curvature();
    let snapshot = Snapshot::new(s.clone()).unwrap();
    let library = Arc::new(MaterialLibrary::with_builtin());
    let c = CompiledSystem::compile(
        snapshot.clone(),
        library.clone(),
        vec![variable.clone()],
        None,
        &JobContext::new(),
    )
    .unwrap();
    let rows = vec![
        Candidate {
            id: 99,
            values: vec![nominal],
        },
        Candidate {
            id: 14,
            values: vec![0.04],
        },
        Candidate {
            id: 89,
            values: vec![f64::NAN],
        },
    ];
    let (out, _) = run(&backend, &c, &rows, 1);
    assert_eq!(out[4].words(), [99, 0, 14, 0, 89, 0]);
    assert_eq!(
        out[6].words(),
        [
            READY,
            DIRTY_ALL,
            OUTSIDE_BOUNDS,
            DIRTY_ALL,
            OUTSIDE_BOUNDS,
            DIRTY_ALL
        ]
    );
    // Thick lens power P=(n-1)(c1-c2)+(n-1)^2*t*c1*c2/n.
    let n = 1.5;
    let power = (n - 1.0) * (nominal - c2) + (n - 1.0) * (n - 1.0) * t * nominal * c2 / n;
    let analytic = -((n - 1.0) + (n - 1.0) * (n - 1.0) * t * c2 / n) / (power * power);
    for h in [1e-5, 1e-6, 1e-7] {
        let rows = [
            Candidate {
                id: 0,
                values: vec![nominal - h],
            },
            Candidate {
                id: 1,
                values: vec![nominal + h],
            },
        ];
        let (out, _) = run(&backend, &c, &rows, 64);
        let BufferData::DoubleFloat(fo) = &out[3] else {
            unreachable!()
        };
        let stride = s.wavelengths.wavelengths.len() * 2 * (8 + 7 * s.surfaces.len());
        let derivative = (fo[stride].to_f64() - fo[0].to_f64()) / (2.0 * h);
        assert!(
            (derivative / analytic - 1.0).abs() < 1e-5,
            "{h}: {derivative} vs {analytic}"
        );
    }
    // Changing imported instruction identity must fail at submission, before
    // any GPU could interpret attacker-controlled indices.
    let uploaded = c.upload(&backend, &JobContext::new()).unwrap();
    let mut plan = ResidentPlan::new(snapshot.clone());
    let req = uploaded
        .append(&backend, &mut plan, &rows, &JobContext::new())
        .unwrap();
    let replacement = backend
        .upload_buffer(
            req.dependencies.desc.clone(),
            BufferData::U32(vec![u32::MAX; req.dependencies.desc.len().unwrap()]),
        )
        .unwrap();
    plan.buffers[req.dependencies.buffer.0] = PlanBuffer::Import(replacement.view());
    assert!(backend.submit_plan(plan, &JobContext::new()).is_err());
    uploaded.release(&backend).unwrap();
    s.solves.push(Solve {
        target: ParamPath::Thickness {
            surface: s.surfaces[2].id,
        },
        kind: SolveKind::MarginalRayHeight { height: 0.0 },
    });
    let snapshot = Snapshot::new(s).unwrap();
    let err = CompiledSystem::compile(
        snapshot.clone(),
        library.clone(),
        vec![variable.clone()],
        None,
        &JobContext::new(),
    )
    .unwrap_err();
    assert!(matches!(err, MkError::Unsupported(_)));
    let fallback = reference_batch(
        &snapshot,
        library,
        &[variable],
        &rows,
        None,
        &err.to_string(),
        &JobContext::new(),
    )
    .unwrap();
    assert_eq!(fallback.fallback_count, 3);
    assert!(fallback.candidates[0].1.is_ok());
    assert!(fallback.candidates[1].1.is_err());
    assert_eq!(fallback.candidates[2].0, 89);
    assert!(
        fallback
            .run_info
            .settings
            .to_string()
            .contains("MarginalRayHeight")
    );
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert!(matches!(
        CompiledSystem::compile(snapshot, testutil::library(), vec![], None, &ctx),
        Err(MkError::Cancelled)
    ));
}
