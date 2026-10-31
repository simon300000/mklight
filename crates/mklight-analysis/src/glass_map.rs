//! Catalogue and design glass coordinates at standard spectral lines.
use mklight_core::job::{Precision, RunTimer};
use mklight_core::materials::agf::GlassStatus;
use mklight_core::materials::{MaterialLibrary, ResolvedMaterial, TransmittancePoint, lines};
use mklight_core::{JobContext, MaterialRef, MkResult, RunInfo, Snapshot};
use serde::{Deserialize, Serialize};

/// Optional overrides of the availability retained on catalogue materials.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlassMetadata {
    pub catalog: String,
    pub name: String,
    pub status: GlassStatus,
    pub exclude_substitution: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GlassMapSettings {
    pub metadata: Vec<GlassMetadata>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlassPoint {
    pub catalog: Option<String>,
    pub name: String,
    pub kind: String,
    #[serde(with = "crate::util::nf")]
    pub nd: f64,
    #[serde(with = "crate::util::nf")]
    pub vd: f64,
    #[serde(with = "crate::util::nf")]
    pub pgf: f64,
    pub status: Option<GlassStatus>,
    pub exclude_substitution: Option<bool>,
    pub relative_cost: Option<f64>,
    pub climate_resistance: Option<f64>,
    pub stain_resistance: Option<f64>,
    pub transmittance: Vec<TransmittancePoint>,
    pub surfaces: Vec<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlassMapResult {
    pub glasses: Vec<GlassPoint>,
    pub run_info: RunInfo,
}
fn coordinates(m: &ResolvedMaterial) -> [f64; 3] {
    let [d, f, c, g] = [lines::D, lines::F, lines::C, lines::G].map(|l| m.index(l));
    [d, (d - 1.0) / (f - c), (g - f) / (f - c)]
}
pub fn glass_map_job(
    snapshot: &Snapshot,
    library: &MaterialLibrary,
    s: &GlassMapSettings,
    ctx: &JobContext,
) -> MkResult<GlassMapResult> {
    let timer = RunTimer::start();
    ctx.check()?;
    let mut glasses = Vec::new();
    let count: usize = library.catalogs().iter().map(|c| c.glasses.len()).sum();
    for cat in library.catalogs() {
        for g in cat.glasses.values() {
            ctx.check()?;
            let pgf = (g.catalog_index(lines::G) - g.catalog_index(lines::F))
                / (g.catalog_index(lines::F) - g.catalog_index(lines::C));
            let meta = s.metadata.iter().find(|m| {
                m.catalog.eq_ignore_ascii_case(&cat.name) && m.name.eq_ignore_ascii_case(&g.name)
            });
            glasses.push(GlassPoint {
                catalog: Some(cat.name.clone()),
                name: g.name.clone(),
                kind: "catalog".into(),
                nd: g.nd,
                vd: g.vd,
                pgf,
                status: Some(meta.map_or(g.substitution.status, |m| m.status)),
                exclude_substitution: Some(meta.map_or(g.substitution.exclude_substitution, |m| {
                    m.exclude_substitution
                })),
                relative_cost: g.relative_cost,
                climate_resistance: g.substitution.climate_resistance,
                stain_resistance: g.substitution.stain_resistance,
                transmittance: g.transmittance.clone(),
                surfaces: Vec::new(),
            });
            ctx.report(
                0.8 * glasses.len() as f64 / count.max(1) as f64,
                "glass map",
            );
        }
    }
    for (i, surface) in snapshot.system().surfaces.iter().enumerate() {
        ctx.check()?;
        match &surface.material {
            MaterialRef::Catalog { catalog, name } => {
                // The entry the design resolves to: the first loaded
                // catalogue (optionally the named one) holding the glass.
                let g = library.find_glass(catalog.as_deref(), name)?;
                glasses[..count]
                    .iter_mut()
                    .find(|v| {
                        v.name == g.name
                            && catalog.as_deref().is_none_or(|c| {
                                v.catalog
                                    .as_deref()
                                    .is_some_and(|v| v.eq_ignore_ascii_case(c))
                            })
                    })
                    .unwrap()
                    .surfaces
                    .push(i);
            }
            MaterialRef::Model { .. } => {
                let m = library.resolve(&surface.material, &Default::default())?;
                let [nd, vd, pgf] = coordinates(&m);
                glasses.push(GlassPoint {
                    catalog: None,
                    name: format!("Model at surface {i}"),
                    kind: "model".into(),
                    nd,
                    vd,
                    pgf,
                    status: None,
                    exclude_substitution: None,
                    relative_cost: None,
                    climate_resistance: None,
                    stain_resistance: None,
                    transmittance: Vec::new(),
                    surfaces: vec![i],
                });
            }
            _ => {}
        }
    }
    ctx.report(1.0, "glass map");
    Ok(GlassMapResult {
        glasses,
        run_info: timer.finish(
            snapshot,
            "glass_map",
            s,
            "cpu-f64",
            Precision::F64Reference,
            None,
            Vec::new(),
        ),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn agf_material_metadata_is_reported_without_overrides() {
        let catalog = mklight_core::materials::agf::parse_agf(
            "FIXTURE",
            "\
NM TEST 5 0 1.5 0 1 2 0
CD 1.5 0 0
LD 0.3 2.5
OD 2 3 4 -1 -1 -1
IT 0.55 0.81 10
",
        )
        .unwrap();
        let mut lib = MaterialLibrary::with_builtin();
        lib.add_catalog(std::sync::Arc::new(catalog));
        let snap = Snapshot::new(mklight_core::samples::singlet()).unwrap();
        let r = glass_map_job(&snap, &lib, &Default::default(), &JobContext::new()).unwrap();
        let g = r.glasses.iter().find(|g| g.name == "TEST").unwrap();
        assert_eq!(g.status, Some(GlassStatus::Obsolete));
        assert_eq!(g.exclude_substitution, Some(true));
        assert_eq!(g.relative_cost, Some(2.0));
        assert_eq!(g.climate_resistance, Some(3.0));
        assert_eq!(g.stain_resistance, Some(4.0));
        assert_eq!(g.transmittance.len(), 1);
        assert_eq!(
            (
                g.transmittance[0].um,
                g.transmittance[0].tau,
                g.transmittance[0].thickness_mm
            ),
            (0.55, 0.81, 10.0)
        );
    }

    #[test]
    fn catalogue_and_model_coordinates_and_markers() {
        let mut sys = mklight_core::samples::singlet();
        sys.surfaces[2].material = MaterialRef::Model {
            nd: 1.6,
            vd: 40.0,
            dpgf: 0.01,
        };
        let snap = Snapshot::new(sys).unwrap();
        let lib = MaterialLibrary::with_builtin();
        let r = glass_map_job(
            &snap,
            &lib,
            &GlassMapSettings {
                metadata: vec![GlassMetadata {
                    catalog: "BUILTIN".into(),
                    name: "N-BK7".into(),
                    status: GlassStatus::Preferred,
                    exclude_substitution: false,
                }],
            },
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(
            r.glasses.len(),
            lib.catalogs()
                .iter()
                .map(|c| c.glasses.len())
                .sum::<usize>()
                + 1
        );
        let bk = r.glasses.iter().find(|g| g.name == "N-BK7").unwrap();
        assert!((bk.nd - 1.5168).abs() < 1e-5);
        assert!((bk.vd - 64.17).abs() < 0.02);
        assert_eq!(bk.surfaces, vec![1]);
        assert_eq!(bk.status, Some(GlassStatus::Preferred));
        let model = r.glasses.last().unwrap();
        assert_eq!(model.surfaces, vec![2]);
        assert!((model.nd - 1.6).abs() < 1e-12);
        assert!((model.vd - 40.0).abs() < 1e-9);
        assert!((model.pgf - mklight_core::materials::normal_pgf(40.0) - 0.01).abs() < 1e-10);
    }
    /// A glass in two catalogues is marked where the design resolves it: the
    /// named catalogue, else the first loaded one.
    #[test]
    fn duplicate_glasses_are_marked_in_the_resolving_catalogue() {
        let mut lib = MaterialLibrary::with_builtin();
        let mut other = (*lib.catalogs()[0]).clone();
        other.name = "OTHER".into();
        lib.add_catalog(std::sync::Arc::new(other));
        let mut sys = mklight_core::samples::cooke_triplet();
        sys.surfaces[1].material = MaterialRef::Catalog {
            catalog: Some("other".into()),
            name: "n-sk16".into(),
        };
        let r = glass_map_job(
            &Snapshot::new(sys).unwrap(),
            &lib,
            &Default::default(),
            &JobContext::new(),
        )
        .unwrap();
        let marks: Vec<_> = r
            .glasses
            .iter()
            .filter(|g| !g.surfaces.is_empty())
            .map(|g| {
                (
                    g.catalog.as_deref().unwrap(),
                    g.name.as_str(),
                    g.surfaces.clone(),
                )
            })
            .collect();
        assert_eq!(
            marks,
            [
                ("BUILTIN", "F2", vec![3]),
                ("BUILTIN", "N-SK16", vec![5]),
                ("OTHER", "N-SK16", vec![1]),
            ]
        );
    }
}
