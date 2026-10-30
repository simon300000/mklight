//! Engine adapters for signed field maps, catalogue maps and image synthesis.
use super::geometric::{crate_json, output, parse};
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::EngineResult;
use mklight_analysis::{field_maps as fm, footprint_overlay as fp, glass_map as gm};
use mklight_core::JobContext;
use mklight_trace::SequentialTracer;
use mklight_wave::image_simulation as im;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

struct FieldMap(fm::FieldAnalysis);
impl Analysis for FieldMap {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn title(&self) -> &str {
        match self.0 {
            fm::FieldAnalysis::PupilAberration => "Pupil aberration",
            fm::FieldAnalysis::RelativeIllumination => "Relative illumination map",
            fm::FieldAnalysis::ChiefRayAngle => "Chief ray angle",
        }
    }
    fn category(&self) -> &str {
        "geometric"
    }
    fn description(&self) -> &str {
        "Signed Cartesian field map or x/y sweep, with image heights and explicitly defined diagnostic columns."
    }
    fn settings_schema(&self) -> Value {
        settings_from_default::<fm::FieldMapSettings>(json!({
            "grid":{"description":"[nx,ny], 1..101 samples per axis; one sample is on axis."},
            "extent":{"description":"Positive normalised field half-widths [x,y]."},
            "mode":{"enum":["grid","x","y"]},
            "wavelength":{"type":["integer","null"],"minimum":0,"description":"Null selects the run spectrum's primary wavelength."},
            "radial_samples":{"minimum":1,"maximum":512},"azimuth_samples":{"minimum":4,"maximum":4096},
            "sensor_limit":{"description":"CRA only: increasing [image radius mm, maximum incidence degrees], linearly interpolated, held at endpoints. Empty means no limit."}
        }))
    }
    fn result_schema(&self) -> Value {
        result(
            &[
                ("shape", tuple(int(""), 2, "[nx,ny]")),
                ("wavelength", int("Run spectrum index.")),
                (
                    "columns",
                    arr(
                        obj(&[
                            ("name", string("")),
                            ("unit", string("")),
                            ("definition", string("")),
                        ]),
                        "Ordered columns in values.",
                    ),
                ),
                ("field_coordinates", string("Buffer name.")),
                ("image_mm", string("Buffer name.")),
                ("values", string("Buffer name.")),
            ],
            vec![
                buffer(
                    "field_coordinates",
                    "f64",
                    &["ny", "nx", "2"],
                    "1",
                    "Normalised [Hx,Hy].",
                ),
                buffer(
                    "image_mm",
                    "f64",
                    &["ny", "nx", "2"],
                    "mm",
                    "Chief hit [x,y]; NaN for unusable fields.",
                ),
                buffer(
                    "values",
                    "f64",
                    &["ny", "nx", "columns"],
                    "column dependent",
                    "Diagnostics; NaN when undefined, including CRA margin without limit.",
                ),
            ],
        )
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let t = SequentialTracer::new(input.resolved()?)?;
        let r = fm::field_map_job(&t, self.0, &parse(settings)?, ctx)?;
        let (mut v, w) = crate_json(&r)?;
        for key in ["field_coordinates", "image_mm", "values"] {
            v[key] = json!(key);
        }
        let [nx, ny] = r.shape;
        let nc = r.columns.len();
        Ok(output(
            v,
            vec![
                Buffer::new(
                    "field_coordinates",
                    vec![ny, nx, 2],
                    BufferData::F64(r.field_coordinates.into_iter().flatten().collect()),
                ),
                Buffer::new(
                    "image_mm",
                    vec![ny, nx, 2],
                    BufferData::F64(r.image_mm.into_iter().flatten().collect()),
                ),
                Buffer::new("values", vec![ny, nx, nc], BufferData::F64(r.values)),
            ],
            w,
        ))
    }
}
struct GlassMap;
impl Analysis for GlassMap {
    fn name(&self) -> &str {
        "glass_map"
    }
    fn title(&self) -> &str {
        "Glass map"
    }
    fn category(&self) -> &str {
        "materials"
    }
    fn description(&self) -> &str {
        "Catalogue n_d / V_d and P_g,F / V_d, with design surface markers and optional retained AGF statuses."
    }
    fn settings_schema(&self) -> Value {
        settings_from_default::<gm::GlassMapSettings>(
            json!({"metadata":{"description":"Optional retained AGF metadata [{catalog,name,status,exclude_substitution}]. MaterialLibrary does not retain AGF availability: unknown status remains null."}}),
        )
    }
    fn result_schema(&self) -> Value {
        result(
            &[
                (
                    "glasses",
                    arr(
                        obj(&[
                            (
                                "catalog",
                                nullable(string("Catalogue name; null for model glass.")),
                            ),
                            ("name", string("")),
                            ("kind", enumeration(&["catalog", "model"], "")),
                            (
                                "nd",
                                nf_u(
                                    "1",
                                    "Catalogue d-line index; model computed at reference conditions.",
                                ),
                            ),
                            ("vd", nf_u("1", "Abbe number.")),
                            (
                                "pgf",
                                nf_u("1", "(n_g-n_F)/(n_F-n_C), computed from dispersion."),
                            ),
                            (
                                "status",
                                json!({"anyOf":[{"type":"string","enum":["standard","preferred","obsolete","special","melt"]},obj(&[("other",int("Unrecognised AGF status code."))]),{"type":"null"}],"description":"Retained AGF status, or null when unavailable."}),
                            ),
                            ("exclude_substitution", nullable(boolean(""))),
                            (
                                "surfaces",
                                arr(
                                    int(""),
                                    "Surface indices whose following medium is this glass.",
                                ),
                            ),
                        ]),
                        "All loaded glasses in catalogue order, then model glasses in surface order.",
                    ),
                ),
                ("coordinates", string("Buffer name.")),
            ],
            vec![buffer(
                "coordinates",
                "f64",
                &["glasses", "3"],
                "1",
                "[n_d,V_d,P_g,F], aligned with glasses metadata.",
            )],
        )
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let r = gm::glass_map_job(&input.snapshot, &input.library, &parse(settings)?, ctx)?;
        let (mut v, w) = crate_json(&r)?;
        v["coordinates"] = json!("coordinates");
        let data = r.glasses.iter().flat_map(|g| [g.nd, g.vd, g.pgf]).collect();
        Ok(output(
            v,
            vec![Buffer::new(
                "coordinates",
                vec![r.glasses.len(), 3],
                BufferData::F64(data),
            )],
            w,
        ))
    }
}

#[derive(Default)]
struct ImageSimulation {
    cache: Mutex<Option<(String, Arc<im::PsfGrid>)>>,
}
impl Analysis for ImageSimulation {
    fn name(&self) -> &str {
        "image_simulation"
    }
    fn title(&self) -> &str {
        "Image simulation"
    }
    fn category(&self) -> &str {
        "wave"
    }
    fn description(&self) -> &str {
        "Linear-light image through field-dependent polychromatic PSFs, chief-ray distortion, relative illumination and optional pixel aperture / shot / read noise. RGB channels share the selected energy spectrum."
    }
    fn settings_schema(&self) -> Value {
        settings_from_default::<im::ImageSimulationSettings>(json!({
            "source":{"description":"{kind:pattern,pattern:grid|siemens_star|slanted_edge|text_chart}, {kind:buffer,width,height,channels:1|3,data:[linear samples]}, or {kind:path,path:...} (PNG/JPEG/PNM, sRGB decoded)."},
            "size":{"description":"Output [width,height], 2..4096 each."},
            "optics":settings_from_default::<im::ImageGridSettings>(json!({
                "field_grid":{"minimum":2,"maximum":33},"field_extent":{"exclusiveMinimum":0.0},
                "pupil":{"description":"Traced pupil settings; radiometric amplitude by default."},
                "psf_size":{"minimum":3,"maximum":513,"description":"Odd size. Increase until truncation warnings disappear."},
                "psf_spacing_mm":{"type":["number","null"],"exclusiveMinimum":0.0,"unit":"mm","description":"Null: half the smallest lambda F/# at the first grid node."},
                "illumination_radial":{"minimum":1,"maximum":512},"illumination_azimuth":{"minimum":4,"maximum":4096}
            })),
            "tile_size":{"minimum":4,"maximum":512},
            "detector_aperture_mm":{"type":["array","null"],"unit":"mm","description":"Optional positive [x,y] active aperture, at most four output pixels; 4x4 quadrature."},
            "read_noise_sigma":{"minimum":0.0},"full_well_electrons":{"type":["number","null"],"exclusiveMinimum":0.0,"maximum":1e12},"seed":{"minimum":0}
        }))
    }
    fn result_schema(&self) -> Value {
        result(
            &[
                ("size", tuple(int(""), 2, "[width,height].")),
                ("channels", int("1 or 3, interleaved linear light.")),
                (
                    "pixel_size_mm",
                    tuple(num("mm", ""), 2, "[x,y] output pitch."),
                ),
                (
                    "origin_mm",
                    tuple(
                        num("mm", ""),
                        2,
                        "Lower-left pixel edge in the image frame.",
                    ),
                ),
                ("field_grid", int("Square PSF grid side.")),
                (
                    "field_mapping",
                    arr(
                        obj(&[
                            ("field", tuple(num("1", ""), 2, "Normalised field.")),
                            (
                                "ideal_mm",
                                tuple(num("mm", ""), 2, "First-order chief image."),
                            ),
                            (
                                "real_mm",
                                tuple(num("mm", ""), 2, "Real primary chief image."),
                            ),
                            (
                                "relative_illumination",
                                nf_u(
                                    "1",
                                    "Spectrally weighted projected-solid-angle irradiance ratio.",
                                ),
                            ),
                        ]),
                        "Row-major field nodes; mapping is bilinear between nodes.",
                    ),
                ),
                ("psf_size", int("")),
                ("psf_spacing_mm", num("mm", "")),
                ("image", string("Image buffer name.")),
            ],
            vec![buffer(
                "image",
                "f64",
                &["height", "width", "channels"],
                "linear light",
                "Unclamped output; read noise can be negative. Zero extension at source boundaries.",
            )],
        )
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let s: im::ImageSimulationSettings = parse(settings)?;
        let key = format!(
            "{}:{}:{}",
            input.snapshot.hash(),
            input.library_hash(),
            serde_json::to_string(&s.optics)?
        );
        let cached = self
            .cache
            .lock()
            .unwrap()
            .as_ref()
            .filter(|(k, _)| k == &key)
            .map(|(_, v)| v.clone());
        let t = SequentialTracer::new(input.resolved()?)?;
        let grid = match cached {
            Some(g) => g,
            None => {
                let g = Arc::new(im::build_psf_grid(&t, &s.optics, &ctx.subrange(0.0, 0.65))?);
                *self.cache.lock().unwrap() = Some((key, g.clone()));
                g
            }
        };
        let mut r = im::image_simulation_job(&t, &s, Some(grid), ctx)?;
        let image = std::mem::take(&mut r.image);
        let (mut v, w) = crate_json(&r)?;
        v["image"] = json!("image");
        Ok(output(
            v,
            vec![Buffer::new(
                "image",
                vec![r.size[1], r.size[0], r.channels],
                BufferData::F64(image),
            )],
            w,
        ))
    }
}

/// Preserve the existing local footprint result and add every surface's
/// global cloud and outline, rather than changing the old buffer meaning.
struct Footprint3d(Arc<dyn Analysis>);
impl Analysis for Footprint3d {
    fn name(&self) -> &str {
        "footprint"
    }
    fn title(&self) -> &str {
        self.0.title()
    }
    fn category(&self) -> &str {
        self.0.category()
    }
    fn description(&self) -> &str {
        "Local selected-surface footprint plus global hits and convex outlines on every surface, per field and wavelength."
    }
    fn settings_schema(&self) -> Value {
        self.0.settings_schema()
    }
    fn seed(&self, s: &Value) -> Option<u64> {
        self.0.seed(s)
    }
    fn result_schema(&self) -> Value {
        let mut schema = self.0.result_schema();
        schema["properties"]["overlays"] = arr(
            obj(&[
                ("surface", int("")),
                ("field", int("")),
                ("wavelength", int("")),
                (
                    "points",
                    tuple(int(""), 2, "[offset,count] in global_points."),
                ),
                (
                    "outline",
                    tuple(
                        int(""),
                        2,
                        "[offset,count] in global_outlines; counterclockwise and unclosed in local x/y.",
                    ),
                ),
            ]),
            "All surfaces except the object; passing hits before downstream clipping.",
        );
        schema["required"]
            .as_array_mut()
            .unwrap()
            .push(json!("overlays"));
        for (name, desc) in [
            (
                "global_points",
                "Global hit points, including surface sag and placement.",
            ),
            (
                "global_outlines",
                "Convex hull vertices per surface/field/wavelength; not a vignetting boundary for nonconvex pupils.",
            ),
        ] {
            schema["x-buffers"].as_array_mut().unwrap().push(buffer(
                name,
                "f64",
                &["N", "3"],
                "mm",
                desc,
            ));
        }
        schema
    }
    fn run(
        &self,
        input: &AnalysisInput<'_>,
        s: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        let mut out = self.0.run(input, s, &ctx.subrange(0.0, 0.4))?;
        let t = SequentialTracer::new(input.resolved()?)?;
        let r = fp::footprint_overlay_job(&t, &parse(s)?, &ctx.subrange(0.4, 1.0))?;
        out.json["overlays"] = serde_json::to_value(r.sets)?;
        out.buffers.push(Buffer::new(
            "global_points",
            vec![r.points.len(), 3],
            BufferData::F64(r.points.into_iter().flatten().collect()),
        ));
        out.buffers.push(Buffer::new(
            "global_outlines",
            vec![r.outlines.len(), 3],
            BufferData::F64(r.outlines.into_iter().flatten().collect()),
        ));
        Ok(out)
    }
}
pub fn register_all(r: &mut AnalysisRegistry) {
    for kind in [
        fm::FieldAnalysis::PupilAberration,
        fm::FieldAnalysis::RelativeIllumination,
        fm::FieldAnalysis::ChiefRayAngle,
    ] {
        r.register(Arc::new(FieldMap(kind)));
    }
    r.register(Arc::new(GlassMap));
    r.register(Arc::new(ImageSimulation::default()));
    let footprint = r.get("footprint").unwrap();
    r.register(Arc::new(Footprint3d(footprint)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::Project;
    use crate::{AnalysisRequest, Engine, EngineConfig};
    use mklight_core::samples;
    fn check(engine: &Engine, result: &super::super::AnalysisResult) {
        let schema = engine
            .analysis(&result.run_info.analysis)
            .unwrap()
            .result_schema();
        validate(&schema, &result.json, true).unwrap();
        let desc: Vec<_> = result
            .buffers
            .iter()
            .map(|b| {
                (
                    b.name.clone(),
                    serde_json::to_value(b.dtype())
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_owned(),
                    b.shape.len(),
                )
            })
            .collect();
        check_buffers(&schema, &desc).unwrap();
    }
    #[test]
    fn asymmetric_maps_and_sweeps_keep_both_image_axes() {
        let e = Engine::new(EngineConfig::default());
        let session = e
            .session_from_project(Project::new(samples::anamorphic_adapter()))
            .unwrap();
        for name in [
            "pupil_aberration",
            "relative_illumination_map",
            "chief_ray_angle",
        ] {
            for mode in ["grid", "x", "y"] {
                let r=session.run_analysis(&AnalysisRequest::new(name,json!({"grid":[3,3],"mode":mode,"spectrum":{"kind":"explicit","wavelengths":[{"um":0.55,"weight":1.0}]}})),&JobContext::new()).unwrap();
                check(&e, &r);
                let b = r.buffers.iter().find(|b| b.name == "image_mm").unwrap();
                assert_eq!(
                    b.shape,
                    match mode {
                        "x" => vec![1, 3, 2],
                        "y" => vec![3, 1, 2],
                        _ => vec![3, 3, 2],
                    }
                );
            }
        }
    }
    #[test]
    fn cached_image_grid_preserves_noisy_rgb_results_and_library_identity() {
        let e = Engine::new(EngineConfig::default());
        let mut session = e
            .session_from_project(Project::new(samples::cooke_triplet()))
            .unwrap();
        let settings = json!({"size":[16,16],"monochrome":false,"source":{"kind":"buffer","width":2,"height":2,"channels":3,"data":[1,0,0,0,1,0,0,0,1,1,1,1]},"optics":{"field_grid":2,"field_extent":0.25,"pupil":{"grid":16,"rays":17},"psf_size":17},"read_noise_sigma":0.01,"full_well_electrons":2000.0,"seed":12});
        let run = |s: &crate::Session| {
            s.run_analysis(
                &AnalysisRequest::new("image_simulation", settings.clone()),
                &JobContext::new(),
            )
            .unwrap()
        };
        let a = run(&session);
        let b = run(&session);
        assert_eq!(a.buffers, b.buffers);
        check(&e, &a);
        assert_eq!(a.run_info.seed, Some(12));
        assert_eq!(a.buffers[0].shape, vec![16, 16, 3]);
        let mut cat = mklight_core::materials::builtin::catalog();
        cat.name = "OTHER".into();
        session.add_catalog(cat, None);
        let c = run(&session);
        assert_ne!(a.library_hash, c.library_hash);
        assert_eq!(a.buffers, c.buffers);
    }
}
