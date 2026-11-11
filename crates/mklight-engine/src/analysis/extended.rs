//! Engine adapters for the 2-D field maps, the glass map and image
//! simulation.
use super::geometric::{crate_json, output, parse};
use super::schema::*;
use super::{Analysis, AnalysisInput, AnalysisOutput, AnalysisRegistry};
use crate::buffer::{Buffer, BufferData};
use crate::error::EngineResult;
use mklight_analysis::{field_maps as fm, glass_map as gm};
use mklight_core::JobContext;
use mklight_wave::image_simulation as im;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

struct FieldMap(fm::FieldAnalysis);
impl Analysis for FieldMap {
    fn memoizable(&self) -> bool {
        true
    }
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
        let mut schema = settings_from_default::<fm::FieldMapSettings>(json!({
            "grid":{"description":"[nx,ny], 1..101 samples per axis; one sample is on axis."},
            "extent":{"description":"Positive normalised field half-widths [x,y]."},
            "mode":{"enum":["grid","x","y"],"description":"grid: nx×ny map; x / y: sweep along one field axis."},
            "wavelength":{"type":["integer","null"],"minimum":0,"description":"Null selects the run spectrum's primary wavelength."},
            "radial_samples":{"minimum":1,"maximum":512},"azimuth_samples":{"minimum":4,"maximum":4096},
            "sensor_limit":{"description":"Increasing [image radius mm, maximum incidence degrees], linearly interpolated, held at endpoints. Empty means no limit."}
        }));
        // Each map publishes only the settings it reads.
        let props = schema["properties"].as_object_mut().unwrap();
        if !matches!(self.0, fm::FieldAnalysis::RelativeIllumination) {
            props.remove("radial_samples");
            props.remove("azimuth_samples");
        }
        if !matches!(self.0, fm::FieldAnalysis::ChiefRayAngle) {
            props.remove("sensor_limit");
        }
        schema
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
        let t = input.tracer(ctx)?;
        let s = parse(settings)?;
        let r = super::gpu::native_job!(
            input,
            |b| mklight_analysis::gpu::field_map_job(
                b,
                &t,
                self.0,
                &s,
                &input.route.geometric(),
                ctx
            ),
            fm::field_map_job(&t, self.0, &s, ctx)
        )?;
        // The maps move into buffers; only the metadata is serialised.
        let mut r = r;
        let field_coordinates = std::mem::take(&mut r.field_coordinates);
        let image_mm = std::mem::take(&mut r.image_mm);
        let values = std::mem::take(&mut r.values);
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
                    BufferData::F64(field_coordinates.into_flattened()),
                ),
                Buffer::new(
                    "image_mm",
                    vec![ny, nx, 2],
                    BufferData::F64(image_mm.into_flattened()),
                ),
                Buffer::new("values", vec![ny, nx, nc], BufferData::F64(values)),
            ],
            w,
        ))
    }
}
struct GlassMap;
impl Analysis for GlassMap {
    fn memoizable(&self) -> bool {
        true
    }
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
        "Catalogue n_d / V_d and P_g,F / V_d, with design surface markers and retained AGF status and quality data."
    }
    fn settings_schema(&self) -> Value {
        settings_from_default::<gm::GlassMapSettings>(
            json!({"metadata":{"description":"Optional availability overrides [{catalog,name,status,exclude_substitution}]; defaults come from each catalogue material."}}),
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
                            ("kind", enumeration(&["catalog", "offset", "model"], "")),
                            ("offset", material_offset()),
                            (
                                "nd",
                                nf_u(
                                    "1",
                                    "Catalogue d-line index; design offset and model glasses computed at catalogue reference conditions.",
                                ),
                            ),
                            ("vd", nf_u("1", "Abbe number.")),
                            (
                                "pgf",
                                nf_u("1", "(n_g-n_F)/(n_F-n_C), computed from dispersion."),
                            ),
                            (
                                "status",
                                json!({"anyOf":[{"type":"string","enum":["standard","preferred","obsolete","special","melt"]},obj(&[("other",int("Unrecognised AGF status code."))]),{"type":"null"}],"description":"Catalogue availability (standard when unspecified); null for model glasses."}),
                            ),
                            ("exclude_substitution", nullable(boolean(""))),
                            ("relative_cost", nullable(num("", "AGF relative cost."))),
                            (
                                "climate_resistance",
                                nullable(num("", "AGF climate class (CR).")),
                            ),
                            (
                                "stain_resistance",
                                nullable(num("", "AGF stain class (FR).")),
                            ),
                            (
                                "transmittance",
                                arr(
                                    obj(&[
                                        ("um", num("", "Wavelength in µm.")),
                                        (
                                            "tau",
                                            num(
                                                "",
                                                "Internal transmittance, excluding interfaces.",
                                            ),
                                        ),
                                        ("thickness_mm", num("", "Sample thickness in mm.")),
                                    ]),
                                    "Catalogue internal transmission samples.",
                                ),
                            ),
                            (
                                "surfaces",
                                arr(
                                    int(""),
                                    "Surface indices whose following medium is this glass.",
                                ),
                            ),
                        ]),
                        "All loaded glasses in catalogue order, then offset and model design glasses in surface order. Offset surfaces mark the shifted point only.",
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
    gpu_cache: Mutex<Option<GpuGrid>>,
}
struct GpuGrid {
    key: String,
    backend: Arc<dyn mklight_gpu::ComputeBackend>,
    grid: Arc<mklight_wave::gpu::image_simulation::ResidentPsfGrid>,
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
            "source":{"description":"{kind:pattern,pattern:grid|siemens_star|slanted_edge|text_chart} (evaluated analytically at each output pixel's source point, averaged over the pixel), {kind:buffer,width,height,channels:1|3,data:[linear samples]}, or {kind:path,path:...} (8- or 16-bit grey or RGB PNG, sRGB decoded, alpha ignored)."},
            "size":{"description":"Output [width,height], 2..4096 each."},
            "optics":settings_from_default::<im::ImageGridSettings>(json!({
                "field_grid":{"minimum":2,"maximum":33},"field_extent":{"exclusiveMinimum":0.0},
                "field_center":{"description":"Normalised field [hx,hy] at the centre of the simulated patch."},
                "pupil":{"description":"Traced pupil settings; automatic phase sampling and radiometric amplitude by default."},
                "psf_size":{"minimum":0,"maximum":2047,"description":"Odd size, or 0 to choose from traced blur plus diffraction (cap 2047)."},
                "psf_spacing_mm":{"type":["number","null"],"exclusiveMinimum":0.0,"unit":"mm","description":"Null: smallest lambda F/# across all grid nodes divided by 2.5."},
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
                            ("pupil_sampling", super::wave::pupil_sampling_schema()),
                            ("field", tuple(num("1", ""), 2, "Normalised field.")),
                            (
                                "ideal_mm",
                                tuple(num("mm", ""), 2, "First-order chief image."),
                            ),
                            (
                                "real_mm",
                                tuple(
                                    num("mm", ""),
                                    2,
                                    "Real image point of the PSF reference (primary chief by default): the PSF centre.",
                                ),
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
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .filter(|(k, _)| k == &key)
            .map(|(_, v)| v.clone());
        let t = input.tracer(ctx)?;
        let mut r = if let Some(d) = super::gpu::device(input) {
            let cached = self
                .gpu_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .filter(|g| g.key == key && Arc::ptr_eq(&g.backend, &d.backend))
                .map(|g| g.grid.clone());
            let grid = match cached {
                Some(grid) => grid,
                None => {
                    let built = mklight_wave::gpu::image_simulation::build_psf_grid(
                        d.backend.as_ref(),
                        &t,
                        &s.optics,
                        &ctx.subrange(0., 0.65),
                    )?;
                    super::gpu::record_native(input, &built.run_info);
                    let grid = Arc::new(built.value);
                    *self.gpu_cache.lock().unwrap_or_else(|e| e.into_inner()) = Some(GpuGrid {
                        key: key.clone(),
                        backend: d.backend.clone(),
                        grid: grid.clone(),
                    });
                    grid
                }
            };
            let r = mklight_wave::gpu::image_simulation::image_simulation_job(
                d.backend.as_ref(),
                &t,
                &s,
                Some(grid),
                &ctx.subrange(0.65, 1.),
            )?;
            super::gpu::record_native(input, &r.run_info);
            r
        } else {
            let grid = match cached {
                Some(g) => g,
                None => {
                    let g = Arc::new(im::build_psf_grid(&t, &s.optics, &ctx.subrange(0.0, 0.65))?);
                    *self.cache.lock().unwrap_or_else(|e| e.into_inner()) = Some((key, g.clone()));
                    g
                }
            };
            im::image_simulation_job(&t, &s, Some(grid), &ctx.subrange(0.65, 1.))?
        };
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
}
