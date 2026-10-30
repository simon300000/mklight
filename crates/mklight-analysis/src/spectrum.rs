//! Run-local spectral quadrature. Wavelengths are µm, weights relative energy.
//! Piecewise-linear responses are integrated exactly; equal-weight nodes are
//! bin energy centroids, uniform nodes use integrated Voronoi-bin weights.
use mklight_core::math::KahanSum;
use mklight_core::model::{Wavelength, WavelengthSet};
use mklight_core::{JobContext, MkError, MkResult};
use serde::{Deserialize, Serialize};

#[path = "spectral_tables.rs"]
mod tables;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampling {
    #[default]
    EqualWeight,
    Uniform,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Spectrum {
    Explicit {
        wavelengths: Vec<Wavelength>,
    },
    Photopic {
        #[serde(default)]
        sampling: SpectralSampling,
    },
    Scotopic {
        #[serde(default)]
        sampling: SpectralSampling,
    },
    Blackbody {
        temperature_k: f64,
        #[serde(default)]
        sampling: SpectralSampling,
    },
    Sensor {
        table: Vec<[f64; 2]>,
        #[serde(default)]
        sampling: SpectralSampling,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpectralSampling {
    pub samples: usize,
    pub range_um: [f64; 2],
    pub spacing: Sampling,
}
impl Default for SpectralSampling {
    fn default() -> Self {
        Self {
            samples: 9,
            range_um: [0.38, 0.78],
            spacing: Sampling::EqualWeight,
        }
    }
}

fn invalid(s: &str) -> MkError {
    MkError::InvalidParam(format!("spectrum: {s}"))
}

impl Spectrum {
    /// Primary = sample closest to the weighted mean wavelength; ties choose
    /// the first input sample. Explicit weights and their ordering are kept.
    pub fn wavelengths(&self) -> MkResult<WavelengthSet> {
        self.wavelengths_job(&JobContext::new())
    }

    pub fn wavelengths_job(&self, ctx: &JobContext) -> MkResult<WavelengthSet> {
        ctx.check()?;
        let wavelengths = match self {
            Self::Explicit { wavelengths } => wavelengths.clone(),
            Self::Photopic { sampling } => sample(&tables::photopic(), sampling, ctx)?,
            Self::Scotopic { sampling } => sample(&tables::scotopic(), sampling, ctx)?,
            Self::Sensor { table, sampling } => sample(table, sampling, ctx)?,
            Self::Blackbody {
                temperature_k: t,
                sampling,
            } => {
                validate_sampling(sampling)?;
                if !(*t > 0.0 && t.is_finite()) {
                    return Err(invalid("temperature must be positive"));
                }
                // Planck spectral radiance B_lambda; c2 = hc/k from exact SI
                // h, c and k (BIPM SI Brochure, 9th ed., 2019). Log evaluation
                // avoids overflow for cold emitters and short wavelengths.
                let [a, b] = sampling.range_um;
                let logs: Vec<_> = (0..=4096)
                    .map(|i| {
                        let l = a + (b - a) * i as f64 / 4096.0;
                        let x = 14387.768775039337 / (l * t);
                        let log_b = -5.0 * l.ln() - if x > 50.0 { x } else { x.exp_m1().ln() };
                        (l, log_b)
                    })
                    .collect();
                let peak = logs.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
                let table: Vec<_> = logs.iter().map(|&(l, v)| [l, (v - peak).exp()]).collect();
                sample(&table, sampling, ctx)?
            }
        };
        if wavelengths.is_empty()
            || wavelengths.iter().any(|w| {
                !(w.um > 0.0 && w.um.is_finite() && w.weight >= 0.0 && w.weight.is_finite())
            })
        {
            return Err(invalid(
                "need positive wavelengths and nonnegative finite weights",
            ));
        }
        let total = wavelengths
            .iter()
            .map(|w| w.weight)
            .collect::<KahanSum>()
            .value();
        if !(total > 0.0 && total.is_finite()) {
            return Err(invalid("zero or overflowing energy"));
        }
        let mean = wavelengths
            .iter()
            .map(|w| w.um * (w.weight / total))
            .collect::<KahanSum>()
            .value();
        let primary = (0..wavelengths.len())
            .min_by(|&a, &b| {
                (wavelengths[a].um - mean)
                    .abs()
                    .total_cmp(&(wavelengths[b].um - mean).abs())
            })
            .unwrap();
        Ok(WavelengthSet {
            wavelengths,
            primary,
        })
    }
}

fn validate_sampling(s: &SpectralSampling) -> MkResult<()> {
    let [a, b] = s.range_um;
    if s.samples == 0 || s.samples > 4096 || !(a > 0.0 && b > a && b.is_finite()) {
        return Err(invalid(
            "need 1..4096 samples and an increasing positive range",
        ));
    }
    Ok(())
}
fn response(table: &[[f64; 2]], x: f64) -> f64 {
    if x < table[0][0] || x > table[table.len() - 1][0] {
        return 0.0;
    }
    let i = table
        .partition_point(|p| p[0] < x)
        .clamp(1, table.len() - 1);
    let [a, u] = table[i - 1];
    let [b, v] = table[i];
    u + (v - u) * ((x - a) / (b - a))
}
/// Exact zeroth and first moments of a piecewise-linear response.
fn integral(table: &[[f64; 2]], a: f64, b: f64) -> (f64, f64) {
    let mut points = vec![a];
    points.extend(table.iter().map(|p| p[0]).filter(|&x| x > a && x < b));
    points.push(b);
    let (mut energy, mut moment) = (KahanSum::default(), KahanSum::default());
    for w in points.windows(2) {
        let (x, y) = (w[0], w[1]);
        let (u, v) = (response(table, x), response(table, y));
        let e = (y - x) * (u + v) * 0.5;
        energy.add(e);
        moment.add(x * e + (y - x).powi(2) * (u + 2.0 * v) / 6.0);
    }
    (energy.value(), moment.value())
}
fn sample(table: &[[f64; 2]], s: &SpectralSampling, ctx: &JobContext) -> MkResult<Vec<Wavelength>> {
    validate_sampling(s)?;
    if table.len() < 2
        || table
            .iter()
            .any(|p| !(p[0] > 0.0 && p[0].is_finite() && p[1] >= 0.0 && p[1].is_finite()))
        || table.windows(2).any(|w| w[0][0] >= w[1][0])
    {
        return Err(invalid(
            "response table must have increasing wavelengths and nonnegative responses",
        ));
    }
    let [a, b] = s.range_um;
    if a < table[0][0] || b > table[table.len() - 1][0] {
        return Err(invalid("sampling range must lie inside response table"));
    }
    let (total, _) = integral(table, a, b);
    if !(total > 0.0) {
        return Err(invalid("response has zero energy in range"));
    }
    let n = s.samples;
    let mut out = Vec::with_capacity(n);
    match s.spacing {
        Sampling::Uniform => {
            for i in 0..n {
                ctx.check()?;
                let um = if n == 1 {
                    (a + b) * 0.5
                } else {
                    a + (b - a) * i as f64 / (n - 1) as f64
                };
                let d = if n == 1 {
                    b - a
                } else {
                    (b - a) / (n - 1) as f64
                };
                let (e, _) = integral(table, (um - d * 0.5).max(a), (um + d * 0.5).min(b));
                out.push(Wavelength {
                    um,
                    weight: if n == 1 { total } else { e },
                });
            }
        }
        Sampling::EqualWeight => {
            let mut left = a;
            for i in 0..n {
                ctx.check()?;
                let target = total * (i + 1) as f64 / n as f64;
                let mut remaining = target;
                let mut right = b;
                for pair in table.windows(2) {
                    let x = pair[0][0].max(a);
                    let y = pair[1][0].min(b);
                    if y <= x {
                        continue;
                    }
                    let u = response(table, x);
                    let v = response(table, y);
                    let energy = (y - x) * (u + v) * 0.5;
                    if remaining > energy {
                        remaining -= energy;
                        continue;
                    }
                    // Stable positive root of u*d + slope*d²/2 = energy.
                    let slope = (v - u) / (y - x);
                    let denom = u + (u * u + 2.0 * slope * remaining).max(0.0).sqrt();
                    right = if denom > 0.0 {
                        x + 2.0 * remaining / denom
                    } else {
                        x
                    };
                    break;
                }
                if i + 1 == n {
                    right = b;
                }
                let (e, m) = integral(table, left, right);
                out.push(Wavelength {
                    um: m / e,
                    weight: e,
                });
                left = right;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cie_integral_and_peak() {
        let table = tables::photopic();
        assert_eq!(response(&table, 0.555), 1.0);
        assert!((integral(&table, 0.36, 0.83).0 - 0.1068569).abs() < 2e-7);
        for spacing in [Sampling::Uniform, Sampling::EqualWeight] {
            let s = SpectralSampling {
                range_um: [0.36, 0.83],
                spacing,
                ..Default::default()
            };
            let ws = Spectrum::Photopic { sampling: s }.wavelengths().unwrap();
            assert!(
                (ws.wavelengths.iter().map(|w| w.weight).sum::<f64>() - 0.1068569).abs() < 2e-7
            );
        }
        assert_eq!(response(&tables::scotopic(), 0.507), 1.0);
    }
    #[test]
    fn triangular_response_centroid_and_energy() {
        let ws = Spectrum::Sensor {
            table: vec![[0.4, 0.0], [0.5, 1.0], [0.6, 0.0]],
            sampling: SpectralSampling {
                samples: 1,
                range_um: [0.4, 0.6],
                ..Default::default()
            },
        }
        .wavelengths()
        .unwrap();
        assert!((ws.wavelengths[0].um - 0.5).abs() < 1e-14);
        assert!((ws.wavelengths[0].weight - 0.1).abs() < 1e-14);
    }
}
