//! Run-local spectra (ANALYSIS.md, custom spectrum): an explicit list, the
//! CIE photopic or scotopic luminous efficiency, a blackbody or a sensor
//! response, sampled into weighted wavelengths (µm, relative energy).
//!
//! A response is piecewise linear and integrated exactly over the sampling
//! range. `equal_weight` splits the range into `N` bins of equal energy and
//! places each node at its bin's energy centroid (equal weights); `uniform`
//! spaces `N` nodes evenly over the range, endpoints included, each weighted
//! by the energy of its Voronoi cell. The primary wavelength is the node
//! nearest the energy-weighted mean wavelength.
use mklight_core::math::KahanSum;
use mklight_core::model::{Wavelength, WavelengthSet};
use mklight_core::{MkError, MkResult};
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
    /// The sampled wavelength set. The primary is the sample closest to the
    /// energy-weighted mean wavelength (the first on ties). Explicit weights
    /// and their order are kept; the design validation (`Snapshot::new`)
    /// checks the wavelengths and weights themselves.
    pub fn wavelengths(&self) -> MkResult<WavelengthSet> {
        let wavelengths = match self {
            Self::Explicit { wavelengths } => wavelengths.clone(),
            Self::Photopic { sampling } => sample(&tables::photopic(), sampling)?,
            Self::Scotopic { sampling } => sample(&tables::scotopic(), sampling)?,
            Self::Sensor { table, sampling } => sample(table, sampling)?,
            Self::Blackbody {
                temperature_k: t,
                sampling,
            } => {
                validate_sampling(sampling)?;
                if !(*t > 0.0 && t.is_finite()) {
                    return Err(invalid("temperature must be positive"));
                }
                // Planck spectral radiance B_λ ∝ λ⁻⁵/(exp(c₂/λT) − 1), with
                // c₂ = hc/k from the exact SI h, c and k (BIPM SI Brochure,
                // 9th ed., 2019), tabulated finely enough that its
                // piecewise-linear integral is exact to ~1e-6. Logarithms
                // avoid overflow for cold emitters and short wavelengths.
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
                sample(&table, sampling)?
            }
        };
        let total: f64 = wavelengths.iter().map(|w| w.weight).sum();
        let mean: f64 = wavelengths.iter().map(|w| w.um * w.weight / total).sum();
        let primary = (0..wavelengths.len())
            .min_by(|&a, &b| {
                (wavelengths[a].um - mean)
                    .abs()
                    .total_cmp(&(wavelengths[b].um - mean).abs())
            })
            .unwrap_or(0);
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
fn sample(table: &[[f64; 2]], s: &SpectralSampling) -> MkResult<Vec<Wavelength>> {
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
            // Nodes on [a, b] (the centre for one node), each weighted by
            // the energy of its Voronoi cell.
            let d = (b - a) / (n.max(2) - 1) as f64;
            for i in 0..n {
                let um = if n == 1 {
                    (a + b) * 0.5
                } else {
                    a + d * i as f64
                };
                let half = if n == 1 { (b - a) * 0.5 } else { d * 0.5 };
                let (e, _) = integral(table, (um - half).max(a), (um + half).min(b));
                out.push(Wavelength { um, weight: e });
            }
        }
        Sampling::EqualWeight => {
            let mut left = a;
            for i in 0..n {
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
    // Published CIE 018:2019 Tables 1/2 and dataset column sums;
    // DOI 10.25039/CIE.DS.dktna2s3 and 10.25039/CIE.DS.gr6w4b5g.
    #[test]
    fn cie_published_values_and_integrals() {
        for (table, start_nm, sum, energy_nm, samples) in [
            (
                tables::photopic(),
                360,
                106.8569171011719,
                106.8569149167669,
                &[
                    (360, 0.000003917),
                    (450, 0.038),
                    (510, 0.503),
                    (555, 1.0),
                    (610, 0.503),
                    (830, 0.00000045181),
                ][..],
            ),
            (
                tables::scotopic(),
                380,
                97.0713165740,
                97.0710220045,
                &[
                    (380, 0.000589),
                    (450, 0.455),
                    (506, 1.0),
                    (507, 1.0),
                    (508, 1.0),
                    (555, 0.402),
                    (650, 0.000677),
                    (780, 0.000000139),
                ][..],
            ),
        ] {
            let end_nm = samples.last().unwrap().0;
            assert_eq!(table.len(), end_nm - start_nm + 1);
            for (i, &[um, _]) in table.iter().enumerate() {
                assert_eq!(um, (start_nm + i) as f64 * 0.001);
            }
            for &(nm, value) in samples {
                assert_eq!(table[nm - start_nm][1], value, "{nm} nm");
            }
            assert!((table.iter().map(|r| r[1]).sum::<f64>() - sum).abs() < 1e-12);
            let range_um = [start_nm as f64 * 0.001, end_nm as f64 * 0.001];
            assert!(
                (integral(&table, range_um[0], range_um[1]).0 * 1000.0 - energy_nm).abs() < 1e-12
            );
            for spacing in [Sampling::Uniform, Sampling::EqualWeight] {
                let wavelengths = sample(
                    &table,
                    &SpectralSampling {
                        range_um,
                        spacing,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(
                    (wavelengths.iter().map(|w| w.weight).sum::<f64>() * 1000.0 - energy_nm).abs()
                        < 1e-12
                );
            }
        }
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
    /// Planck's law between nodes, and the Wien peak 2897.77 µm·K / T.
    #[test]
    fn blackbody_follows_planck_and_wien() {
        for t in [3000.0, 5000.0] {
            let s = SpectralSampling {
                samples: 451,
                range_um: [0.3, 1.2],
                spacing: Sampling::Uniform,
            };
            let ws = Spectrum::Blackbody {
                temperature_k: t,
                sampling: s,
            }
            .wavelengths()
            .unwrap()
            .wavelengths;
            let planck = |l: f64| l.powi(-5) / (14387.768775039337 / (l * t)).exp_m1();
            let (a, b) = (&ws[50], &ws[300]);
            let ratio = (a.weight / b.weight) / (planck(a.um) / planck(b.um));
            // Cell integrals differ from point values by O(d²B″/24B).
            assert!((ratio - 1.0).abs() < 1e-4, "{ratio}");
            let peak = ws[1..450]
                .iter()
                .max_by(|a, b| a.weight.total_cmp(&b.weight))
                .unwrap();
            assert!((peak.um - 2897.771955 / t).abs() < 0.002, "{}", peak.um);
        }
    }
}
