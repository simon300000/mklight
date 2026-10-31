use super::{algebra::E, compile::Compiler};
use crate::{PackedSystemSet, consts::*, layout::PackedSurface};
use mklight_core::{
    MkResult, SurfaceShape,
    param::{CoefficientSet, ParamPath, ShapeParameter},
};

impl Compiler {
    pub(super) fn shape_curvatures(&mut self, i: usize, shape: &SurfaceShape) -> MkResult<[E; 2]> {
        let id = self.model.surfaces[i].id;
        let h = mklight_core::surface_math::vertex_curvatures(shape);
        let nominal = shape.curvature();
        let cy = self
            .param(&ParamPath::Curvature { surface: id })
            .unwrap_or_else(|_| self.c(nominal));
        let mut out = [
            self.c(h[0] - nominal) + cy.clone(),
            self.c(h[2] - nominal) + cy.clone(),
        ];
        if matches!(
            shape,
            SurfaceShape::Toroid { .. } | SurfaceShape::Biconic { .. }
        ) {
            out[0] = self.param(&ParamPath::ShapeParam {
                surface: id,
                parameter: ShapeParameter::CurvatureX,
            })?;
        }
        if let Some(y) = self.params.get(&ParamPath::ShapeParam {
            surface: id,
            parameter: ShapeParameter::CurvatureY,
        }) {
            out[1] = out[1].clone() - cy.clone() + y.clone();
        }
        let changes: Vec<_> = self
            .params
            .keys()
            .filter(|p| matches!(p,ParamPath::ShapeCoefficient{surface,..} if *surface==id))
            .cloned()
            .collect();
        for path in changes {
            let mut zero = self.model.clone();
            path.set(&mut zero, 0.0)?;
            let h0 = mklight_core::surface_math::vertex_curvatures(&zero.surfaces[i].shape);
            path.set(&mut zero, 1.0)?;
            let h1 = mklight_core::surface_math::vertex_curvatures(&zero.surfaces[i].shape);
            let delta = self.param(&path)? - path.get(&self.model)?;
            for axis in 0..2 {
                out[axis] = out[axis].clone() + delta.clone() * (h1[axis * 2] - h0[axis * 2]);
            }
        }
        if let Some(r) = self
            .params
            .get(&ParamPath::ShapeParam {
                surface: id,
                parameter: ShapeParameter::NormRadius,
            })
            .cloned()
        {
            if matches!(
                shape,
                SurfaceShape::QBfs { .. }
                    | SurfaceShape::XYPolynomial { .. }
                    | SurfaceShape::ZernikeSag { .. }
            ) {
                let nominal = ParamPath::ShapeParam {
                    surface: id,
                    parameter: ShapeParameter::NormRadius,
                }
                .get(&self.model)?;
                self.positive(r.clone());
                let scale = self.c(nominal * nominal) / (r.clone() * r);
                for v in &mut out {
                    *v = cy.clone() + (v.clone() - cy.clone()) * scale.clone();
                }
            }
        }
        Ok(out)
    }
    pub(super) fn pack_shape(
        &mut self,
        row: usize,
        i: usize,
        set: &mut PackedSystemSet,
    ) -> MkResult<()> {
        let sf = self.model.surfaces[i].clone();
        let id = sf.id;
        let p = set.surfaces[row];
        let curvature = self
            .param(&ParamPath::Curvature { surface: id })
            .unwrap_or_else(|_| self.c(sf.shape.curvature()));
        let curvature = self
            .params
            .get(&ParamPath::ShapeParam {
                surface: id,
                parameter: ShapeParameter::CurvatureY,
            })
            .cloned()
            .unwrap_or(curvature);
        let conic = self
            .param(&ParamPath::Conic { surface: id })
            .unwrap_or_else(|_| self.c(sf.shape.conic()));
        let conic = self
            .params
            .get(&ParamPath::ShapeParam {
                surface: id,
                parameter: ShapeParameter::ConicY,
            })
            .cloned()
            .unwrap_or(conic);
        self.surf(
            row,
            std::mem::offset_of!(PackedSurface, curvature),
            std::mem::offset_of!(PackedSurface, curvature_lo),
            curvature,
        );
        self.surf(
            row,
            std::mem::offset_of!(PackedSurface, conic),
            std::mem::offset_of!(PackedSurface, conic_lo),
            conic,
        );
        if matches!(sf.shape, SurfaceShape::Paraxial { .. }) {
            let f = self.param(&ParamPath::ParaxialFocalLength { surface: id })?;
            self.positive(f.clone().abs());
            self.surf(
                row,
                std::mem::offset_of!(PackedSurface, paraxial_power),
                std::mem::offset_of!(PackedSurface, paraxial_power_lo),
                self.c(1.0) / f,
            );
        }
        let radius = 1.0 / (p.asphere_inv_norm2 as f64).sqrt();
        for j in 0..p.num_asphere as usize {
            let a = self.param(&ParamPath::Asphere {
                surface: id,
                index: j,
            })? * radius.powi(2 * j as i32 + 4);
            self.surf(
                row,
                std::mem::offset_of!(PackedSurface, asphere) + j * 4,
                std::mem::offset_of!(PackedSurface, asphere_lo) + j * 4,
                a,
            );
        }
        if p.coefficient_count == 0 {
            return Ok(());
        }
        let off = p.coefficient_offset as usize;
        let data = &set.coefficients[off..off + p.coefficient_count as usize];
        if p.flags & MKL_SURF_HAS_SAG_OVERLAY != 0 {
            return Err(mklight_core::MkError::Unsupported(
                "P4: reference preparation for sag overlays".into(),
            ));
        }
        let na = data[MKL_COEFF_NA as usize].hi as usize;
        let nb = data[MKL_COEFF_NB as usize].hi as usize;
        let r = data[MKL_COEFF_RADIUS as usize].to_f64();
        let norm = matches!(
            p.shape,
            MKL_SHAPE_Q_CON
                | MKL_SHAPE_Q_BFS
                | MKL_SHAPE_XY
                | MKL_SHAPE_ZERNIKE
                | MKL_SHAPE_DIFFRACTIVE
        );
        if norm {
            let r = self.param(&ParamPath::ShapeParam {
                surface: id,
                parameter: ShapeParameter::NormRadius,
            })?;
            self.positive(r.clone());
            self.coefficient(off + MKL_COEFF_RADIUS as usize, r);
        }
        if matches!(p.shape, MKL_SHAPE_TOROID | MKL_SHAPE_BICONIC) {
            let x = self.param(&ParamPath::ShapeParam {
                surface: id,
                parameter: ShapeParameter::CurvatureX,
            })?;
            self.coefficient(off + MKL_COEFF_CX as usize, x);
            if p.shape == MKL_SHAPE_BICONIC {
                let x = self.param(&ParamPath::ShapeParam {
                    surface: id,
                    parameter: ShapeParameter::ConicX,
                })?;
                self.coefficient(off + MKL_COEFF_KX as usize, x);
            }
        }
        let family = match p.shape {
            MKL_SHAPE_ODD_ASPHERE => CoefficientSet::Odd,
            MKL_SHAPE_Q_CON | MKL_SHAPE_Q_BFS => CoefficientSet::Q,
            MKL_SHAPE_XY => CoefficientSet::Xy,
            MKL_SHAPE_ZERNIKE => CoefficientSet::Zernike,
            MKL_SHAPE_DIFFRACTIVE => CoefficientSet::Phase,
            _ => CoefficientSet::XAsphere,
        };
        let mut a = Vec::new();
        for j in 0..na {
            let mut value = self.param(&ParamPath::ShapeCoefficient {
                surface: id,
                set: family,
                index: j,
            })?;
            if family == CoefficientSet::Odd {
                value = value * r.powi(j as i32 + 1);
            }
            if family == CoefficientSet::XAsphere {
                value = value * r.powi(2 * j as i32 + 4);
            }
            a.push(value);
        }
        if p.shape == MKL_SHAPE_Q_BFS {
            let (mut f, mut g, mut h) = ([0.0f64; 20], [0.0f64; 20], [0.0f64; 20]);
            f[0] = 2.0;
            f[1] = 19f64.sqrt() / 2.0;
            g[0] = -0.5;
            for m in 2..20 {
                h[m - 2] = -((m * (m - 1)) as f64) / (2.0 * f[m - 2]);
                g[m - 1] = -(1.0 + g[m - 2] * h[m - 2]) / f[m - 1];
                f[m] =
                    ((m * (m + 1) + 3) as f64 - g[m - 1] * g[m - 1] - h[m - 2] * h[m - 2]).sqrt();
            }
            let mut b = vec![self.c(0.0); na + 2];
            for m in (0..na).rev() {
                b[m] = (a[m].clone() - b[m + 1].clone() * g[m] - b[m + 2].clone() * h[m]) / f[m];
            }
            a = b[..na].to_vec();
        }
        let start = off
            + MKL_COEFF_PREFIX as usize
            + if p.shape == MKL_SHAPE_DIFFRACTIVE {
                crate::layout::MAX_WAVELENGTHS
            } else {
                0
            };
        for (j, v) in a.into_iter().enumerate() {
            self.coefficient(start + j, v);
        }
        for j in 0..nb {
            let v = self.param(&ParamPath::ShapeCoefficient {
                surface: id,
                set: CoefficientSet::YAsphere,
                index: j,
            })? * r.powi(2 * j as i32 + 4);
            self.coefficient(start + na + j, v);
        }
        // Coefficients may cross zero without changing topology or selecting a
        // nominal exact-conic fast path for a non-conic candidate.
        self.coefficient(off + MKL_COEFF_EXACT_CONIC as usize, self.c(0.0));
        Ok(())
    }
}
