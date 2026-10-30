# Surface mathematics and the CPU reference

All geometry is local to a surface. Lengths and sag amplitudes are in mm,
curvatures in mm⁻¹, wavelengths in µm in the model (converted to mm in ray
physics). `surface_math::evaluate(shape, x, y)` returns sag, gradient
`[z_x,z_y]`, and Hessian `[z_xx,z_xy,z_yy]`. It evaluates a validated model.
`None` means outside the real sag domain. `SurfaceShape::sag(r²)` remains
backward compatible: it is the **positive y meridian**, `z(0,√r²)`; use the
Cartesian API for freeforms. No core dependency on the wave crate is needed.

## 1. Base conics and power series

With `s=x²+y²`, `κ=1+k`, `w=√(1−κc²s)`,

```
B(c,k,s) = c s/(1+w)
B_s = c/(2w),  B_ss = κ c³/(4w³)
z_x=2x B_s, z_y=2y B_s
z_xx=2 B_s+4x² B_ss, z_xy=4xy B_ss, z_yy=2 B_s+4y² B_ss.
```

This rationalised sag avoids subtracting nearly equal quantities. `c=0`
is exactly a plane. Plane, ideal lens and coordinate break have zero sag;
a coordinate break has no intersection geometry. The real conic domain is
`1−κc²s≥0`. At the rim the sag exists but its derivatives can be infinite
or undefined. The legacy implicit conic normal remains finite there; an
extended shape needs a finite gradient, so a hit exactly on a vertical rim
is `OutsideDomain`.

`even_asphere` is `B+Σ a_i r^(2i+4)` (zero-based i). Existing coefficients
and parameter paths keep their meanings and units `mm^(−2i−3)`.
`odd_asphere` is `B+Σ β_i r^i`, **i=1 onward**, with units `mm^(1−i)`.
Both use Horner evaluation. Odd coefficients have at most 20 terms.
The odd `r¹` term is a cone: at exactly the vertex we define the normal as
`+z` (zero gradient), but **the Hessian does not exist**, and is returned
as NaN. The formal first-order curvature omits that term and is `c+2β₂`;
it is not a limit of real-ray focal length when β₁ is nonzero.

## 2. Forbes Q surfaces

Let `u=r/r_norm`, `t=u²`. Coefficients `a_m` are in mm and start at m=0.
Both families allow 20 terms and require `r_norm>0`.

```
q_con: z = B(c,k,r²) + t² Σ a_m Q_m^con(t)
Q_m^con(t) = P_m^(0,4)(2t−1)
Q_0=1; Q_1=6t−5; Q_2=28t²−42t+15.

q_bfs: z = B(c_bfs,0,r²) + t(1−t) Σ a_m Q_m^bfs(t)/√(1−c_bfs²r²)
Q_0=1
Q_1=(13−16t)/√19
Q_2=(29−100t+76t²)/(√190/2).
```

QCon uses Clenshaw with `Q_(k+1)=A_k Q_k−B_k Q_(k−1)`:

```
A_0=6t−5, B_0=0
v=2k+4, d=2(k+1)(k+5)v
A_k=(v+1)[(v+2)v(2t−1)−16]/d
B_k=2k(k+4)(v+2)/d                          (k≥0; gives A_0, B_0 above)
b_k=a_k+A_k b_(k+1)−B_(k+1)b_(k+2); ΣaQ=b_0.
```

QBfs uses Forbes' auxiliary basis and triangular conversion:

```
P_0=2, P_1=6−8t, P_(m+1)=(2−4t)P_m−P_(m−1)
P_m=f_m Q_m+g_(m−1)Q_(m−1)+h_(m−2)Q_(m−2)
f_0=2, f_1=√19/2, g_0=−1/2
h_(m−2)=−m(m−1)/(2f_(m−2))
g_(m−1)=−[1+g_(m−2)h_(m−2)]/f_(m−1)
f_m=√[m(m+1)+3−g_(m−1)²−h_(m−2)²]         (m≥2)
b_m=[a_m−g_m b_(m+1)−h_m b_(m+2)]/f_m     (backward)
α_m=b_m+(2−4t)α_(m+1)−α_(m+2)             (backward)
ΣaQ=2(α_0+α_1).
```

Out-of-range recurrence states are zero. The second-order jets differentiate
these same recurrences; for a step `b=a+A(t)b₁−B b₂`,
`b'=A'b₁+Ab₁'−Bb₂'` and `b''=2A'b₁'+Ab₁''−Bb₂''`.
No monomial conversion or numerical derivatives are used. The QBfs square
root must be positive for a nonzero departure. With zero coefficients it
reduces to the sphere, including its rim. QCon leaves the vertex curvature
unchanged; QBfs has vertex curvature `c_bfs+2Σa_m Q_m(0)/r_norm²`.

Definitions: [Forbes, Opt. Express 15, 5218–5226 (2007)](https://doi.org/10.1364/OE.15.005218).
Recurrence methods: [Forbes, Opt. Express 18, 13851–13862 (2010)](https://doi.org/10.1364/OE.18.013851),
and QBfs auxiliary basis: [Forbes, Opt. Express 18, 19700–19712 (2010)](https://doi.org/10.1364/OE.18.019700), §§2–3, Appendix A.

## 3. Diffractive phase on a base surface

`diffractive` contains `base: SurfaceShape` restricted to `plane`, `conic`,
or `even_asphere`, and `phase`:

| field | meaning |
|---|---|
| `coefficients` | A₁, A₂, …, at most 10, in radians |
| `norm_radius` | positive normalisation radius, mm |
| `design_order` | nonzero signed integer order used for design |
| `design_wavelength_um` | positive design wavelength λ₀, µm |
| `trace_order` | signed integer order selected for tracing; zero is ordinary refraction/reflection |

`Φ(x,y)=Σ A_i [(x²+y²)/r_norm²]^i`, i=1 onward. Coefficients describe
**order +1** and remain fixed as wavelength or selected order changes.
Design order and wavelength set the relief depth for scalar efficiency;
neither rescales the trace phase or groove period.

The time convention is `exp[i(ωt−k·r)]` (CONVENTIONS §9). Here Φ denotes a
**phase lag**: an order-m screen multiplies the field by `exp(−imΦ)`.
The eikonal, whose field factor is `exp(−i2πS/λ)`, therefore jumps by
`ΔS=mλΦ/(2π)`. Differentiating this boundary condition *along the actual
base surface* gives

```
n' t'_parallel = n t_parallel + m λ ∇_surface Φ/(2π).
```

For a graph `z(x,y)`, with unit axial normal N, the surface gradient is
`g=(Φ_x,Φ_y,0)−N[N·(Φ_x,Φ_y,0)]`. This is the metric-correct tangent
vector: its dot product with `(dx,dy,z_x dx+z_y dy)` equals `dΦ`.
Let `v=[n(t−N(t·N))+mλg/(2π)]/n'`. Then
`t'=v+σN√(1−v·v)`, where σ selects the incident normal hemisphere for
transmission and its opposite for reflection. Reflecting DOEs use n'=n.
`v·v>1` gives `Failed { reason: evanescent_order }`, distinct from TIR.
Dummy and image surfaces have no phase interaction, as for existing shapes.

The same `ΔS` is added to compensated OPL at the interaction, before the
surface record is stored. Consequently OPD is still `OPL_chief−OPL_ray`.
A focusing plane DOE has negative A₁: for +z propagation in air,

```
φ_DOE = −m λ A₁/(π r_norm²),    f=1/φ_DOE.
```

This reduced power is added to the refractive base power. Signed reduced
indices give the same power expression on reflected paths. Fixed periods
mean `φ∝λ`, not `1/λ` and not an index-dispersion scaling. Thus its effective
d/F/C Abbe number is `λ_d/(λ_F−λ_C)≈−3.453`, opposite to ordinary glass.
A perfect converging phase at λ₀ and order m is
`Φ=−2π[√(f²+r²)−f]/(mλ₀)`; its polynomial expansion is tested to a
controlled truncation error against constant total OPL at the focus.

### 3.1 Scalar diffraction efficiency

Every `Surface` has `diffraction_efficiency`: `{"type":"ideal"}` (default,
unit efficiency in `trace_order`), `{"type":"kinoform"}`, or
`{"type":"multilevel","levels":N}` (1–255; N=1 is an unetched screen).
This is a thin-element, scalar, locally periodic, normal-incidence relief
model. It omits vector/resonance effects, shadowing and fabrication errors.
It is an incoherent power partition, not a coherent wave-imaging model.
Coating R/T and bulk absorption remain separate factors.

Let u run from 0 to 1 over one order-1 structure period. A continuous
manufactured ramp has complex factor `exp(−i2πbu)`, b=αm₀. Its Fourier
coefficient in the trace convention is

```
c_m = ∫₀¹ exp[−i2π(b−m)u] du
η_m = |c_m|² = sinc²(m−αm₀),     sinc(x)=sin(πx)/(πx).
```

For a relief separating two media, define the manufactured depth using
`D₀=|n_after(λ₀)−n_before(λ₀)|`. At λ, `D=|n_after(λ)−n_before(λ)|`, so
`α=(λ₀/λ)D/D₀`. Both materials are evaluated at their actual environment,
including λ₀ even when it is absent from the wavelength list. For a glass
in air this is `(λ₀/λ)[n(λ)−1]/[n(λ₀)−1]`. Reversing transmission leaves
the ratio unchanged. For a designed reflecting relief use `D₀=2n_inc(λ₀)`
and `D=2n_inc(λ)`: the relief changes a round-trip optical path. A Fresnel
reflection at a *transmitting* DOE uses `D=2n_inc(λ)` over the original
transmission depth D₀; it generally blazes into a different order. A
same-medium mathematical phase screen has no material depth; its defined
limit is α=λ₀/λ. This convention needs no extra fictitious substrate.

For N equal-width steps with heights proportional to j/N, integrate each
cell before summing:

```
c_m = exp(iπm/N) sinc(m/N) (1/N) Σ[j=0..N−1] exp[−i2π(b−m)j/N]
η_m = sinc²(m/N) |sin[π(b−m)] / {N sin[π(b−m)/N]}|².
```

The removable singularity is evaluated with reduced-argument sinc ratios.
At λ₀ for m₀=1, `η₁=sinc²(1/N)`: binary 0.4052847346, four-level
0.8105694691, eight-level 0.9496412036. A continuous kinoform has η_m₀=1
at λ₀; a multilevel ramp retains its quantisation loss in other orders.
Parseval gives `Σ[m∈Z]η_m=1` for either unit-modulus phase ramp. Any finite
order window sums to at most one; its omitted tail must not be normalised
away or called material absorption. The MC samples integer orders with a
rounded-Cauchy rejection envelope, including negative orders and zero,
without a finite order window. Non-propagating orders terminate in the
explicit `diffraction` energy sink: this scalar approximation does not
redistribute their power into propagating modes.

Derivation and scalar-design background: [Mait, JOSA A 12, 2145–2158
(1995)](https://doi.org/10.1364/JOSAA.12.002145). Quantisation and material
scaling: [Considering Grayscale Process and Material Properties for Robust
Multilevel Diffractive Flat Optics, Photonics 11, 1200
(2024)](https://doi.org/10.3390/photonics11121200).

`materials::diffraction` is shared by primary radiometry, explicit ghosts
and non-sequential transport. System transmission multiplies the selected
order's η at every DOE (also for polarised transmission and Jones pupil
maps). Ghost/stray order ownership is CONVENTIONS §10 and GHOST §2.1.
Ghost paths and stray geometry now support the extended surfaces; their
entries in §8's earlier guard list are superseded by this implementation.

## 4. Toroid and biconic

Axis-asphere coefficients begin with y⁴ or x⁴, in physical mm coordinates,
just like `even_asphere`; at most 10 per axis. They do not need a
normalisation radius. Both types have vertex curvatures `(c_x,c_y)`.

For `toroid`, `z_y=B(c_y,k_y,y²)+Σb_i y^(2i+4)`. Put `a=1−c_x z_y`:

```
z = z_y + c_x x²/[a+√(a²−c_x²x²)].
```

This is the rationalised vertex-connected branch of the rotation about
the line parallel to y through z=1/c_x. For positive c_x it equals
`R_x−√[(R_x−z_y)²−x²]`. For negative c_x the square-root sign reverses
so the sheet still passes through the vertex. Domain: `a>0` and
`a²−c_x²x²≥0`. At c_x=0 it is exactly `z_y`, without dividing by c_x;
at c_y=0 with zero y coefficients it is an x cylinder. Equal curvatures,
k_y=0 and zero coefficients give a sphere.

For `biconic`,

```
w=√[1−(1+k_x)c_x²x²−(1+k_y)c_y²y²]
z=(c_x x²+c_y y²)/(1+w)+Σa_i x^(2i+4)+Σb_i y^(2i+4).
```

The square-root argument must be nonnegative. Equal c and k with zero
polynomials reduce to a conic. Setting one curvature and its polynomial
to zero gives a cylinder; a cylinder is a preset, not another shape type.

## 5. XY polynomial

`xy_polynomial` is `B(c,k,r²)+Σ C_j X^m Y^n`, with `X=x/r_norm`,
`Y=y/r_norm`, coefficients in mm. All terms are present, without symmetry
restrictions. Total degree d=m+n increases from 1 through 10; within a
degree m decreases from d to 0. A coefficient vector may end within a
degree; missing trailing coefficients are zero. There is no piston term.

| zero-based index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| term | X | Y | X² | XY | Y² | X³ | X²Y | XY² | Y³ |

`j=d(d+1)/2−1+n`; degree D has `D(D+3)/2` coefficients (65 at D=10).
Evaluation uses Horner within each degree. Vertex Hessian is
`[c+2C₂/r_norm², C₃/r_norm², c+2C₄/r_norm²]` (indices zero-based).

## 6. Zernike sag

`zernike_sag` is `B(c,k,r²)+Σc_j Z_j(ρ,θ)`. Sag coefficients are **mm**,
not the wave crate's waves. `ρ=r/r_norm`, and θ runs from +x toward +y.
The first vector coefficient is j=1 (piston); piston and tilt are retained.

A term `(n,m)` uses `R_n^|m|(ρ) cos(mθ)` for m>0, sine for m<0, and the
radial polynomial for m=0. `R=ρ^|m| P_((n−|m|)/2)^(0,|m|)(2ρ²−1)`;
`ρ^|m| exp(i|m|θ)=(X+iY)^|m|`. The Jacobi recurrence and complex polynomial
multiplication avoid polar-coordinate singularities at the origin.

| j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fringe (n,m) | 0,0 | 1,1 | 1,−1 | 2,0 | 2,2 | 2,−2 | 3,1 | 3,−1 | 4,0 | 3,3 | 3,−3 |
| Standard (n,m) | 0,0 | 1,1 | 1,−1 | 2,0 | 2,−2 | 2,2 | 3,−1 | 3,1 | 3,−3 | 3,3 | 4,0 |

Default `fringe` is unnormalised (`R(1)=1`), 37 terms: for j≤36,
`k=floor(√(j−1))`, `o=j−1−k²`, `|m|=k−floor(o/2)`, `n=2k−|m|`, with
cosine for even o and sine for odd o. j=37 is `(12,0)`.
`standard` (alias `noll`) permits 45 terms. Choose n such that
`n(n+1)/2<j≤(n+1)(n+2)/2`, `q=j−n(n+1)/2−1`;
`|m|=2ceil(q/2)` for even n, otherwise `2floor(q/2)+1`. Even j is cosine,
odd j sine. Multiply by `√(n+1)` for m=0, otherwise `√[2(n+1)]`.
These definitions match `mklight-wave::zernike`.
See [Noll, JOSA 66, 207–211 (1976)](https://doi.org/10.1364/JOSA.66.000207).

## 7. Domains, derivatives, validation and parameters

Normalised polynomials (Q, XY, Zernike and phase) are evaluated only for
`ρ≤1+10⁻¹²`, including a rounding margin, never clamped. Odd and physical
axis series have only their base geometric domains. The sag, gradient and
Hessian share one expression via second-order Cartesian jets. For a
product `fg`, `H= g H_f+f H_g+∇f∇gᵀ+∇g∇fᵀ`; for composition F(f),
`H=F' H_f+F''∇f∇fᵀ`. Derivative tests cross-check this against differences
of independent sag/gradient evaluations, including the vertex and limits.

Structural validation extends CONVENTIONS §13: finite scalar/coefficients,
positive radius and design wavelength, nonzero design order, signed i32
orders, supported base, and the limits above. Codes are `non_finite`,
`invalid_norm_radius`, `coefficient_count`, `invalid_design_wavelength`,
`invalid_diffraction_order`, `invalid_diffractive_base`.

`ParamPath::ShapeCoefficient {surface,set,index}` uses sets `odd`, `q`,
`phase`, `x_asphere`, `y_asphere`, `xy`, `zernike`. A missing in-range
coefficient reads zero; writing extends with zeros. Out-of-range indices
are rejected before allocation. Values are range-checked only by structural
validation, never by the path itself. `ShapeParam {surface,parameter}` supports
`curvature_x`, `conic_x`, `curvature_y`, `conic_y`, `norm_radius`,
`design_order`, `design_wavelength_um`, `trace_order`. JSON keeps the legacy
`param` tag, so the scalar selector is named **parameter**. Order edits
must be integral. Orders are discrete, not continuous optimisation variables.
Legacy `Curvature`, `Conic` and `Asphere` paths retain their meanings, also
addressing a DOE's base where applicable.

## 8. Reference tracing, first order and symmetry

New sag intersections start from the conic/BFS solution in the closest
approach frame. Safeguarded Newton uses `g=z−sag`,
`g'=d_z−z_x d_x−z_y d_y`, a crossing bracket and residual backtracking.
Normals are `normalize(−z_x,−z_y,1)`. Clipping, signed transfers and OPL
retain the existing behavior. `Miss`, `NoConvergence` and `OutsideDomain`
keep the existing distinction between no evidence of crossing, an
unconverged bracket/converging iteration, and leaving the real sag domain.

`FirstOrder` and `ParaxialModel::new` keep the y-axis sweep. The separate
`ParaxialXY` reports x and y first-order tables at every wavelength, plus
**signed squeeze = EFL_y/EFL_x**, primary wavelength, symmetry and RunInfo.
Its job variant checks cancellation between wavelengths and reports monotone
progress. The existing EFL convention is unit-index referenced; image-space
focal length is n' EFL. BFL is the axial distance from the last optical
surface. Tilts and decenters are ignored by these unfolded sweeps.

First-order powers use the vertex Hessian diagonal and signed reduced
indices, plus DOE power. Higher-order freeforms have only this local
quadratic meaning; no Seidel interpretation is supplied. Mixed xy power
couples the axes: the reported diagonal sweeps are projections and emit a
warning, not principal focal lengths. Piston/tip/tilt terms need an actual
chief-ray linearisation for physical first order away from the nominal
vertex. A nonzero odd cone term likewise has only formal quadratic data.

`system_symmetry` classifies geometry as `rotational`, `meridional` (invariant
under x→−x, the y-z plane), or `none`; placements and aperture shapes are
included, fields are sampling requests. This classification is conservative
and can be used to disable symmetry shortcuts. It makes no claim about
arbitrary coating, mechanics or scatter symmetry.

GPU packing, stray geometry, ghost paths, Seidel, Coddington, rotational
layout and merit evaluation reject extended shapes until those solvers are
implemented, with the single error of `OpticalSystem::require_legacy_shapes`
(the GPU packer and stray patches reject them in their shape matches). CPU tracing and these mathematical APIs
are the reference for that work. The app is unchanged.

## 9. Reference samples and verification

`samples::all()` includes the original designs and these CPU fixtures:

| identifier | purpose |
|---|---|
| `hybrid_doe` | 3 mm N-BK7 plano-convex singlet with rear phase; F/C back focus equal including glass transfer |
| `anamorphic` | finite-thickness Galilean cylindrical adapter in constant-index glass, followed by the N-BK7 singlet; EFL_y/EFL_x=2 |
| `biconic`, `toroid` | unequal-axis elements with axis polynomial departures |
| `q_con`, `q_bfs` | nonzero low-order Forbes departures on a singlet |
| `odd_asphere` | nonzero r³ with a regular vertex (β₁=0) |
| `xy_freeform` | unequal quadratic powers plus odd and mixed cubic terms |
| `zernike_sag` | Fringe astigmatism and trefoil departures |

The reference tests trace every sample field, wavelength and pupil with
paraxial and real aiming. They check edge separation around the entire rim,
known conic reductions, exact cylindrical line focus, narrow-fan first-order
focal lengths, grating momentum (including a tilted tangent plane), reflective
DOE signs, F/C correction, evanescence, constant design-wave OPL, and steep
Q/freeform intersections. Core tests compare analytic gradients/Hessians to
finite differences, Q functions to explicit polynomials (Q^bfs to m=4,
Q^con to m=3) and all twenty Q^bfs terms to their slope orthonormality,
and every supported Zernike term to its independent factorial definition;
the wave crate checks that Zernike sag equals its own basis term by term. Legacy-only solver
cross-validation sets explicitly exclude extended shapes; engine tests check
their unsupported-analysis errors and CPU fallback from GPU packing.
