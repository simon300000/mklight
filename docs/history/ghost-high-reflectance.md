# The high-reflectance ghost flux discrepancy: pupil quadrature, not transport

The investigation behind the ghost/stray partition test of GHOST.md §5
(`mklight-stray/tests/partition.rs`). A record of finished work; names and
numbers are those of the time of writing.

A singlet with ideal 30 % reflectors, a 24 × 24 mm detector, one wavelength
(0.55 µm), and ghost path `[2, 1]` exposed aliasing of the detector boundary
by the old cell-centre grid. Relative to entrance-pupil flux, on axis:

| base grid | old centre-only flux | boundary-refined flux |
|---|---:|---:|
| 120² | 2.63592417e-3 | 2.57960188e-3 |
| 240² | 2.63592417e-3 | 2.57579215e-3 |
| 480² | 2.59473785e-3 | 2.57778067e-3 |
| 960² | 2.57043793e-3 | 2.57764714e-3 |

The unchanged 120²/240² answer was false convergence. A sensor accepts only
a small part of this broad ghost, so whole-cell acceptance inflated its
pupil area. The same effect explains the small negative z scores at 5 %;
it scales with the ideal coating weight `(1 − R)² R²`.

Targeted isolation: replay exactly the stray solver's Philox launch rays in
the sequence tracer, with identical apertures, one wavelength, no scatter,
no sensor reflection, and enough splitting to follow every order-2 branch.
At 200,000 rays, seeds 7–9 and both singlet fields, the two order-2 fluxes
agree within 4e-14 relative. The coefficients are angle independent and
identical for s/p; both sums are incoherent. Thus neither angle/polarisation,
sensor absorption, roulette, clipping geometry nor source normalisation
causes this discrepancy. Compare **order 2**, not all ghost orders: at 30 %,
higher orders contribute measurable flux.

`mklight-stray/tests/partition.rs` retains the paired flux and exact per-ray
standard-error checks, requires the 120²/480² refined fluxes to agree within
0.2 %, and requires |z| < 3 for each seed and the combined mean. The normal
suite uses 60,000 rays × three seeds × two fields; the ignored large test
uses 2,000,000 rays × four seeds × two fields. The large run gave
z from −0.73 to +1.54, with combined-mean z = +1.12 (on axis) and
+0.76 (off axis):

```sh
cargo test --release -p mklight-stray --test tests partition::high_reflectance_ghost_flux_large_run -- --ignored --nocapture
```
