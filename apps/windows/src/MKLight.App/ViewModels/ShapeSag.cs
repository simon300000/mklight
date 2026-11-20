using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>
/// Sag z(x, y) of every surface shape, for the inspector's profile previews and departure maps only
/// (docs/SURFACES.md §1–6; the Mac app's <c>ShapeSag</c>). Display geometry: no ray tracing, no first
/// order. The engine's <c>surface_math</c> is the definition; these follow the same formulas.
/// </summary>
public static class ShapeSag
{
    /// <summary>Rationalised conic sag B(c, k, s) with s = r², or null outside its real domain.</summary>
    public static double? Conic(double c, double k, double s)
    {
        var q = 1 - (1 + k) * c * c * s;
        if (q < 0) return null;
        return c * s / (1 + Math.Sqrt(q));
    }

    /// <summary>The sag at local (x, y) in mm, or null outside the shape's domain (normalised polynomials beyond their radius).</summary>
    public static double? Sag(SurfaceShape shape, double x, double y)
    {
        var s = x * x + y * y;
        switch (shape)
        {
            case PlaneShape or ParaxialShape or CoordinateBreakShape:
                return 0;
            case ConicShape c:
                return Conic(c.VertexCurvature, c.Conic, s);
            case EvenAsphereShape e:
            {
                return Conic(e.VertexCurvature, e.Conic, s) is { } b ? b + Horner(e.Coefficients, s) * s * s : null;
            }
            case OddAsphereShape o:
            {
                if (Conic(o.VertexCurvature, o.Conic, s) is not { } b) return null;
                var r = Math.Sqrt(s);
                var z = b;
                for (var i = 0; i < o.Coefficients.Count; i++) z += o.Coefficients[i] * Math.Pow(r, i + 1);
                return z;
            }
            case QConShape q:
            {
                if (Conic(q.VertexCurvature, q.Conic, s) is not { } b || Normalised(s, q.NormRadiusMm) is not { } t) return null;
                return b + t * t * QCon(q.Coefficients, t);
            }
            case QBfsShape q:
            {
                if (Conic(q.VertexCurvature, 0, s) is not { } b || Normalised(s, q.NormRadiusMm) is not { } t) return null;
                if (q.Coefficients.All(a => a == 0)) return b;
                var w = 1 - q.VertexCurvature * q.VertexCurvature * s;
                if (w <= 0) return null;
                return b + t * (1 - t) * QBfs(q.Coefficients, t) / Math.Sqrt(w);
            }
            case DiffractiveShape d:
                return Sag(d.Base, x, y);
            case SagOverlayShape o:
            {
                // SURFACES §10: one fringe is λ/2 PV sag.
                if (Sag(o.Base, x, y) is not { } b) return null;
                double u = x / o.NormRadiusMm, v = y / o.NormRadiusMm, t = u * u + v * v;
                var theta = 2 * o.AstigmatismAngleDeg * Math.PI / 180;
                var astig = (u - v) * (u + v) * Math.Cos(theta) + 2 * u * v * Math.Sin(theta);
                return b + o.TestWavelengthUm * 1e-3 * (o.Astigmatism * astig / 4 + 2 * o.Spherical * (t * t - t));
            }
            case ToroidShape tor:
            {
                var yy = y * y;
                if (Conic(tor.CurvatureY, tor.ConicY, yy) is not { } zb) return null;
                var zy = zb + Horner(tor.YAsphere, yy) * yy * yy;
                var cx = tor.CurvatureX;
                if (cx == 0) return zy;
                var a = 1 - cx * zy;
                var d = a * a - cx * cx * x * x;
                if (a <= 0 || d < 0) return null;
                return zy + cx * x * x / (a + Math.Sqrt(d));
            }
            case BiconicShape bi:
            {
                double xx = x * x, yy = y * y;
                var w = 1 - (1 + bi.ConicX) * bi.CurvatureX * bi.CurvatureX * xx - (1 + bi.ConicY) * bi.CurvatureY * bi.CurvatureY * yy;
                if (w < 0) return null;
                return (bi.CurvatureX * xx + bi.CurvatureY * yy) / (1 + Math.Sqrt(w)) + Horner(bi.XAsphere, xx) * xx * xx + Horner(bi.YAsphere, yy) * yy * yy;
            }
            case XyPolynomialShape p:
            {
                if (Conic(p.VertexCurvature, p.Conic, s) is not { } b || Normalised(s, p.NormRadiusMm) is null) return null;
                return b + XyDeparture(p.Coefficients, x / p.NormRadiusMm, y / p.NormRadiusMm);
            }
            case ZernikeSagShape z:
            {
                if (Conic(z.VertexCurvature, z.Conic, s) is not { } b || Normalised(s, z.NormRadiusMm) is null) return null;
                return b + ZernikeDeparture(z.Coefficients, z.Ordering, x / z.NormRadiusMm, y / z.NormRadiusMm);
            }
            default:
                return 0;
        }
    }

    /// <summary>The sag minus its base conic: what the polynomial or freeform terms add.</summary>
    public static double? Departure(SurfaceShape shape, double x, double y)
    {
        if (Sag(shape, x, y) is not { } z) return null;
        var s = x * x + y * y;
        double? base0 = shape switch
        {
            XyPolynomialShape p => Conic(p.VertexCurvature, p.Conic, s),
            ZernikeSagShape p => Conic(p.VertexCurvature, p.Conic, s),
            QConShape p => Conic(p.VertexCurvature, p.Conic, s),
            OddAsphereShape p => Conic(p.VertexCurvature, p.Conic, s),
            EvenAsphereShape p => Conic(p.VertexCurvature, p.Conic, s),
            QBfsShape p => Conic(p.VertexCurvature, 0, s),
            _ => 0,
        };
        return base0 is { } b ? z - b : null;
    }

    // ---- Polynomials ----

    static double Horner(IReadOnlyList<double> a, double s)
    {
        var v = 0.0;
        for (var i = a.Count - 1; i >= 0; i--) v = v * s + a[i];
        return v;
    }

    /// <summary>t = (r / r_norm)², or null beyond the normalisation radius.</summary>
    static double? Normalised(double s, double rn)
    {
        if (!(rn > 0)) return null;
        var t = s / (rn * rn);
        return t <= 1 + 1e-12 ? t : null;
    }

    /// <summary>Σ a_m Q_m^con(t) with Q_m^con(t) = P_m^(0,4)(2t − 1).</summary>
    public static double QCon(IReadOnlyList<double> a, double t)
    {
        double x = 2 * t - 1, alpha = 0, beta = 4;
        double sum = 0, p0 = 1, p1 = alpha + 1 + (alpha + beta + 2) * (x - 1) / 2;
        for (var m = 0; m < a.Count; m++)
        {
            if (m == 0) sum += a[m] * p0;
            else if (m == 1) sum += a[m] * p1;
            else
            {
                double n = m, ab = alpha + beta;
                var c1 = 2 * n * (n + ab) * (2 * n + ab - 2);
                var c2 = (2 * n + ab - 1) * ((2 * n + ab) * (2 * n + ab - 2) * x + alpha * alpha - beta * beta);
                var c3 = 2 * (n + alpha - 1) * (n + beta - 1) * (2 * n + ab);
                var p2 = (c2 * p1 - c3 * p0) / c1;
                p0 = p1;
                p1 = p2;
                sum += a[m] * p2;
            }
        }
        return sum;
    }

    /// <summary>Σ a_m Q_m^bfs(t) through Forbes' auxiliary basis (SURFACES §2).</summary>
    public static double QBfs(IReadOnlyList<double> a, double t)
    {
        var n = a.Count;
        if (n == 0) return 0;
        double[] f = new double[n], g = new double[n], h = new double[n];
        f[0] = 2;
        if (n > 1) f[1] = Math.Sqrt(19.0) / 2;
        g[0] = -0.5;
        for (var m = 2; m < n; m++)
        {
            h[m - 2] = -(double)(m * (m - 1)) / (2 * f[m - 2]);
            g[m - 1] = -(1 + g[m - 2] * h[m - 2]) / f[m - 1];
            f[m] = Math.Sqrt(m * (m + 1) + 3 - g[m - 1] * g[m - 1] - h[m - 2] * h[m - 2]);
        }
        var b = new double[n + 2];
        for (var m = n - 1; m >= 0; m--)
            b[m] = (a[m] - (m + 1 < n ? g[m] * b[m + 1] : 0) - (m + 2 < n ? h[m] * b[m + 2] : 0)) / f[m];
        var al = new double[n + 2];
        for (var m = n - 1; m >= 0; m--) al[m] = b[m] + (2 - 4 * t) * al[m + 1] - al[m + 2];
        return 2 * (al[0] + (n > 1 ? al[1] : 0));
    }

    /// <summary>Σ C_j X^m Y^n in the engine's term order (SURFACES §5).</summary>
    static double XyDeparture(IReadOnlyList<double> c, double x, double y)
    {
        var z = 0.0;
        for (var j = 0; j < c.Count; j++)
        {
            if (c[j] == 0) continue;
            if (XYTerm.FromIndex(j) is not { } t) break;
            z += c[j] * Math.Pow(x, t.M) * Math.Pow(y, t.N);
        }
        return z;
    }

    /// <summary>Σ c_j Z_j(ρ, θ) with θ from +x toward +y (SURFACES §6).</summary>
    static double ZernikeDeparture(IReadOnlyList<double> c, ZernikeOrdering ordering, double x, double y)
    {
        double rho = Math.Sqrt(x * x + y * y), theta = Math.Atan2(y, x);
        var z = 0.0;
        for (var k = 0; k < c.Count; k++)
        {
            if (c[k] == 0) continue;
            if (ZernikeTerm.FromJ(k + 1, ordering) is not { } t) break;
            z += c[k] * Zernike(t, ordering, rho, theta);
        }
        return z;
    }

    /// <summary>One Zernike term: Fringe unnormalised, Standard (Noll) normalised.</summary>
    public static double Zernike(ZernikeTerm t, ZernikeOrdering ordering, double rho, double theta)
    {
        var m = Math.Abs(t.M);
        var r = 0.0;
        for (var k = 0; k <= (t.N - m) / 2; k++)
        {
            var num = Factorial(t.N - k);
            var den = Factorial(k) * Factorial((t.N + m) / 2 - k) * Factorial((t.N - m) / 2 - k);
            r += (k % 2 == 0 ? 1 : -1) * num / den * Math.Pow(rho, t.N - 2 * k);
        }
        var angular = t.M > 0 ? Math.Cos(m * theta) : t.M < 0 ? Math.Sin(m * theta) : 1;
        var norm = ordering == ZernikeOrdering.Standard ? (t.M == 0 ? Math.Sqrt(t.N + 1) : Math.Sqrt(2.0 * (t.N + 1))) : 1;
        return norm * r * angular;
    }

    static double Factorial(int n)
    {
        var v = 1.0;
        for (var i = 2; i <= n; i++) v *= i;
        return v;
    }

    // ---- Diffractive phase ----

    /// <summary>Phase Φ(r) = Σ A_i (r/r_norm)^(2i) in radians (order +1).</summary>
    public static double Phase(DiffractivePhase p, double r)
    {
        if (!(p.NormRadius > 0)) return 0;
        var rho2 = r / p.NormRadius * (r / p.NormRadius);
        var v = 0.0;
        for (var i = 0; i < p.Coefficients.Count; i++) v += p.Coefficients[i] * Math.Pow(rho2, i + 1);
        return v;
    }

    /// <summary>dΦ/dr in rad/mm.</summary>
    public static double PhaseSlope(DiffractivePhase p, double r)
    {
        if (!(p.NormRadius > 0)) return 0;
        var rn2 = p.NormRadius * p.NormRadius;
        var rho2 = r * r / rn2;
        var v = 0.0;
        for (var k = 0; k < p.Coefficients.Count; k++)
        {
            double i = k + 1;
            v += p.Coefficients[k] * i * Math.Pow(rho2, i - 1) * 2 * r / rn2;
        }
        return v;
    }

    /// <summary>Paraxial reduced power of the phase in order m at λ (µm): φ = −m λ A₁ / (π r_norm²) in mm⁻¹ (SURFACES §3).</summary>
    public static double PhasePower(DiffractivePhase p, int order, double wavelengthUm)
    {
        if (p.Coefficients.Count == 0 || !(p.NormRadius > 0)) return 0;
        return -order * wavelengthUm * 1e-3 * p.Coefficients[0] / (Math.PI * p.NormRadius * p.NormRadius);
    }

    /// <summary>The smallest zone period 2π / |dΦ/dr| (mm) over 0 &lt; r ≤ h, and where it occurs; null without a phase gradient.</summary>
    public static (double Period, double Radius)? MinimumPeriod(DiffractivePhase p, double h, int samples = 400)
    {
        if (!(h > 0) || !double.IsFinite(h)) return null;
        (double, double)? best = null;
        for (var k = 1; k <= samples; k++)
        {
            var r = h * k / samples;
            var slope = Math.Abs(PhaseSlope(p, r));
            if (!(slope > 0) || !double.IsFinite(slope)) continue;
            var period = 2 * Math.PI / slope;
            if (best is null || period < best.Value.Item1) best = (period, r);
        }
        return best;
    }
}
