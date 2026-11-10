import Foundation
import MKLightModel

/// Synthetic analyses built on `MockOptics`. See its documentation: these are
/// illustrative stand-ins for UI development, not engine results.
extension MockOptics {
    func fieldValue(_ i: Int) -> Point2 {
        let f = system.fields.fields[i]
        return Point2(f.x, f.y)
    }

    func linspace(_ a: Double, _ b: Double, _ n: Int) -> [Double] {
        guard n > 1 else { return [(a + b) / 2] }
        return (0..<n).map { k in
            let t = Double(k) / Double(n - 1)
            return a.addingProduct(t, b - a)
        }
    }

    // MARK: Layout

    /// Clear semi-diameters: fixed values, else the largest marginal/chief
    /// ray height over every field at the primary wavelength.
    func semiDiameters() -> [Double] {
        let n = system.surfaces.count
        var traced = [Double](repeating: 0, count: n)
        for f in system.fields.fields.indices {
            let fp = fieldPoint(f)
            for (px, py) in [(0.0, 0.0), (0, 1), (0, -1), (1, 0), (-1, 0)] {
                for (i, h) in trace(makeRay(fp, px: px, py: py), w: primary).heights { traced[i] = max(traced[i], h) }
            }
        }
        return system.surfaces.indices.map { i in
            let s = system.surfaces[i]
            if let v = s.semiDiameter.fixedValue { return v }
            if let a = s.aperture { return a.boundingRadius }
            return traced[i] > 0 ? traced[i] : epRadius
        }
    }

    func profile(_ i: Int, from: Double, to: Double, samples: Int, clampTo sd: Double, axis: Int) -> [Point3] {
        let shape = system.surfaces[i].shape
        return linspace(from, to, samples).compactMap { r in
            let rc = min(abs(r), sd)
            guard let (z, _) = Self.sag(shape, rc * rc) else { return nil }
            let zz = vertexZ[i] + z
            return axis == 1 ? Point3(0, r, zz) : Point3(r, 0, zz)
        }
    }

    func layout(_ s: LayoutSettings) throws -> LayoutScene {
        let n = system.surfaces.count
        let sd = semiDiameters()
        let m = max(s.profileSamples | 1, 5)
        var lines: [LayoutScene.Polyline] = []
        var elements: [LayoutScene.Element] = []
        let image = n - 1

        // Rays.
        var imageHeight = 0.0
        let wls = s.wavelengths.indices(in: system.wavelengths)
        for f in s.fields.indices(count: system.fields.fields.count) {
            try Task.checkCancellation()
            let fp = fieldPoint(f)
            let meridional = system.fields.fields[f].x == 0
            for w in wls {
                var pupil = linspace(-1, 1, max(s.raysPerField, 1)).map { (0.0, $0) }
                if s.include3D { pupil += [(-1, 0), (1, 0)] }
                for (px, py) in pupil {
                    let t = trace(makeRay(fp, px: px, py: py), w: w)
                    if let h = t.heights[image] { imageHeight = max(imageHeight, h) }
                    lines.append(
                        .init(
                            kind: .ray, points: t.points.map { Point3($0.x, $0.y, $0.z) },
                            inSection: meridional && px == 0, field: f, wavelength: w, vignetted: !t.ok))
                }
            }
        }

        // Surfaces.
        for i in 0..<n where !system.surfaces[i].isCoordinateBreak {
            if i == 0 && objectDistance == nil { continue }
            var r = sd[i]
            if i == image { r = max(imageHeight, abs(chiefImagePerUnit * fieldParameterY(fieldPoint(x: 0, y: maxField)))) }
            if i == 0 {
                r = system.fields.fields.map { f -> Double in
                    if case .height(let x, let y) = fieldPoint(x: f.x, y: f.y) { return hypot(x, y) }
                    return 0
                }.max() ?? 1
            }
            r = max(r, 1e-3)
            let kind: LayoutScene.PolylineKind = i == system.stop ? .stop : (i == image ? .imagePlane : .surfaceProfile)
            lines.append(.init(kind: kind, points: profile(i, from: -r, to: r, samples: m, clampTo: r, axis: 1), inSection: true, surface: i))
            if i == system.stop {
                // Stop marks: short radial ticks outside the clear aperture.
                for sgn in [-1.0, 1.0] {
                    lines.append(.init(kind: .stop, points: [Point3(0, sgn * r, vertexZ[i]), Point3(0, sgn * r * 1.3, vertexZ[i])], inSection: true, surface: i))
                }
            }
            if s.include3D {
                lines.append(.init(kind: kind, points: profile(i, from: -r, to: r, samples: m, clampTo: r, axis: 0), inSection: false, surface: i))
                if let (z, _) = Self.sag(system.surfaces[i].shape, r * r) {
                    let rim = linspace(0, 2 * .pi, 65).map { a in Point3(r * cos(a), r * sin(a), vertexZ[i] + z) }
                    lines.append(.init(kind: .surfaceRim, points: rim, inSection: false, surface: i))
                }
            }
        }

        // Lens elements: a glass-filled gap between two refracting surfaces.
        for i in 1..<max(image - 1, 1) {
            let a = system.surfaces[i], b = system.surfaces[i + 1]
            guard a.interaction == .refract, !a.isCoordinateBreak, b.interaction == .refract, !b.isCoordinateBreak,
                !a.material.isAir, a.material != .vacuum
            else { continue }
            let h = max(sd[i], sd[i + 1])
            let front = profile(i, from: -h, to: h, samples: m, clampTo: sd[i], axis: 1)
            let back = profile(i + 1, from: -h, to: h, samples: m, clampTo: sd[i + 1], axis: 1)
            guard front.count == back.count, let f0 = front.first, let f1 = front.last, let b0 = back.first, let b1 = back.last else { continue }
            elements.append(.init(firstSurface: i, lastSurface: i + 1, front: front, back: back))
            lines.append(.init(kind: .elementEdge, points: [f0, b0], inSection: true, surface: i))
            lines.append(.init(kind: .elementEdge, points: [f1, b1], inSection: true, surface: i))
            // Flats where one surface is smaller than the element's outer radius.
            if sd[i] < h {
                lines.append(.init(kind: .elementEdge, points: Array(front.prefix(2)), inSection: true, surface: i))
            }
        }

        let all = lines.flatMap(\.points)
        var bounds = Bounds3.enclosing(all)
        lines.append(.init(kind: .axis, points: [Point3(0, 0, bounds.min.z), Point3(0, 0, bounds.max.z)], inSection: true))
        bounds = Bounds3.enclosing(all + [Point3(0, 0, bounds.min.z)])

        let infos = system.surfaces.indices.map { i in
            LayoutScene.SurfaceInfo(
                index: i, id: system.surfaces[i].id, vertex: Point3(0, 0, i == 0 && objectDistance == nil ? 0 : vertexZ[i]),
                semiDiameter: sd[i], semiDiameterIsAuto: system.surfaces[i].semiDiameter == .auto)
        }
        return LayoutScene(
            polylines: lines, elements: elements, surfaces: infos, bounds: bounds, wavelengthsUm: wavelengthsUm,
            fieldCount: system.fields.fields.count)
    }

    // MARK: Spot diagram

    static func pupilSamples(_ pattern: PupilPattern, rings: Int, rng: inout SplitMix64) -> [(Double, Double)] {
        let r = max(rings, 1)
        switch pattern {
        case .hexapolar:
            var out = [(0.0, 0.0)]
            for k in 1...r {
                let rad = Double(k) / Double(r)
                for j in 0..<(6 * k) {
                    let a = 2 * Double.pi * Double(j) / Double(6 * k)
                    out.append((rad * cos(a), rad * sin(a)))
                }
            }
            return out
        case .square:
            var out: [(Double, Double)] = []
            for i in -r...r {
                for j in -r...r {
                    let x = Double(i) / Double(r), y = Double(j) / Double(r)
                    if x * x + y * y <= 1 + 1e-12 { out.append((x, y)) }
                }
            }
            return out
        case .random:
            return (0..<(3 * r * (r + 1) + 1)).map { _ in
                let rad = rng.unit().squareRoot(), a = 2 * Double.pi * rng.unit()
                return (rad * cos(a), rad * sin(a))
            }
        }
    }

    func workingFNumber(_ w: Int) -> Double {
        let m = marginalStart()
        let pr = paraxial(y1: m.y1, u0: m.u0, w: w)
        let nu = abs(pr.nu[pr.nu.count - 2])
        return nu > 1e-12 ? 1 / (2 * nu) : .infinity
    }

    func spot(_ s: SpotDiagramSettings) throws -> SpotDiagramResult {
        let wls = s.wavelengths.indices(in: system.wavelengths)
        var fields: [SpotDiagramResult.FieldSpot] = []
        for f in s.fields.indices(count: system.fields.fields.count) {
            try Task.checkCancellation()
            let fp = fieldPoint(f)
            var rng = SplitMix64(seed: s.execution.seed ^ (UInt64(f) &* 0x9E37_79B9))
            let pupil = Self.pupilSamples(s.pattern, rings: s.rings, rng: &rng)
            var raw: [(Double, Double, Int)] = []
            for w in wls {
                for (px, py) in pupil {
                    if let r = trace(makeRay(fp, px: px, py: py), w: w).atImage { raw.append((r.p.x, r.p.y, w)) }
                }
            }
            var cx = CompensatedSum(), cy = CompensatedSum()
            for p in raw {
                cx.add(p.0)
                cy.add(p.1)
            }
            let count = Double(max(raw.count, 1))
            let centroid = Point2(cx.value / count, cy.value / count)
            let chief = trace(makeRay(fp, px: 0, py: 0), w: primary).atImage.map { Point2($0.p.x, $0.p.y) } ?? centroid
            let ref = s.reference == .chief ? chief : centroid
            var ss = CompensatedSum(), geo = 0.0
            let pts = raw.map { p -> SpotDiagramResult.Point in
                let dx = p.0 - ref.x, dy = p.1 - ref.y
                ss.add(dx * dx + dy * dy)
                geo = max(geo, hypot(dx, dy))
                return .init(x: dx, y: dy, wavelength: p.2)
            }
            fields.append(
                .init(
                    field: f, fieldValue: fieldValue(f), points: pts, reference: ref, centroid: centroid,
                    rmsRadius: (ss.value / count).squareRoot(), geoRadius: geo))
        }
        let airy = 1.22 * wavelengthsUm[primary] * 1e-3 * workingFNumber(primary)
        return SpotDiagramResult(fields: fields, wavelengthsUm: wavelengthsUm, airyRadius: airy)
    }

    // MARK: Fans

    func rayFan(_ s: FanSettings) throws -> FanResult {
        let wls = s.wavelengths.indices(in: system.wavelengths)
        let pupil = linspace(-1, 1, max(s.samples | 1, 3))
        let fields = try s.fields.indices(count: system.fields.fields.count).map { f -> FanResult.FieldFan in
            try Task.checkCancellation()
            let fp = fieldPoint(f)
            var tan: [FanResult.Curve] = [], sag: [FanResult.Curve] = []
            for w in wls {
                let chief = trace(makeRay(fp, px: 0, py: 0), w: w).atImage?.p
                func eps(_ px: Double, _ py: Double, _ axis: Int) -> Double? {
                    guard let c = chief, let p = trace(makeRay(fp, px: px, py: py), w: w).atImage?.p else { return nil }
                    let e = axis == 1 ? p.y - c.y : p.x - c.x
                    return e.isFinite ? e : nil
                }
                tan.append(.init(wavelength: w, pupil: pupil, values: pupil.map { eps(0, $0, 1) }))
                sag.append(.init(wavelength: w, pupil: pupil, values: pupil.map { eps($0, 0, 0) }))
            }
            return .init(field: f, fieldValue: fieldValue(f), tangential: tan, sagittal: sag)
        }
        return FanResult(quantity: .transverseAberration, fields: fields, wavelengthsUm: wavelengthsUm)
    }

    /// Exit-pupil radius and its distance to the image surface (primary).
    func exitPupil() -> (radius: Double, distance: Double) {
        let w = primary, k = system.surfaces.count - 2
        let nk = imageN(w)
        let chief = paraxial(y1: -epZ, u0: 1, w: w)
        let m0 = marginalStart()
        let marg = paraxial(y1: m0.y1, u0: m0.u0, w: w)
        let tk = system.surfaces[k].thickness
        let yk = chief.y[k], uk = chief.nu[k] / nk
        guard abs(uk) > 1e-15 else { return (epRadius, .infinity) }
        let dxp = -yk / uk // from surface k along the axis
        let ym = marg.y[k] + dxp * marg.nu[k] / nk
        return (abs(ym), abs(tk - dxp))
    }

    /// OPD from the transverse fan: `∂W/∂P = −(n' r_xp / R) ε(P)`, integrated
    /// outward from the chief ray (trapezoid rule), in waves.
    func opdFan(_ s: FanSettings) throws -> FanResult {
        let fan = try rayFan(s)
        let (rxp, dist) = exitPupil()
        let nImg = abs(imageN(primary))
        let scale = dist.isFinite && dist > 0 ? nImg * rxp / dist : 0
        func integrate(_ c: FanResult.Curve) -> FanResult.Curve {
            let lambdaMM = wavelengthsUm[c.wavelength] * 1e-3
            let mid = c.pupil.count / 2
            var out = [Double?](repeating: nil, count: c.pupil.count)
            out[mid] = c.values[mid] == nil ? nil : 0
            for dir in [1, -1] {
                var acc = 0.0, k = mid
                while k + dir >= 0 && k + dir < c.pupil.count {
                    guard let a = c.values[k], let b = c.values[k + dir] else { break }
                    acc += 0.5 * (a + b) * (c.pupil[k + dir] - c.pupil[k])
                    k += dir
                    let w = -scale * acc / lambdaMM
                    out[k] = w.isFinite ? w : nil
                }
            }
            return .init(wavelength: c.wavelength, pupil: c.pupil, values: out)
        }
        let fields = fan.fields.map {
            FanResult.FieldFan(
                field: $0.field, fieldValue: $0.fieldValue, tangential: $0.tangential.map(integrate),
                sagittal: $0.sagittal.map(integrate))
        }
        return FanResult(quantity: .opd, fields: fields, wavelengthsUm: wavelengthsUm)
    }

    // MARK: Field curvature and distortion

    func fieldCurvature(_ s: FieldCurvatureSettings) throws -> FieldCurvatureResult {
        let values = linspace(0, maxField, max(s.fieldSamples, 2))
        let wls = s.wavelengths.indices(in: system.wavelengths)
        let delta = 0.02
        func crossing(_ a: Ray?, _ b: Ray?, axis: Int) -> Double? {
            guard let a, let b, abs(a.d.z) > 1e-12, abs(b.d.z) > 1e-12 else { return nil }
            let sa = (axis == 1 ? a.d.y : a.d.x) / a.d.z, sb = (axis == 1 ? b.d.y : b.d.x) / b.d.z
            let ya = axis == 1 ? a.p.y : a.p.x, yb = axis == 1 ? b.p.y : b.p.x
            guard abs(sa - sb) > 1e-15 else { return nil }
            return -(ya - yb) / (sa - sb)
        }
        let curves = try wls.map { w -> FieldCurvatureResult.Curve in
            var t: [Double?] = [], sg: [Double?] = [], dist: [Double?] = []
            for v in values {
                try Task.checkCancellation()
                let fp = fieldPoint(x: 0, y: v)
                let chief = trace(makeRay(fp, px: 0, py: 0), w: w).atImage
                t.append(crossing(trace(makeRay(fp, px: 0, py: delta), w: w).atImage, trace(makeRay(fp, px: 0, py: -delta), w: w).atImage, axis: 1))
                sg.append(crossing(trace(makeRay(fp, px: delta, py: 0), w: w).atImage, trace(makeRay(fp, px: -delta, py: 0), w: w).atImage, axis: 0))
                let yPar = chiefImagePerUnit * fieldParameterY(fp)
                if let c = chief, abs(yPar) > 1e-12 {
                    dist.append(100 * (c.p.y - yPar) / yPar)
                } else {
                    dist.append(chief == nil ? nil : 0)
                }
            }
            return .init(wavelength: w, tangential: t, sagittal: sg, distortionPercent: dist)
        }
        return FieldCurvatureResult(fieldValues: values, fieldUnit: system.fields.kind.unit, curves: curves, wavelengthsUm: wavelengthsUm)
    }

    // MARK: MTF and PSF (diffraction limit × geometric blur)

    /// RMS spot extent along x and y about the polychromatic centroid (mm).
    func blur(field f: Int, wavelengths: [Int]) -> (sx: Double, sy: Double) {
        let fp = fieldPoint(f)
        var rng = SplitMix64(seed: 0)
        let pupil = Self.pupilSamples(.hexapolar, rings: 6, rng: &rng)
        var pts: [(Double, Double)] = []
        for w in wavelengths {
            for (px, py) in pupil {
                if let r = trace(makeRay(fp, px: px, py: py), w: w).atImage { pts.append((r.p.x, r.p.y)) }
            }
        }
        guard !pts.isEmpty else { return (0, 0) }
        let n = Double(pts.count)
        var mx = CompensatedSum(), my = CompensatedSum()
        for p in pts {
            mx.add(p.0)
            my.add(p.1)
        }
        let cx = mx.value / n, cy = my.value / n
        var vx = CompensatedSum(), vy = CompensatedSum()
        for p in pts {
            vx.add((p.0 - cx) * (p.0 - cx))
            vy.add((p.1 - cy) * (p.1 - cy))
        }
        return ((vx.value / n).squareRoot(), (vy.value / n).squareRoot())
    }

    /// Diffraction-limited MTF of a uniform circular pupil; even in `nu`, so a
    /// negative frequency gives the same value as its magnitude.
    static func diffractionMTF(_ nu: Double, cutoff: Double) -> Double {
        guard cutoff > 0, nu.isFinite, abs(nu) < cutoff else { return 0 }
        let x = abs(nu) / cutoff
        return 2 / .pi * (acos(x) - x * ((1 - x) * (1 + x)).squareRoot())
    }

    func mtf(_ s: MTFSettings) throws -> MTFResult {
        let wls = s.wavelengths.indices(in: system.wavelengths)
        let weights = wls.map { system.wavelengths.wavelengths[$0].weight }
        let wsum = max(weights.reduce(0, +), 1e-300)
        let nw = workingFNumber(primary)
        let freqs = linspace(0, s.maxFrequency, max(s.frequencySamples, 2))
        let cutoffs = wls.map { 1 / (wavelengthsUm[$0] * 1e-3 * nw) }
        let dl = freqs.map { nu in zip(weights, cutoffs).reduce(0) { $0 + $1.0 * Self.diffractionMTF(nu, cutoff: $1.1) } / wsum }
        var curves: [MTFResult.Curve] = []
        let fieldIdx = s.fields.indices(count: system.fields.fields.count)
        for f in fieldIdx {
            try Task.checkCancellation()
            let (sx, sy) = blur(field: f, wavelengths: wls)
            for (o, sigma) in [(MTFResult.Orientation.tangential, sy), (.sagittal, sx)] {
                let g = freqs.map { nu in exp(-2 * .pi * .pi * sigma * sigma * nu * nu) }
                curves.append(.init(field: f, orientation: o, modulation: zip(dl, g).map { $0 * $1 }))
            }
        }
        let cutoff = 1 / (wavelengthsUm[primary] * 1e-3 * nw)
        return MTFResult(
            frequencies: freqs, curves: curves, diffractionLimit: dl, cutoffFrequency: cutoff.isFinite ? cutoff : 0,
            fieldValues: fieldIdx.map(fieldValue))
    }

    func psf(_ s: PSFSettings) throws -> PSFResult {
        let f = min(max(s.field, 0), system.fields.fields.count - 1)
        let wls = s.wavelengths.indices(in: system.wavelengths)
        let weights = wls.map { system.wavelengths.wavelengths[$0].weight }
        let wsum = max(weights.reduce(0, +), 1e-300)
        let nw = workingFNumber(primary)
        let size = max(min(s.imageSampling, 512), 16)
        let (sx, sy) = blur(field: f, wavelengths: wls)
        let airy = 1.22 * wavelengthsUm[primary] * 1e-3 * (nw.isFinite ? nw : 10)
        let half = max(4 * airy, 4 * max(sx, sy))
        let pitch = 2 * half / Double(size)
        var img = [Double](repeating: 0, count: size * size)
        for row in 0..<size {
            if row % 16 == 0 { try Task.checkCancellation() }
            let y = (Double(row) + 0.5) * pitch - half
            for col in 0..<size {
                let x = (Double(col) + 0.5) * pitch - half
                let r = hypot(x, y)
                var acc = 0.0
                for (k, w) in wls.enumerated() {
                    let v = Double.pi * r / (wavelengthsUm[w] * 1e-3 * nw)
                    let a = v < 1e-8 ? 1 : 2 * j1(v) / v
                    acc += weights[k] * a * a
                }
                img[row * size + col] = acc / wsum
            }
        }
        img = Self.gaussianBlur(img, size: size, sigmaX: sx / pitch, sigmaY: sy / pitch)
        let peak = img.max() ?? 0
        return PSFResult(field: f, width: size, height: size, pixelPitch: pitch, values: img.map { Float($0) }, strehl: min(peak, 1))
    }

    static func gaussianBlur(_ img: [Double], size: Int, sigmaX: Double, sigmaY: Double) -> [Double] {
        func kernel(_ s: Double) -> [Double]? {
            guard s > 0.3 else { return nil }
            let r = min(Int((3 * s).rounded(.up)), size)
            let k = (-r...r).map { exp(-0.5 * Double($0 * $0) / (s * s)) }
            let sum = k.reduce(0, +)
            return k.map { $0 / sum }
        }
        var out = img
        if let k = kernel(sigmaX) {
            let r = k.count / 2
            var tmp = out
            for row in 0..<size {
                for col in 0..<size {
                    var acc = 0.0
                    for (j, kv) in k.enumerated() {
                        let c = col + j - r
                        if c >= 0 && c < size { acc += kv * out[row * size + c] }
                    }
                    tmp[row * size + col] = acc
                }
            }
            out = tmp
        }
        if let k = kernel(sigmaY) {
            let r = k.count / 2
            var tmp = out
            for row in 0..<size {
                for col in 0..<size {
                    var acc = 0.0
                    for (j, kv) in k.enumerated() {
                        let rr = row + j - r
                        if rr >= 0 && rr < size { acc += kv * out[rr * size + col] }
                    }
                    tmp[row * size + col] = acc
                }
            }
            out = tmp
        }
        return out
    }

    // MARK: First-order data

    func firstOrder(_ s: FirstOrderSettings) -> FirstOrderResult {
        let w = s.wavelength.indices(in: system.wavelengths).first ?? primary
        let k = system.surfaces.count - 2
        let nk = imageN(w)
        var e: [FirstOrderResult.Entry] = []
        // ±∞ is a legitimate value (an afocal system's EFL); NaN means the
        // quantity is undefined for this system and is left out.
        func add(_ key: String, _ label: String, _ v: Double?, _ unit: String) {
            if let v, !v.isNaN { e.append(.init(key: key, label: label, value: v, unit: unit)) }
        }
        let efl = efl(w)
        add("efl", "Effective focal length", efl ?? .infinity, "mm")
        let a = paraxial(y1: 1, u0: 0, w: w)
        if abs(a.nu[k]) > 1e-15 { add("bfl", "Back focal length", -a.y[k] * nk / a.nu[k], "mm") }
        add("entrance_pupil_diameter", "Entrance pupil diameter", 2 * epRadius, "mm")
        add("ep_position", "Entrance pupil position (from surface 1)", epZ, "mm")
        let (rxp, dxp) = exitPupil()
        add("xp_diameter", "Exit pupil diameter", 2 * rxp, "mm")
        add("xp_distance", "Exit pupil distance (to image)", dxp, "mm")
        if objectDistance == nil, let efl { add("image_f_number", "Image-space F/#", abs(efl) / (2 * epRadius), "") }
        let nw = workingFNumber(w)
        add("working_f_number", "Paraxial working F/#", nw, "")
        add("image_na", "Image-space NA", 1 / (2 * nw), "")
        add("max_field", "Maximum field", maxField, system.fields.kind.unit)
        add("paraxial_image_height", "Paraxial image height", chiefImagePerUnit * fieldParameterY(fieldPoint(x: 0, y: maxField)), "mm")
        add("total_track", "Total track (surface 1 → image)", trackLength, "mm")
        if let d = objectDistance {
            let u0 = epRadius / objectToPupil(d)
            let m = paraxial(y1: u0 * d, u0: u0, w: w)
            let nuk = m.nu[k]
            if abs(nuk) > 1e-15 { add("magnification", "Paraxial magnification", nAfter[0][w] * u0 / nuk, "") }
        }
        // Petzval sum Σ c (n' − n) / (n n') and the Petzval radius −1/(n'·P).
        var petz = CompensatedSum()
        for i in 1..<(system.surfaces.count - 1) {
            let sf = system.surfaces[i]
            guard !sf.isCoordinateBreak, sf.interaction != .dummy else { continue }
            let n1 = nBefore(i, w) * (i > 1 ? travelSign[i - 1] : 1), n2 = nAfter[i][w] * travelSign[i]
            petz.add(sf.shape.curvature * (n2 - n1) / (n1 * n2))
        }
        if abs(petz.value) > 1e-15 { add("petzval_radius", "Petzval radius", -1 / (nk * petz.value), "mm") }
        return FirstOrderResult(entries: e, wavelengthUm: wavelengthsUm[w])
    }
}
