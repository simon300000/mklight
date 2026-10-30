import Foundation

/// Structural edits of the surface list. They keep the stop index and surface
/// identities consistent; they never compute optics.
extension OpticalSystem {
    /// Inserts a plane air surface at `index` (1…image index) and returns its
    /// id. Surfaces at and after `index` move down by one.
    @discardableResult
    public mutating func insertSurface(at index: Int, thickness: Double = 0) -> SurfaceID {
        let i = min(max(index, 1), imageIndex)
        let id = nextSurfaceID
        surfaces.insert(Surface(id: id, shape: .plane, thickness: thickness, material: .air), at: i)
        if i <= stop { stop += 1 }
        return id
    }

    /// Removes the surface at `index`. The object, the image and the last
    /// remaining surface between them cannot be removed. Deleting the stop
    /// keeps the stop at the same position (clamped). Configuration overrides,
    /// solves and mechanical parts that refer to the removed surface are
    /// removed too.
    @discardableResult
    public mutating func deleteSurface(at index: Int) -> Bool {
        guard index > 0, index < imageIndex, surfaces.count > 3 else { return false }
        let removed = surfaces.remove(at: index)
        if index < stop { stop -= 1 }
        stop = min(max(stop, 1), imageIndex - 1)
        for k in configurations.indices {
            configurations[k].overrides.removeAll { $0.first.surface == removed.id }
        }
        mechanics.removeAll { $0.surfaces.contains(removed.id) }
        solves.removeAll { $0.surfaces.contains(removed.id) }
        return true
    }

    /// Removes field `index` (at least one field remains). Overrides of that
    /// field are dropped and those of later fields renumbered, so every
    /// index-addressed `ParamPath` keeps naming the same field.
    @discardableResult
    public mutating func removeField(at index: Int) -> Bool {
        guard fields.fields.indices.contains(index), fields.fields.count > 1 else { return false }
        fields.fields.remove(at: index)
        remapIndexedOverrides(removed: index) { $0.fieldIndex }
        return true
    }

    /// Removes wavelength `index` (at least one remains), keeping the primary
    /// wavelength and index-addressed overrides pointing at the same entries.
    @discardableResult
    public mutating func removeWavelength(at index: Int) -> Bool {
        guard wavelengths.wavelengths.indices.contains(index), wavelengths.wavelengths.count > 1 else { return false }
        wavelengths.wavelengths.remove(at: index)
        if wavelengths.primary > index || (wavelengths.primary == index && index == wavelengths.wavelengths.count) {
            wavelengths.primary -= 1
        }
        remapIndexedOverrides(removed: index) { $0.wavelengthIndex }
        return true
    }

    private mutating func remapIndexedOverrides(removed: Int, index: (ParamPath) -> Int?) {
        for k in configurations.indices {
            configurations[k].overrides = configurations[k].overrides.compactMap { o in
                guard let i = index(o.first) else { return o }
                if i == removed { return nil }
                return i > removed ? Pair(o.first.withIndex(i - 1), o.second) : o
            }
        }
        // Solves whose target or pickup source names the removed entry go;
        // later entries are renumbered.
        func remap(_ p: ParamPath) -> ParamPath? {
            guard let i = index(p) else { return p }
            if i == removed { return nil }
            return i > removed ? p.withIndex(i - 1) : p
        }
        solves = solves.compactMap { s in
            guard let target = remap(s.target) else { return nil }
            var out = Solve(target: target, kind: s.kind)
            if case .pickup(let source, let scale, let offset) = s.kind {
                guard let src = remap(source) else { return nil }
                out.kind = .pickup(source: src, scale: scale, offset: offset)
            }
            return out
        }
    }

    /// Drops configuration overrides, solves and mechanical parts that resolved in
    /// `previous` but no longer resolve here. Call after an edit (shape,
    /// material, semi-diameter or aperture changes can invalidate them); the
    /// edit stays undoable as a whole. References that were already broken in
    /// `previous` are kept, so `structuralIssues()` still reports them.
    public mutating func dropReferencesInvalidated(since previous: OpticalSystem) {
        let current = self
        for k in configurations.indices {
            configurations[k].overrides.removeAll { o in
                o.first.unresolvedReason(in: current) != nil && o.first.unresolvedReason(in: previous) == nil
            }
        }
        func broken(_ s: Solve, in sys: OpticalSystem) -> Bool {
            let fixesSemiDiameter = if case .semiDiameter = s.target { true } else { false }
            if !(fixesSemiDiameter && s.target.surface.flatMap(sys.surfaceIndex(of:)) != nil),
                s.target.unresolvedReason(in: sys) != nil
            {
                return true
            }
            if case .pickup(let source, _, _) = s.kind, sys.solve(for: source) == nil,
                source.unresolvedReason(in: sys) != nil
            {
                return true
            }
            return s.surfaces.contains { sys.surfaceIndex(of: $0) == nil }
        }
        solves.removeAll { broken($0, in: current) && !broken($0, in: previous) }
        let ids = Set(surfaces.map(\.id)), before = Set(previous.surfaces.map(\.id))
        mechanics.removeAll { m in
            m.surfaces.contains { !ids.contains($0) } && m.surfaces.allSatisfy(before.contains)
        }
    }

    /// Makes `index` the aperture stop if it is between object and image.
    public mutating func setStop(_ index: Int) {
        guard index > 0, index < imageIndex else { return }
        stop = index
    }
}

extension ParamPath {
    /// Field index of a field-addressed path.
    public var fieldIndex: Int? {
        switch self {
        case .fieldX(let i), .fieldY(let i), .fieldWeight(let i): i
        default: nil
        }
    }

    /// Wavelength index of a wavelength-addressed path.
    public var wavelengthIndex: Int? {
        switch self {
        case .wavelengthUm(let i), .wavelengthWeight(let i): i
        default: nil
        }
    }

    /// The same parameter for another field or wavelength index.
    func withIndex(_ i: Int) -> ParamPath {
        switch self {
        case .fieldX: .fieldX(index: i)
        case .fieldY: .fieldY(index: i)
        case .fieldWeight: .fieldWeight(index: i)
        case .wavelengthUm: .wavelengthUm(index: i)
        case .wavelengthWeight: .wavelengthWeight(index: i)
        default: self
        }
    }
}

extension MechanicalPart {
    /// Surfaces the part is attached to.
    public var surfaces: [SurfaceID] {
        switch self {
        case .barrel(let from, let to, _, _): [from, to]
        case .annularBaffle(let s, _, _, _, _): [s]
        }
    }
}
