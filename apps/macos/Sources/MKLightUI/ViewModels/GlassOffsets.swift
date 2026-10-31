import Foundation
import MKLightEngine
import MKLightModel

// Index and Abbe offsets of catalogue glasses (docs/MATERIALS.md,
// "Catalogue index and Abbe offsets"): a melt's Δn_d and ΔV_d on top of the
// catalogue glass, shown subtly in the lens table, edited in the
// inspector, marked on the glass map and dropped by glass substitution.

/// A catalogue glass's d-line numbers from the engine's `glass_info`: with
/// its offset (`nd`, `vd`) and as catalogued.
public struct GlassNumbers: Sendable, Hashable {
    public var catalog: String?
    public var nd: Double
    public var vd: Double
    public var catalogNd: Double
    public var catalogVd: Double

    public init(catalog: String? = nil, nd: Double, vd: Double, catalogNd: Double, catalogVd: Double) {
        self.catalog = catalog
        self.nd = nd
        self.vd = vd
        self.catalogNd = catalogNd
        self.catalogVd = catalogVd
    }

    public init?(json: JSONValue) {
        guard let nd = json["nd"]?.nf, let vd = json["vd"]?.nf, nd.isFinite, vd.isFinite else { return nil }
        self.nd = nd
        self.vd = vd
        catalogNd = json["catalog_nd"]?.nf ?? nd
        catalogVd = json["catalog_vd"]?.nf ?? vd
        catalog = json["catalog"]?.string
    }
}

extension Workspace {
    /// V_d of the catalogue glass after each surface, offset included, as
    /// far as the engine has given them (the rest are asked for).
    public func abbeNumbers(of system: OpticalSystem) -> [SurfaceID: Double] {
        var out: [SurfaceID: Double] = [:]
        for s in system.surfaces {
            if case .catalog = s.material, let g = glass(s.material) { out[s.id] = g.vd }
        }
        return out
    }
}

extension MaterialOffset {
    /// The table's subtle suffix: `+0.0012/−0.8` (`""` when zero); a zero
    /// component reads `0`.
    public var compactText: String {
        guard !isZero else { return "" }
        return Self.signed(dnD, decimals: 4) + "/" + Self.signed(dvD, decimals: 2)
    }

    /// `Δn_d +0.0012, ΔV_d −0.8`.
    public var longText: String {
        "Δn_d \(Self.signed(dnD, decimals: 5)), ΔV_d \(Self.signed(dvD, decimals: 2))"
    }

    /// A signed number with a true minus and no trailing zeros; `0` for zero.
    static func signed(_ v: Double, decimals: Int) -> String {
        guard v != 0 else { return "0" }
        var s = String(format: "%.\(decimals)f", abs(v))
        if s.contains(".") {
            while s.hasSuffix("0") { s.removeLast() }
            if s.hasSuffix(".") { s.removeLast() }
        }
        return (v < 0 ? "\u{2212}" : "+") + s
    }

    /// Why `v` cannot be an offset component (`|Δn_d| ≤ 0.05`,
    /// `|ΔV_d| ≤ 10`, the engine's validation), if it cannot.
    public static func rejection(dnD v: Double) -> String? {
        v.isFinite && abs(v) <= maxDnD ? nil : "Δn_d must lie within ±\(Fmt.short(maxDnD))"
    }

    public static func rejection(dvD v: Double) -> String? {
        v.isFinite && abs(v) <= maxDvD ? nil : "ΔV_d must lie within ±\(Fmt.short(maxDvD))"
    }
}

extension MaterialRef {
    /// The catalogue offset, if the material is a catalogue glass.
    public var catalogOffset: MaterialOffset? {
        if case .catalog(_, _, let o) = self { return o }
        return nil
    }

    /// The same catalogue glass with offset `o`; `nil` for other materials.
    public func withOffset(_ o: MaterialOffset) -> MaterialRef? {
        guard case .catalog(let cat, let name, _) = self else { return nil }
        return .catalog(catalog: cat, name: name, offset: o)
    }

    /// `new` typed over this material: the same catalogue glass (by name,
    /// in the same or an unnamed catalogue) keeps its melt offset; any other
    /// glass starts without one, since the offset belonged to this melt.
    public func replaced(by new: MaterialRef) -> MaterialRef {
        guard case .catalog(let cat, let name, let o) = self, !o.isZero, case .catalog(let nc, let nn, let no) = new, no.isZero,
            nn.uppercased() == name.uppercased(), nc == nil || cat == nil || nc?.uppercased() == cat?.uppercased()
        else { return new }
        return .catalog(catalog: nc ?? cat, name: nn, offset: o)
    }
}

extension LensValueParser {
    /// A Δn_d, within the engine's ±0.05.
    public static func offsetDnD(_ text: String) -> Result<Double, Failure> {
        finite(text).flatMap { v in MaterialOffset.rejection(dnD: v).map { .failure(.outOfRange($0)) } ?? .success(v) }
    }

    /// A ΔV_d, within the engine's ±10.
    public static func offsetDvD(_ text: String) -> Result<Double, Failure> {
        finite(text).flatMap { v in MaterialOffset.rejection(dvD: v).map { .failure(.outOfRange($0)) } ?? .success(v) }
    }
}

extension GlassSubstitutionSession {
    /// The glasses that may change and carry a melt offset, with a note:
    /// a replacement glass is installed without the offset (even the same
    /// glass), and offset variables on those surfaces are left out of the
    /// re-optimisation. `nil` when no ticked glass has an offset.
    public func offsetNote(system: OpticalSystem) -> String? {
        let slots = GlassSlot.all(in: system).filter { selected.contains($0.id) && !($0.offset?.isZero ?? true) }
        guard !slots.isEmpty else { return nil }
        let names = slots.map { "S\($0.index) \($0.material) \($0.offset?.compactText ?? "")" }
        let list = ListFormatter.localizedString(byJoining: names)
        let one = slots.count == 1
        return "\(list) \(one ? "carries a melt offset" : "carry melt offsets"). A replacement glass is installed without "
            + "\(one ? "it" : "them"), even the same glass: the offset belonged to the original melt."
    }

    /// Material parameters substitution may not vary: the offsets and model
    /// numbers of the glasses it replaces (the engine refuses them).
    func isSubstitutedMaterialParam(_ p: ParamPath) -> Bool {
        switch p {
        case .materialDnD(let id), .materialDvD(let id), .modelNd(let id), .modelVd(let id), .modelDpgf(let id): selected.contains(id)
        default: false
        }
    }
}
