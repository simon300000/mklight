import Foundation

/// Sample systems for new documents and tests.
public enum SampleSystems {
    /// One of the engine's built-in samples (`mklight_core::samples`).
    public struct Entry: Codable, Sendable, Hashable, Identifiable {
        /// Short identifier, e.g. `achromat`.
        public var id: String
        public var system: OpticalSystem
        public init(id: String, system: OpticalSystem) {
            self.id = id
            self.system = system
        }
    }

    /// The engine's built-in samples, shipped as `Resources/samples.json`
    /// (written by `cargo run -p mklight-core --example dump_fixtures`). They
    /// validate in the engine and use only built-in glasses.
    public static let bundled: [Entry] = {
        guard let url = Bundle.module.url(forResource: "samples", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let entries = try? MKLightJSON.decode([Entry].self, from: data)
        else { return [] }
        return entries
    }()

    /// A bundled sample by identifier.
    public static func bundled(_ id: String) -> OpticalSystem? { bundled.first { $0.id == id }?.system }

    /// The classic Cooke triplet (F/4.7, EFL ≈ 47.1 mm, ±20° field). Identical to
    /// the `cooke_triplet.json` engine fixture.
    public static func cookeTriplet() -> OpticalSystem {
        func s(_ id: UInt64, _ r: Double, _ t: Double, _ m: MaterialRef, coated: Bool = true) -> Surface {
            Surface(
                id: SurfaceID(id), shape: .conic(curvature: RadiusDisplay.curvature(fromRadius: r), conic: 0),
                thickness: t, material: m, coating: coated ? "AR" : nil)
        }
        let surfaces = [
            Surface(id: SurfaceID(0), shape: .plane, thickness: .infinity),
            s(1, 22.01359, 3.25896, .glass("N-SK16")),
            s(2, -435.76044, 6.00755, .air),
            s(3, -22.21328, 0.99997, .glass("F2")),
            s(4, 20.29192, 4.75041, .air),
            Surface(id: SurfaceID(5), label: "STOP", shape: .plane, thickness: 2.0),
            s(6, 79.68360, 2.95208, .glass("N-SK16")),
            s(7, -18.39533, 41.36144, .air),
            Surface(id: SurfaceID(8), shape: .plane, thickness: 0),
        ]
        return OpticalSystem(
            name: "Cooke triplet",
            notes: "Classic three-element anastigmat, EFL ≈ 47.1 mm, F/4.7, image at the paraxial focus.",
            aperture: .entrancePupilDiameter(10),
            fields: FieldSet(kind: .objectAngle, fields: [Field(y: 0), Field(y: 14), Field(y: 20)]),
            surfaces: surfaces,
            stop: 5,
            coatings: ["AR": .ideal(reflectance: 0.005, transmittance: 0.995)])
    }

    /// A biconvex N-BK7 singlet, f ≈ 100 mm, object at infinity, stop on the
    /// first surface.
    public static func singlet() -> OpticalSystem {
        OpticalSystem(
            name: "Singlet",
            aperture: .entrancePupilDiameter(20),
            fields: FieldSet(kind: .objectAngle, fields: [Field(y: 0), Field(y: 3.5), Field(y: 5)]),
            surfaces: [
                Surface(id: SurfaceID(0), shape: .plane, thickness: .infinity),
                Surface(id: SurfaceID(1), label: "STOP", shape: .conic(curvature: 1 / 103.4, conic: 0), thickness: 5, material: .glass("N-BK7")),
                Surface(id: SurfaceID(2), shape: .conic(curvature: -1 / 103.4, conic: 0), thickness: 97.5),
                Surface(id: SurfaceID(3), shape: .plane, thickness: 0),
            ],
            stop: 1)
    }
}
