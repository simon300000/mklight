// swift-tools-version: 6.0
//
// MKLight macOS application.
//
//   MKLightModel  – Codable mirror of the engine's JSON model (no UI, no optics).
//   MKLightEngine – the engine boundary: `EngineClient` protocol, snapshots,
//                   analysis requests/results, async jobs, and `MockEngine`.
//   CMKLight      – C module over crates/mklight-ffi/include/mklight.h
//                   (system library; links libmklight_ffi.a).
//   MKLightFFI    – `FFIEngine`: `EngineClient` over the Rust engine's C ABI.
//   MKLightUI     – SwiftUI/AppKit document app, lens editor, Metal viewport.
//   MKLight       – the executable (`@main`), a thin shell around MKLightUI.
//
// The Rust engine library is built by `scripts/build-macos.sh` (or
// `scripts/build-macos.sh --engine-only`) into Frameworks/libmklight_ffi.a.
// When that file exists, CMKLight is linked and `FFIEngine` is live;
// otherwise the package still builds, `FFIEngine` reports the library as
// unavailable and its integration tests are skipped. SwiftPM caches the
// evaluated manifest, so the build script clears that cache when the
// library appears or disappears (see README.md).

import Foundation
import PackageDescription

let strict: [SwiftSetting] = [
    .enableUpcomingFeature("ExistentialAny"),
]

let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let engineLibDir = packageDir.appendingPathComponent("Frameworks")
let engineLinked = FileManager.default.fileExists(
    atPath: engineLibDir.appendingPathComponent("libmklight_ffi.a").path)

var ffiDependencies: [Target.Dependency] = ["MKLightModel", "MKLightEngine"]
var engineTargets: [Target] = []
var ffiLinker: [LinkerSetting] = []
if engineLinked {
    engineTargets.append(.systemLibrary(name: "CMKLight", path: "Sources/CMKLight"))
    ffiDependencies.append("CMKLight")
    // Frameworks the engine's GPU backends may use; harmless otherwise.
    ffiLinker = [
        .unsafeFlags(["-L", engineLibDir.path]),
        .linkedFramework("Metal"), .linkedFramework("Foundation"), .linkedFramework("CoreGraphics"),
    ]
}

let package = Package(
    name: "MKLight",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "MKLight", targets: ["MKLight"]),
        .library(name: "MKLightModel", targets: ["MKLightModel"]),
        .library(name: "MKLightEngine", targets: ["MKLightEngine"]),
        .library(name: "MKLightFFI", targets: ["MKLightFFI"]),
        .library(name: "MKLightUI", targets: ["MKLightUI"]),
    ],
    targets: engineTargets + [
        .target(name: "MKLightModel", resources: [.copy("Resources/samples.json")], swiftSettings: strict),
        .target(name: "MKLightEngine", dependencies: ["MKLightModel"], swiftSettings: strict),
        .target(
            name: "MKLightFFI", dependencies: ffiDependencies, swiftSettings: strict, linkerSettings: ffiLinker),
        .target(
            name: "MKLightUI",
            dependencies: ["MKLightModel", "MKLightEngine"],
            resources: [.copy("Viewport/Shaders/Viewport.metal")],
            swiftSettings: strict
        ),
        .executableTarget(
            name: "MKLight", dependencies: ["MKLightUI", "MKLightEngine", "MKLightFFI"], swiftSettings: strict),
        .testTarget(
            name: "MKLightModelTests",
            dependencies: ["MKLightModel"],
            resources: [.copy("Fixtures")],
            swiftSettings: strict
        ),
        .testTarget(
            name: "MKLightEngineTests",
            dependencies: ["MKLightModel", "MKLightEngine"],
            swiftSettings: strict
        ),
        .testTarget(
            name: "MKLightFFITests",
            dependencies: ["MKLightModel", "MKLightEngine", "MKLightFFI"],
            swiftSettings: strict
        ),
        .testTarget(
            name: "MKLightUITests",
            dependencies: ["MKLightModel", "MKLightEngine", "MKLightUI"],
            swiftSettings: strict
        ),
    ],
    swiftLanguageModes: [.v6]
)
