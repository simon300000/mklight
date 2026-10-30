// swift-tools-version: 6.0
//
// MKLight macOS application.
//
//   MKLightModel  – Codable mirror of the engine's JSON model (no UI, no optics).
//   MKLightEngine – the engine boundary: `EngineClient` protocol, snapshots,
//                   analysis requests/results, async jobs, and `MockEngine`.
//   MKLightUI     – SwiftUI/AppKit document app, lens editor, Metal viewport.
//   MKLight       – the executable (`@main`), a thin shell around MKLightUI.
//
// The Rust engine (crates/mklight-ffi) plugs in later as another
// `EngineClient` implementation; see README.md.

import PackageDescription

let strict: [SwiftSetting] = [
    .enableUpcomingFeature("ExistentialAny"),
]

let package = Package(
    name: "MKLight",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "MKLight", targets: ["MKLight"]),
        .library(name: "MKLightModel", targets: ["MKLightModel"]),
        .library(name: "MKLightEngine", targets: ["MKLightEngine"]),
        .library(name: "MKLightUI", targets: ["MKLightUI"]),
    ],
    targets: [
        .target(name: "MKLightModel", resources: [.copy("Resources/samples.json")], swiftSettings: strict),
        .target(name: "MKLightEngine", dependencies: ["MKLightModel"], swiftSettings: strict),
        .target(
            name: "MKLightUI",
            dependencies: ["MKLightModel", "MKLightEngine"],
            resources: [.copy("Viewport/Shaders/Viewport.metal")],
            swiftSettings: strict
        ),
        .executableTarget(name: "MKLight", dependencies: ["MKLightUI", "MKLightEngine"], swiftSettings: strict),
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
            name: "MKLightUITests",
            dependencies: ["MKLightModel", "MKLightEngine", "MKLightUI"],
            swiftSettings: strict
        ),
    ],
    swiftLanguageModes: [.v6]
)
