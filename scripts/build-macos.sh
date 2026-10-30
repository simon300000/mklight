#!/usr/bin/env bash
# Builds the MKLight macOS app bundle.
#
#   scripts/build-macos.sh [--debug] [--open]
#
# Output: build/macos/MKLight.app (ad-hoc signed).
#
# Steps:
#   1. (hook) build the Rust engine static library for the C ABI;
#   2. build the Swift package (apps/macos) in release;
#   3. assemble MKLight.app: executable, Info.plist with the .mklight document
#      type, SwiftPM resource bundles, precompiled viewport shaders when the
#      Metal toolchain is installed (the app otherwise compiles the bundled
#      shader source at run time);
#   4. ad-hoc code signing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/apps/macos"
OUT="$ROOT/build/macos"
APP="$OUT/MKLight.app"
CONFIG=release
OPEN_APP=0
for arg in "$@"; do
  case "$arg" in
    --debug) CONFIG=debug ;;
    --open) OPEN_APP=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

VERSION="0.1.0"
BUILD_NUMBER="$(git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo 1)"

# ---------------------------------------------------------------------------
# 1. Rust engine (HOOK — not active yet).
#
# When crates/mklight-ffi exists, build its static library here and link it
# into the Swift package, e.g.:
#
#   (cd "$ROOT" && cargo build --release -p mklight-ffi --target aarch64-apple-darwin)
#   FFI_LIB="$ROOT/target/aarch64-apple-darwin/release/libmklight_ffi.a"
#   FFI_HEADER_DIR="$ROOT/crates/mklight-ffi/include"   # cbindgen output
#
# and expose them to SwiftPM through a system-library / binary target
# (module map over mklight.h) plus linker flags such as
#   -Xlinker "$FFI_LIB" -Xlinker -framework -Xlinker Metal
# The Swift side then gets an `FFIEngine: EngineClient` in a new MKLightFFI
# target, and Sources/MKLight/main.swift switches from MockEngine to it.
# Set MKLIGHT_WITH_FFI=1 once that exists.
if [[ "${MKLIGHT_WITH_FFI:-0}" == "1" ]]; then
  echo "error: MKLIGHT_WITH_FFI=1 but the Rust FFI hook is not implemented yet" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Swift package.
echo "==> swift build -c $CONFIG"
swift build --package-path "$PKG" -c "$CONFIG" --product MKLight
BIN_DIR="$(swift build --package-path "$PKG" -c "$CONFIG" --show-bin-path)"

# ---------------------------------------------------------------------------
# 3. Bundle.
echo "==> assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/MKLight" "$APP/Contents/MacOS/MKLight"
for bundle in "$BIN_DIR"/*.bundle; do
  [[ -e "$bundle" ]] || continue
  case "$(basename "$bundle")" in
    *Tests.bundle) continue ;;  # test fixtures do not ship
  esac
  cp -R "$bundle" "$APP/Contents/Resources/"
done

# Precompiled shaders (optional): needs the Metal toolchain component
# (`xcodebuild -downloadComponent MetalToolchain`). Compiled with IEEE-safe
# math, matching the runtime-compiled fallback.
SHADER="$PKG/Sources/MKLightUI/Viewport/Shaders/Viewport.metal"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if xcrun metal -c -target air64-apple-macos15.0 -fmetal-math-mode=safe "$SHADER" -o "$TMP/Viewport.air" 2>/dev/null \
   && xcrun metallib "$TMP/Viewport.air" -o "$APP/Contents/Resources/MKLightViewport.metallib" 2>/dev/null; then
  echo "    precompiled MKLightViewport.metallib"
else
  echo "    Metal toolchain not installed: shaders will be compiled at run time"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>MKLight</string>
  <key>CFBundleIdentifier</key><string>com.mklight.MKLight</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>MKLight</string>
  <key>CFBundleDisplayName</key><string>MKLight</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${BUILD_NUMBER}</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSSupportsAutomaticTermination</key><false/>
  <key>NSSupportsSuddenTermination</key><false/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>MKLight Project</string>
      <key>CFBundleTypeRole</key><string>Editor</string>
      <key>LSHandlerRank</key><string>Owner</string>
      <key>LSItemContentTypes</key><array><string>com.mklight.project</string></array>
    </dict>
    <dict>
      <key>CFBundleTypeName</key><string>JSON Optical System</string>
      <key>CFBundleTypeRole</key><string>Editor</string>
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>LSItemContentTypes</key><array><string>public.json</string></array>
    </dict>
  </array>
  <key>UTExportedTypeDeclarations</key>
  <array>
    <dict>
      <key>UTTypeIdentifier</key><string>com.mklight.project</string>
      <key>UTTypeDescription</key><string>MKLight Project</string>
      <key>UTTypeConformsTo</key><array><string>public.json</string></array>
      <key>UTTypeTagSpecification</key>
      <dict>
        <key>public.filename-extension</key><array><string>mklight</string></array>
        <key>public.mime-type</key><array><string>application/vnd.mklight.project+json</string></array>
      </dict>
    </dict>
  </array>
</dict>
</plist>
PLIST
plutil -lint "$APP/Contents/Info.plist" >/dev/null

# Icon placeholder: drop an AppIcon.icns into apps/macos/Resources to use it.
if [[ -f "$PKG/Resources/AppIcon.icns" ]]; then
  cp "$PKG/Resources/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
  /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string AppIcon" "$APP/Contents/Info.plist"
fi

# ---------------------------------------------------------------------------
# 4. Signing (ad hoc; replace "-" with a Developer ID for distribution).
codesign --force --sign - --timestamp=none "$APP" >/dev/null
codesign --verify --strict "$APP"
echo "==> built $APP"

if [[ "$OPEN_APP" == "1" ]]; then
  open "$APP"
fi
