#!/usr/bin/env bash
# Builds the Rust engine library and the MKLight macOS app bundle.
#
#   scripts/build-macos.sh [--debug] [--open] [--universal] [--engine-only] [--no-engine]
#
#   --debug        debug Swift build (the engine library is always release)
#   --open         open the app when done
#   --universal    engine library for arm64 + x86_64 (needs the rustup target
#                  x86_64-apple-darwin); default arm64 only
#   --engine-only  only build the engine library into apps/macos/Frameworks,
#                  so that plain `swift build` / `swift test` link it
#   --no-engine    remove the engine library from apps/macos/Frameworks (the
#                  package then builds without it; debug builds of the app
#                  fall back to MockEngine, the FFI tests are skipped)
#
# Output: build/macos/MKLight.app (ad-hoc signed).
#
# Steps:
#   1. build the Rust engine static library (crates/mklight-ffi, release) and
#      copy it to apps/macos/Frameworks/libmklight_ffi.a, where Package.swift
#      finds it (C module CMKLight over crates/mklight-ffi/include/mklight.h);
#   2. build the Swift package (apps/macos) in release;
#   3. assemble MKLight.app: executable, Info.plist with the .mklight document
#      type and the imported .zmx/.seq lens file types,
#      SwiftPM resource bundles, precompiled viewport shaders when the
#      Metal toolchain is installed (the app otherwise compiles the bundled
#      shader source at run time);
#   4. ad-hoc code signing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/apps/macos"
OUT="$ROOT/build/macos"
APP="$OUT/MKLight.app"
LIB_DIR="$PKG/Frameworks"
LIB="$LIB_DIR/libmklight_ffi.a"
CONFIG=release
OPEN_APP=0
UNIVERSAL=0
ENGINE_ONLY=0
NO_ENGINE=0
for arg in "$@"; do
  case "$arg" in
    --debug) CONFIG=debug ;;
    --open) OPEN_APP=1 ;;
    --universal) UNIVERSAL=1 ;;
    --engine-only) ENGINE_ONLY=1 ;;
    --no-engine) NO_ENGINE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

VERSION="0.1.0"
BUILD_NUMBER="$(git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo 1)"

# SwiftPM caches the evaluated Package.swift by its contents, so it does not
# notice the engine library appearing or disappearing. Clear that cache
# whenever the library's presence changes.
refresh_manifest_cache() {
  echo "    engine library presence changed: clearing the SwiftPM manifest cache"
  swift package --package-path "$PKG" purge-cache >/dev/null
}

# ---------------------------------------------------------------------------
# 1. Rust engine library.
if [[ "$NO_ENGINE" == "1" ]]; then
  if [[ -e "$LIB" ]]; then
    rm -f "$LIB"
    refresh_manifest_cache
  fi
  echo "==> engine library removed; the package builds without it"
  [[ "$ENGINE_ONLY" == "1" ]] && exit 0
else
  if ! command -v cargo >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    [[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
  fi
  command -v cargo >/dev/null 2>&1 || { echo "error: cargo not found (install Rust via rustup)" >&2; exit 1; }
  TARGETS=(aarch64-apple-darwin)
  [[ "$UNIVERSAL" == "1" ]] && TARGETS+=(x86_64-apple-darwin)
  LIBS=()
  for t in "${TARGETS[@]}"; do
    if ! rustup target list --installed 2>/dev/null | grep -qx "$t"; then
      echo "error: Rust target $t is not installed (rustup target add $t)" >&2
      exit 1
    fi
    echo "==> cargo build --release -p mklight-ffi --target $t"
    (cd "$ROOT" && MACOSX_DEPLOYMENT_TARGET=27.0 cargo build --release -p mklight-ffi --target "$t")
    LIBS+=("$ROOT/target/$t/release/libmklight_ffi.a")
  done
  mkdir -p "$LIB_DIR"
  HAD_LIB=0
  [[ -e "$LIB" ]] && HAD_LIB=1
  TMP_LIB="$LIB_DIR/.libmklight_ffi.a.tmp"
  if [[ "${#LIBS[@]}" -gt 1 ]]; then
    lipo -create "${LIBS[@]}" -output "$TMP_LIB"
  else
    cp "${LIBS[0]}" "$TMP_LIB"
  fi
  # Replace only when changed, so SwiftPM does not relink needlessly.
  if [[ "$HAD_LIB" == "1" ]] && cmp -s "$TMP_LIB" "$LIB"; then
    rm -f "$TMP_LIB"
  else
    mv -f "$TMP_LIB" "$LIB"
  fi
  [[ "$HAD_LIB" == "0" ]] && refresh_manifest_cache
  echo "    $LIB ($(lipo -archs "$LIB"))"
  if [[ "$ENGINE_ONLY" == "1" ]]; then
    echo "==> engine library ready: swift build / swift test in apps/macos now link it"
    exit 0
  fi
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
if xcrun metal -c -target air64-apple-macos27.0 -fmetal-math-mode=safe "$SHADER" -o "$TMP/Viewport.air" 2>/dev/null \
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
  <key>LSMinimumSystemVersion</key><string>27.0</string>
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
    <dict>
      <key>CFBundleTypeName</key><string>Lens Prescription (.zmx)</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>LSItemContentTypes</key><array><string>com.mklight.zmx-prescription</string></array>
    </dict>
    <dict>
      <key>CFBundleTypeName</key><string>Lens Prescription (.seq)</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>LSItemContentTypes</key><array><string>com.mklight.seq-prescription</string></array>
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
  <key>UTImportedTypeDeclarations</key>
  <array>
    <dict>
      <key>UTTypeIdentifier</key><string>com.mklight.zmx-prescription</string>
      <key>UTTypeDescription</key><string>Lens Prescription (.zmx)</string>
      <key>UTTypeConformsTo</key><array><string>public.data</string></array>
      <key>UTTypeTagSpecification</key>
      <dict><key>public.filename-extension</key><array><string>zmx</string></array></dict>
    </dict>
    <dict>
      <key>UTTypeIdentifier</key><string>com.mklight.seq-prescription</string>
      <key>UTTypeDescription</key><string>Lens Prescription (.seq)</string>
      <key>UTTypeConformsTo</key><array><string>public.data</string></array>
      <key>UTTypeTagSpecification</key>
      <dict><key>public.filename-extension</key><array><string>seq</string></array></dict>
    </dict>
  </array>
</dict>
</plist>
PLIST
plutil -lint "$APP/Contents/Info.plist" >/dev/null

# App icon: apps/macos/Resources/AppIcon.icns, built from AppIcon.svg with
# rsvg-convert (16-512 pt at 1x and 2x into AppIcon.iconset) and
# `iconutil -c icns`.
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
