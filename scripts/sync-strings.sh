#!/usr/bin/env bash
# Updates the macOS app's string catalog from its sources.
#
#   scripts/sync-strings.sh
#
# Compiles the Swift package from scratch in release (as shipped; in a temporary build directory,
# so every file is compiled; the compiler extracts the localizable strings
# of every file into .stringsdata, as in Xcode) and syncs the extracted strings into
# apps/macos/Sources/MKLightUI/Resources/Localizable.xcstrings: new strings
# are added in the "new" state, strings no longer in the sources become
# "stale". Translate the new strings and delete the stale ones; the
# localization test fails until every string has a translation in every
# language (docs/APP.md, "Languages").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/apps/macos"
CATALOG="$PKG/Sources/MKLightUI/Resources/Localizable.xcstrings"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> compiling with string extraction"
swift build --package-path "$PKG" -c release --scratch-path "$TMP/build" > "$TMP/build.log" 2>&1 || { cat "$TMP/build.log"; exit 1; }
STRINGSDATA=()
while IFS= read -r f; do STRINGSDATA+=("$f"); done < <(find "$TMP/build" -name '*.stringsdata' -not -path '*Tests*')
[[ "${#STRINGSDATA[@]}" -gt 0 ]] || { echo "error: the build extracted no strings" >&2; exit 1; }
echo "==> syncing ${#STRINGSDATA[@]} files into the catalog"
xcrun xcstringstool sync "$CATALOG" --stringsdata "${STRINGSDATA[@]}"
python3 - "$CATALOG" <<'EOF'
import json, sys
strings = json.load(open(sys.argv[1]))["strings"]
stale = [k for k, v in strings.items() if v.get("extractionState") == "stale"]
untranslated = [k for k, v in strings.items()
                if v.get("shouldTranslate", True) and k not in stale
                and "zh-Hans" not in v.get("localizations", {})]
print(f"    {len(strings)} strings, {len(untranslated)} without a translation, {len(stale)} stale")
EOF
