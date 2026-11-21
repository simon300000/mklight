#!/usr/bin/env bash
# Checks the engine translation catalogs (locales/<id>/engine.json) against
# the text the engine emits, and lists what is missing or stale.
#
#   scripts/i18n-check.sh
#
# Runs the `i18n` test of mklight-engine (crates/mklight-engine/tests/i18n.rs).
# On failure it prints, per locale, the missing entries as JSON to paste into
# the catalog (with where each string comes from) and the stale ones to
# remove. Either way it writes every English string, enum value and
# diagnostic code with its sources to target/i18n/engine-en.json, the
# starting point for a new language (docs/I18N.md).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"

status=0
cargo test -p mklight-engine --test tests i18n:: -- --nocapture || status=$?
mkdir -p "$TARGET/i18n"
if [[ -f "$TARGET/tmp/engine-en.json" ]]; then
    cp "$TARGET/tmp/engine-en.json" "$TARGET/i18n/engine-en.json"
    echo "English engine text: $TARGET/i18n/engine-en.json"
fi
exit $status
