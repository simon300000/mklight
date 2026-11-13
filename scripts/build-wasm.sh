#!/usr/bin/env bash
# Build the MKLight engine for the web (docs/WEB.md): crates/mklight-wasm →
# apps/web/engine/pkg/ (wasm-bindgen `--target web` output, gitignored).
#
#   scripts/build-wasm.sh            release build, wasm-opt -O3, size report
#   scripts/build-wasm.sh --dev      debug build (fast, large, no wasm-opt)
#
# For measurements only (docs/WEB.md "Speed"):
#   --no-simd       without WebAssembly SIMD (simd128 and rustfft's wasm_simd)
#   --no-fft-simd   simd128, but rustfft's scalar algorithms
#   --libm-fma      the compiler's integer `fma` instead of mklight_core::platform::fma
#   OUT=dir         write the package elsewhere than apps/web/engine/pkg
#
# Threads need the wasm `atomics` and `bulk-memory` features and a standard
# library rebuilt with them (`-Z build-std`), hence a pinned nightly. Native
# builds keep using the stable toolchain of rust-toolchain.toml.
#
# Tools: rustup (installs the nightly below on first use), wasm-bindgen-cli
# of the same version as the `wasm-bindgen` crate in Cargo.lock
# (`cargo install wasm-bindgen-cli --version <v> --locked`), and binaryen's
# wasm-opt (`brew install binaryen`) for release builds.
set -euo pipefail

# The pinned nightly for the wasm build. Bump deliberately, together with a
# full run of the web tests (apps/web/engine: npm test).
NIGHTLY="nightly-2026-11-01"
# Shared-memory maximum: 2 GiB. A shared memory's maximum is reserved up
# front by some engines; 4 GiB fails on iOS/iPadOS Safari, 2 GiB does not.
MAX_MEMORY=$((2 * 1024 * 1024 * 1024))
# Each engine and rayon thread's shadow stack (wasm-bindgen's default is
# 1 MiB; the trace recursion stays far below).
STACK_SIZE=$((2 * 1024 * 1024))

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/apps/web/engine/pkg}"
PROFILE=release
SIMD=",+simd128"
FEATURES=()
EXTRA_CFG=""
for a in "$@"; do
  case "$a" in
    --dev) PROFILE=dev ;;
    --no-simd) SIMD=""; FEATURES=(--no-default-features) ;;
    --no-fft-simd) FEATURES=(--no-default-features) ;;
    --libm-fma) EXTRA_CFG=" --cfg mklight_libm_fma" ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

if ! rustup run "$NIGHTLY" rustc --version >/dev/null 2>&1; then
  rustup toolchain install "$NIGHTLY" --profile minimal --component rust-src \
    --target wasm32-unknown-unknown
fi

LOCKED_BINDGEN=$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/version = |"/,""); print; exit}' "$ROOT/Cargo.lock")
HAVE_BINDGEN=$(wasm-bindgen --version 2>/dev/null | awk '{print $2}' || true)
if [[ "$HAVE_BINDGEN" != "$LOCKED_BINDGEN" ]]; then
  echo "wasm-bindgen-cli $LOCKED_BINDGEN is required (found '${HAVE_BINDGEN:-none}'):" >&2
  echo "  cargo install wasm-bindgen-cli --version $LOCKED_BINDGEN --locked" >&2
  exit 1
fi

export CARGO_TARGET_DIR="$ROOT/target/wasm"
# simd128: fixed-width SIMD (Chrome 91+, Safari 16.4+), so LLVM vectorises
# the tracer's packet kernel and rustfft uses its wasm SIMD algorithms.
export RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+nontrapping-fptoint,+sign-ext$SIMD$EXTRA_CFG \
-C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=$MAX_MEMORY \
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base \
-C link-arg=-zstack-size=$STACK_SIZE"
# Release: the workspace profile (thin LTO, one codegen unit) without debug info.
export CARGO_PROFILE_RELEASE_DEBUG=0

cd "$ROOT"
if [[ $PROFILE == release ]]; then
  cargo +"$NIGHTLY" build -p mklight-wasm --release --target wasm32-unknown-unknown \
    -Z build-std=panic_abort,std ${FEATURES[@]+"${FEATURES[@]}"}
  WASM="$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/mklight_wasm.wasm"
else
  cargo +"$NIGHTLY" build -p mklight-wasm --target wasm32-unknown-unknown \
    -Z build-std=panic_abort,std ${FEATURES[@]+"${FEATURES[@]}"}
  WASM="$CARGO_TARGET_DIR/wasm32-unknown-unknown/debug/mklight_wasm.wasm"
fi

rm -rf "$OUT"
wasm-bindgen --target web --out-dir "$OUT" --out-name mklight_wasm "$WASM"

BG="$OUT/mklight_wasm_bg.wasm"
if [[ $PROFILE == release ]]; then
  if ! command -v wasm-opt >/dev/null; then
    echo "wasm-opt not found (brew install binaryen)" >&2
    exit 1
  fi
  wasm-opt -O3 --enable-threads --enable-bulk-memory --enable-mutable-globals \
    --enable-nontrapping-float-to-int --enable-sign-ext ${SIMD:+--enable-simd} \
    "$BG" -o "$BG.opt"
  mv "$BG.opt" "$BG"
fi

# A marker the worker checks: the pkg matches this build's threaded ABI.
echo "{\"profile\": \"$PROFILE\", \"nightly\": \"$NIGHTLY\", \"bindgen\": \"$LOCKED_BINDGEN\"}" > "$OUT/build.json"

raw=$(wc -c < "$BG" | tr -d ' ')
gz=$(gzip -9 -c "$BG" | wc -c | tr -d ' ')
if command -v brotli >/dev/null; then
  br=$(brotli -q 11 -c "$BG" | wc -c | tr -d ' ')
else
  br="n/a (brotli not installed)"
fi
echo "mklight_wasm_bg.wasm: $raw bytes, gzip -9 $gz, brotli -q 11 $br"
