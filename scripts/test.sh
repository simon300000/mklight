#!/usr/bin/env bash
# Runs the workspace's Rust tests with the test programs side by side.
#
#   scripts/test.sh [cargo test selection...] [-- test program arguments...]
#
#   scripts/test.sh                          every test of the workspace
#   scripts/test.sh -p mklight-trace         one crate (any `cargo test`
#                                            selection: -p, --test, --lib, ...)
#   scripts/test.sh -- --ignored             arguments for every test program
#   JOBS=4 scripts/test.sh                   programs at a time (default 8)
#
# `cargo test` runs the workspace's test programs (about 30: per crate the
# unit tests and one program for `tests/`) one after another. Run together
# they finish in about two minutes on an M2 Pro instead of four and a
# half; more than 8 at a time gains nothing, the tests already use every
# core. This script
#
#   1. builds the test programs with cargo (compile errors print as usual);
#   2. runs them JOBS at a time, each in its package directory as cargo
#      does, with the output of each in a log file. The programs that took
#      longest in the previous run start first (times are kept in the
#      target directory), and the doc tests run through cargo at the same
#      time (skipped when test program arguments are given);
#   3. runs every program that failed once more on its own: a test that
#      fails only because the machine is busy passes there, any other
#      failure is printed and fails the run.
#
# The tests inside one program still run on threads, as with `cargo test`.
# Exit status: 0 when everything passed, 1 otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
JOBS="${JOBS:-8}"

select=()
program_args=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        program_args=("$@")
        break
    fi
    select+=("$1")
    shift
done
[[ ${#select[@]} -gt 0 ]] || select=(--workspace)

LOGS="$(mktemp -d "${TMPDIR:-/tmp}/mklight-test.XXXXXX")"
started=$SECONDS

# 1. Build. The second call is up to date and only lists the programs: one
#    JSON line per artifact, with its package manifest and executable.
cargo test "${select[@]}" --no-run
cargo test "${select[@]}" --no-run --message-format=json 2>/dev/null \
    | grep '"reason":"compiler-artifact"' \
    | grep -E '"profile":\{[^}]*"test":true' \
    | sed -n 's/.*"manifest_path":"\([^"]*\)\/Cargo\.toml".*"executable":"\([^"]*\)".*/\1	\2/p' \
    > "$LOGS/programs"
count=$(wc -l < "$LOGS/programs" | tr -d ' ')
[[ "$count" -gt 0 ]] || { echo "no test programs selected" >&2; exit 1; }
built=$((SECONDS - started))

# Longest first, by the seconds of the previous run (unknown programs
# first of all). A program's key is its package and its name without the
# build hash.
TIMES="$(dirname "$(dirname "$(head -1 "$LOGS/programs" | cut -f2)")")/mklight-test-times"
touch "$TIMES"
awk -F '\t' '
    function key(dir, exe,    d, e) {
        d = dir; sub(/.*\//, "", d)
        e = exe; sub(/.*\//, "", e); sub(/-[0-9a-f]+$/, "", e)
        return d "/" e
    }
    FILENAME == ARGV[1] { seconds[$1] = $2; next }
    { k = key($1, $2); print (k in seconds ? seconds[k] : 1000000) "\t" k "\t" $1 "\t" $2 }
' "$TIMES" "$LOGS/programs" | sort -t "$(printf '\t')" -k1,1nr -k2,2 | cut -f2- > "$LOGS/ordered"

# One program, as cargo runs it: in its package directory, with the
# libraries of the target directory on the loader path. The test program
# arguments are read from a file, one per line.
: > "$LOGS/args"
for a in ${program_args[@]+"${program_args[@]}"}; do
    printf '%s\n' "$a" >> "$LOGS/args"
done
run_program() {
    local dir="$1" exe="$2" deps a
    local args=()
    while IFS= read -r a; do
        args+=("$a")
    done < "$LOGS/args"
    deps="$(dirname "$exe")"
    cd "$dir"
    CARGO_MANIFEST_DIR="$dir" \
        DYLD_FALLBACK_LIBRARY_PATH="$deps:$(dirname "$deps")${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}" \
        "$exe" ${args[@]+"${args[@]}"}
}
export -f run_program
export LOGS

# 2. Side by side, with the doc tests next to them.
echo "running $count test programs, $JOBS at a time (logs in $LOGS)"
doc_pid=
if [[ ${#program_args[@]} -eq 0 ]]; then
    cargo test "${select[@]}" --doc > "$LOGS/doc.log" 2>&1 &
    doc_pid=$!
fi
tr '\t' '\n' < "$LOGS/ordered" \
    | xargs -P "$JOBS" -n 3 bash -c '
        if ! run_program "$2" "$3" > "$LOGS/$(basename "$3").log" 2>&1; then
            printf "%s\t%s\n" "$2" "$3" >> "$LOGS/failed"
        fi
        printf "%s\t%s\n" "$1" "$SECONDS" >> "$LOGS/times"' _ \
    || true
# New times first, then those of programs that did not run this time.
cat "$LOGS/times" "$TIMES" | awk -F '\t' '!seen[$1]++' > "$LOGS/times.all"
cp "$LOGS/times.all" "$TIMES"

# 3. Failed programs again, one at a time.
status=0
if [[ -s "$LOGS/failed" ]]; then
    while IFS=$'\t' read -r dir exe; do
        name="$(basename "$exe")"
        echo "again, on its own: ${dir#"$ROOT"/} $name"
        mv "$LOGS/$name.log" "$LOGS/$name.first"
        grep -E '^test .* FAILED$' "$LOGS/$name.first" | head -20 | sed 's/^/    /'
        if (run_program "$dir" "$exe") > "$LOGS/$name.log" 2>&1; then
            echo "  passed (it failed only while other programs ran)"
        else
            status=1
            echo "  FAILED: ${dir#"$ROOT"/} $name"
            grep -E '^test .* FAILED$|panicked at|^error' "$LOGS/$name.log" | head -40 | sed 's/^/    /'
            echo "    full output: $LOGS/$name.log"
        fi
    done < "$LOGS/failed"
fi

if [[ -n "$doc_pid" ]]; then
    if ! wait "$doc_pid"; then
        if grep -q 'test result: FAILED' "$LOGS/doc.log"; then
            status=1
            echo "FAILED: doc tests"
            grep -E '^test .* FAILED$|panicked at' "$LOGS/doc.log" | head -40 | sed 's/^/    /'
            echo "    full output: $LOGS/doc.log"
        else
            # A selection of single targets has no doc tests to run.
            echo "doc tests not run: $(grep -m1 '^error' "$LOGS/doc.log" || tail -1 "$LOGS/doc.log")"
        fi
    fi
fi

totals=$(cat "$LOGS"/*.log | awk '
    /^test result:/ {
        for (i = 1; i <= NF; i++) {
            if ($i == "passed;") p += $(i - 1)
            if ($i == "failed;") f += $(i - 1)
            if ($i == "ignored;") g += $(i - 1)
        }
    }
    END { printf "%d passed, %d failed, %d ignored", p, f, g }')
echo "$totals in $count programs; build ${built} s, tests $((SECONDS - started - built)) s"
exit $status
