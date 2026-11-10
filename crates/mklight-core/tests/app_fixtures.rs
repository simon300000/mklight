//! The macOS app's fixtures and bundled samples are exactly what
//! `examples/dump_fixtures` writes, so the app's mirror of the model and of
//! its validation rules is tested against the engine as it is.

use std::collections::BTreeSet;

// `main` writes the files; this test reads what it would write.
#[allow(dead_code)]
#[path = "../examples/dump_fixtures.rs"]
mod generator;

use generator::Place;

const REGENERATE: &str = "regenerate with `cargo run -p mklight-core --example dump_fixtures`";

/// The first line at which two texts differ, for the failure message.
fn first_difference(committed: &str, generated: &str) -> String {
    let mut lines = committed.lines().zip(generated.lines()).enumerate();
    match lines.find(|(_, (a, b))| a != b) {
        Some((k, (a, b))) => format!("line {}: committed `{a}`, generated `{b}`", k + 1),
        None => format!(
            "{} committed lines, {} generated",
            committed.lines().count(),
            generated.lines().count()
        ),
    }
}

#[test]
fn committed_files_are_what_the_generator_writes() {
    let mut stale = Vec::new();
    for f in generator::fixtures() {
        let path = f.place.directory().join(f.name);
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        if committed != f.json {
            stale.push(format!(
                "{}: {}",
                path.display(),
                first_difference(&committed, &f.json)
            ));
        }
    }
    assert!(stale.is_empty(), "{REGENERATE}:\n{}", stale.join("\n"));
}

/// Each directory holds the generated files and nothing else: a file the
/// generators no longer write would otherwise stay, unchecked. The solve
/// reports come from `mklight-trace`'s `dump_solve_fixtures`, which that
/// crate checks.
#[test]
fn the_directories_hold_only_generated_files() {
    for (place, others) in [
        (Place::TestFixture, &["solve_reports.json"][..]),
        (Place::AppResource, &[][..]),
    ] {
        let dir = place.directory();
        let found: BTreeSet<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        let expected: BTreeSet<String> = generator::fixtures()
            .iter()
            .filter(|f| f.place == place)
            .map(|f| f.name.to_string())
            .chain(others.iter().map(|name| name.to_string()))
            .collect();
        assert_eq!(found, expected, "{}", dir.display());
    }
}
