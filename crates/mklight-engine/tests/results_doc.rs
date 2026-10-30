//! docs/RESULTS.md's analysis reference is generated from the registry's
//! settings and result schemas; this test fails when the two drift apart.
//! Regenerate after an intended change and review the diff:
//!
//! ```text
//! MKLIGHT_UPDATE_RESULTS_DOC=1 cargo test -p mklight-engine --test results_doc
//! ```

use mklight_engine::Engine;
use mklight_engine::analysis::schema::render_markdown;

const BEGIN: &str = "<!-- BEGIN GENERATED: analysis reference (tests/results_doc.rs) -->";
const END: &str = "<!-- END GENERATED -->";

/// Category order of the reference.
const CATEGORIES: &[&str] = &[
    "model",
    "materials",
    "geometric",
    "radiometry",
    "polarization",
    "ghost",
    "stray",
    "wave",
    "diagnostic",
];

fn generated() -> String {
    let mut infos = Engine::default().analyses();
    let rank = |c: &str| CATEGORIES.iter().position(|x| *x == c).unwrap_or(usize::MAX);
    infos.sort_by(|a, b| {
        rank(&a.category)
            .cmp(&rank(&b.category))
            .then(a.name.cmp(&b.name))
    });
    let mut out = String::from("\n");
    for i in &infos {
        assert!(
            rank(&i.category) != usize::MAX,
            "category '{}' of {} missing from CATEGORIES",
            i.category,
            i.name
        );
        out.push_str(&render_markdown(i));
    }
    out
}

#[test]
fn results_doc_matches_the_registry() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/RESULTS.md");
    let doc = std::fs::read_to_string(path).expect("docs/RESULTS.md");
    let (Some(a), Some(b)) = (doc.find(BEGIN), doc.find(END)) else {
        panic!("docs/RESULTS.md lacks the generated-section markers");
    };
    let start = a + BEGIN.len();
    let current = &doc[start..b];
    let want = generated();
    if std::env::var("MKLIGHT_UPDATE_RESULTS_DOC").is_ok_and(|v| v == "1") {
        let new = format!("{}{}{}", &doc[..start], want, &doc[b..]);
        std::fs::write(path, new).unwrap();
        return;
    }
    if current != want {
        let line = current
            .lines()
            .zip(want.lines())
            .position(|(x, y)| x != y)
            .unwrap_or(0);
        panic!(
            "docs/RESULTS.md is out of date (first difference at generated line {line}); \
             run MKLIGHT_UPDATE_RESULTS_DOC=1 cargo test -p mklight-engine --test results_doc"
        );
    }
}
