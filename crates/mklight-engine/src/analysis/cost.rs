//! Recorded full-chain cost decisions. Unknown device/build/input combinations
//! prefer qualified Metal; an explicit backend bypasses this table.
use super::AnalysisInput;
use crate::gpu::GpuDevice;
use serde_json::{Value, json};
use std::sync::OnceLock;

pub(crate) fn key(analysis: &str, snapshot: &str, library: &str, settings: &Value) -> String {
    blake3::hash(
        serde_json::to_string(
            &json!({"analysis":analysis,"snapshot":snapshot,"library":library,"settings":settings}),
        )
        .unwrap()
        .as_bytes(),
    )
    .to_hex()
    .to_string()
}
fn evidence() -> &'static Value {
    static DATA: OnceLock<Value> = OnceLock::new();
    DATA.get_or_init(|| serde_json::from_str(include_str!("cost-evidence.json")).unwrap())
}
fn host_build() -> &'static str {
    static BUILD: OnceLock<String> = OnceLock::new();
    BUILD.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("/usr/bin/sw_vers")
                .arg("-buildVersion")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().into())
                .unwrap_or_default()
        }
        #[cfg(not(target_os = "macos"))]
        {
            String::new()
        }
    })
}
pub(crate) fn decision(
    input: &AnalysisInput<'_>,
    analysis: &str,
    settings: &Value,
    device: &GpuDevice,
) -> Value {
    let table = evidence();
    let key = key(
        analysis,
        &input.snapshot.hash().to_hex(),
        input.library_hash(),
        settings,
    );
    let matching = table["os_build"].as_str() == Some(host_build())
        && table["device"].as_str() == Some(device.backend.info().name.as_str())
        && table["kernel_build"].as_str() == Some(device.backend.precision().kernel_build.as_str());
    let row = matching
        .then(|| {
            table["observations"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["key"].as_str() == Some(&key))
        })
        .flatten();
    match row {
        Some(row) => {
            json!({"version":table["version"],"evidence":table["evidence"],"key":key,"decision":row["decision"],"measurement":row})
        }
        None => {
            json!({"version":table["version"],"key":key,"decision":"qualified_metal","reason":"unknown full-chain cost"})
        }
    }
}
