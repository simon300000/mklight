//! Immutable model snapshots. Analyses and optimisations always run against a
//! `Snapshot`, never against live, mutable model state (ARCHITECTURE §State).

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::sync::Arc;

use crate::error::MkResult;
use crate::model::OpticalSystem;

/// BLAKE3 hash of a system's canonical JSON. Equal hashes mean the systems
/// are physically identical.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SnapshotHash(pub [u8; 32]);

impl SnapshotHash {
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
    pub fn short(&self) -> String {
        self.to_hex()[..12].to_string()
    }
}

impl fmt::Debug for SnapshotHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SnapshotHash({})", self.short())
    }
}
impl fmt::Display for SnapshotHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}
impl Serialize for SnapshotHash {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}
impl<'de> Deserialize<'de> for SnapshotHash {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        let err = || serde::de::Error::custom("snapshot hash must be 64 hex chars");
        let digit = |b: u8| (b as char).to_digit(16).map(|v| v as u8);
        let b = s.as_bytes();
        if b.len() != 64 {
            return Err(err());
        }
        let mut out = [0u8; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = match (digit(b[2 * i]), digit(b[2 * i + 1])) {
                (Some(hi), Some(lo)) => hi << 4 | lo,
                _ => return Err(err()),
            };
        }
        Ok(SnapshotHash(out))
    }
}

/// An immutable, validated, content-addressed optical system. Cheap to clone.
#[derive(Clone)]
pub struct Snapshot {
    system: Arc<OpticalSystem>,
    hash: SnapshotHash,
}

impl fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Snapshot")
            .field("name", &self.system.name)
            .field("hash", &self.hash)
            .finish()
    }
}

impl PartialEq for Snapshot {
    fn eq(&self, o: &Self) -> bool {
        self.hash == o.hash
    }
}
impl Eq for Snapshot {}

impl Snapshot {
    /// Validate and freeze a system.
    pub fn new(system: OpticalSystem) -> MkResult<Self> {
        system.validate()?;
        let hash = hash_system(&system)?;
        Ok(Snapshot {
            system: Arc::new(system),
            hash,
        })
    }
    pub fn system(&self) -> &OpticalSystem {
        &self.system
    }
    pub fn system_arc(&self) -> Arc<OpticalSystem> {
        self.system.clone()
    }
    pub fn hash(&self) -> SnapshotHash {
        self.hash
    }
    /// Make a modified copy (copy-on-write).
    pub fn modified(&self, f: impl FnOnce(&mut OpticalSystem)) -> MkResult<Snapshot> {
        let mut s = (*self.system).clone();
        f(&mut s);
        Snapshot::new(s)
    }
}

pub fn hash_system(system: &OpticalSystem) -> MkResult<SnapshotHash> {
    let bytes = serde_json::to_vec(system)?;
    Ok(SnapshotHash(*blake3::hash(&bytes).as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_stable_and_sensitive() {
        let a = Snapshot::new(OpticalSystem::new_empty()).unwrap();
        let b = Snapshot::new(OpticalSystem::new_empty()).unwrap();
        assert_eq!(a.hash(), b.hash());
        let c = a
            .modified(|s| s.surfaces[1].thickness = 10.000000001)
            .unwrap();
        assert_ne!(a.hash(), c.hash());
        let j = serde_json::to_string(&a.hash()).unwrap();
        assert_eq!(serde_json::from_str::<SnapshotHash>(&j).unwrap(), a.hash());
    }

    #[test]
    fn hash_text_is_exactly_64_hex_digits() {
        let hex = "0123456789abcdefABCDEF".repeat(3);
        let ok = serde_json::from_str::<SnapshotHash>(&format!("\"{}\"", &hex[..64])).unwrap();
        assert_eq!(ok.to_hex(), hex[..64].to_lowercase());
        for bad in [
            // 64 bytes, but not 64 characters.
            format!("{}é", &hex[..62]),
            format!("é{}", &hex[..62]),
            format!("{}é0", &hex[..61]),
            // A sign is not a hex digit.
            format!("+f{}", &hex[..62]),
            format!("{}g", &hex[..63]),
            hex[..63].to_string(),
            hex[..66].to_string(),
        ] {
            assert!(
                serde_json::from_str::<SnapshotHash>(&format!("\"{bad}\"")).is_err(),
                "{bad}"
            );
        }
    }
}
