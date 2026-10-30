//! Serde helpers. JSON cannot represent infinities, so `±∞` are written as the
//! strings `"inf"` / `"-inf"` (see CONVENTIONS §2).

use serde::{Deserialize, Deserializer, Serializer};

pub mod f64_inf {
    use super::*;

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_infinite() {
            s.serialize_str(if *v > 0.0 { "inf" } else { "-inf" })
        } else {
            s.serialize_f64(*v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Num {
            F(f64),
            S(String),
        }
        match Num::deserialize(d)? {
            Num::F(f) => Ok(f),
            Num::S(s) => match s.trim().to_ascii_lowercase().as_str() {
                "inf" | "+inf" | "infinity" | "+infinity" => Ok(f64::INFINITY),
                "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
                other => other.parse::<f64>().map_err(serde::de::Error::custom),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct T {
        #[serde(with = "super::f64_inf")]
        v: f64,
    }

    #[test]
    fn roundtrip_inf() {
        for v in [f64::INFINITY, f64::NEG_INFINITY, 1.25, -3.0] {
            let s = serde_json::to_string(&T { v }).unwrap();
            assert_eq!(serde_json::from_str::<T>(&s).unwrap(), T { v });
        }
    }
}
