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

/// Deserializes an internally tagged enum whose tag-only (unit) variants
/// must not carry other keys. serde's `deny_unknown_fields` applies to struct
/// variants only, so without this `{"mode": "auto", "value": 5}` would load
/// as `Auto` and silently drop `value`. `inner` is the derived
/// deserializer (`#[serde(remote = "Self")]`).
pub(crate) fn deserialize_strict_unit_variants<'de, D, T>(
    d: D,
    tag: &str,
    unit_variants: &[&str],
    inner: impl FnOnce(serde_json::Value) -> Result<T, serde_json::Error>,
) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    let v = serde_json::Value::deserialize(d)?;
    if let Some(obj) = v.as_object()
        && let Some(t) = obj.get(tag).and_then(|t| t.as_str())
        && unit_variants.contains(&t)
        && let Some(k) = obj.keys().find(|k| *k != tag)
    {
        return Err(D::Error::custom(format!(
            "unknown field `{k}`: `{tag}: {t}` takes no other fields"
        )));
    }
    inner(v).map_err(D::Error::custom)
}

/// `Serialize`/`Deserialize` for an internally tagged enum derived with
/// `#[serde(remote = "Self")]`, rejecting extra keys on its unit variants
/// ([`deserialize_strict_unit_variants`]). The JSON form is unchanged.
macro_rules! strict_unit_variants {
    ($ty:ty, $tag:literal, [$($unit:literal),+ $(,)?]) => {
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                <$ty>::serialize(self, s)
            }
        }
        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                $crate::serde_util::deserialize_strict_unit_variants(
                    d,
                    $tag, &[$($unit),+], <$ty>::deserialize)
            }
        }
    };
}
pub(crate) use strict_unit_variants;

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
