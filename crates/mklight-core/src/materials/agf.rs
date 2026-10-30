//! Reader for AGF-format glass catalog text files (the de-facto interchange
//! format manufacturers publish). Users supply the catalog files, and
//! MKLight does not ship vendor catalogs.

use crate::error::{MkError, MkResult};
use crate::materials::GlassCatalog;

/// Parse AGF text into a catalog named `name`.
pub fn parse_agf(_name: &str, _text: &str) -> MkResult<GlassCatalog> {
    Err(MkError::Unsupported(
        "AGF parsing not implemented yet".into(),
    ))
}
