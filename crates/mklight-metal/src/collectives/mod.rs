//! Metal encoder seam for collectives; enabled by its owning package.
use mklight_core::MkResult;
use mklight_gpu::{collectives::Request, pending::JobControl};
pub(crate) fn execute(
    _: &Request,
    _: &mut crate::resident::Execution<'_>,
    _: &JobControl,
) -> MkResult<()> {
    mklight_gpu::resident::unsupported("collectives")
}
