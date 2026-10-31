//! Metal encoder seam for aiming; enabled by its owning package.
use mklight_core::MkResult;
use mklight_gpu::{aiming::Request, pending::JobControl};
pub(crate) fn execute(
    _: &Request,
    _: &mut crate::resident::Execution<'_>,
    _: &JobControl,
) -> MkResult<()> {
    mklight_gpu::resident::unsupported("aiming")
}
