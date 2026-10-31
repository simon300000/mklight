//! Metal encoder seam for transforms; enabled by its owning package.
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, transforms::Request};
pub(crate) fn execute(
    _: &Request,
    _: &mut crate::resident::Execution<'_>,
    _: &JobControl,
) -> MkResult<()> {
    mklight_gpu::resident::unsupported("transforms")
}
