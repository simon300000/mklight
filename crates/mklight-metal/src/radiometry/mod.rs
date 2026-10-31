//! Metal encoder seam for radiometry; enabled by its owning package.
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, radiometry::Request};
pub(crate) fn execute(
    _: &Request,
    _: &mut crate::resident::Execution<'_>,
    _: &JobControl,
) -> MkResult<()> {
    mklight_gpu::resident::unsupported("radiometry")
}
