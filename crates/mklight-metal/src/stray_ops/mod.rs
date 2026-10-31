//! Metal encoder seam for stray_ops; enabled by its owning package.
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, stray_ops::Request};
pub(crate) fn execute(
    _: &Request,
    _: &mut crate::resident::Execution<'_>,
    _: &JobControl,
) -> MkResult<()> {
    mklight_gpu::resident::unsupported("stray_ops")
}
