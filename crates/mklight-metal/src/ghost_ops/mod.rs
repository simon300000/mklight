//! Metal encoder seam for ghost_ops; enabled by its owning package.
use mklight_core::MkResult;
use mklight_gpu::{ghost_ops::Request, pending::JobControl};
pub(crate) fn execute(
    _: &Request,
    _: &mut crate::resident::Execution<'_>,
    _: &JobControl,
) -> MkResult<()> {
    mklight_gpu::resident::unsupported("ghost_ops")
}
