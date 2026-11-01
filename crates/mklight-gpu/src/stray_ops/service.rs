//! Source reuse of P7's single coating evaluator; no independent coating model.
pub use crate::radiometry::{CoatingLayer, CoatingSpec, CoatingValue};
#[path = "../radiometry/coating.rs"]
pub mod coating;
#[path = "../radiometry/math.rs"]
pub mod math;
