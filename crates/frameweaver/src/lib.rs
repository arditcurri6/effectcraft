//! FrameWeaver: the algorithms behind the `ec.time.frameweaver` retiming effect.
//!
//! The pipeline, from a pair of source frames A and B to an in-between frame at fraction `w`:
//!
//! 1. [`prep`]: an analysis plane per frame (luma × alpha, optional matte, optional Image Prep),
//!    downscaled to the preset's analysis size.
//! 2. [`flow`]: coarse-to-fine dense optical flow (DIS, Kroeger et al. 2016) with an optional
//!    variational refinement, in both directions.
//! 3. [`consistency`]: forward-backward consistency, occlusion and confidence maps, and repair
//!    of the unreliable vectors.
//! 4. [`guides`] / [`vectors`]: user guide points and imported vectors steer or replace the flow.
//! 5. [`field`]: per-output-pixel *time fields* (where each frame is sampled for fraction `w`,
//!    how visible it is and how far to trust the result), by inverse or forward warping.
//! 6. [`synth`]: the frame itself, motion blur, foreground over background, diagnostics.
//!
//! [`cache`] keeps estimated flows by content. [`backend`] is the seam for other flow estimators.
//!
//! Everything here is pure Rust, allocation-bounded and panic-free on hostile input (empty or
//! huge images, NaN pixels and parameters).

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod backend;
pub mod cache;
pub mod consistency;
pub mod field;
pub mod flow;
pub mod guides;
pub mod plane;
pub mod prep;
pub mod synth;
pub mod vectors;

pub use backend::{ClassicDis, FlowBackend};
pub use consistency::FlowPair;
pub use field::{TimeField, Warping};
pub use flow::{FlowField, FlowParams, Quality};
pub use plane::Plane;

/// Why an estimation was refused. The effect passes its input through (or falls back to a
/// frame blend) rather than failing the render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowError {
    /// A frame has no pixels.
    Empty,
    /// The two frames differ in size.
    SizeMismatch,
    /// The frames exceed the analysis limit.
    TooLarge,
}

impl std::fmt::Display for FlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FlowError::Empty => "frame has no pixels",
            FlowError::SizeMismatch => "frames differ in size",
            FlowError::TooLarge => "frame exceeds the analysis size limit",
        })
    }
}

impl std::error::Error for FlowError {}

/// Largest analysis plane side (pixels).
pub const MAX_SIDE: usize = 4096;

#[cfg(test)]
pub(crate) mod testkit;
