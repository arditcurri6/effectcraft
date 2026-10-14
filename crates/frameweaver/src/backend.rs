//! The seam for flow estimators.
//!
//! [`ClassicDis`] (the DIS pipeline in [`flow`](crate::flow) with the consistency analysis in
//! [`consistency`](crate::consistency)) is the only backend and needs no model files, no GPU
//! and no network. Another estimator — a neural network such as RAFT or GMFlow — would
//! implement [`FlowBackend`] and return the same [`FlowPair`] (flows in both directions plus
//! occlusion and confidence), after which the field building, synthesis, guides and cache work
//! unchanged. Nothing implements it yet: see `docs/frameweaver.md` ("Neural flow") for what is
//! required (a pure-Rust inference path, model weights with a licence that allows
//! redistribution, an explicit download and verification step) and what was evaluated.

use crate::FlowError;
use crate::consistency::FlowPair;
use crate::flow::FlowParams;
use crate::plane::Plane;

/// An optical-flow estimator producing a [`FlowPair`] between two analysis planes.
pub trait FlowBackend: Send + Sync {
    /// A stable identifier (shown in diagnostics).
    fn name(&self) -> &'static str;
    /// Estimate A → B and B → A with occlusion and confidence.
    fn estimate(&self, a: &Plane, b: &Plane, params: &FlowParams) -> Result<FlowPair, FlowError>;
}

/// Dense Inverse Search with variational refinement: the classical CPU estimator.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassicDis;

impl FlowBackend for ClassicDis {
    fn name(&self) -> &'static str {
        "classic-dis"
    }

    fn estimate(&self, a: &Plane, b: &Plane, params: &FlowParams) -> Result<FlowPair, FlowError> {
        FlowPair::estimate(a, b, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Quality;

    #[test]
    fn classic_backend_estimates_through_the_trait() {
        let backend: &dyn FlowBackend = &ClassicDis;
        let a = crate::testkit::render(64, 48, |x, y| (x, y));
        let b = crate::testkit::render(64, 48, |x, y| (x - 3.0, y));
        let pair = backend.estimate(&a, &b, &FlowParams::for_quality(Quality::Draft, 0.7)).unwrap();
        assert_eq!(backend.name(), "classic-dis");
        assert!((pair.fab.at(32, 24)[0] - 3.0).abs() < 0.2);
    }
}
