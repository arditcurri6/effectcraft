//! The whole pipeline for two frames in one call: analysis planes, bidirectional flow,
//! consistency, time field, synthesis. The effect adapter composes the same stages itself (it
//! adds caching, guides, mattes and vectors); this is for tests, benchmarks and other callers
//! that just want the in-between frame.

use effectcraft_raster::Image;

use crate::consistency::FlowPair;
use crate::field::{self, TimeField, Warping};
use crate::flow::{FlowParams, Quality};
use crate::prep::analysis_plane;
use crate::{FlowError, synth};

/// What [`interpolate`] does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub quality: Quality,
    /// Motion Sensitivity 0…1.
    pub sensitivity: f32,
    pub warping: Warping,
    pub smart_blend: bool,
    /// Contrast/Edge Enhance on the analysis plane.
    pub image_prep: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options { quality: Quality::Balanced, sensitivity: 0.7, warping: Warping::Inverse, smart_blend: true, image_prep: false }
    }
}

/// The flow pair between two frames.
pub fn estimate_pair(a: &Image, b: &Image, o: &Options) -> Result<(FlowPair, f32), FlowError> {
    let p = FlowParams::for_quality(o.quality, o.sensitivity);
    let pa = analysis_plane(a, p.max_long, None, o.image_prep).ok_or(FlowError::Empty)?;
    let pb = analysis_plane(b, p.max_long, None, o.image_prep).ok_or(FlowError::Empty)?;
    let pair = FlowPair::estimate(&pa.plane, &pb.plane, &p)?;
    Ok((pair, pa.scale))
}

/// The time field of `pair` at fraction `w`.
pub fn time_field(pair: &FlowPair, a: &Image, b: &Image, w: f32, o: &Options) -> Result<TimeField, FlowError> {
    let frame = (a.width, a.height);
    let p = FlowParams::for_quality(o.quality, o.sensitivity);
    let pa = analysis_plane(a, p.max_long, None, o.image_prep).ok_or(FlowError::Empty)?;
    let pb = analysis_plane(b, p.max_long, None, o.image_prep).ok_or(FlowError::Empty)?;
    match o.warping {
        Warping::Inverse => field::inverse(pair, &pa.plane, &pb.plane, w, frame).ok_or(FlowError::Empty),
        Warping::Forward => field::forward(pair, &pa.plane, &pb.plane, w, frame).ok_or(FlowError::Empty),
    }
}

/// The frame at fraction `w` ∈ 0…1 between `a` and `b`.
pub fn interpolate(a: &Image, b: &Image, w: f32, o: &Options) -> Result<Image, FlowError> {
    if a.is_empty() || b.is_empty() {
        return Err(FlowError::Empty);
    }
    if a.width != b.width || a.height != b.height {
        return Err(FlowError::SizeMismatch);
    }
    let (pair, _) = estimate_pair(a, b, o)?;
    let tf = time_field(&pair, a, b, w, o)?;
    Ok(synth::synthesize(a, b, &tf, w, o.smart_blend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;

    #[test]
    fn interpolate_matches_the_stages_and_rejects_bad_input() {
        let a = render_image(96, 64, |x, y| (x, y));
        let b = render_image(96, 64, |x, y| (x - 5.0, y));
        let truth = render_image(96, 64, |x, y| (x - 2.5, y));
        let o = Options { quality: Quality::Best, ..Options::default() };
        let out = interpolate(&a, &b, 0.5, &o).unwrap();
        assert!(psnr(&out, &truth, 12) > 35.0);
        assert_eq!(interpolate(&Image::new(0, 0), &b, 0.5, &o), Err(FlowError::Empty));
        assert_eq!(interpolate(&a, &Image::new(10, 10), 0.5, &o), Err(FlowError::SizeMismatch));
        // A NaN fraction is treated as 0, not propagated.
        let nan = interpolate(&a, &b, f32::NAN, &o).unwrap();
        assert!(nan.data.iter().all(|p| p.iter().all(|v| v.is_finite())));
    }
}
