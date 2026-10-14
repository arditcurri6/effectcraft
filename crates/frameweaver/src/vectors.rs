//! External motion vectors: a layer whose colour encodes per-pixel motion (a 3D render's
//! motion-vector pass, or vectors exported from another tool) replaces the estimated flow.
//!
//! * **Signed RG** (0.5 = still): `v = (c.rg − 0.5) · 2 · scale`.
//! * **Float RG**: `v = c.rg · scale` (the channels hold the vector directly, in a 32 bpc
//!   project).
//!
//! `c` is the straight (un-premultiplied) colour; `scale` is in frame pixels (a full-scale
//! signed channel moves the pixel by `scale`). Flip Y negates `v.y`. The layer holds the motion
//! to the next frame (`ToNext`, read at the time of source frame A) or to the previous frame
//! (`ToPrevious`, read at B). The reverse field is built by splatting the decoded one negated
//! ([`FlowPair::from_forward`]), so occlusion detection still applies.

use rayon::prelude::*;

use effectcraft_raster::Image;

use crate::consistency::FlowPair;
use crate::flow::{FlowField, FlowParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    SignedRg,
    FloatRg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    ToNext,
    ToPrevious,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorOpts {
    pub encoding: Encoding,
    pub scale: f32,
    pub direction: Direction,
    pub flip_y: bool,
}

/// Decode a vector layer to a flow field at the layer's resolution (vectors in frame pixels).
/// Non-finite values decode as no motion. `None` when the layer is empty or too large.
pub fn decode(img: &Image, o: &VectorOpts) -> Option<FlowField> {
    let (w, h) = (img.width as usize, img.height as usize);
    let mut f = FlowField::new(w, h)?;
    if img.data.len() != w * h {
        return None;
    }
    let scale = if o.scale.is_finite() { o.scale } else { 0.0 };
    f.v.par_iter_mut().zip(img.data.par_iter()).for_each(|(v, p)| {
        let a = p[3];
        let (r, g) = if a > 1e-6 { (p[0] / a, p[1] / a) } else { (0.0, 0.0) };
        let (mut x, mut y) = match o.encoding {
            Encoding::SignedRg => ((r - 0.5) * 2.0 * scale, (g - 0.5) * 2.0 * scale),
            Encoding::FloatRg => (r * scale, g * scale),
        };
        if o.flip_y {
            y = -y;
        }
        if !(x.is_finite() && y.is_finite()) {
            x = 0.0;
            y = 0.0;
        }
        // Signed encoding of an empty pixel means "still", not full-scale negative motion.
        if a <= 1e-6 {
            x = 0.0;
            y = 0.0;
        }
        *v = [x, y];
    });
    Some(f)
}

/// Encode a flow field (frame pixels) the way [`decode`] reads it (opaque pixels).
pub fn encode(f: &FlowField, encoding: Encoding, scale: f32) -> Image {
    let mut img = Image::new(f.w as u32, f.h as u32);
    let s = if scale.is_finite() && scale.abs() > 1e-12 { scale } else { 1.0 };
    for (p, v) in img.data.iter_mut().zip(&f.v) {
        *p = match encoding {
            Encoding::SignedRg => [0.5 + 0.5 * v[0] / s, 0.5 + 0.5 * v[1] / s, 0.0, 1.0],
            Encoding::FloatRg => [v[0] / s, v[1] / s, 0.0, 1.0],
        };
    }
    img
}

/// A [`FlowPair`] at `field_w × field_h` from a vector layer: decode, resample to the field's
/// resolution (scaling the vectors), reverse, analyse. For `ToPrevious` the layer is B → A.
pub fn pair_from_layer(layer: &Image, o: &VectorOpts, field_w: usize, field_h: usize, params: &FlowParams) -> Option<FlowPair> {
    let decoded = decode(layer, o)?;
    let at_field = if decoded.w == field_w && decoded.h == field_h { decoded } else { decoded.resize(field_w, field_h)? };
    match o.direction {
        Direction::ToNext => FlowPair::from_forward(at_field, params),
        Direction::ToPrevious => FlowPair::from_forward(at_field, params).map(FlowPair::swapped),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Quality;

    fn opts(encoding: Encoding) -> VectorOpts {
        VectorOpts { encoding, scale: 16.0, direction: Direction::ToNext, flip_y: false }
    }

    #[test]
    fn round_trip_both_encodings() {
        let mut f = FlowField::new(8, 6).unwrap();
        for (i, v) in f.v.iter_mut().enumerate() {
            *v = [(i as f32 - 20.0) * 0.4, 5.0 - i as f32 * 0.3];
        }
        for enc in [Encoding::SignedRg, Encoding::FloatRg] {
            let back = decode(&encode(&f, enc, 16.0), &opts(enc)).unwrap();
            for (a, b) in f.v.iter().zip(&back.v) {
                assert!((a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4, "{enc:?} {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn flip_y_negates_and_still_is_half() {
        let img = Image::filled(2, 2, [0.5, 0.75, 0.0, 1.0]);
        let v = decode(&img, &VectorOpts { flip_y: true, ..opts(Encoding::SignedRg) }).unwrap();
        assert!((v.v[0][0]).abs() < 1e-6 && (v.v[0][1] + 8.0).abs() < 1e-5, "{:?}", v.v[0]);
        // Premultiplied input is un-premultiplied first.
        let half = Image::filled(2, 2, [0.25, 0.25, 0.0, 0.5]);
        assert!(decode(&half, &opts(Encoding::SignedRg)).unwrap().v[0][0].abs() < 1e-6);
    }

    #[test]
    fn direction_decides_which_frame_the_layer_describes() {
        let p = FlowParams::for_quality(Quality::Draft, 0.7);
        let f = FlowField { w: 48, h: 32, v: vec![[3.0, 0.0]; 48 * 32] };
        let layer = encode(&f, Encoding::SignedRg, 16.0);
        let next = pair_from_layer(&layer, &opts(Encoding::SignedRg), 48, 32, &p).unwrap();
        assert!((next.fab.at(24, 16)[0] - 3.0).abs() < 1e-3 && (next.fba.at(24, 16)[0] + 3.0).abs() < 1e-3);
        let prev = pair_from_layer(&layer, &VectorOpts { direction: Direction::ToPrevious, ..opts(Encoding::SignedRg) }, 48, 32, &p).unwrap();
        assert!((prev.fba.at(24, 16)[0] - 3.0).abs() < 1e-3 && (prev.fab.at(24, 16)[0] + 3.0).abs() < 1e-3);
    }

    #[test]
    fn vectors_are_rescaled_to_the_field_resolution() {
        let p = FlowParams::for_quality(Quality::Draft, 0.7);
        // A FloatRg layer of scale 1 holding (8, 4) frame pixels, read at half resolution.
        let f = FlowField { w: 96, h: 64, v: vec![[8.0, 4.0]; 96 * 64] };
        let layer = encode(&f, Encoding::FloatRg, 1.0);
        let o = VectorOpts { scale: 1.0, ..opts(Encoding::FloatRg) };
        let pair = pair_from_layer(&layer, &o, 48, 32, &p).unwrap();
        let v = pair.fab.at(20, 14);
        assert!((v[0] - 4.0).abs() < 1e-3 && (v[1] - 2.0).abs() < 1e-3, "{v:?}");
    }

    #[test]
    fn hostile_layers_are_refused_or_neutral() {
        assert!(decode(&Image::new(0, 0), &opts(Encoding::SignedRg)).is_none());
        let mut img = Image::filled(4, 4, [0.5, 0.5, 0.0, 1.0]);
        img.data[3] = [f32::NAN, f32::INFINITY, 0.0, 1.0];
        let f = decode(&img, &VectorOpts { scale: f32::NAN, ..opts(Encoding::SignedRg) }).unwrap();
        assert!(f.v.iter().all(|v| *v == [0.0, 0.0]));
        let f = decode(&img, &opts(Encoding::SignedRg)).unwrap();
        assert!(f.v.iter().all(|v| v[0].is_finite() && v[1].is_finite()));
    }
}
