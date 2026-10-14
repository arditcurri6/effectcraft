//! Frame synthesis from a [`TimeField`]: the in-between frame, motion blur and the
//! foreground-over-background composite.

use rayon::prelude::*;

use effectcraft_raster::{Image, Px};

use crate::field::TimeField;

/// The nearest source frame to fraction `w` (A below 0.5).
pub fn nearest(a: &Image, b: &Image, w: f32) -> Image {
    if w < 0.5 { a.clone() } else { b.clone() }
}

/// A linear cross-fade (premultiplied).
pub fn blend(a: &Image, b: &Image, w: f32) -> Image {
    effectcraft_raster::flow::mix(a, b, w)
}

/// Whether continuous frame position (x, y) is inside a `w × h` frame.
#[inline]
fn inside(x: f32, y: f32, w: u32, h: u32) -> bool {
    x >= 0.0 && y >= 0.0 && x <= w as f32 && y <= h as f32
}

/// The in-between frame at fraction `w` ("Motion Weighted Blend"):
///
/// `O = [(1 − w)·v_a·A(x + F_ta) + w·v_b·B(x + F_tb)] / [(1 − w)·v_a + w·v_b]`
///
/// on premultiplied RGBA, then mixed with the nearest unwarped frame by `1 − conf`, so where
/// neither frame can be reconstructed with confidence the output is a real source pixel and
/// not a warped guess. With `smart_blend`, a sample position outside its frame sets that
/// frame's visibility to 0 for the pixel (its weight moves to the other frame) instead of
/// clamping to the edge.
pub fn synthesize(a: &Image, b: &Image, field: &TimeField, w: f32, smart_blend: bool) -> Image {
    if a.width != b.width || a.height != b.height || a.is_empty() || field.is_empty() {
        return blend(a, b, w);
    }
    let w = if w.is_finite() { w.clamp(0.0, 1.0) } else { 0.0 };
    let mut out = Image::new(a.width, a.height);
    let (fw, fh) = (a.width, a.height);
    out.rows_mut().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let c = [x as f32 + 0.5, y as f32 + 0.5];
            let s = field.sample(c[0], c[1]);
            let pa = [c[0] + s.to_a[0], c[1] + s.to_a[1]];
            let pb = [c[0] + s.to_b[0], c[1] + s.to_b[1]];
            let (mut va, mut vb) = (s.vis_a, s.vis_b);
            if smart_blend {
                if !inside(pa[0], pa[1], fw, fh) {
                    va = 0.0;
                }
                if !inside(pb[0], pb[1], fw, fh) {
                    vb = 0.0;
                }
            }
            let (wa, wb) = ((1.0 - w) * va, w * vb);
            // The fallback is the nearest frame that can show this pixel (a covered pixel is
            // visible in A only, a revealed one in B only).
            let near: Px = if wa >= wb { a.get_clamped(x as i64, y as i64) } else { b.get_clamped(x as i64, y as i64) };
            let den = wa + wb;
            let warped: Px = if den > 1e-4 {
                let sa = a.sample_bilinear_clamped(pa[0] as f64, pa[1] as f64);
                let sb = b.sample_bilinear_clamped(pb[0] as f64, pb[1] as f64);
                let mut o = [0.0; 4];
                for k in 0..4 {
                    o[k] = (wa * sa[k] + wb * sb[k]) / den;
                }
                o
            } else {
                near
            };
            let cf = if den > 1e-4 { s.conf.clamp(0.0, 1.0) } else { 0.0 };
            for k in 0..4 {
                px[k] = near[k] + (warped[k] - near[k]) * cf;
            }
        }
    });
    out
}

/// Motion blur along the per-frame motion at the output time: `samples` taps at
/// `x + (k / (K − 1) − ½)·span·F_t(x)` with `F_t = F_tb − F_ta` (frame pixels per source
/// frame), box-weighted. `span` is the shutter's extent in source frames
/// (shutter angle / 360 × source frames per output frame). A no-op for one sample or a zero span.
pub fn motion_blur(img: &Image, field: &TimeField, span: f32, samples: u32) -> Image {
    let k = samples.clamp(1, 64);
    if k == 1 || !span.is_finite() || span.abs() < 1e-6 || field.is_empty() || img.is_empty() {
        return img.clone();
    }
    let mut out = Image::new(img.width, img.height);
    out.rows_mut().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let c = [x as f32 + 0.5, y as f32 + 0.5];
            let s = field.sample(c[0], c[1]);
            let m = [(s.to_b[0] - s.to_a[0]) * span, (s.to_b[1] - s.to_a[1]) * span];
            let mut acc = [0.0f32; 4];
            for i in 0..k {
                let t = i as f32 / (k - 1) as f32 - 0.5;
                let p = img.sample_bilinear_clamped((c[0] + t * m[0]) as f64, (c[1] + t * m[1]) as f64);
                for j in 0..4 {
                    acc[j] += p[j];
                }
            }
            for j in 0..4 {
                px[j] = acc[j] / k as f32;
            }
        }
    });
    out
}

/// `fg` over `bg` (premultiplied).
pub fn over(fg: &Image, bg: &Image) -> Image {
    if fg.width != bg.width || fg.height != bg.height {
        return fg.clone();
    }
    let mut o = bg.clone();
    o.data.par_iter_mut().zip(fg.data.par_iter()).for_each(|(p, q)| {
        let k = 1.0 - q[3];
        for c in 0..4 {
            p[c] = q[c] + p[c] * k;
        }
    });
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consistency::FlowPair;
    use crate::field::{forward, inverse};
    use crate::flow::{FlowParams, Quality};
    use crate::prep::analysis_plane;
    use crate::testkit::*;

    const W: u32 = 160;
    const H: u32 = 112;

    /// Frames of a texture translating by `d` px per frame, and the pair estimated on them.
    fn scene(d: (f32, f32), frames: &[f32], q: Quality) -> (Vec<Image>, FlowPair, crate::plane::Plane, crate::plane::Plane) {
        let imgs: Vec<Image> = frames.iter().map(|&t| render_image(W, H, move |x, y| (x - d.0 * t, y - d.1 * t))).collect();
        let p = FlowParams::for_quality(q, 0.7);
        let (pa, pb) = (analysis_plane(&imgs[0], p.max_long, None, false).unwrap(), analysis_plane(&imgs[1], p.max_long, None, false).unwrap());
        let pair = FlowPair::estimate(&pa.plane, &pb.plane, &p).unwrap();
        (imgs, pair, pa.plane, pb.plane)
    }

    #[test]
    fn motion_weighted_blend_beats_frame_blend_on_translation() {
        // Frames at t = 0 and 1, truth at t = 0.5.
        let (imgs, pair, _, _) = scene((12.0, 3.0), &[0.0, 1.0, 0.5], Quality::Best);
        let tf = inverse(&pair, 0.5, (W, H)).unwrap();
        let mw = synthesize(&imgs[0], &imgs[1], &tf, 0.5, true);
        let bl = blend(&imgs[0], &imgs[1], 0.5);
        let (pm, pb) = (psnr(&mw, &imgs[2], 20), psnr(&bl, &imgs[2], 20));
        assert!(pm >= 35.0, "motion-weighted {pm} dB");
        assert!(pm >= pb + 8.0, "motion-weighted {pm} dB vs blend {pb} dB");
        // Nearest is exactly a source frame, and differs from both others.
        let nr = nearest(&imgs[0], &imgs[1], 0.3);
        assert_eq!(nr, imgs[0]);
    }

    #[test]
    fn forward_and_inverse_agree_on_rigid_motion() {
        let (imgs, pair, pa, pb) = scene((7.0, -2.0), &[0.0, 1.0], Quality::Best);
        let inv = synthesize(&imgs[0], &imgs[1], &inverse(&pair, 0.4, (W, H)).unwrap(), 0.4, true);
        let fwd = synthesize(&imgs[0], &imgs[1], &forward(&pair, &pa, &pb, 0.4, (W, H)).unwrap(), 0.4, true);
        let mut s = 0.0;
        let mut n = 0.0;
        for y in 16..H - 16 {
            for x in 16..W - 16 {
                let (p, q) = (inv.get(x as i64, y as i64), fwd.get(x as i64, y as i64));
                s += (0..3).map(|c| (p[c] - q[c]).abs()).sum::<f32>();
                n += 3.0;
            }
        }
        assert!(s / n <= 0.01, "mean abs diff {}", s / n);
    }

    #[test]
    fn interpolation_at_the_ends_returns_the_frames() {
        let (imgs, pair, _, _) = scene((9.0, 0.0), &[0.0, 1.0], Quality::Balanced);
        for (w, which) in [(0.0f32, 0usize), (1.0, 1)] {
            let o = synthesize(&imgs[0], &imgs[1], &inverse(&pair, w, (W, H)).unwrap(), w, true);
            let p = psnr(&o, &imgs[which], 12);
            assert!(p >= 40.0, "w={w}: {p} dB");
        }
    }

    #[test]
    fn motion_blur_smears_along_motion_only() {
        let (imgs, pair, _, _) = scene((10.0, 0.0), &[0.0, 1.0], Quality::Balanced);
        let tf = inverse(&pair, 0.5, (W, H)).unwrap();
        let sharp = synthesize(&imgs[0], &imgs[1], &tf, 0.5, true);
        let none = motion_blur(&sharp, &tf, 0.0, 8);
        assert_eq!(none, sharp);
        let blur = motion_blur(&sharp, &tf, 1.0, 9);
        // Horizontal detail drops, vertical structure along the motion axis is the same texture.
        let var = |im: &Image| {
            let mut s = 0.0;
            for y in 20..H - 20 {
                for x in 20..W - 21 {
                    s += (im.get(x as i64 + 1, y as i64)[0] - im.get(x as i64, y as i64)[0]).abs();
                }
            }
            s
        };
        assert!(var(&blur) < var(&sharp) * 0.9, "{} vs {}", var(&blur), var(&sharp));
    }

    #[test]
    fn over_composites_premultiplied() {
        let fg = Image::filled(2, 2, [0.5, 0.0, 0.0, 0.5]);
        let bg = Image::filled(2, 2, [0.0, 1.0, 0.0, 1.0]);
        let o = over(&fg, &bg);
        assert_eq!(o.data[0], [0.5, 0.5, 0.0, 1.0]);
    }

    /// Objects crossing in front of one another: a square moving over a static background.
    /// The revealed and covered strips must be reconstructed from the one frame that shows
    /// them, not a ghost of both.
    #[test]
    fn occluding_square_is_reconstructed_without_ghosting() {
        let d = 14.0f32;
        let scene_at = |t: f32| {
            let sx = 30.0 + d * t;
            render_image(W, H, move |x, y| if x >= sx && x < sx + 44.0 && (30.0..78.0).contains(&y) { (x - sx + 500.0, y + 300.0) } else { (x, y) })
        };
        let (a, b, truth) = (scene_at(0.0), scene_at(1.0), scene_at(0.5));
        let p = FlowParams::for_quality(Quality::Best, 0.7);
        let (pa, pb) = (analysis_plane(&a, p.max_long, None, false).unwrap(), analysis_plane(&b, p.max_long, None, false).unwrap());
        let pair = FlowPair::estimate(&pa.plane, &pb.plane, &p).unwrap();
        let inv = synthesize(&a, &b, &inverse(&pair, 0.5, (W, H)).unwrap(), 0.5, true);
        let fwd = synthesize(&a, &b, &forward(&pair, &pa.plane, &pb.plane, 0.5, (W, H)).unwrap(), 0.5, true);
        let bl = blend(&a, &b, 0.5);
        // At t = ½ the square spans x ∈ [37, 81): background x ∈ [31, 36) is being revealed
        // (only B shows it), background x ∈ [82, 87) is being covered (only A shows it).
        let strip = |im: &Image, x0: i64| {
            let mut c = Image::new(5, 40);
            for y in 0..40u32 {
                for x in 0..5u32 {
                    c.set(x, y, im.get(x0 + x as i64, 34 + y as i64));
                }
            }
            c
        };
        for (name, x0) in [("revealed", 31), ("covered", 82)] {
            let ti = strip(&truth, x0);
            let (pi, pf, pbl) = (psnr(&strip(&inv, x0), &ti, 0), psnr(&strip(&fwd, x0), &ti, 0), psnr(&strip(&bl, x0), &ti, 0));
            assert!(pi > pbl + 4.0, "{name}: inverse {pi} dB vs blend {pbl} dB");
            assert!(pf > pbl + 4.0, "{name}: forward {pf} dB vs blend {pbl} dB");
        }
        // Whole frame: both warps beat the blend clearly.
        assert!(psnr(&inv, &truth, 8) > psnr(&bl, &truth, 8) + 3.0);
    }
}
