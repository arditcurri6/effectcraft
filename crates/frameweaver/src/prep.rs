//! The analysis plane flow is estimated on: luma × alpha of the frame (optionally weighted by
//! a matte), scaled to the preset's analysis size, optionally contrast/edge enhanced.

use rayon::prelude::*;

use effectcraft_raster::Image;

use crate::plane::Plane;

/// An analysis plane and how it relates to the frame it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    pub plane: Plane,
    /// Frame pixels per analysis pixel (≥ 1 when the plane was downscaled).
    pub scale: f32,
}

/// Luma (Rec.709) of a premultiplied pixel, which already carries the alpha factor. Non-finite
/// values read as 0; the result is clamped to 0…1.
#[inline]
pub fn luma(p: [f32; 4]) -> f32 {
    let v = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

/// Build the analysis plane of `img`. `matte` (frame-sized, 0…1) multiplies the plane so that
/// pixels outside it do not vote. The plane's long side is at most `max_long`.
pub fn analysis_plane(img: &Image, max_long: u32, matte: Option<&[f32]>, enhance: bool) -> Option<Analysis> {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || img.data.len() != w * h {
        return None;
    }
    let d: Vec<f32> = match matte.filter(|m| m.len() == w * h) {
        Some(m) => img.data.par_iter().zip(m.par_iter()).map(|(p, &k)| luma(*p) * if k.is_finite() { k.clamp(0.0, 1.0) } else { 0.0 }).collect(),
        None => img.data.par_iter().map(|p| luma(*p)).collect(),
    };
    let long = w.max(h) as f32;
    let cap = (max_long.max(16) as usize).min(crate::MAX_SIDE) as f32;
    let k = if long > cap { cap / long } else { 1.0 };
    let (nw, nh) = (((w as f32 * k).round() as usize).max(1), ((h as f32 * k).round() as usize).max(1));
    let raw = RawPlane { w, h, d };
    let mut plane = raw.resize(nw, nh)?;
    if enhance {
        plane = image_prep(&plane)?;
    }
    Some(Analysis { plane, scale: w as f32 / nw as f32 })
}

/// A plane that may be larger than [`crate::MAX_SIDE`] (the full-resolution luma).
struct RawPlane {
    w: usize,
    h: usize,
    d: Vec<f32>,
}

impl RawPlane {
    fn resize(self, nw: usize, nh: usize) -> Option<Plane> {
        if self.w == nw && self.h == nh {
            return Plane::from_vec(nw, nh, self.d);
        }
        // Reduce by exact 2× box steps through plain vectors first, so the intermediate planes
        // never trip the side limit.
        let (mut w, mut h, mut d) = (self.w, self.h, self.d);
        while w >= nw * 2 && h >= nh * 2 {
            let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
            let src = &d;
            let nd: Vec<f32> = (0..hw * hh)
                .into_par_iter()
                .map(|i| {
                    let (x, y) = ((i % hw) * 2, (i / hw) * 2);
                    let g = |dx: usize, dy: usize| src.get((y + dy).min(h - 1) * w + (x + dx).min(w - 1)).copied().unwrap_or(0.0);
                    0.25 * (g(0, 0) + g(1, 0) + g(0, 1) + g(1, 1))
                })
                .collect();
            d = nd;
            w = hw;
            h = hh;
        }
        Plane::from_vec(w, h, d)?.resize(nw, nh)
    }
}

/// Contrast/Edge Enhance: `I' = 0.5 + (I − mean₇ₓ₇) / (4·std₇ₓ₇ + 0.02) · 0.25 + 0.5·|∇I|`,
/// clamped to 0…1. It normalises local contrast so flow survives exposure changes and flat
/// regions.
pub fn image_prep(p: &Plane) -> Option<Plane> {
    let mean = p.box_mean(3)?;
    let sq = Plane::from_vec(p.w, p.h, p.d.iter().map(|v| v * v).collect())?;
    let mean_sq = sq.box_mean(3)?;
    let (gx, gy) = p.gradient()?;
    let mut out = Plane::new(p.w, p.h)?;
    out.d.par_iter_mut().enumerate().for_each(|(i, o)| {
        let (m, ms) = (mean.d.get(i).copied().unwrap_or(0.0), mean_sq.d.get(i).copied().unwrap_or(0.0));
        let std = (ms - m * m).max(0.0).sqrt();
        let g = gx.d.get(i).copied().unwrap_or(0.0).hypot(gy.d.get(i).copied().unwrap_or(0.0));
        let v = p.d.get(i).copied().unwrap_or(0.0);
        *o = (0.5 + (v - m) / (4.0 * std + 0.02) * 0.25 + 0.5 * g).clamp(0.0, 1.0);
    });
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_is_downscaled_and_scale_reported() {
        let img = Image::filled(200, 100, [0.5, 0.5, 0.5, 1.0]);
        let a = analysis_plane(&img, 50, None, false).unwrap();
        assert_eq!((a.plane.w, a.plane.h), (50, 25));
        assert!((a.scale - 4.0).abs() < 1e-3);
        assert!((a.plane.d[10] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn matte_excludes_pixels_and_nonfinite_reads_zero() {
        let mut img = Image::filled(8, 8, [1.0, 1.0, 1.0, 1.0]);
        img.data[0] = [f32::NAN, f32::INFINITY, 0.0, 1.0];
        let matte: Vec<f32> = (0..64).map(|i| if i % 8 < 4 { 1.0 } else { 0.0 }).collect();
        let a = analysis_plane(&img, 64, Some(&matte), false).unwrap();
        assert_eq!(a.plane.d[0], 0.0);
        assert_eq!(a.plane.d[1], 1.0);
        assert_eq!(a.plane.d[6], 0.0);
    }

    #[test]
    fn empty_and_degenerate_frames_are_refused() {
        assert!(analysis_plane(&Image::new(0, 0), 64, None, false).is_none());
        let thin = analysis_plane(&Image::filled(1, 4000, [0.3; 4]), 960, None, true).unwrap();
        assert_eq!(thin.plane.w, 1);
    }

    #[test]
    fn image_prep_lifts_flat_contrast() {
        let mut p = Plane::filled(32, 32, 0.1).unwrap();
        for y in 0..32 {
            for x in 16..32 {
                p.d[y * 32 + x] = 0.12;
            }
        }
        let q = image_prep(&p).unwrap();
        let step = (q.at(16, 16) - q.at(15, 16)).abs();
        assert!(step > 0.05, "step {step}");
    }
}
