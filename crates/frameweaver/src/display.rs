//! Diagnostic images (the Diagnostics ▸ Display modes): the flow as a colour wheel, the
//! visibility / confidence of a time field, and guide markers.

use rayon::prelude::*;

use effectcraft_raster::Image;

use crate::field::TimeField;
use crate::flow::FlowField;

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = (h.rem_euclid(1.0)) * 6.0;
    let i = h.floor();
    let f = h - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match i as i32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// The flow as a colour wheel at `frame` size: hue = direction, saturation = magnitude
/// normalised to the 99th percentile of the magnitudes (still = white).
pub fn flow_wheel(f: &FlowField, frame: (u32, u32)) -> Image {
    let mut out = Image::new(frame.0, frame.1);
    if f.is_empty() || out.is_empty() {
        return out;
    }
    let mut mags: Vec<f32> = f.v.iter().map(|v| v[0].hypot(v[1])).filter(|m| m.is_finite()).collect();
    mags.sort_by(f32::total_cmp);
    let norm = mags.get(mags.len().saturating_sub(1) * 99 / 100).copied().unwrap_or(1.0).max(1e-3);
    let (sx, sy) = (f.w as f32 / frame.0 as f32, f.h as f32 / frame.1 as f32);
    out.rows_mut().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let v = f.sample((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5);
            let m = (v[0].hypot(v[1]) / norm).clamp(0.0, 1.0);
            let hue = v[1].atan2(v[0]) / std::f32::consts::TAU;
            let c = hsv(hue, m, 1.0);
            *px = [c[0], c[1], c[2], 1.0];
        }
    });
    out
}

/// What of a [`TimeField`] to show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldView {
    /// Red = how visible A is, green = how visible B is, blue = confidence. White is fully
    /// visible and trusted; cyan marks pixels A cannot show (revealed), magenta pixels B cannot
    /// show (covered), yellow low confidence.
    Occlusion,
    /// Grey = confidence.
    Confidence,
}

pub fn field_view(tf: &TimeField, frame: (u32, u32), view: FieldView) -> Image {
    let mut out = Image::new(frame.0, frame.1);
    if tf.is_empty() || out.is_empty() {
        return out;
    }
    out.rows_mut().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let s = tf.sample(x as f32 + 0.5, y as f32 + 0.5);
            *px = match view {
                FieldView::Occlusion => {
                    [1.0 - (1.0 - s.vis_a).clamp(0.0, 1.0), 1.0 - (1.0 - s.vis_b).clamp(0.0, 1.0), 1.0 - (1.0 - s.conf).clamp(0.0, 1.0), 1.0]
                }
                FieldView::Confidence => {
                    let c = s.conf.clamp(0.0, 1.0);
                    [c, c, c, 1.0]
                }
            };
        }
    });
    out
}

/// Draw a guide on `img`: a dot at `from` (frame pixels) and a line to `to`.
pub fn draw_guide(img: &mut Image, from: [f32; 2], to: [f32; 2], color: [f32; 3]) {
    if img.is_empty() || ![from[0], from[1], to[0], to[1]].iter().all(|v| v.is_finite()) {
        return;
    }
    let n = (from[0] - to[0]).hypot(from[1] - to[1]).ceil().clamp(1.0, 4096.0) as usize;
    let mut plot = |x: f32, y: f32, r: i64| {
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (x.round() as i64 + dx, y.round() as i64 + dy);
                if px >= 0 && py >= 0 && px < img.width as i64 && py < img.height as i64 {
                    img.set(px as u32, py as u32, [color[0], color[1], color[2], 1.0]);
                }
            }
        }
    };
    for i in 0..=n {
        let t = i as f32 / n as f32;
        plot(from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, 0);
    }
    plot(from[0], from[1], 3);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wheel_encodes_direction_and_zero_is_white() {
        let still = FlowField { w: 8, h: 8, v: vec![[0.0; 2]; 64] };
        assert_eq!(flow_wheel(&still, (8, 8)).data[0], [1.0, 1.0, 1.0, 1.0]);
        let right = FlowField { w: 8, h: 8, v: vec![[5.0, 0.0]; 64] };
        let left = FlowField { w: 8, h: 8, v: vec![[-5.0, 0.0]; 64] };
        let (r, l) = (flow_wheel(&right, (8, 8)).data[0], flow_wheel(&left, (8, 8)).data[0]);
        assert_ne!(r, l);
        assert!(r[0] > 0.99 && r[1] < 0.2, "{r:?}");
    }

    #[test]
    fn draw_guide_marks_and_ignores_nonfinite() {
        let mut img = Image::new(20, 20);
        draw_guide(&mut img, [5.0, 5.0], [15.0, 5.0], [1.0, 0.0, 0.0]);
        assert_eq!(img.get(10, 5), [1.0, 0.0, 0.0, 1.0]);
        let before = img.clone();
        draw_guide(&mut img, [f32::NAN, 1.0], [2.0, 2.0], [0.0, 1.0, 0.0]);
        assert_eq!(img, before);
        draw_guide(&mut img, [-1e9, 0.0], [1e9, 0.0], [0.0, 1.0, 0.0]);
    }
}
