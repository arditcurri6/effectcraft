//! Guide points: the user states where a point of the image goes between two source frames,
//! and the flow near it is pulled to that motion.
//!
//! Guide `i` has positions `P_i(a)` and `P_i(b)` (analysis pixels, from the guide's animated
//! position at the two source frame times) and so the motion `d_i = P_i(b) − P_i(a)`. With
//! `σ = radius / 2` and `g_i(x) = exp(−|x − c_i|² / 2σ²)` (`c_i` = `P_i(a)` for the A → B flow
//! and `P_i(b)` for the B → A flow, whose vector is `−d_i`), the flow becomes
//!
//! `F(x) ← F(x)·(1 − G(x)) + [Σ g_i d_i / max(Σ g_i, ε)]·G(x)`, with `G = min(1, Σ g_i)`.
//!
//! Inside a guide's influence the pixel is also declared reliable: occlusion is faded out and
//! confidence raised by `G`, since the user has stated the correspondence.

use rayon::prelude::*;

use crate::consistency::FlowPair;
use crate::flow::FlowField;

/// One guide: where a point is in A and in B (analysis pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Guide {
    pub from: [f32; 2],
    pub to: [f32; 2],
}

impl Guide {
    fn valid(&self) -> bool {
        self.from.iter().chain(self.to.iter()).all(|v| v.is_finite())
    }
}

/// The influence map and the blended target vector of `guides` (each as centre + vector) over a
/// `w × h` field.
fn influence(w: usize, h: usize, centres: &[([f32; 2], [f32; 2])], radius: f32) -> Option<(Vec<f32>, Vec<[f32; 2]>)> {
    if centres.is_empty() || w == 0 || h == 0 {
        return None;
    }
    let sigma = (radius.clamp(1.0, 4000.0) * 0.5).max(0.5);
    let inv = 1.0 / (2.0 * sigma * sigma);
    let reach = (3.0 * sigma).ceil();
    let mut sum_g = vec![0.0f32; w * h];
    let mut sum_gd = vec![[0.0f32; 2]; w * h];
    for &(c, d) in centres {
        let (x0, x1) = (((c[0] - reach).floor().max(0.0)) as usize, ((c[0] + reach).ceil().min(w as f32 - 1.0)).max(0.0) as usize);
        let (y0, y1) = (((c[1] - reach).floor().max(0.0)) as usize, ((c[1] + reach).ceil().min(h as f32 - 1.0)).max(0.0) as usize);
        if c[0] + reach < 0.0 || c[1] + reach < 0.0 || c[0] - reach > w as f32 || c[1] - reach > h as f32 {
            continue;
        }
        for y in y0..=y1.min(h - 1) {
            for x in x0..=x1.min(w - 1) {
                let g = (-((x as f32 - c[0]).powi(2) + (y as f32 - c[1]).powi(2)) * inv).exp();
                let i = y * w + x;
                if let (Some(s), Some(v)) = (sum_g.get_mut(i), sum_gd.get_mut(i)) {
                    *s += g;
                    v[0] += g * d[0];
                    v[1] += g * d[1];
                }
            }
        }
    }
    Some((sum_g, sum_gd))
}

fn steer(f: &mut FlowField, sum_g: &[f32], sum_gd: &[[f32; 2]]) {
    f.v.par_iter_mut().zip(sum_g.par_iter()).zip(sum_gd.par_iter()).for_each(|((v, &s), d)| {
        let big = s.min(1.0);
        if big <= 0.0 {
            return;
        }
        let k = s.max(1e-6);
        v[0] = v[0] * (1.0 - big) + d[0] / k * big;
        v[1] = v[1] * (1.0 - big) + d[1] / k * big;
    });
}

/// Steer `pair`'s flows with `guides` (analysis pixels), `radius` in analysis pixels. Non-finite
/// guides are ignored; a guide with `from == to` pins its neighbourhood to zero motion.
pub fn apply(pair: &mut FlowPair, guides: &[Guide], radius: f32) {
    let valid: Vec<&Guide> = guides.iter().filter(|g| g.valid()).collect();
    if valid.is_empty() || pair.w == 0 || pair.h == 0 {
        return;
    }
    let ab: Vec<([f32; 2], [f32; 2])> = valid.iter().map(|g| (g.from, [g.to[0] - g.from[0], g.to[1] - g.from[1]])).collect();
    let ba: Vec<([f32; 2], [f32; 2])> = valid.iter().map(|g| (g.to, [g.from[0] - g.to[0], g.from[1] - g.to[1]])).collect();
    for (centres, flow, occ, conf) in [(&ab, &mut pair.fab, &mut pair.occ_ab, &mut pair.conf_ab), (&ba, &mut pair.fba, &mut pair.occ_ba, &mut pair.conf_ba)] {
        let Some((sum_g, sum_gd)) = influence(pair.w, pair.h, centres, radius) else { continue };
        steer(flow, &sum_g, &sum_gd);
        for ((o, c), &s) in occ.iter_mut().zip(conf.iter_mut()).zip(&sum_g) {
            let big = s.min(1.0);
            *o *= 1.0 - big;
            *c += (1.0 - *c) * big;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Quality;

    fn still_pair(w: usize, h: usize) -> FlowPair {
        let z = FlowField { w, h, v: vec![[0.0; 2]; w * h] };
        FlowPair::from_flows(z.clone(), z, &crate::flow::FlowParams::for_quality(Quality::Draft, 0.7)).unwrap()
    }

    #[test]
    fn flow_at_the_guide_equals_its_motion_and_fades_out() {
        let mut pair = still_pair(160, 100);
        let g = Guide { from: [60.0, 50.0], to: [68.0, 47.0] };
        apply(&mut pair, &[g], 40.0);
        let v = pair.fab.at(60, 50);
        assert!((v[0] - 8.0).abs() < 0.05 && (v[1] + 3.0).abs() < 0.05, "{v:?}");
        // The B → A flow carries the reverse motion at the guide's position in B.
        let r = pair.fba.at(68, 47);
        assert!((r[0] + 8.0).abs() < 0.05 && (r[1] - 3.0).abs() < 0.05, "{r:?}");
        // σ = 20: at 3σ the influence is below 2 %.
        let far = pair.fab.at(60 + 60, 50);
        assert!(far[0].abs() < 0.02 * 8.0 + 1e-3, "{far:?}");
        // Occlusion is cleared and confidence is full under the guide.
        assert_eq!(pair.occ_ab[50 * 160 + 60], 0.0);
        assert!(pair.conf_ab[50 * 160 + 60] > 0.99);
    }

    #[test]
    fn two_guides_blend_and_zero_length_guides_pin() {
        let mut pair = still_pair(120, 60);
        let moving = Guide { from: [30.0, 30.0], to: [40.0, 30.0] };
        let pin = Guide { from: [90.0, 30.0], to: [90.0, 30.0] };
        pair.fab.v.iter_mut().for_each(|v| *v = [5.0, 5.0]);
        apply(&mut pair, &[moving, pin], 20.0);
        assert!((pair.fab.at(30, 30)[0] - 10.0).abs() < 0.2);
        assert!(pair.fab.at(90, 30)[0].abs() < 0.2 && pair.fab.at(90, 30)[1].abs() < 0.2);
        // Between them, untouched flow remains.
        let mid = pair.fab.at(60, 30);
        assert!((mid[1] - 5.0).abs() < 1.0, "{mid:?}");
    }

    #[test]
    fn hostile_guides_are_ignored() {
        let mut pair = still_pair(40, 40);
        let before = pair.clone();
        apply(&mut pair, &[Guide { from: [f32::NAN, 1.0], to: [2.0, 2.0] }, Guide { from: [1.0, f32::INFINITY], to: [2.0, 2.0] }], 30.0);
        assert_eq!(pair, before);
        apply(&mut pair, &[Guide { from: [-1e9, 1e9], to: [1e9, -1e9] }], f32::NAN);
        apply(&mut pair, &[Guide { from: [5.0, 5.0], to: [6.0, 6.0] }], -3.0);
        assert!(pair.fab.v.iter().all(|v| v[0].is_finite() && v[1].is_finite()));
    }
}
