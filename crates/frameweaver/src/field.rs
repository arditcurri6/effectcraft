//! Time fields: for an output frame at fraction `w` between A and B, where each output pixel
//! reads A and B, how visible each frame is there, and how far to trust the result.
//!
//! * **Inverse warping** finds, for each output pixel `x`, the point `u` of A with
//!   `x = u + w·F_ab(u)` and the point `q` of B with `x = q + (1 − w)·F_ba(q)` by fixed-point
//!   iteration. Smooth and hole-free, but where surfaces cross, the fixed point picks one.
//! * **Forward warping** splats every A pixel to `p + w·F_ab(p)` and every B pixel to
//!   `q + (1 − w)·F_ba(q)`, and resolves collisions by a depth test (an occluded pixel loses
//!   against a visible one; ties go to the smaller photometric error). It handles crossing
//!   surfaces correctly; holes are filled by push-pull.
//!
//! **Visibility** `v_a` is 0 where A cannot supply the pixel: the content is revealed
//! (`occ_ba` at the B point `q` — it has no counterpart in A). `v_b` is 0 where B cannot:
//! the content is being covered (`occ_ab` at the A point `u`). **Confidence** `conf` is the
//! trust in the warped result: 1 where both frames agree on the correspondence (their
//! confidences times how closely `u + F_ab(u)` meets `q`), the visible frame's confidence where
//! only one frame can supply the pixel, and 0 where neither can. The synthesis falls back to
//! the nearest source frame by `1 − conf`.

use rayon::prelude::*;

use crate::consistency::{FlowPair, push_pull};
use crate::flow::FlowField;
use crate::plane::Plane;

/// How the time field is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Warping {
    Inverse,
    Forward,
}

/// See the [module documentation](self).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimeField {
    pub width: usize,
    pub height: usize,
    /// Frame pixels per field pixel, per axis.
    pub scale: [f32; 2],
    /// Offsets (in frame pixels) from an output pixel to its sample position in A / B.
    pub to_a: Vec<[f32; 2]>,
    pub to_b: Vec<[f32; 2]>,
    pub vis_a: Vec<f32>,
    pub vis_b: Vec<f32>,
    pub conf: Vec<f32>,
}

/// A bilinear sample of a [`TimeField`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FieldSample {
    pub to_a: [f32; 2],
    pub to_b: [f32; 2],
    pub vis_a: f32,
    pub vis_b: f32,
    pub conf: f32,
}

impl TimeField {
    pub fn is_empty(&self) -> bool {
        let n = self.width * self.height;
        n == 0 || self.to_a.len() != n || self.to_b.len() != n || self.vis_a.len() != n || self.vis_b.len() != n || self.conf.len() != n
    }

    /// The field at frame position (`px`, `py`) (continuous, pixel centres at +0.5).
    #[inline]
    pub fn sample(&self, px: f32, py: f32) -> FieldSample {
        if self.is_empty() || !(px.is_finite() && py.is_finite()) {
            return FieldSample { vis_a: 1.0, vis_b: 1.0, conf: 0.0, ..FieldSample::default() };
        }
        let fx = (px / self.scale[0] - 0.5).clamp(-1.0, self.width as f32);
        let fy = (py / self.scale[1] - 0.5).clamp(-1.0, self.height as f32);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let (xi, yi) = (x0 as i64, y0 as i64);
        let idx = |x: i64, y: i64| y.clamp(0, self.height as i64 - 1) as usize * self.width + x.clamp(0, self.width as i64 - 1) as usize;
        let ids = [idx(xi, yi), idx(xi + 1, yi), idx(xi, yi + 1), idx(xi + 1, yi + 1)];
        let ws = [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty];
        let mut o = FieldSample::default();
        for (&i, &k) in ids.iter().zip(&ws) {
            let (a, b) = (self.to_a.get(i).copied().unwrap_or([0.0; 2]), self.to_b.get(i).copied().unwrap_or([0.0; 2]));
            o.to_a[0] += a[0] * k;
            o.to_a[1] += a[1] * k;
            o.to_b[0] += b[0] * k;
            o.to_b[1] += b[1] * k;
            o.vis_a += self.vis_a.get(i).copied().unwrap_or(1.0) * k;
            o.vis_b += self.vis_b.get(i).copied().unwrap_or(1.0) * k;
            o.conf += self.conf.get(i).copied().unwrap_or(0.0) * k;
        }
        o
    }

    pub fn bytes(&self) -> usize {
        self.width * self.height * (8 + 8 + 4 + 4 + 4)
    }
}

#[inline]
fn bilerp(v: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if !(x.is_finite() && y.is_finite()) || w == 0 || h == 0 {
        return 0.0;
    }
    let (x0, y0) = (x.floor(), y.floor());
    let (tx, ty) = (x - x0, y - y0);
    let g = |xx: i64, yy: i64| v.get(yy.clamp(0, h as i64 - 1) as usize * w + xx.clamp(0, w as i64 - 1) as usize).copied().unwrap_or(0.0);
    let (xi, yi) = (x0.clamp(-1e6, 1e6) as i64, y0.clamp(-1e6, 1e6) as i64);
    let top = g(xi, yi) + (g(xi + 1, yi) - g(xi, yi)) * tx;
    let bot = g(xi, yi + 1) + (g(xi + 1, yi + 1) - g(xi, yi + 1)) * tx;
    top + (bot - top) * ty
}

/// Residual (field pixels) at which two frames' correspondence is trusted at 1/e^0.5.
const RESIDUAL_SIGMA: f32 = 1.5;
/// Residual (field pixels) above which `u` and `q` are different scene points.
const RESIDUAL_CONFLICT: f32 = 2.0;
/// Residual (field pixels) of `x = u + w·F(u)` within which a fixed point counts as a solution.
const FIXED_POINT_TOLERANCE: f32 = 1.0;

/// Inverse-warped time field at fraction `w` ∈ 0…1. `a` / `b` are the analysis planes the pair
/// was estimated on; `frame` is the frame size; the field is at the pair's (analysis)
/// resolution.
///
/// Per output pixel `x`, two candidate scene points: `u` in A (`x = u + w·F_ab(u)`) and `q` in
/// B (`x = q + (1 − w)·F_ba(q)`), by fixed-point iteration. When they agree (the residual
/// `|u + F_ab(u) − q|` is small) both frames are sampled there. Where they disagree — surfaces
/// crossing, an object about to cover the background — they are *different* scene points and
/// blending their samples would mix surfaces: the pixel takes a single **winner**, the
/// candidate that survives in the other frame (or, if both do, the better photometric match),
/// and both samples come from that point's own correspondence. If neither survives, neither
/// frame can show the pixel and the synthesis falls back to the nearest frame.
pub fn inverse(pair: &FlowPair, a: &Plane, b: &Plane, w: f32, frame: (u32, u32)) -> Option<TimeField> {
    let (fw, fh) = (pair.w, pair.h);
    if fw == 0 || fh == 0 || frame.0 == 0 || frame.1 == 0 || a.w != fw || a.h != fh || b.w != fw || b.h != fh {
        return None;
    }
    let w = if w.is_finite() { w.clamp(0.0, 1.0) } else { 0.0 };
    let n = fw * fh;
    let scale = [frame.0 as f32 / fw as f32, frame.1 as f32 / fh as f32];
    let rows: Vec<([f32; 2], [f32; 2], f32, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i % fw) as f32, (i / fw) as f32);
            let mut u = [x, y];
            let mut q = [x, y];
            for _ in 0..5 {
                let f = pair.fab.sample(u[0], u[1]);
                u = [x - w * f[0], y - w * f[1]];
                let g = pair.fba.sample(q[0], q[1]);
                q = [x - (1.0 - w) * g[0], y - (1.0 - w) * g[1]];
            }
            let fu = pair.fab.sample(u[0], u[1]);
            let gq = pair.fba.sample(q[0], q[1]);
            let (u_b, q_a) = ([u[0] + fu[0], u[1] + fu[1]], [q[0] + gq[0], q[1] + gq[1]]);
            let r = ((u_b[0] - q[0]).powi(2) + (u_b[1] - q[1]).powi(2)).sqrt();
            let (occ_u, occ_q) = (bilerp(&pair.occ_ab, fw, fh, u[0], u[1]), bilerp(&pair.occ_ba, fw, fh, q[0], q[1]));
            let (ca, cb) = (bilerp(&pair.conf_ab, fw, fh, u[0], u[1]), bilerp(&pair.conf_ba, fw, fh, q[0], q[1]));
            let px = |p: [f32; 2]| [(p[0] - x) * scale[0], (p[1] - y) * scale[1]];
            // A candidate only counts if it solves its own equation: where the flow jumps, the
            // fixed-point iteration can oscillate between two surfaces and end on a point that
            // does not land on `x` at all.
            let res_u = ((u[0] + w * fu[0] - x).powi(2) + (u[1] + w * fu[1] - y).powi(2)).sqrt();
            let res_q = ((q[0] + (1.0 - w) * gq[0] - x).powi(2) + (q[1] + (1.0 - w) * gq[1] - y).powi(2)).sqrt();
            let (valid_u, valid_q) = (res_u <= FIXED_POINT_TOLERANCE, res_q <= FIXED_POINT_TOLERANCE);
            if valid_u && valid_q && r <= RESIDUAL_CONFLICT {
                // One scene point seen from both sides.
                let (va, vb) = (1.0 - occ_q, 1.0 - occ_u);
                let rfac = (-(r * r) / (2.0 * RESIDUAL_SIGMA * RESIDUAL_SIGMA)).exp();
                let conf = va * vb * ca * cb * rfac + va * (1.0 - vb) * ca + (1.0 - va) * vb * cb;
                return (px(u), px(q), va, vb, conf.clamp(0.0, 1.0));
            }
            // One scene point, or two that disagree: take the one that is really at this pixel.
            let pick_u = match (valid_u, valid_q) {
                (false, false) => None,
                (true, false) => Some(true),
                (false, true) => Some(false),
                (true, true) => match (occ_u < 0.5, occ_q < 0.5) {
                    // Surfaces crossing: the one that survives in the other frame is in front.
                    (true, false) => Some(true),
                    (false, true) => Some(false),
                    (false, false) => None,
                    (true, true) => {
                        let err_u = (b.sample(u_b[0], u_b[1]) - a.sample(u[0], u[1])).abs();
                        let err_q = (a.sample(q_a[0], q_a[1]) - b.sample(q[0], q[1])).abs();
                        Some(err_u <= err_q)
                    }
                },
            };
            match pick_u {
                // The A point, and where it goes in B (A only if B hides it).
                Some(true) => (px(u), px(u_b), 1.0, 1.0 - occ_u, ca),
                // The B point, and where it came from in A (B only if A lacks it).
                Some(false) => (px(q_a), px(q), 1.0 - occ_q, 1.0, cb),
                None => (px(u), px(q), 0.0, 0.0, 0.0),
            }
        })
        .collect();
    let mut f = TimeField { width: fw, height: fh, scale, ..TimeField::default() };
    f.to_a.reserve(n);
    for (ta, tb, va, vb, c) in rows {
        f.to_a.push(ta);
        f.to_b.push(tb);
        f.vis_a.push(va);
        f.vis_b.push(vb);
        f.conf.push(c);
    }
    Some(f)
}

/// Per-pixel accumulator of the depth-tested splats of one frame.
struct Splat {
    key: Vec<f32>,
    wt: Vec<f32>,
    off: Vec<[f32; 2]>,
    conf: Vec<f32>,
}

impl Splat {
    fn new(n: usize) -> Splat {
        Splat { key: vec![f32::INFINITY; n], wt: vec![0.0; n], off: vec![[0.0; 2]; n], conf: vec![0.0; n] }
    }

    /// Splat a pixel whose sample point is `src` (field px) to `dst` with bilinear weights;
    /// `key` is its depth priority (lower wins).
    fn add(&mut self, w: usize, h: usize, src: [f32; 2], dst: [f32; 2], key: f32, conf: f32) {
        if !(dst[0].is_finite() && dst[1].is_finite()) {
            return;
        }
        let (x0, y0) = (dst[0].floor(), dst[1].floor());
        let (fx, fy) = (dst[0] - x0, dst[1] - y0);
        for (dx, dy, k) in [(0i64, 0i64, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            let (px, py) = (x0 as i64 + dx, y0 as i64 + dy);
            if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 || k <= 1e-4 {
                continue;
            }
            let j = py as usize * w + px as usize;
            let (Some(bk), Some(wt), Some(off), Some(cf)) = (self.key.get_mut(j), self.wt.get_mut(j), self.off.get_mut(j), self.conf.get_mut(j)) else {
                continue;
            };
            if key < *bk - 0.05 {
                *bk = key;
                *wt = 0.0;
                *off = [0.0; 2];
                *cf = 0.0;
            } else if key > *bk + 0.05 {
                continue;
            }
            *wt += k;
            // Relative to where the splat actually landed, not to the pixel it spreads onto:
            // the sample position is then `pixel − motion` however little of the pixel the
            // splat covers (exact for locally uniform motion).
            off[0] += (src[0] - dst[0]) * k;
            off[1] += (src[1] - dst[1]) * k;
            *cf += conf * k;
        }
    }
}

/// The depth-test priority of a splat (lower wins): occluded pixels rank behind visible ones;
/// between two visible pixels the better photometric match wins. An occluded pixel's match
/// error compares it with a point it does not correspond to, so it takes no part in the
/// ranking (otherwise neighbouring occluded splats discard each other).
fn depth_key(occ: f32, err: f32) -> f32 {
    if occ >= 0.5 { occ * 10.0 } else { occ * 10.0 + err }
}

/// Forward-warped time field at fraction `w`. `a` / `b` are the analysis planes the pair was
/// estimated on (for the photometric tie-break).
pub fn forward(pair: &FlowPair, a: &Plane, b: &Plane, w: f32, frame: (u32, u32)) -> Option<TimeField> {
    let (fw, fh) = (pair.w, pair.h);
    if fw == 0 || fh == 0 || frame.0 == 0 || frame.1 == 0 || a.w != fw || a.h != fh || b.w != fw || b.h != fh {
        return None;
    }
    let w = if w.is_finite() { w.clamp(0.0, 1.0) } else { 0.0 };
    let n = fw * fh;
    let scale = [frame.0 as f32 / fw as f32, frame.1 as f32 / fh as f32];
    let mut sa = Splat::new(n);
    let mut sb = Splat::new(n);
    for y in 0..fh {
        for x in 0..fw {
            let i = y * fw + x;
            let p = [x as f32, y as f32];
            let f = pair.fab.at(x as i64, y as i64);
            let err = (b.sample(p[0] + f[0], p[1] + f[1]) - a.d.get(i).copied().unwrap_or(0.0)).abs();
            let key = depth_key(pair.occ_ab.get(i).copied().unwrap_or(0.0), err);
            sa.add(fw, fh, p, [p[0] + w * f[0], p[1] + w * f[1]], key, pair.conf_ab.get(i).copied().unwrap_or(0.0));
            let g = pair.fba.at(x as i64, y as i64);
            let err = (a.sample(p[0] + g[0], p[1] + g[1]) - b.d.get(i).copied().unwrap_or(0.0)).abs();
            let key = depth_key(pair.occ_ba.get(i).copied().unwrap_or(0.0), err);
            sb.add(fw, fh, p, [p[0] + (1.0 - w) * g[0], p[1] + (1.0 - w) * g[1]], key, pair.conf_ba.get(i).copied().unwrap_or(0.0));
        }
    }
    let finish = |s: &Splat| -> Option<(FlowField, Plane, Vec<f32>)> {
        let avg = FlowField { w: fw, h: fh, v: s.off.iter().zip(&s.wt).map(|(o, &k)| if k > 1e-6 { [o[0] / k, o[1] / k] } else { [0.0; 2] }).collect() };
        let weight = Plane::from_vec(fw, fh, s.wt.iter().map(|&k| k.min(1.0)).collect())?;
        let filled = push_pull(&avg, &weight, [0.0; 2])?;
        let conf: Vec<f32> = s.conf.iter().zip(&s.wt).map(|(&c, &k)| if k > 1e-6 { c / k } else { 0.0 }).collect();
        Some((filled, weight, conf))
    };
    let (fa, wa, ca) = finish(&sa)?;
    let (fb, wb, cb) = finish(&sb)?;
    let mut f = TimeField { width: fw, height: fh, scale, ..TimeField::default() };
    for i in 0..n {
        let (ka, kb) = (wa.d.get(i).copied().unwrap_or(0.0), wb.d.get(i).copied().unwrap_or(0.0));
        let (ca, cb) = (ca.get(i).copied().unwrap_or(0.0), cb.get(i).copied().unwrap_or(0.0));
        // A frame that splatted onto the pixel can show it: a quarter of the pixel covered is
        // enough (partial coverage at an edge only means the splat landed between pixels).
        // A hole in both frames: both count as visible (hallucinated motion, confidence 0).
        let (va, vb) = if ka < 1e-3 && kb < 1e-3 { (1.0, 1.0) } else { ((ka * 4.0).min(1.0), (kb * 4.0).min(1.0)) };
        let conf = if ka < 1e-3 && kb < 1e-3 { 0.0 } else { va * vb * ca * cb + va * (1.0 - vb) * ca + (1.0 - va) * vb * cb };
        let o = |v: [f32; 2]| [v[0] * scale[0], v[1] * scale[1]];
        f.to_a.push(o(fa.v.get(i).copied().unwrap_or([0.0; 2])));
        f.to_b.push(o(fb.v.get(i).copied().unwrap_or([0.0; 2])));
        f.vis_a.push(va);
        f.vis_b.push(vb);
        f.conf.push(conf.clamp(0.0, 1.0));
    }
    Some(f)
}
