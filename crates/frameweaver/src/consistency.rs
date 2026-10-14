//! Forward-backward consistency of a flow pair: occlusion and confidence maps, repair of the
//! unreliable vectors, and a global-motion fallback.
//!
//! For a pixel `x` of A with flow `F_ab(x)`, the opposite flow at its target should return it:
//! `F_ab(x) + F_ba(x + F_ab(x)) ≈ 0`. A violation means one of
//!
//! * **occlusion** — the target is outside the frame, or the pixel sits at a motion boundary
//!   (the flow jumps within a few pixels) where another surface covers it. The pixel has no
//!   counterpart: its flow is meaningless and the pixel is visible in one frame only;
//! * **unreliable motion** — inconsistent without a motion boundary (flat texture, blur,
//!   repeated pattern): a correspondence exists but is not trustworthy, which lowers the
//!   *confidence* (and the synthesis then falls back towards the nearest frame);
//!
//! and a pair that is consistent has confidence 1.
//!
//! The unreliable and occluded vectors are then *repaired*: replaced by a confidence-weighted
//! push-pull fill from the reliable surroundings (or by the global affine motion when almost
//! nothing in the frame is reliable), so that the synthesis can warp the one frame that does
//! show a covered or revealed region along its surroundings' motion.

use rayon::prelude::*;

use crate::flow::{FlowField, FlowParams};
use crate::plane::Plane;

/// A flow pair at analysis resolution with its occlusion and confidence maps.
///
/// `occ_ab[i]` ∈ 0…1 is how surely A's pixel `i` is hidden in B (covered, or leaving the
/// frame); `occ_ba[i]` how surely B's pixel `i` is absent from A (revealed, or entering the
/// frame). `conf_*` ∈ 0…1 is the trust in the (repaired) vector at that pixel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlowPair {
    pub w: usize,
    pub h: usize,
    pub fab: FlowField,
    pub fba: FlowField,
    pub occ_ab: Vec<f32>,
    pub occ_ba: Vec<f32>,
    pub conf_ab: Vec<f32>,
    pub conf_ba: Vec<f32>,
}

/// Occlusion and confidence of one direction.
struct Maps {
    occ: Vec<f32>,
    conf: Vec<f32>,
    /// Hard occlusion mask (1 = occluded).
    hard: Vec<f32>,
    /// Hard occlusion or low confidence (1 = not reliable), for repair.
    bad: Vec<f32>,
}

/// Motion-boundary strength: the flow's central-difference gradient magnitude (px per px),
/// dilated by 2 pixels.
fn motion_boundary(f: &FlowField) -> Option<Plane> {
    let (w, h) = (f.w, f.h);
    let mut g = Plane::new(w, h)?;
    g.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let (xi, yi) = (x as i64, y as i64);
            let (l, r, t, d) = (f.at(xi - 1, yi), f.at(xi + 1, yi), f.at(xi, yi - 1), f.at(xi, yi + 1));
            let gx = (r[0] - l[0]).hypot(r[1] - l[1]) * 0.5;
            let gy = (d[0] - t[0]).hypot(d[1] - t[1]) * 0.5;
            *o = gx.max(gy);
        }
    });
    // Max-dilate by 2 (separable).
    let mut tmp = Plane::new(w, h)?;
    tmp.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = (-2..=2).map(|d| g.at(x as i64 + d, y as i64)).fold(0.0, f32::max);
        }
    });
    let mut out = Plane::new(w, h)?;
    out.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = (-2..=2).map(|d| tmp.at(x as i64, y as i64 + d)).fold(0.0, f32::max);
        }
    });
    Some(out)
}

/// Threshold on the dilated motion-boundary strength above which an inconsistency counts as an
/// occlusion rather than as unreliable motion.
const BOUNDARY: f32 = 0.35;

/// Confidence given to an occluded pixel (visible in one frame only, vector repaired).
const OCCLUDED_TRUST: f32 = 1.0;

fn analyze(f: &FlowField, g: &FlowField) -> Option<Maps> {
    let (w, h) = (f.w, f.h);
    let boundary = motion_boundary(f)?;
    let n = w * h;
    let rows: Vec<(f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let v = f.v.get(i).copied().unwrap_or([0.0; 2]);
            let (tx, ty) = (x + v[0], y + v[1]);
            let oof = tx < -1.0 || ty < -1.0 || tx > w as f32 || ty > h as f32;
            let gb = g.sample(tx, ty);
            let r2 = (v[0] + gb[0]).powi(2) + (v[1] + gb[1]).powi(2);
            let mag = v[0] * v[0] + v[1] * v[1] + gb[0] * gb[0] + gb[1] * gb[1];
            let thr2 = 0.01 * mag + 0.5;
            let inconsistent = r2 > thr2;
            let near_boundary = boundary.d.get(i).copied().unwrap_or(0.0) > BOUNDARY;
            let occluded = oof || (inconsistent && near_boundary);
            let conf = if !inconsistent { 1.0 } else { (-(r2 / thr2).sqrt().min(8.0) + 1.0).exp().clamp(0.0, 1.0) };
            (if occluded { 1.0 } else { 0.0 }, conf)
        })
        .collect();
    let bad: Vec<f32> = rows.iter().map(|&(o, c)| if o > 0.5 || c < 0.5 { 1.0 } else { 0.0 }).collect();
    // An explained occlusion is not an unreliable correspondence: the pixel is simply visible
    // in one frame, whose own sample (along the repaired vector) is trusted almost fully.
    let conf: Vec<f32> = rows.iter().map(|&(o, c)| if o > 0.5 { OCCLUDED_TRUST } else { c }).collect();
    let hard = Plane::from_vec(w, h, rows.iter().map(|r| r.0).collect())?;
    let occ = hard.box_mean(1)?.d;
    let hard_v = hard.d.clone();
    Some(Maps { occ, conf, hard: hard_v, bad })
}

/// Average each consistent vector with the inverted opposite flow (precision).
fn symmetrize(f: &FlowField, g: &FlowField, conf: &[f32]) -> FlowField {
    let mut out = f.clone();
    let w = f.w;
    out.v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            if conf.get(y * w + x).copied().unwrap_or(0.0) < 0.99 {
                continue;
            }
            let v = f.at(x as i64, y as i64);
            let gb = g.sample(x as f32 + v[0], y as f32 + v[1]);
            *o = [0.5 * (v[0] - gb[0]), 0.5 * (v[1] - gb[1])];
        }
    });
    out
}

/// Weighted push-pull fill: pixels with weight ≥ 1 keep their value, the rest take the
/// weight-averaged value of their surroundings, found on a pyramid (down: sum weighted values
/// and weights; up: blend the coarser fill in where the weight is missing). With no weight
/// anywhere the result is `fallback`.
pub fn push_pull(f: &FlowField, weight: &Plane, fallback: [f32; 2]) -> Option<FlowField> {
    if f.w != weight.w || f.h != weight.h || f.is_empty() {
        return None;
    }
    // Level data: sums of value × weight, and weights.
    struct Level {
        w: usize,
        h: usize,
        sum: Vec<[f32; 2]>,
        wt: Vec<f32>,
    }
    let mut levels = vec![Level { w: f.w, h: f.h, sum: f.v.iter().zip(&weight.d).map(|(v, &k)| [v[0] * k, v[1] * k]).collect(), wt: weight.d.clone() }];
    while let Some(l) = levels.last() {
        if l.w <= 1 && l.h <= 1 {
            break;
        }
        let (nw, nh) = (l.w.div_ceil(2), l.h.div_ceil(2));
        let mut sum = vec![[0.0f32; 2]; nw * nh];
        let mut wt = vec![0.0f32; nw * nh];
        for y in 0..l.h {
            for x in 0..l.w {
                let (i, j) = (y * l.w + x, (y / 2) * nw + x / 2);
                if let (Some(s), Some(t), Some(ds), Some(dw)) = (l.sum.get(i), l.wt.get(i), sum.get_mut(j), wt.get_mut(j)) {
                    ds[0] += s[0];
                    ds[1] += s[1];
                    *dw += *t;
                }
            }
        }
        levels.push(Level { w: nw, h: nh, sum, wt });
    }
    // Up: from the coarsest, each level's dense fill.
    let mut coarse: Option<(usize, usize, Vec<[f32; 2]>)> = None;
    for l in levels.iter().rev() {
        let mut fill = vec![[0.0f32; 2]; l.w * l.h];
        for y in 0..l.h {
            for x in 0..l.w {
                let i = y * l.w + x;
                let up = match &coarse {
                    Some((cw, ch, cv)) => {
                        let (fx, fy) = ((x as f32 + 0.5) / 2.0 - 0.5, (y as f32 + 0.5) / 2.0 - 0.5);
                        let (x0, y0) = (fx.floor(), fy.floor());
                        let (tx, ty) = (fx - x0, fy - y0);
                        let g = |xx: i64, yy: i64| {
                            let xx = xx.clamp(0, *cw as i64 - 1) as usize;
                            let yy = yy.clamp(0, *ch as i64 - 1) as usize;
                            cv.get(yy * cw + xx).copied().unwrap_or(fallback)
                        };
                        let (a, b, c, d) = (g(x0 as i64, y0 as i64), g(x0 as i64 + 1, y0 as i64), g(x0 as i64, y0 as i64 + 1), g(x0 as i64 + 1, y0 as i64 + 1));
                        let mut o = [0.0; 2];
                        for k in 0..2 {
                            let top = a[k] + (b[k] - a[k]) * tx;
                            let bot = c[k] + (d[k] - c[k]) * tx;
                            o[k] = top + (bot - top) * ty;
                        }
                        o
                    }
                    None => fallback,
                };
                let (s, k) = (l.sum.get(i).copied().unwrap_or([0.0; 2]), l.wt.get(i).copied().unwrap_or(0.0));
                let miss = (1.0 - k).max(0.0);
                let den = k.max(1.0);
                if let Some(o) = fill.get_mut(i) {
                    *o = [(s[0] + miss * up[0]) / den, (s[1] + miss * up[1]) / den];
                }
            }
        }
        coarse = Some((l.w, l.h, fill));
    }
    let (_, _, v) = coarse?;
    Some(FlowField { w: f.w, h: f.h, v })
}

/// A global affine motion `F(p) = [a b; c d]·(p − centre) + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub m: [f32; 4],
    pub t: [f32; 2],
    pub centre: [f32; 2],
}

impl Affine {
    pub fn eval(&self, x: f32, y: f32) -> [f32; 2] {
        let (dx, dy) = (x - self.centre[0], y - self.centre[1]);
        [self.m[0] * dx + self.m[1] * dy + self.t[0], self.m[2] * dx + self.m[3] * dy + self.t[1]]
    }
}

/// Robust (iteratively re-weighted) least-squares affine fit of the flow over the pixels with
/// `weight > 0.5`, sampled on a grid of about 64×64 points. `None` with too few reliable pixels.
pub fn global_affine(f: &FlowField, weight: &[f32]) -> Option<Affine> {
    let (w, h) = (f.w, f.h);
    if f.is_empty() || weight.len() != w * h {
        return None;
    }
    let centre = [(w as f32 - 1.0) * 0.5, (h as f32 - 1.0) * 0.5];
    let (sx, sy) = ((w / 64).max(1), (h / 64).max(1));
    let mut pts: Vec<(f32, f32, [f32; 2])> = vec![];
    for y in (0..h).step_by(sy) {
        for x in (0..w).step_by(sx) {
            if weight.get(y * w + x).copied().unwrap_or(0.0) > 0.5 {
                pts.push((x as f32 - centre[0], y as f32 - centre[1], f.at(x as i64, y as i64)));
            }
        }
    }
    if pts.len() < 12 {
        return None;
    }
    let mut cur: Option<Affine> = None;
    for _ in 0..4 {
        // Normal equations of [dx dy 1] · p = v for each output component.
        let mut ata = [[0.0f64; 3]; 3];
        let mut atb = [[0.0f64; 3]; 2];
        for &(dx, dy, v) in &pts {
            let k = match &cur {
                Some(a) => {
                    let e = a.eval(dx + centre[0], dy + centre[1]);
                    1.0 / (1.0 + ((e[0] - v[0]).powi(2) + (e[1] - v[1]).powi(2)) as f64)
                }
                None => 1.0,
            };
            let row = [dx as f64, dy as f64, 1.0];
            for i in 0..3 {
                for j in 0..3 {
                    ata[i][j] += k * row[i] * row[j];
                }
                atb[0][i] += k * row[i] * v[0] as f64;
                atb[1][i] += k * row[i] * v[1] as f64;
            }
        }
        let p0 = solve3(ata, atb[0])?;
        let p1 = solve3(ata, atb[1])?;
        cur = Some(Affine { m: [p0[0] as f32, p0[1] as f32, p1[0] as f32, p1[1] as f32], t: [p0[2] as f32, p1[2] as f32], centre });
    }
    cur
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    if d.abs() < 1e-9 || !d.is_finite() {
        return None;
    }
    let mut out = [0.0; 3];
    for k in 0..3 {
        let mut m = a;
        for r in 0..3 {
            m[r][k] = b[r];
        }
        out[k] = det(m) / d;
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// Replace the unreliable vectors (`bad > 0.5`) of `f`.
///
/// * Occluded pixels (`hard > 0.5`) take the motion of the *dominant surface*: a weighted
///   push-pull fill from the reliable pixels whose vector is within 2 px of the global affine
///   fit. A covered or revealed region belongs to the surface that stays behind, which is the
///   image's dominant motion, not the occluder crossing it; averaging both sides would invent
///   a motion neither surface has.
/// * Merely low-confidence pixels take an unbiased push-pull fill of their reliable
///   neighbours (they may sit inside the moving object).
/// * With `global` set and under a quarter of the pixels reliable, all of them take the
///   global affine motion instead.
fn repair(f: &FlowField, hard: &[f32], bad: &[f32], global: bool) -> Option<FlowField> {
    let good: Vec<f32> = bad.iter().map(|&b| if b > 0.5 { 0.0 } else { 1.0 }).collect();
    let frac = good.iter().sum::<f32>() / good.len().max(1) as f32;
    if frac > 0.999 {
        return Some(f.clone());
    }
    let dominant = global_affine(f, &good);
    let w = f.w;
    if global
        && frac < 0.25
        && let Some(a) = dominant
    {
        let mut out = f.clone();
        out.v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                if good.get(y * w + x).copied().unwrap_or(0.0) < 0.5 {
                    *o = a.eval(x as f32, y as f32);
                }
            }
        });
        return Some(out);
    }
    let centre = dominant.map_or([0.0; 2], |a| a.eval(a.centre[0], a.centre[1]));
    let plain = push_pull(f, &Plane::from_vec(f.w, f.h, good.clone())?, centre)?;
    let background = Plane::from_vec(
        f.w,
        f.h,
        f.v.iter()
            .enumerate()
            .zip(&good)
            .map(|((i, v), &g)| {
                let d = dominant.map_or([0.0; 2], |a| a.eval((i % w) as f32, (i / w) as f32));
                if g > 0.5 && (v[0] - d[0]).powi(2) + (v[1] - d[1]).powi(2) < 4.0 { 1.0 } else { 0.0 }
            })
            .collect(),
    )?;
    let bg = push_pull(f, &background, centre)?;
    let mut out = f.clone();
    out.v.par_iter_mut().enumerate().for_each(|(i, o)| {
        if good.get(i).copied().unwrap_or(1.0) < 0.5 {
            let src = if hard.get(i).copied().unwrap_or(0.0) > 0.5 { &bg } else { &plain };
            *o = src.v.get(i).copied().unwrap_or([0.0; 2]);
        }
    });
    Some(out)
}

impl FlowPair {
    /// Build from independently estimated flows: consistency analysis (twice with Best's
    /// symmetrisation), then repair.
    pub fn from_flows(fab: FlowField, fba: FlowField, p: &FlowParams) -> Option<FlowPair> {
        if fab.is_empty() || fab.w != fba.w || fab.h != fba.h {
            return None;
        }
        let (mut fab, mut fba) = (fab, fba);
        let mut ab = analyze(&fab, &fba)?;
        let mut ba = analyze(&fba, &fab)?;
        if p.symmetrize {
            let (sab, sba) = (symmetrize(&fab, &fba, &ab.conf), symmetrize(&fba, &fab, &ba.conf));
            fab = sab;
            fba = sba;
            ab = analyze(&fab, &fba)?;
            ba = analyze(&fba, &fab)?;
        }
        let rab = repair(&fab, &ab.hard, &ab.bad, p.global_fallback)?;
        let rba = repair(&fba, &ba.hard, &ba.bad, p.global_fallback)?;
        // Confidence describes the vectors that will be used, so it is measured again on the
        // repaired pair: a pixel whose own flow is right but whose counterpart in the other
        // frame was corrupted (and has since been repaired) is not unreliable.
        let (ab2, ba2) = (analyze(&rab, &rba)?, analyze(&rba, &rab)?);
        let final_conf =
            |hard: &[f32], second: Vec<f32>| -> Vec<f32> { hard.iter().zip(second).map(|(&h, c)| if h > 0.5 { OCCLUDED_TRUST } else { c }).collect() };
        let (conf_ab, conf_ba) = (final_conf(&ab.hard, ab2.conf), final_conf(&ba.hard, ba2.conf));
        Some(FlowPair { w: fab.w, h: fab.h, fab: rab, fba: rba, occ_ab: ab.occ, occ_ba: ba.occ, conf_ab, conf_ba })
    }

    /// Estimate both directions between analysis planes (in parallel) and analyse them.
    pub fn estimate(a: &Plane, b: &Plane, p: &FlowParams) -> Result<FlowPair, crate::FlowError> {
        let (fab, fba) = rayon::join(|| crate::flow::estimate(a, b, p), || crate::flow::estimate(b, a, p));
        FlowPair::from_flows(fab?, fba?, p).ok_or(crate::FlowError::Empty)
    }

    /// Build from a forward flow alone (imported vectors): the reverse flow is the forward one
    /// negated and splatted to its targets; holes are filled by push-pull.
    pub fn from_forward(fab: FlowField, p: &FlowParams) -> Option<FlowPair> {
        let fba = reverse(&fab)?;
        FlowPair::from_flows(fab, fba, &FlowParams { symmetrize: false, ..*p })
    }

    /// The pair with A and B exchanged.
    pub fn swapped(self) -> FlowPair {
        FlowPair { w: self.w, h: self.h, fab: self.fba, fba: self.fab, occ_ab: self.occ_ba, occ_ba: self.occ_ab, conf_ab: self.conf_ba, conf_ba: self.conf_ab }
    }

    /// Approximate heap size, for the cache.
    pub fn bytes(&self) -> usize {
        self.w * self.h * (8 + 8 + 4 * 4)
    }
}

/// The flow B → A implied by `f` (A → B): each vector negated and bilinearly splatted onto its
/// target, holes filled.
pub fn reverse(f: &FlowField) -> Option<FlowField> {
    let (w, h) = (f.w, f.h);
    let mut sum = vec![[0.0f32; 2]; w * h];
    let mut wt = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = f.at(x as i64, y as i64);
            let (tx, ty) = (x as f32 + v[0], y as f32 + v[1]);
            if !(tx.is_finite() && ty.is_finite()) {
                continue;
            }
            let (x0, y0) = (tx.floor(), ty.floor());
            let (fx, fy) = (tx - x0, ty - y0);
            for (dx, dy, k) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
                let (px, py) = (x0 as i64 + dx, y0 as i64 + dy);
                if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 || k <= 0.0 {
                    continue;
                }
                let j = py as usize * w + px as usize;
                if let (Some(s), Some(t)) = (sum.get_mut(j), wt.get_mut(j)) {
                    s[0] -= v[0] * k;
                    s[1] -= v[1] * k;
                    *t += k;
                }
            }
        }
    }
    let avg = FlowField { w, h, v: sum.iter().zip(&wt).map(|(s, &k)| if k > 1e-6 { [s[0] / k, s[1] / k] } else { [0.0; 2] }).collect() };
    let weight = Plane::from_vec(w, h, wt.iter().map(|&k| k.min(1.0)).collect())?;
    push_pull(&avg, &weight, [0.0; 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{Quality, estimate};
    use crate::testkit::*;

    fn constant(w: usize, h: usize, v: [f32; 2]) -> FlowField {
        FlowField { w, h, v: vec![v; w * h] }
    }

    #[test]
    fn constant_motion_is_fully_consistent() {
        let p = FlowParams::for_quality(Quality::Draft, 0.7);
        let pair = FlowPair::from_flows(constant(60, 40, [3.0, 0.0]), constant(60, 40, [-3.0, 0.0]), &p).unwrap();
        // Only pixels whose target leaves the frame are occluded.
        let inside = (10..40).map(|x| pair.occ_ab[20 * 60 + x]).fold(0.0, f32::max);
        assert_eq!(inside, 0.0);
        assert!(pair.conf_ab[20 * 60 + 20] > 0.99);
    }

    #[test]
    fn motion_leaving_the_frame_is_occluded() {
        let p = FlowParams::for_quality(Quality::Draft, 0.7);
        let pair = FlowPair::from_flows(constant(60, 40, [8.0, 0.0]), constant(60, 40, [-8.0, 0.0]), &p).unwrap();
        // A's right edge leaves; B's left edge was not in A.
        assert!(pair.occ_ab[20 * 60 + 59] > 0.5, "{}", pair.occ_ab[20 * 60 + 59]);
        assert!(pair.occ_ba[20 * 60] > 0.5);
    }

    #[test]
    fn push_pull_fills_holes_and_uses_fallback() {
        let mut f = constant(32, 32, [2.0, -1.0]);
        let mut wgt = Plane::filled(32, 32, 1.0).unwrap();
        for y in 10..20 {
            for x in 10..20 {
                f.v[y * 32 + x] = [99.0, 99.0];
                wgt.d[y * 32 + x] = 0.0;
            }
        }
        let r = push_pull(&f, &wgt, [0.0; 2]).unwrap();
        let c = r.at(15, 15);
        assert!((c[0] - 2.0).abs() < 0.05 && (c[1] + 1.0).abs() < 0.05, "{c:?}");
        let none = push_pull(&f, &Plane::filled(32, 32, 0.0).unwrap(), [5.0, 6.0]).unwrap();
        assert_eq!(none.at(3, 3), [5.0, 6.0]);
    }

    #[test]
    fn global_affine_recovers_a_zoom_with_outliers() {
        let (w, h) = (80, 60);
        let mut f = FlowField::new(w, h).unwrap();
        let c = [(w as f32 - 1.0) / 2.0, (h as f32 - 1.0) / 2.0];
        for y in 0..h {
            for x in 0..w {
                f.v[y * w + x] = [0.03 * (x as f32 - c[0]) + 1.0, 0.03 * (y as f32 - c[1]) - 2.0];
            }
        }
        for i in (0..w * h).step_by(7) {
            f.v[i] = [30.0, -30.0];
        }
        let a = global_affine(&f, &vec![1.0; w * h]).unwrap();
        assert!((a.m[0] - 0.03).abs() < 0.01 && (a.t[0] - 1.0).abs() < 0.8 && (a.t[1] + 2.0).abs() < 0.8, "{a:?}");
    }

    #[test]
    fn reverse_of_a_translation_is_its_negation() {
        let r = reverse(&constant(48, 32, [4.0, 1.0])).unwrap();
        let v = r.at(24, 16);
        assert!((v[0] + 4.0).abs() < 1e-3 && (v[1] + 1.0).abs() < 1e-3);
    }

    /// A square moving over a static textured background: the strip it uncovers is flagged
    /// occluded in B, the strip it covers occluded in A, and the flow in them is repaired to
    /// the background's (zero) motion.
    #[test]
    fn moving_square_over_background_marks_both_strips() {
        let (w, h, d) = (128usize, 96usize, 10.0f32);
        let scene = |sq_x: f32| {
            render(w, h, move |x, y| if x >= sq_x && x < sq_x + 40.0 && (28.0..68.0).contains(&y) { (x - sq_x + 500.0, y + 300.0) } else { (x, y) })
        };
        let (a, b) = (scene(30.0), scene(30.0 + d));
        let p = FlowParams::for_quality(Quality::Best, 0.7);
        let fab = estimate(&a, &b, &p).unwrap();
        let fba = estimate(&b, &a, &p).unwrap();
        let pair = FlowPair::from_flows(fab, fba, &p).unwrap();
        // Covered strip in A: x ∈ [70, 80); revealed strip in B: x ∈ [30, 40).
        let frac = |occ: &[f32], x0: usize, x1: usize| {
            let (mut s, mut n) = (0.0, 0.0);
            for y in 34..62 {
                for x in x0..x1 {
                    s += occ[y * w + x];
                    n += 1.0;
                }
            }
            s / n
        };
        let covered = frac(&pair.occ_ab, 72, 78);
        let revealed = frac(&pair.occ_ba, 32, 38);
        let elsewhere = frac(&pair.occ_ab, 4, 20).max(frac(&pair.occ_ba, 100, 120));
        assert!(covered >= 0.7, "covered strip {covered}");
        assert!(revealed >= 0.7, "revealed strip {revealed}");
        assert!(elsewhere <= 0.1, "elsewhere {elsewhere}");
        // Repaired flow in the revealed strip of B follows the background (≈ 0), not the square.
        let v = pair.fba.at(35, 48);
        assert!(v[0].abs() < 4.0, "repaired reveal flow {v:?}");
    }
}
