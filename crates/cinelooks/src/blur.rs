//! A deterministic Gaussian blur for large radii: halve the image with 2×2 box averages until
//! the remaining sigma is small, blur there with a separable Gaussian, and come back up with
//! bilinear sampling. Borders clamp to the edge (no dark frame edges). The GPU kernels
//! (`fcl_down`, `fcl_blur`, `fcl_up`) run the same plan with the same weights.

use rayon::prelude::*;

/// Most halvings (sigma up to ~256 px blurred at a 4.0 px working sigma).
pub const MAX_LEVELS: u32 = 6;
/// Largest kernel radius at the working level.
pub const MAX_RADIUS: usize = 32;

/// How a blur of `sigma` pixels runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Number of halvings.
    pub levels: u32,
    /// Normalised kernel weights, centre first (`weights[0]` is the centre tap).
    pub weights: Vec<f32>,
}

impl Plan {
    pub fn new(sigma: f32) -> Plan {
        let mut s = if sigma.is_finite() { sigma.max(0.0) } else { 0.0 };
        let mut levels = 0;
        while s > 4.0 && levels < MAX_LEVELS {
            s *= 0.5;
            levels += 1;
        }
        let r = ((s * 3.0).ceil() as usize).clamp(0, MAX_RADIUS);
        let mut w: Vec<f32> = (0..=r).map(|i| if s < 1e-3 { if i == 0 { 1.0 } else { 0.0 } } else { (-((i * i) as f32) / (2.0 * s * s)).exp() }).collect();
        let total: f32 = w.iter().skip(1).sum::<f32>() * 2.0 + w.first().copied().unwrap_or(1.0);
        for v in &mut w {
            *v /= total.max(1e-9);
        }
        Plan { levels, weights: w }
    }

    pub fn is_identity(&self) -> bool {
        self.levels == 0 && self.weights.len() <= 1
    }
}

/// Level sizes from `w`×`h` down `levels` halvings (ceil).
pub fn sizes(w: usize, h: usize, levels: u32) -> Vec<(usize, usize)> {
    let mut v = vec![(w, h)];
    for _ in 0..levels {
        let &(a, b) = v.last().unwrap_or(&(w, h));
        v.push((a.div_ceil(2).max(1), b.div_ceil(2).max(1)));
    }
    v
}

fn down<const N: usize>(src: &[[f32; N]], w: usize, h: usize) -> (Vec<[f32; N]>, usize, usize) {
    let (dw, dh) = (w.div_ceil(2).max(1), h.div_ceil(2).max(1));
    let mut out = vec![[0.0f32; N]; dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let x0 = (2 * x).min(w - 1);
            let x1 = (2 * x + 1).min(w - 1);
            let y0 = (2 * y).min(h - 1);
            let y1 = (2 * y + 1).min(h - 1);
            let g = |xx: usize, yy: usize| src.get(yy * w + xx).copied().unwrap_or([0.0; N]);
            let (a, b, c, d) = (g(x0, y0), g(x1, y0), g(x0, y1), g(x1, y1));
            for k in 0..N {
                px[k] = (a[k] + b[k] + c[k] + d[k]) * 0.25;
            }
        }
    });
    (out, dw, dh)
}

fn pass<const N: usize>(src: &[[f32; N]], w: usize, h: usize, wt: &[f32], horizontal: bool) -> Vec<[f32; N]> {
    let mut out = vec![[0.0f32; N]; w * h];
    let r = wt.len().saturating_sub(1) as i64;
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let mut acc = [0.0f32; N];
            for i in -r..=r {
                let wv = wt.get(i.unsigned_abs() as usize).copied().unwrap_or(0.0);
                let (sx, sy) =
                    if horizontal { ((x as i64 + i).clamp(0, w as i64 - 1) as usize, y) } else { (x, (y as i64 + i).clamp(0, h as i64 - 1) as usize) };
                let s = src.get(sy * w + sx).copied().unwrap_or([0.0; N]);
                for k in 0..N {
                    acc[k] += s[k] * wv;
                }
            }
            *px = acc;
        }
    });
    out
}

/// Bilinear sample of a `w`×`h` buffer at continuous position (pixel centres at +0.5),
/// clamped to the edge.
#[inline]
pub fn sample<const N: usize>(src: &[[f32; N]], w: usize, h: usize, x: f32, y: f32) -> [f32; N] {
    if w == 0 || h == 0 {
        return [0.0; N];
    }
    let fx = (x - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (y - 0.5).clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let g = |xx: usize, yy: usize| src.get(yy * w + xx).copied().unwrap_or([0.0; N]);
    let (a, b, c, d) = (g(x0, y0), g(x1, y0), g(x0, y1), g(x1, y1));
    let mut o = [0.0f32; N];
    for k in 0..N {
        let top = a[k] + (b[k] - a[k]) * tx;
        let bot = c[k] + (d[k] - c[k]) * tx;
        o[k] = top + (bot - top) * ty;
    }
    o
}

/// Blur `src` (`w`×`h`) with `plan`; same size out.
pub fn blur<const N: usize>(src: &[[f32; N]], w: usize, h: usize, plan: &Plan) -> Vec<[f32; N]> {
    if w == 0 || h == 0 || src.len() != w * h {
        return src.to_vec();
    }
    if plan.is_identity() {
        return src.to_vec();
    }
    let mut cur = src.to_vec();
    let (mut cw, mut ch) = (w, h);
    for _ in 0..plan.levels {
        let (d, dw, dh) = down(&cur, cw, ch);
        (cur, cw, ch) = (d, dw, dh);
    }
    if plan.weights.len() > 1 {
        cur = pass(&cur, cw, ch, &plan.weights, true);
        cur = pass(&cur, cw, ch, &plan.weights, false);
    }
    if plan.levels == 0 {
        return cur;
    }
    let (sx, sy) = (cw as f32 / w as f32, ch as f32 / h as f32);
    let mut out = vec![[0.0f32; N]; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            *px = sample(&cur, cw, ch, (x as f32 + 0.5) * sx, (y as f32 + 0.5) * sy);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_constant_and_energy() {
        let (w, h) = (97, 61);
        let src = vec![[0.25f32, 1.0]; w * h];
        for sigma in [0.0, 1.5, 6.0, 40.0] {
            let out = blur(&src, w, h, &Plan::new(sigma));
            assert!(out.iter().all(|p| (p[0] - 0.25).abs() < 1e-5 && (p[1] - 1.0).abs() < 1e-5), "sigma {sigma}");
        }
        // A point spreads and keeps (roughly) its energy at small sigma.
        let mut pt = vec![[0.0f32; 1]; w * h];
        pt[30 * w + 40] = [1.0];
        let out = blur(&pt, w, h, &Plan::new(2.0));
        let sum: f32 = out.iter().map(|p| p[0]).sum();
        assert!((sum - 1.0).abs() < 1e-3, "{sum}");
        assert!(out[30 * w + 40][0] < 0.2 && out[30 * w + 43][0] > 0.0);
    }

    #[test]
    fn plans_bound_work() {
        let p = Plan::new(1e9);
        assert_eq!(p.levels, MAX_LEVELS);
        assert!(p.weights.len() <= MAX_RADIUS + 1);
        assert!(Plan::new(f32::NAN).is_identity());
        assert_eq!(sizes(5, 3, 2), vec![(5, 3), (3, 2), (2, 1)]);
    }
}
