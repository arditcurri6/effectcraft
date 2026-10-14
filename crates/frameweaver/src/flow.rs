//! Dense optical flow: coarse-to-fine Dense Inverse Search (Kroeger, Timofte, Dai, Van Gool,
//! "Fast Optical Flow using Dense Inverse Search", ECCV 2016) with an optional variational
//! refinement (brightness + gradient constancy, robust smoothness; Brox et al. style data
//! terms, solved by red-black SOR).
//!
//! Per pyramid level, coarse to fine: the coarser flow initialises a regular grid of patches;
//! each patch runs inverse-compositional Lucas–Kanade (template gradients and Hessian fixed,
//! only the warped frame re-sampled per iteration); the patch vectors are densified by an
//! error-weighted average; an optional variational pass polishes the dense field.
//!
//! What the four quality presets change is spelled out in [`FlowParams::for_quality`].

use rayon::prelude::*;

use crate::plane::Plane;
use crate::{FlowError, MAX_SIDE};

/// Quality presets. Each one runs a different amount of the same pipeline: see
/// [`FlowParams::for_quality`] for the exact differences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quality {
    Draft,
    Balanced,
    High,
    Best,
}

impl Quality {
    pub const ALL: [Quality; 4] = [Quality::Draft, Quality::Balanced, Quality::High, Quality::Best];

    pub fn from_index(i: u32) -> Quality {
        match i {
            0 => Quality::Draft,
            1 => Quality::Balanced,
            2 => Quality::High,
            _ => Quality::Best,
        }
    }
}

/// Every knob of the estimator (a [`Quality`] fills them in).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowParams {
    /// Long side of the analysis plane.
    pub max_long: u32,
    /// Patch side (px at every pyramid level).
    pub patch: usize,
    /// Patch overlap 0…0.9 (stride = patch × (1 − overlap)).
    pub overlap: f32,
    /// Lucas–Kanade iterations per patch.
    pub iters: u32,
    /// Variational refinement: outer (re-linearisation) and inner (SOR) iterations per level.
    pub refine_outer: u32,
    pub refine_inner: u32,
    /// Variational smoothness weight.
    pub alpha: f32,
    /// 3×3 median passes on each level's dense field.
    pub median: u32,
    /// Average each vector with the inverted opposite flow where the two agree (precision).
    pub symmetrize: bool,
    /// Fit a global affine motion and use it where local vectors are unreliable.
    pub global_fallback: bool,
}

impl FlowParams {
    /// The preset's parameters, adjusted by Motion Sensitivity `m` (0…1): the patch size is
    /// scaled by `1.5 − m` (clamped to 6…16 px) and the smoothness weight by `2 − 1.8m`, so a
    /// high sensitivity gives finer, more independent vectors and a low one smoother, more
    /// rigid motion.
    ///
    /// | | analysis | patch | overlap | LK iters | variational (outer×inner) | median | symmetrize | global fallback |
    /// |---|---|---|---|---|---|---|---|---|
    /// | Draft | 640 | 12 | 0.3 | 4 | – | – | – | – |
    /// | Balanced | 960 | 10 | 0.4 | 6 | 2×6 | 1 | – | – |
    /// | High | 1440 | 8 | 0.5 | 10 | 3×8 | 1 | – | yes |
    /// | Best | 1920 | 8 | 0.6 | 12 | 5×10 | 1 | yes | yes |
    pub fn for_quality(q: Quality, sensitivity: f32) -> FlowParams {
        let m = if sensitivity.is_finite() { sensitivity.clamp(0.0, 1.0) } else { 0.7 };
        let (max_long, patch, overlap, iters, ro, ri, median, sym, global) = match q {
            Quality::Draft => (640, 12.0, 0.3, 4, 0, 0, 0, false, false),
            Quality::Balanced => (960, 10.0, 0.4, 6, 2, 6, 1, false, false),
            Quality::High => (1440, 8.0, 0.5, 10, 3, 8, 1, false, true),
            Quality::Best => (1920, 8.0, 0.6, 12, 5, 10, 1, true, true),
        };
        FlowParams {
            max_long,
            patch: ((patch * (1.5 - m)).round() as usize).clamp(6, 16),
            overlap,
            iters,
            refine_outer: ro,
            refine_inner: ri,
            alpha: 0.1 * (2.0 - 1.8 * m),
            median,
            symmetrize: sym,
            global_fallback: global,
        }
    }
}

/// A dense motion field: one vector per pixel, in pixels of this field (A → B).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlowField {
    pub w: usize,
    pub h: usize,
    pub v: Vec<[f32; 2]>,
}

impl FlowField {
    pub fn new(w: usize, h: usize) -> Option<FlowField> {
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
            return None;
        }
        Some(FlowField { w, h, v: vec![[0.0; 2]; w * h] })
    }

    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0 || self.v.len() != self.w * self.h
    }

    /// The vector at pixel (x, y), edges clamped.
    #[inline]
    pub fn at(&self, x: i64, y: i64) -> [f32; 2] {
        if self.is_empty() {
            return [0.0; 2];
        }
        let x = x.clamp(0, self.w as i64 - 1) as usize;
        let y = y.clamp(0, self.h as i64 - 1) as usize;
        self.v.get(y * self.w + x).copied().unwrap_or([0.0; 2])
    }

    /// Bilinear sample at continuous (x, y) (pixel centres at integers).
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> [f32; 2] {
        if !(x.is_finite() && y.is_finite()) {
            return [0.0; 2];
        }
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let (xi, yi) = (x0.clamp(-1e6, 1e6) as i64, y0.clamp(-1e6, 1e6) as i64);
        let (a, b, c, d) = (self.at(xi, yi), self.at(xi + 1, yi), self.at(xi, yi + 1), self.at(xi + 1, yi + 1));
        let mut o = [0.0; 2];
        for k in 0..2 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (d[k] - c[k]) * tx;
            o[k] = top + (bot - top) * ty;
        }
        o
    }

    /// Resample to `nw × nh`, scaling the vectors by the size ratio.
    pub fn resize(&self, nw: usize, nh: usize) -> Option<FlowField> {
        if self.is_empty() {
            return None;
        }
        let mut out = FlowField::new(nw, nh)?;
        let (sx, sy) = (self.w as f32 / nw as f32, self.h as f32 / nh as f32);
        out.v.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let s = self.sample((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5);
                *o = [s[0] / sx, s[1] / sy];
            }
        });
        Some(out)
    }

    /// Mean endpoint error against a constant vector, away from a `margin`-pixel border.
    pub fn mean_error_to(&self, truth: impl Fn(f32, f32) -> [f32; 2] + Sync, margin: usize) -> f32 {
        let mut s = 0.0f64;
        let mut n = 0u64;
        for y in margin..self.h.saturating_sub(margin) {
            for x in margin..self.w.saturating_sub(margin) {
                let t = truth(x as f32, y as f32);
                let v = self.at(x as i64, y as i64);
                s += ((v[0] - t[0]).hypot(v[1] - t[1])) as f64;
                n += 1;
            }
        }
        if n == 0 { 0.0 } else { (s / n as f64) as f32 }
    }
}

/// Dense flow from `a` to `b` (same size).
pub fn estimate(a: &Plane, b: &Plane, p: &FlowParams) -> Result<FlowField, FlowError> {
    if a.is_empty() || b.is_empty() {
        return Err(FlowError::Empty);
    }
    if a.w != b.w || a.h != b.h {
        return Err(FlowError::SizeMismatch);
    }
    let patch = p.patch.clamp(4, 32);
    // Pyramid: halve until the long side is about 40 px or the patch no longer fits twice.
    let mut pa = vec![a.clone()];
    let mut pb = vec![b.clone()];
    while pa.len() < 12 {
        let (la, lb) = match (pa.last(), pb.last()) {
            (Some(x), Some(y)) => (x, y),
            _ => break,
        };
        if la.w.max(la.h) <= 40 || la.w.min(la.h) < patch * 2 {
            break;
        }
        match (la.half(), lb.half()) {
            (Some(x), Some(y)) => {
                pa.push(x);
                pb.push(y);
            }
            _ => break,
        }
    }
    let stride = ((patch as f32 * (1.0 - p.overlap.clamp(0.0, 0.9))).round() as usize).max(1);
    let mut flow: Option<FlowField> = None;
    for level in (0..pa.len()).rev() {
        let (Some(la), Some(lb)) = (pa.get(level), pb.get(level)) else { continue };
        let init = match flow.take() {
            Some(f) => f.resize(la.w, la.h).ok_or(FlowError::TooLarge)?,
            None => FlowField::new(la.w, la.h).ok_or(FlowError::TooLarge)?,
        };
        // The coarsest level has no initial flow: search integer displacements exhaustively.
        let search: i64 = if level + 1 == pa.len() {
            ((la.w.max(la.h) / 6) as i64).clamp(3, 12)
        } else if level > 0 {
            7
        } else {
            4
        };
        let mut f = dis_level(la, lb, &init, patch, stride, p.iters, search).unwrap_or(init);
        for _ in 0..p.median {
            f = median3(&f);
        }
        if p.refine_outer > 0
            && p.refine_inner > 0
            && let Some(r) = refine(la, lb, &f, p.refine_outer, p.refine_inner, p.alpha)
        {
            f = r;
        }
        flow = Some(f);
    }
    flow.ok_or(FlowError::Empty)
}

/// The inverse-compositional Lucas–Kanade solver of one pyramid level: the template (A) with
/// its gradients, and the frame it is matched in (B).
struct Patches<'a> {
    a: &'a Plane,
    b: &'a Plane,
    gx: Plane,
    gy: Plane,
    ps: usize,
}

impl Patches<'_> {
    /// Sum of squared differences of the patch at (px, py) displaced by `u`.
    fn sse(&self, px: usize, py: usize, u: [f32; 2]) -> f32 {
        let mut sse = 0.0f32;
        for y in py..py + self.ps {
            for x in px..px + self.ps {
                let e = self.b.sample(x as f32 + u[0], y as f32 + u[1]) - self.a.at(x as i64, y as i64);
                sse += e * e;
            }
        }
        sse
    }

    /// Up to `iters` Lucas–Kanade updates from `start`; an update that makes the patch match
    /// worse is undone. The template Hessian is fixed (the "inverse" in Dense Inverse Search).
    fn lk(&self, px: usize, py: usize, start: [f32; 2], iters: u32) -> [f32; 2] {
        let ps = self.ps;
        let (mut hxx, mut hxy, mut hyy) = (0.0f32, 0.0f32, 0.0f32);
        for y in py..py + ps {
            for x in px..px + ps {
                let (dx, dy) = (self.gx.at(x as i64, y as i64), self.gy.at(x as i64, y as i64));
                hxx += dx * dx;
                hxy += dx * dy;
                hyy += dy * dy;
            }
        }
        let det = hxx * hyy - hxy * hxy;
        if det < 1e-8 * (ps * ps) as f32 || !det.is_finite() {
            return start;
        }
        let (ixx, ixy, iyy) = (hyy / det, -hxy / det, hxx / det);
        let lim = (self.a.w + self.a.h) as f32;
        let mut u = start;
        let mut prev_u = start;
        let mut prev_sse = f32::INFINITY;
        for _ in 0..iters.max(1) {
            let (mut bx, mut by, mut sse) = (0.0f32, 0.0f32, 0.0f32);
            for y in py..py + ps {
                for x in px..px + ps {
                    let e = self.b.sample(x as f32 + u[0], y as f32 + u[1]) - self.a.at(x as i64, y as i64);
                    bx += e * self.gx.at(x as i64, y as i64);
                    by += e * self.gy.at(x as i64, y as i64);
                    sse += e * e;
                }
            }
            if sse > prev_sse {
                // The last update made the patch match worse: keep the previous vector.
                return prev_u;
            }
            prev_sse = sse;
            prev_u = u;
            let (du, dv) = (ixx * bx + ixy * by, ixy * bx + iyy * by);
            if !(du.is_finite() && dv.is_finite()) {
                break;
            }
            u = [(u[0] - du).clamp(-lim, lim), (u[1] - dv).clamp(-lim, lim)];
            if du * du + dv * dv < 1e-4 {
                break;
            }
        }
        u
    }
}

/// One level of Dense Inverse Search starting from `init`.
///
/// 1. Each patch starts from the best of the initial flow at itself and around it (and, if it
///    still matches poorly, of every integer displacement within `search` — always on the
///    coarsest level), then runs Lucas–Kanade.
/// 2. Two propagation passes: each patch tries its eight neighbours' converged vectors and
///    re-polishes the best. A fast small object that the coarser levels smeared away is found
///    by the few patches that lie fully on it and spreads to the rest.
/// 3. The patch vectors are densified by an error-weighted average.
fn dis_level(a: &Plane, b: &Plane, init: &FlowField, patch: usize, stride: usize, iters: u32, search: i64) -> Option<FlowField> {
    let (w, h) = (a.w, a.h);
    let ps = patch.min(w).min(h);
    let top = init.v.iter().all(|v| *v == [0.0; 2]);
    if ps < 3 {
        return None;
    }
    let (gx, gy) = a.gradient()?;
    let solver = Patches { a, b, gx, gy, ps };
    let nx = (w - ps).div_ceil(stride) + 1;
    let ny = (h - ps).div_ceil(stride) + 1;
    let pos = |k: usize, n: usize, len: usize| -> usize { if k + 1 == n { len - ps } else { (k * stride).min(len - ps) } };
    let origin = |i: usize| (pos(i % nx, nx, w), pos(i / nx, ny, h));
    let mse_floor = 4e-4 * (ps * ps) as f32;
    let (step, half) = (ps as i64, (ps / 2) as i64);
    let offsets = [
        (-step, 0),
        (step, 0),
        (0, -step),
        (0, step),
        (-half, 0),
        (half, 0),
        (0, -half),
        (0, half),
        (-half, -half),
        (half, -half),
        (-half, half),
        (half, half),
    ];
    let mut vecs: Vec<[f32; 2]> = (0..nx * ny)
        .into_par_iter()
        .map(|i| {
            let (px, py) = origin(i);
            let (cx, cy) = ((px + ps / 2) as i64, (py + ps / 2) as i64);
            let mut c = init.at(cx, cy);
            let mut best = solver.sse(px, py, c);
            for (dx, dy) in offsets {
                let cand = init.at(cx + dx, cy + dy);
                let e = solver.sse(px, py, cand);
                if e < best {
                    best = e;
                    c = cand;
                }
            }
            if search > 0 && (best > mse_floor || top) {
                for dy in -search..=search {
                    for dx in -search..=search {
                        let cand = [dx as f32, dy as f32];
                        let e = solver.sse(px, py, cand);
                        if e < best {
                            best = e;
                            c = cand;
                        }
                    }
                }
            }
            solver.lk(px, py, c, iters)
        })
        .collect();
    for _ in 0..2 {
        let cur = &vecs;
        let next: Vec<[f32; 2]> = (0..nx * ny)
            .into_par_iter()
            .map(|i| {
                let (px, py) = origin(i);
                let own = cur.get(i).copied().unwrap_or([0.0; 2]);
                let mut best = solver.sse(px, py, own);
                if best <= mse_floor * 0.25 {
                    return own;
                }
                let mut c = own;
                let (kx, ky) = ((i % nx) as i64, (i / nx) as i64);
                for (dx, dy) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (qx, qy) = (kx + dx, ky + dy);
                    if qx < 0 || qy < 0 || qx >= nx as i64 || qy >= ny as i64 {
                        continue;
                    }
                    let cand = cur.get(qy as usize * nx + qx as usize).copied().unwrap_or(own);
                    let e = solver.sse(px, py, cand);
                    if e < best {
                        best = e;
                        c = cand;
                    }
                }
                if c == own { own } else { solver.lk(px, py, c, 3) }
            })
            .collect();
        vecs = next;
    }
    // Densify: error-weighted average of the patches covering each pixel.
    let mut out = FlowField::new(w, h)?;
    out.v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let ky_lo = (y + 1).saturating_sub(ps) / stride;
        let ky_hi = (y / stride + 1).min(ny - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let kx_lo = (x + 1).saturating_sub(ps) / stride;
            let kx_hi = (x / stride + 1).min(nx - 1);
            let (mut sx, mut sy, mut sw) = (0.0f32, 0.0f32, 0.0f32);
            let av = a.at(x as i64, y as i64);
            let mut visit = |kx: usize, ky: usize| {
                let (px, py) = (pos(kx, nx, w), pos(ky, ny, h));
                if x < px || x >= px + ps || y < py || y >= py + ps {
                    return;
                }
                let Some(u) = vecs.get(ky * nx + kx) else { return };
                let err = (b.sample(x as f32 + u[0], y as f32 + u[1]) - av).abs() * 255.0;
                let wt = 1.0 / err.max(1.0);
                sx += u[0] * wt;
                sy += u[1] * wt;
                sw += wt;
            };
            for ky in ky_lo..=ky_hi {
                for kx in kx_lo..=kx_hi {
                    visit(kx, ky);
                }
                if kx_hi < nx - 1 {
                    visit(nx - 1, ky);
                }
            }
            if ky_hi < ny - 1 {
                for kx in kx_lo..=kx_hi {
                    visit(kx, ny - 1);
                }
                visit(nx - 1, ny - 1);
            }
            *o = if sw > 0.0 { [sx / sw, sy / sw] } else { init.at(x as i64, y as i64) };
        }
    });
    Some(out)
}

/// 3×3 component-wise median.
pub fn median3(f: &FlowField) -> FlowField {
    let Some(mut out) = FlowField::new(f.w, f.h) else { return f.clone() };
    let w = f.w;
    out.v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut xs = [0.0f32; 9];
            let mut ys = [0.0f32; 9];
            let mut n = 0;
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let v = f.at(x as i64 + dx, y as i64 + dy);
                    if let (Some(a), Some(b)) = (xs.get_mut(n), ys.get_mut(n)) {
                        *a = v[0];
                        *b = v[1];
                    }
                    n += 1;
                }
            }
            xs.sort_by(f32::total_cmp);
            ys.sort_by(f32::total_cmp);
            *o = [xs[4], ys[4]];
        }
    });
    out
}

/// Variational refinement of `flow` (A → B) on one level: `outer` re-linearisations of
/// brightness constancy (weight 1) and gradient constancy (weight γ = 5) with Charbonnier
/// robust penalties, smoothness weighted by `alpha` and the flow gradient, each solved with
/// `inner` red-black SOR sweeps (ω = 1.6).
pub fn refine(a: &Plane, b: &Plane, flow: &FlowField, outer: u32, inner: u32, alpha: f32) -> Option<FlowField> {
    let (w, h) = (a.w, a.h);
    if flow.w != w || flow.h != h || w < 3 || h < 3 {
        return None;
    }
    const GAMMA: f32 = 5.0;
    const EPS2: f32 = 1e-4;
    const OMEGA: f32 = 1.6;
    let (ax, ay) = a.gradient()?;
    let (bx, by) = b.gradient()?;
    let (bxx, bxy) = bx.gradient()?;
    let (_, byy) = by.gradient()?;
    let mut u = flow.clone();
    let n = w * h;
    for _ in 0..outer {
        // Linearised data terms about the current flow: per pixel A (3 numbers) and b (2).
        let data: Vec<[f32; 5]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let f = u.v.get(i).copied().unwrap_or([0.0; 2]);
                let (sx, sy) = (x + f[0], y + f[1]);
                let it = b.sample(sx, sy) - a.d.get(i).copied().unwrap_or(0.0);
                let (ix, iy) = (bx.sample(sx, sy), by.sample(sx, sy));
                let (ixx, ixy, iyy) = (bxx.sample(sx, sy), bxy.sample(sx, sy), byy.sample(sx, sy));
                let ixt = ix - ax.d.get(i).copied().unwrap_or(0.0);
                let iyt = iy - ay.d.get(i).copied().unwrap_or(0.0);
                let pd = 1.0 / (it * it + EPS2).sqrt();
                let pg = GAMMA / (ixt * ixt + iyt * iyt + EPS2).sqrt();
                let a11 = pd * ix * ix + pg * (ixx * ixx + ixy * ixy);
                let a12 = pd * ix * iy + pg * (ixx * ixy + ixy * iyy);
                let a22 = pd * iy * iy + pg * (ixy * ixy + iyy * iyy);
                let b1 = -(pd * it * ix + pg * (ixt * ixx + iyt * ixy));
                let b2 = -(pd * it * iy + pg * (ixt * ixy + iyt * iyy));
                [a11, a12, a22, b1, b2]
            })
            .collect();
        // Smoothness weights from the flow's own gradient.
        let sw: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let (l, r, t, d) = (u.at(x - 1, y), u.at(x + 1, y), u.at(x, y - 1), u.at(x, y + 1));
                let gu = ((r[0] - l[0]) * 0.5).powi(2) + ((d[0] - t[0]) * 0.5).powi(2);
                let gv = ((r[1] - l[1]) * 0.5).powi(2) + ((d[1] - t[1]) * 0.5).powi(2);
                alpha / (gu + gv + 1e-3).sqrt()
            })
            .collect();
        let mut du: Vec<[f32; 2]> = vec![[0.0; 2]; n];
        for _ in 0..inner {
            for color in 0..2usize {
                let cur = &du;
                let next: Vec<[f32; 2]> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let (x, y) = (i % w, i / w);
                        let old = cur.get(i).copied().unwrap_or([0.0; 2]);
                        if (x + y) % 2 != color {
                            return old;
                        }
                        let Some(d) = data.get(i) else { return old };
                        let si = sw.get(i).copied().unwrap_or(0.0);
                        let u0 = u.v.get(i).copied().unwrap_or([0.0; 2]);
                        let (mut s, mut tu, mut tv) = (0.0f32, 0.0f32, 0.0f32);
                        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                                continue;
                            }
                            let j = ny as usize * w + nx as usize;
                            let e = 0.5 * (si + sw.get(j).copied().unwrap_or(0.0));
                            let uj = u.v.get(j).copied().unwrap_or([0.0; 2]);
                            let dj = cur.get(j).copied().unwrap_or([0.0; 2]);
                            s += e;
                            tu += e * (uj[0] + dj[0] - u0[0]);
                            tv += e * (uj[1] + dj[1] - u0[1]);
                        }
                        let (m11, m22) = (d[0] + s + 1e-6, d[2] + s + 1e-6);
                        let det = m11 * m22 - d[1] * d[1];
                        if det.abs() < 1e-12 || !det.is_finite() {
                            return old;
                        }
                        let (r1, r2) = (d[3] + tu, d[4] + tv);
                        let nu = (m22 * r1 - d[1] * r2) / det;
                        let nv = (m11 * r2 - d[1] * r1) / det;
                        if !(nu.is_finite() && nv.is_finite()) {
                            return old;
                        }
                        [old[0] + OMEGA * (nu - old[0]), old[1] + OMEGA * (nv - old[1])]
                    })
                    .collect();
                du = next;
            }
        }
        u.v.par_iter_mut().zip(du.par_iter()).for_each(|(f, d)| {
            f[0] += d[0];
            f[1] += d[1];
        });
    }
    Some(u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;

    fn flow_for(map: impl Fn(f32, f32) -> (f32, f32) + Sync, q: Quality) -> (FlowField, usize) {
        let (w, h) = (160, 120);
        let a = render(w, h, |x, y| (x, y));
        let b = render(w, h, &map);
        let f = estimate(&a, &b, &FlowParams::for_quality(q, 0.7)).unwrap();
        (f, 16)
    }

    #[test]
    fn integer_translation_best_and_draft() {
        // B(x) = A(x − d): flow is +d.
        for (q, tol) in [(Quality::Best, 0.25), (Quality::High, 0.35), (Quality::Draft, 0.75)] {
            let (f, m) = flow_for(|x, y| (x - 6.0, y - 2.0), q);
            let e = f.mean_error_to(|_, _| [6.0, 2.0], m);
            assert!(e <= tol, "{q:?}: {e}");
        }
    }

    #[test]
    fn subpixel_translation() {
        let (f, m) = flow_for(|x, y| (x - 2.3, y + 1.4), Quality::Best);
        let e = f.mean_error_to(|_, _| [2.3, -1.4], m);
        assert!(e <= 0.25, "{e}");
    }

    #[test]
    fn rotation_and_zoom() {
        let (cx, cy) = (80.0f32, 60.0f32);
        let th = 3.0f32.to_radians();
        // B(x) = A(R⁻¹(x − c) + c): a point p of A lands at R(p − c) + c.
        let rot = move |x: f32, y: f32| {
            let (dx, dy) = (x - cx, y - cy);
            (cx + th.cos() * dx + th.sin() * dy, cy - th.sin() * dx + th.cos() * dy)
        };
        let (f, m) = flow_for(rot, Quality::Best);
        let truth = move |x: f32, y: f32| {
            let (dx, dy) = (x - cx, y - cy);
            [th.cos() * dx - th.sin() * dy - dx, th.sin() * dx + th.cos() * dy - dy]
        };
        let e = f.mean_error_to(truth, m);
        assert!(e <= 0.4, "rotation {e}");
        let zoom = 1.05f32;
        let (f, m) = flow_for(move |x, y| (cx + (x - cx) / zoom, cy + (y - cy) / zoom), Quality::Best);
        let e = f.mean_error_to(move |x, y| [(x - cx) * (zoom - 1.0), (y - cy) * (zoom - 1.0)], m);
        assert!(e <= 0.4, "zoom {e}");
    }

    #[test]
    fn presets_differ_and_quality_is_ordered() {
        let ps: Vec<FlowParams> = Quality::ALL.iter().map(|q| FlowParams::for_quality(*q, 0.7)).collect();
        for i in 0..3 {
            assert!(ps[i].max_long < ps[i + 1].max_long);
            assert!(ps[i].iters <= ps[i + 1].iters);
            assert!(ps[i].refine_outer <= ps[i + 1].refine_outer);
        }
        assert!(ps[3].symmetrize && !ps[0].symmetrize);
        // Sensitivity moves patch size and smoothness in opposite senses.
        let lo = FlowParams::for_quality(Quality::Best, 0.0);
        let hi = FlowParams::for_quality(Quality::Best, 1.0);
        assert!(lo.patch > hi.patch && lo.alpha > hi.alpha);
    }

    #[test]
    fn hostile_inputs_do_not_panic() {
        let p = FlowParams::for_quality(Quality::Best, f32::NAN);
        let empty = Plane::default();
        assert_eq!(estimate(&empty, &empty, &p), Err(FlowError::Empty));
        let a = Plane::filled(10, 10, 0.5).unwrap();
        let b = Plane::filled(9, 10, 0.5).unwrap();
        assert_eq!(estimate(&a, &b, &p), Err(FlowError::SizeMismatch));
        // Tiny, thin, flat and NaN planes.
        for (w, h) in [(1, 1), (2, 2), (1, 400), (3, 5), (64, 2)] {
            let a = Plane::filled(w, h, 0.4).unwrap();
            let f = estimate(&a, &a, &p).unwrap();
            assert!(f.v.iter().all(|v| v[0].is_finite() && v[1].is_finite()));
        }
        let mut n = Plane::filled(40, 40, 0.5).unwrap();
        n.d[100] = f32::NAN;
        n.d[101] = f32::INFINITY;
        let f = estimate(&n, &n, &p).unwrap();
        assert_eq!(f.v.len(), 1600);
    }

    #[test]
    fn static_scene_has_zero_flow() {
        let a = render(96, 64, |x, y| (x, y));
        let f = estimate(&a, &a, &FlowParams::for_quality(Quality::Balanced, 0.7)).unwrap();
        assert!(f.mean_error_to(|_, _| [0.0, 0.0], 8) < 0.02);
    }
}
