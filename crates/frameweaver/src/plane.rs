//! A single-channel f32 plane with clamped access, bilinear sampling and resizing.
//!
//! Plane coordinates put pixel `i`'s centre at coordinate `i` (not `i + 0.5`).

use rayon::prelude::*;

use crate::MAX_SIDE;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Plane {
    /// A zero plane; `None` when a side is 0 or over [`MAX_SIDE`].
    pub fn new(w: usize, h: usize) -> Option<Plane> {
        Plane::filled(w, h, 0.0)
    }

    pub fn filled(w: usize, h: usize, v: f32) -> Option<Plane> {
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
            return None;
        }
        Some(Plane { w, h, d: vec![v; w * h] })
    }

    /// Wrap existing data; `None` when its length is not `w × h`.
    pub fn from_vec(w: usize, h: usize, d: Vec<f32>) -> Option<Plane> {
        (w > 0 && h > 0 && w <= MAX_SIDE && h <= MAX_SIDE && d.len() == w * h).then_some(Plane { w, h, d })
    }

    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0 || self.d.len() != self.w * self.h
    }

    /// The value at (x, y), edge pixels repeated outside.
    #[inline]
    pub fn at(&self, x: i64, y: i64) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let x = x.clamp(0, self.w as i64 - 1) as usize;
        let y = y.clamp(0, self.h as i64 - 1) as usize;
        self.d.get(y * self.w + x).copied().unwrap_or(0.0)
    }

    /// Bilinear sample at continuous (x, y), edges clamped. Non-finite coordinates read 0.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        if !(x.is_finite() && y.is_finite()) {
            return 0.0;
        }
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let (xi, yi) = (x0.clamp(-1e6, 1e6) as i64, y0.clamp(-1e6, 1e6) as i64);
        let a = self.at(xi, yi);
        let b = self.at(xi + 1, yi);
        let c = self.at(xi, yi + 1);
        let d = self.at(xi + 1, yi + 1);
        let top = a + (b - a) * tx;
        let bot = c + (d - c) * tx;
        top + (bot - top) * ty
    }

    /// Central-difference gradient (∂x, ∂y).
    pub fn gradient(&self) -> Option<(Plane, Plane)> {
        let mut gx = Plane::new(self.w, self.h)?;
        let mut gy = Plane::new(self.w, self.h)?;
        let w = self.w;
        gx.d.par_chunks_mut(w).zip(gy.d.par_chunks_mut(w)).enumerate().for_each(|(y, (rx, ry))| {
            for x in 0..w {
                let (xi, yi) = (x as i64, y as i64);
                rx[x] = 0.5 * (self.at(xi + 1, yi) - self.at(xi - 1, yi));
                ry[x] = 0.5 * (self.at(xi, yi + 1) - self.at(xi, yi - 1));
            }
        });
        Some((gx, gy))
    }

    /// 2× box-filtered downscale (odd sizes round up; the last row / column is repeated).
    pub fn half(&self) -> Option<Plane> {
        let (nw, nh) = (self.w.div_ceil(2).max(1), self.h.div_ceil(2).max(1));
        let mut out = Plane::new(nw, nh)?;
        out.d.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let (sx, sy) = (x as i64 * 2, y as i64 * 2);
                *o = 0.25 * (self.at(sx, sy) + self.at(sx + 1, sy) + self.at(sx, sy + 1) + self.at(sx + 1, sy + 1));
            }
        });
        Some(out)
    }

    /// Resize to `nw × nh`: repeated 2× box reduction while the target is at most half the
    /// size, then a bilinear resample (pixel-centre aligned).
    pub fn resize(&self, nw: usize, nh: usize) -> Option<Plane> {
        if self.is_empty() || nw == 0 || nh == 0 {
            return None;
        }
        let mut cur = std::borrow::Cow::Borrowed(self);
        while cur.w >= nw * 2 && cur.h >= nh * 2 {
            cur = std::borrow::Cow::Owned(cur.half()?);
        }
        if cur.w == nw && cur.h == nh {
            return Some(cur.into_owned());
        }
        let mut out = Plane::new(nw, nh)?;
        let (sx, sy) = (cur.w as f32 / nw as f32, cur.h as f32 / nh as f32);
        out.d.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = cur.sample((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5);
            }
        });
        Some(out)
    }

    /// Box mean over a `(2r+1)²` window, edges clamped (separable running sums).
    pub fn box_mean(&self, r: usize) -> Option<Plane> {
        let (w, h) = (self.w, self.h);
        let mut tmp = Plane::new(w, h)?;
        let k = 1.0 / (2 * r + 1) as f32;
        let ri = r as i64;
        tmp.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let yi = y as i64;
            let mut s: f32 = (-ri..=ri).map(|d| self.at(d, yi)).sum();
            for (x, o) in row.iter_mut().enumerate() {
                *o = s * k;
                let xi = x as i64;
                s += self.at(xi + ri + 1, yi) - self.at(xi - ri, yi);
            }
        });
        let mut out = Plane::new(w, h)?;
        out.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let yi = y as i64;
            for (x, o) in row.iter_mut().enumerate() {
                let xi = x as i64;
                let s: f32 = (-ri..=ri).map(|d| tmp.at(xi, yi + d)).sum();
                *o = s * k;
            }
        });
        Some(out)
    }
}

/// Mean absolute difference of two planes (0 when they differ in size or are empty).
pub fn mean_abs_diff(a: &Plane, b: &Plane) -> f32 {
    if a.w != b.w || a.h != b.h || a.d.is_empty() {
        return 0.0;
    }
    a.d.iter().zip(&b.d).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.d.len() as f32
}
