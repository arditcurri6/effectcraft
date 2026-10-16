//! Spatial operations: tools that read neighbouring pixels (glows, streaks, local tone,
//! resampling). They run on the working buffer — straight RGB in the chain's domain plus alpha
//! — in buffer pixels; [`crate::program`] converts their layer-space parameters. Glows are
//! light *added within the layer's alpha* (blurred from premultiplied highlights), so a
//! transparent layer keeps its alpha.

use rayon::prelude::*;

use crate::blur::{Plan, blur, sample};
use crate::cmath::{Working, mix3, smoothstep};

/// A compiled spatial operation (pixel units of the buffer).
#[derive(Clone, Debug, PartialEq)]
pub enum SpatialOp {
    /// Halation: highlights above `thr` (linear) blurred at a tight and a wide radius
    /// (`falloff` mixes them), tinted and added.
    Halation { thr: f32, tight: Plan, wide: Plan, falloff: f32, strength: f32, color: [f32; 3], mix: f32 },
    /// Diffusion: a coloured bloom of highlights plus a soft mist of the whole image.
    Diffusion { thr: f32, bloom: Plan, mist: Plan, intensity: f32, softness: f32, color: [f32; 3], mix: f32 },
    /// Horizontal streaks: a two-sided exponential filter with per-pixel decay `decay`.
    Streak { thr: f32, decay: f32, intensity: f32, color: [f32; 3], mix: f32 },
    /// Red/cyan and blue/yellow displacement (pixels), radial around `center` or uniform.
    Chroma { rc: f32, by: f32, radial: f32, center: [f32; 2], inv_ref: f32, mix: f32 },
    /// Radial distortion `r' = r (1 + k r²) / zoom` around `center` (r in units of `ref_r`).
    Distort { k: f32, inv_zoom: f32, center: [f32; 2], ref_r: f32, extend: bool, mix: f32 },
    /// Blur towards the edges of the layer rectangle `rect` (pixels).
    EdgeSoft { blur: Plan, rect: [f32; 4], size: f32, feather: f32, mix: f32 },
    /// Local shadow / highlight recovery from the blurred luminance.
    Local { blur: Plan, shadows: f32, highlights: f32, width: f32, mix: f32 },
}

/// The working buffer: straight colour + alpha, row-major.
pub struct Buffer {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[f32; 4]>,
}

#[inline]
fn y(l: [f32; 3], c: [f32; 4]) -> f32 {
    l[0] * c[0] + l[1] * c[1] + l[2] * c[2]
}

/// Premultiplied highlight energy above `thr`: the luma of the excess (scalar).
#[inline]
pub fn excess_luma(c: [f32; 4], thr: f32, l: [f32; 3]) -> f32 {
    let e = [(c[0] - thr).max(0.0), (c[1] - thr).max(0.0), (c[2] - thr).max(0.0)];
    (l[0] * e[0] + l[1] * e[1] + l[2] * e[2]) * c[3].clamp(0.0, 1.0)
}

/// `[l x]` per pixel, normalised streak filter `(1−d)/(1+d) Σ d^|i| x`.
pub fn streak_rows(src: &[f32], w: usize, h: usize, decay: f32) -> Vec<f32> {
    let d = decay.clamp(0.0, 0.9999);
    let norm = (1.0 - d) / (1.0 + d);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w.max(1)).enumerate().for_each(|(yy, row)| {
        let Some(s) = src.get(yy * w..(yy + 1) * w) else { return };
        let mut acc = 0.0f32;
        for (x, o) in row.iter_mut().enumerate() {
            acc = s[x] + d * acc;
            *o = acc;
        }
        acc = 0.0;
        for x in (0..w).rev() {
            acc = s[x] + d * acc;
            row[x] = (row[x] + acc - s[x]) * norm;
        }
    });
    out
}

impl SpatialOp {
    pub fn run(&self, b: &Buffer, w: &Working) -> Vec<[f32; 4]> {
        let l = w.luma;
        let (bw, bh) = (b.w, b.h);
        let src = &b.px;
        let mix = |a: [f32; 4], o: [f32; 4], m: f32| [a[0] + (o[0] - a[0]) * m, a[1] + (o[1] - a[1]) * m, a[2] + (o[2] - a[2]) * m, a[3] + (o[3] - a[3]) * m];
        match self {
            SpatialOp::Halation { thr, tight, wide, falloff, strength, color, mix: m } => {
                let e: Vec<[f32; 1]> = src.iter().map(|c| [excess_luma(*c, *thr, l)]).collect();
                let t = blur(&e, bw, bh, tight);
                let wd = blur(&e, bw, bh, wide);
                src.par_iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let g = t.get(i).map_or(0.0, |v| v[0]) * (1.0 - falloff) + wd.get(i).map_or(0.0, |v| v[0]) * falloff;
                        let o = [c[0] + strength * g * color[0], c[1] + strength * g * color[1], c[2] + strength * g * color[2], c[3]];
                        mix(*c, o, *m)
                    })
                    .collect()
            }
            SpatialOp::Diffusion { thr, bloom, mist, intensity, softness, color, mix: m } => {
                let e: Vec<[f32; 3]> = src
                    .iter()
                    .map(|c| {
                        let a = c[3].clamp(0.0, 1.0);
                        [(c[0] - thr).max(0.0) * a, (c[1] - thr).max(0.0) * a, (c[2] - thr).max(0.0) * a]
                    })
                    .collect();
                let bl = blur(&e, bw, bh, bloom);
                let pre: Vec<[f32; 4]> = src
                    .iter()
                    .map(|c| {
                        let a = c[3].clamp(0.0, 1.0);
                        [c[0] * a, c[1] * a, c[2] * a, a]
                    })
                    .collect();
                let soft = blur(&pre, bw, bh, mist);
                src.par_iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let s = soft.get(i).copied().unwrap_or(*c);
                        let sa = s[3].max(1e-4);
                        let sc = [s[0] / sa, s[1] / sa, s[2] / sa];
                        let base = mix3([c[0], c[1], c[2]], sc, softness * intensity);
                        let g = bl.get(i).copied().unwrap_or([0.0; 3]);
                        let o = [base[0] + intensity * g[0] * color[0], base[1] + intensity * g[1] * color[1], base[2] + intensity * g[2] * color[2], c[3]];
                        mix(*c, o, *m)
                    })
                    .collect()
            }
            SpatialOp::Streak { thr, decay, intensity, color, mix: m } => {
                let e: Vec<f32> = src.iter().map(|c| excess_luma(*c, *thr, l)).collect();
                let s = streak_rows(&e, bw, bh, *decay);
                src.par_iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let g = s.get(i).copied().unwrap_or(0.0) * 3.0;
                        let o = [c[0] + intensity * g * color[0], c[1] + intensity * g * color[1], c[2] + intensity * g * color[2], c[3]];
                        mix(*c, o, *m)
                    })
                    .collect()
            }
            SpatialOp::Chroma { rc, by, radial, center, inv_ref, mix: m } => {
                let mut out = vec![[0.0f32; 4]; bw * bh];
                out.par_chunks_mut(bw.max(1)).enumerate().for_each(|(yy, row)| {
                    for (x, o) in row.iter_mut().enumerate() {
                        let q = [x as f32 + 0.5, yy as f32 + 0.5];
                        let v = [(q[0] - center[0]) * inv_ref, (q[1] - center[1]) * inv_ref];
                        let dr = [rc * (1.0 - radial) + v[0] * rc * radial, v[1] * rc * radial];
                        let db = [by * (1.0 - radial) + v[0] * by * radial, v[1] * by * radial];
                        let c = src.get(yy * bw + x).copied().unwrap_or([0.0; 4]);
                        let r = sample(src, bw, bh, q[0] - dr[0], q[1] - dr[1]);
                        let b = sample(src, bw, bh, q[0] - db[0], q[1] - db[1]);
                        *o = mix(c, [r[0], c[1], b[2], c[3]], *m);
                    }
                });
                out
            }
            SpatialOp::Distort { k, inv_zoom, center, ref_r, extend, mix: m } => {
                let mut out = vec![[0.0f32; 4]; bw * bh];
                out.par_chunks_mut(bw.max(1)).enumerate().for_each(|(yy, row)| {
                    for (x, o) in row.iter_mut().enumerate() {
                        let q = [x as f32 + 0.5, yy as f32 + 0.5];
                        let v = [(q[0] - center[0]) / ref_r, (q[1] - center[1]) / ref_r];
                        let f = (1.0 + k * (v[0] * v[0] + v[1] * v[1])) * inv_zoom;
                        let s = [center[0] + v[0] * ref_r * f, center[1] + v[1] * ref_r * f];
                        let c = src.get(yy * bw + x).copied().unwrap_or([0.0; 4]);
                        *o = mix(c, sample_premul(src, bw, bh, s[0], s[1], *extend), *m);
                    }
                });
                out
            }
            SpatialOp::EdgeSoft { blur: plan, rect, size, feather, mix: m } => {
                let pre: Vec<[f32; 4]> = src
                    .iter()
                    .map(|c| {
                        let a = c[3].clamp(0.0, 1.0);
                        [c[0] * a, c[1] * a, c[2] * a, a]
                    })
                    .collect();
                let bl = blur(&pre, bw, bh, plan);
                let (hw, hh) = (((rect[2] - rect[0]) * 0.5).max(1.0), ((rect[3] - rect[1]) * 0.5).max(1.0));
                let mut out = vec![[0.0f32; 4]; bw * bh];
                out.par_chunks_mut(bw.max(1)).enumerate().for_each(|(yy, row)| {
                    for (x, o) in row.iter_mut().enumerate() {
                        let q = [x as f32 + 0.5, yy as f32 + 0.5];
                        let ex = ((q[0] - rect[0]).min(rect[2] - q[0]) / hw).max(0.0);
                        let ey = ((q[1] - rect[1]).min(rect[3] - q[1]) / hh).max(0.0);
                        let e = ex.min(ey);
                        let msk = 1.0 - smoothstep(size * (1.0 - feather), size + 1e-4, e);
                        let i = yy * bw + x;
                        let c = src.get(i).copied().unwrap_or([0.0; 4]);
                        let s = bl.get(i).copied().unwrap_or(c);
                        let sa = s[3].max(1e-4);
                        let sc = [s[0] / sa, s[1] / sa, s[2] / sa];
                        let r = mix3([c[0], c[1], c[2]], sc, msk);
                        *o = mix(c, [r[0], r[1], r[2], c[3]], *m);
                    }
                });
                out
            }
            SpatialOp::Local { blur: plan, shadows, highlights, width, mix: m } => {
                let ly: Vec<[f32; 1]> = src.iter().map(|c| [y(l, *c).max(0.0)]).collect();
                let base = blur(&ly, bw, bh, plan);
                src.par_iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let b = base.get(i).map_or(0.0, |v| v[0]);
                        let lv = (b.max(1e-5) / 0.18).log2();
                        let ws = 1.0 - smoothstep(-2.0 - 4.0 * width, 0.0, lv);
                        let wh = smoothstep(0.3, 1.0 + 2.5 * width, lv);
                        let g = (shadows * 2.0 * ws - highlights * 2.0 * wh).exp2();
                        mix(*c, [c[0] * g, c[1] * g, c[2] * g, c[3]], *m)
                    })
                    .collect()
            }
        }
    }
}

/// Bilinear sample of a straight-colour buffer as premultiplied colour (so transparent texels
/// don't darken edges), un-premultiplied again; outside is transparent unless `extend`.
#[inline]
pub fn sample_premul(src: &[[f32; 4]], w: usize, h: usize, x: f32, y: f32, extend: bool) -> [f32; 4] {
    if w == 0 || h == 0 {
        return [0.0; 4];
    }
    let fx = x - 0.5;
    let fy = y - 0.5;
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let g = |xx: i64, yy: i64| -> [f32; 4] {
        let (xx, yy) = if extend {
            (xx.clamp(0, w as i64 - 1), yy.clamp(0, h as i64 - 1))
        } else if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
            return [0.0; 4];
        } else {
            (xx, yy)
        };
        let c = src.get(yy as usize * w + xx as usize).copied().unwrap_or([0.0; 4]);
        let a = c[3].clamp(0.0, 1.0);
        [c[0] * a, c[1] * a, c[2] * a, a]
    };
    let (xi, yi) = (x0 as i64, y0 as i64);
    let (a, b, c, d) = (g(xi, yi), g(xi + 1, yi), g(xi, yi + 1), g(xi + 1, yi + 1));
    let mut o = [0.0f32; 4];
    for k in 0..4 {
        let top = a[k] + (b[k] - a[k]) * tx;
        let bot = c[k] + (d[k] - c[k]) * tx;
        o[k] = top + (bot - top) * ty;
    }
    if o[3] > 1e-6 { [o[0] / o[3], o[1] / o[3], o[2] / o[3], o[3]] } else { [0.0; 4] }
}
