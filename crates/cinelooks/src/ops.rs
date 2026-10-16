//! Per-pixel operations: what a look's colour, film, grain, vignette, mask and LUT tools
//! compile to (see [`crate::program`]). Every op works on straight RGB in a known domain and
//! may read the pixel's layer-space position. `fx_cinelooks.wgsl` (`fcl_pixel`) interprets the
//! same ops with the same arithmetic, in the same order.

use std::sync::Arc;

use effectcraft_color::Conversion;

use crate::cmath::{Working, hue_dist, hue_sat, mat3, mix3, rotate_hue, smoothstep, spow, value_noise};
use crate::curve::Table;
use crate::lut::Lut;

/// Hue centres of the HSL tool's ranges (degrees), in order.
pub const HSL_CENTERS: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0];

/// A shape that selects where a masked correction applies (layer space, pixels).
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    /// Full at `a`, none at `b`; `feather` 0..1 is the soft fraction of the way.
    Linear { a: [f32; 2], b: [f32; 2], feather: f32 },
    /// Inside an ellipse with half-axes `r`, rotated by `rot` (cos, sin); `feather` 0..1 of
    /// the radius is soft (inwards).
    Ellipse { c: [f32; 2], r: [f32; 2], rot: [f32; 2], feather: f32 },
}

impl Shape {
    #[inline]
    pub fn eval(&self, p: [f32; 2]) -> f32 {
        match *self {
            Shape::Linear { a, b, feather } => {
                let d = [b[0] - a[0], b[1] - a[1]];
                let l2 = (d[0] * d[0] + d[1] * d[1]).max(1e-6);
                let t = ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2;
                let f = feather.clamp(0.0, 1.0);
                1.0 - smoothstep(0.5 - 0.5 * f, 0.5 + 0.5 * f + 1e-4, t)
            }
            Shape::Ellipse { c, r, rot, feather } => {
                let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
                // Rotate into the ellipse's frame.
                let lx = dx * rot[0] + dy * rot[1];
                let ly = -dx * rot[1] + dy * rot[0];
                let (ex, ey) = (lx / r[0].max(1e-3), ly / r[1].max(1e-3));
                let d = (ex * ex + ey * ey).sqrt();
                let f = feather.clamp(0.001, 1.0);
                1.0 - smoothstep(1.0 - f, 1.0, d)
            }
        }
    }
}

/// A compiled per-pixel operation.
#[derive(Clone, Debug)]
pub enum PixelOp {
    /// Encoded → linear with the working curve.
    Decode,
    /// Linear → encoded.
    Encode,
    /// Remember the colour (a node with strength < 100 % starts).
    Begin,
    /// Mix the remembered colour towards the current one by the fraction.
    End(f32),
    Exposure {
        gain: f32,
        gamma: f32,
        protect: f32,
        knee: f32,
    },
    Contrast {
        k: f32,
        pivot: f32,
        filmic: f32,
    },
    Matrix([[f32; 3]; 3]),
    Mono([f32; 3]),
    Wheels {
        offset: [f32; 3],
        lift: [f32; 3],
        gexp: [f32; 3],
        gain: [f32; 3],
        shadows: [f32; 3],
        mids: [f32; 3],
        highs: [f32; 3],
    },
    /// Master, red, green, blue tables (`None` = identity).
    Curves(Arc<[Option<Table>; 4]>),
    /// Global hue rotation (°), saturation factor, lightness (−1..1) and per-range
    /// (hue shift °, saturation −1..1, luminance −1..1).
    Hsl {
        rot: f32,
        sat: f32,
        light: f32,
        ranges: [[f32; 3]; 8],
    },
    HueRemap {
        center: f32,
        range: f32,
        falloff: f32,
        shift: f32,
        sat: f32,
        lum: f32,
    },
    /// A film tone response (Film Response, Film Print): optional log-space matrix and offsets,
    /// a per-channel sigmoid in stops, then saturation in the display domain.
    Film {
        pre_gain: f32,
        matrix: Option<[[f32; 3]; 3]>,
        offsets: [f32; 3],
        k: f32,
        grey: f32,
        p_lo: f32,
        p_hi: f32,
        sat: f32,
    },
    Bleach {
        contrast: f32,
        desat: f32,
    },
    Fade {
        contrast: f32,
        blacks: f32,
        sat: f32,
        shadow: [f32; 3],
        high: [f32; 3],
        amount: f32,
    },
    Grain {
        amount: f32,
        cell: f32,
        soft: f32,
        luma_resp: f32,
        colored: bool,
        color_amt: f32,
        seed: u32,
    },
    Vignette {
        c: [f32; 2],
        inv_r: [f32; 2],
        feather: f32,
        stops: f32,
        keep: f32,
    },
    /// Exposure (gain), saturation and tint through a shape.
    Masked {
        shape: Shape,
        invert: bool,
        opacity: f32,
        gain: f32,
        sat: f32,
        tint: [f32; 3],
    },
    Qualifier(Box<Qualifier>),
    Lut {
        lut: Arc<Lut>,
        trilinear: bool,
        pre: Option<Conversion>,
        post: Option<Conversion>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qualifier {
    pub hue: f32,
    pub hue_width: f32,
    pub hue_soft: f32,
    /// Saturation and luminance bands in percent, softness in percent.
    pub sat: [f32; 2],
    pub lum: [f32; 2],
    pub soft: f32,
    pub invert: bool,
    pub show_mask: bool,
    pub opacity: f32,
    pub shift: f32,
    pub saturation: f32,
    pub gain: f32,
    pub tint: [f32; 3],
}

#[inline]
fn band(x: f32, lo: f32, hi: f32, soft: f32) -> f32 {
    let a = if lo <= 0.0 { 1.0 } else { smoothstep(lo - soft, lo + 1e-4, x) };
    let b = if hi >= 100.0 { 1.0 } else { 1.0 - smoothstep(hi - 1e-4, hi + soft, x) };
    a * b
}

/// Rational sigmoid: `x / (1 + |x|^p)^(1/p)` (−1..1, slope 1 at 0; larger p = harder knee).
#[inline]
pub fn rsig(x: f32, p: f32) -> f32 {
    x / (1.0 + x.abs().powf(p)).powf(1.0 / p)
}

#[inline]
fn dot(l: [f32; 3], c: [f32; 3]) -> f32 {
    l[0] * c[0] + l[1] * c[1] + l[2] * c[2]
}

#[inline]
fn saturate_by(c: [f32; 3], l: [f32; 3], f: f32) -> [f32; 3] {
    let y = dot(l, c);
    [y + (c[0] - y) * f, y + (c[1] - y) * f, y + (c[2] - y) * f]
}

impl PixelOp {
    /// Apply to straight RGB `c` at layer position `p`. `saved` is the Begin/End register.
    #[inline]
    pub fn apply(&self, c: [f32; 3], p: [f32; 2], w: &Working, saved: &mut [f32; 3]) -> [f32; 3] {
        let l = w.luma;
        match self {
            PixelOp::Decode => c.map(|v| w.decode(v)),
            PixelOp::Encode => c.map(|v| w.encode(v)),
            PixelOp::Begin => {
                *saved = c;
                c
            }
            PixelOp::End(s) => mix3(*saved, c, *s),
            PixelOp::Exposure { gain, gamma, protect, knee } => {
                let mut c = c.map(|v| v * gain);
                if *gamma != 1.0 {
                    c = c.map(|v| if v <= 1.0 { spow(v, *gamma) } else { 1.0 + gamma * (v - 1.0) });
                }
                if *protect > 0.0 {
                    let m = c[0].max(c[1]).max(c[2]);
                    if m > *knee {
                        let x = (m - knee) / (1.0 - knee).max(1e-4);
                        let t = knee + (1.0 - knee) * (x / (1.0 + x));
                        let r = t / m;
                        c = mix3(c, c.map(|v| v * r), *protect);
                    }
                }
                c
            }
            PixelOp::Contrast { k, pivot, filmic } => c.map(|v| {
                let lin = pivot + (v - pivot) * k;
                let fil = if v <= 0.0 || v >= 1.0 {
                    v
                } else if v < *pivot {
                    pivot * (v / pivot).powf(*k)
                } else {
                    1.0 - (1.0 - pivot) * ((1.0 - v) / (1.0 - pivot)).powf(*k)
                };
                lin + (fil - lin) * filmic
            }),
            PixelOp::Matrix(m) => mat3(m, c),
            PixelOp::Mono(m) => {
                let g = dot(*m, c);
                [g, g, g]
            }
            PixelOp::Wheels { offset, lift, gexp, gain, shadows, mids, highs } => {
                let mut c = [0, 1, 2].map(|i| {
                    let v = c[i] + offset[i];
                    let v = v + lift[i] * (1.0 - v);
                    spow(v, gexp[i]) * gain[i]
                });
                let y = dot(l, c);
                let ws = 1.0 - smoothstep(0.0, 0.55, y);
                let wh = smoothstep(0.45, 1.0, y);
                let wm = (1.0 - ws - wh).max(0.0);
                for i in 0..3 {
                    c[i] += shadows[i] * ws + mids[i] * wm + highs[i] * wh;
                }
                c
            }
            PixelOp::Curves(t) => {
                let [m, r, g, b] = &**t;
                let mut c = c;
                if let Some(m) = m {
                    c = c.map(|v| m.eval(v));
                }
                for (i, t) in [r, g, b].into_iter().enumerate() {
                    if let Some(t) = t {
                        c[i] = t.eval(c[i]);
                    }
                }
                c
            }
            PixelOp::Hsl { rot, sat, light, ranges } => {
                let (h, s) = hue_sat(c);
                // Interpolate the two neighbouring ranges' adjustments around the hue circle.
                let mut i = 7;
                for (k, ck) in HSL_CENTERS.iter().enumerate() {
                    if h >= *ck {
                        i = k;
                    }
                }
                let c0 = HSL_CENTERS[i];
                let c1 = if i == 7 { 360.0 } else { HSL_CENTERS[i + 1] };
                let t = ((h - c0) / (c1 - c0)).clamp(0.0, 1.0);
                let (a, b) = (ranges[i], ranges[(i + 1) % 8]);
                let ws = smoothstep(0.0, 0.2, s);
                let adj = [0, 1, 2].map(|k| (a[k] + (b[k] - a[k]) * t) * ws);
                let mut c = rotate_hue(c, rot + adj[0], l);
                c = saturate_by(c, l, sat * (1.0 + adj[1]));
                let d = adj[2] * 0.2;
                c = c.map(|v| v + d);
                if *light >= 0.0 { c.map(|v| v + (1.0 - v) * light * 0.6) } else { c.map(|v| v * (1.0 + light * 0.6)) }
            }
            PixelOp::HueRemap { center, range, falloff, shift, sat, lum } => {
                let (h, s) = hue_sat(c);
                let d = hue_dist(h, *center);
                let wgt = (1.0 - smoothstep(*range, range + falloff + 1e-3, d)) * smoothstep(0.03, 0.2, s);
                if wgt <= 0.0 {
                    return c;
                }
                let c2 = rotate_hue(c, shift * wgt, l);
                let c2 = saturate_by(c2, l, 1.0 + sat * wgt);
                c2.map(|v| v + lum * 0.25 * wgt)
            }
            PixelOp::Film { pre_gain, matrix, offsets, k, grey, p_lo, p_hi, sat } => {
                let mut s = c.map(|v| ((v * pre_gain).max(1e-6) / 0.18).log2());
                if let Some(m) = matrix {
                    s = mat3(m, s);
                }
                let d = [0, 1, 2].map(|i| {
                    let x = s[i] + offsets[i];
                    if x >= 0.0 { grey + (1.0 - grey) * rsig(k * x / (1.0 - grey), *p_hi) } else { grey + grey * rsig(k * x / grey, *p_lo) }
                });
                let d = saturate_by(d, l, *sat);
                d.map(|v| w.decode(v))
            }
            PixelOp::Bleach { contrast, desat } => {
                let yy = dot(l, c).clamp(0.0, 1.0);
                let ov = c.map(|v| if v < 0.5 { 2.0 * v * yy } else { 1.0 - 2.0 * (1.0 - v) * (1.0 - yy) });
                let r = mix3(c, ov, *contrast);
                saturate_by(r, l, 1.0 - desat)
            }
            PixelOp::Fade { contrast, blacks, sat, shadow, high, amount } => {
                let c = c.map(|v| 0.5 + (v - 0.5) * (1.0 - 0.5 * contrast));
                let c = c.map(|v| blacks * 0.3 + v * (1.0 - blacks * 0.3));
                let c = saturate_by(c, l, 1.0 + sat);
                let y = dot(l, c).clamp(0.0, 1.0);
                let (ws, wh) = ((1.0 - y) * (1.0 - y), y * y);
                [0, 1, 2].map(|i| c[i] + amount * 0.5 * (shadow[i] * ws + high[i] * wh))
            }
            PixelOp::Grain { amount, cell, soft, luma_resp, colored, color_amt, seed } => {
                // A rotated, offset lattice: no axis-aligned blocks, and pixel centres never sit
                // at the same place in every cell (so softness acts at any size).
                let (x, y) = ((0.8 * p[0] - 0.6 * p[1]) / cell + 0.371, (0.6 * p[0] + 0.8 * p[1]) / cell + 0.137);
                let n = |s: u32| 0.8 * value_noise(x, y, s, *soft) + 0.45 * value_noise(x * 2.0 + 17.31, y * 2.0 - 9.73, s ^ 0x68bc_21eb, *soft);
                let mono = n(*seed);
                let yc = dot(l, c).clamp(0.0, 1.0);
                let wgt = (1.0 - luma_resp) + luma_resp * (4.0 * yc * (1.0 - yc)).clamp(0.0, 1.0);
                let a = amount * 0.12 * wgt;
                if *colored {
                    [0u32, 1, 2].map(|i| {
                        let ch = n(seed ^ 0x02e5_be93u32.wrapping_mul(i + 1));
                        c[i as usize] + a * (mono + (ch - mono) * color_amt)
                    })
                } else {
                    c.map(|v| v + a * mono)
                }
            }
            PixelOp::Vignette { c: ctr, inv_r, feather, stops, keep } => {
                let dx = (p[0] - ctr[0]) * inv_r[0];
                let dy = (p[1] - ctr[1]) * inv_r[1];
                let d = (dx * dx + dy * dy).sqrt();
                let m = smoothstep(1.0 - feather, 1.0, d);
                let mut g = (stops * m).exp2();
                if *stops < 0.0 && *keep > 0.0 {
                    g += (1.0 - g) * keep * smoothstep(0.5, 2.0, dot(l, c));
                }
                c.map(|v| v * g)
            }
            PixelOp::Masked { shape, invert, opacity, gain, sat, tint } => {
                let mut m = shape.eval(p);
                if *invert {
                    m = 1.0 - m;
                }
                m *= opacity;
                if m <= 0.0 {
                    return c;
                }
                let c2 = [0, 1, 2].map(|i| c[i] * gain * tint[i]);
                let c2 = saturate_by(c2, l, 1.0 + sat);
                mix3(c, c2, m)
            }
            PixelOp::Qualifier(q) => {
                let (h, s) = hue_sat(c);
                let y = dot(l, c);
                let wh = if q.hue_width >= 180.0 { 1.0 } else { 1.0 - smoothstep(q.hue_width, q.hue_width + q.hue_soft + 1e-3, hue_dist(h, q.hue)) };
                let mut m = wh * band(s * 100.0, q.sat[0], q.sat[1], q.soft) * band(y * 100.0, q.lum[0], q.lum[1], q.soft);
                if q.invert {
                    m = 1.0 - m;
                }
                m *= q.opacity;
                if q.show_mask {
                    return [m, m, m];
                }
                if m <= 0.0 {
                    return c;
                }
                let c2 = rotate_hue(c, q.shift, l);
                let c2 = saturate_by(c2, l, 1.0 + q.saturation);
                let c2 = if q.gain != 1.0 { c2.map(|v| w.encode(w.decode(v) * q.gain)) } else { c2 };
                let c2 = [0, 1, 2].map(|i| c2[i] * q.tint[i]);
                mix3(c, c2, m)
            }
            PixelOp::Lut { lut, trilinear, pre, post } => {
                let c = pre.as_ref().map_or(c, |cv| cv.apply(c));
                let c = lut.apply(c, *trilinear);
                post.as_ref().map_or(c, |cv| cv.apply(c))
            }
        }
    }
}

/// Run a list of ops on one pixel.
#[inline]
pub fn run(ops: &[PixelOp], c: [f32; 3], p: [f32; 2], w: &Working) -> [f32; 3] {
    let mut saved = c;
    let mut c = c;
    for op in ops {
        c = op.apply(c, p, w, &mut saved);
    }
    // Hostile parameters can't leak NaN into the comp.
    c.map(|v| if v.is_finite() { v } else { 0.0 })
}
