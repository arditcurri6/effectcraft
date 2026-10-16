//! Colour maths shared by the tools (and mirrored operation for operation in
//! `effectcraft-gpu`'s `fx_cinelooks.wgsl`).

use effectcraft_color::ColorSpace;
use effectcraft_color::space::{Curve, bradford, invert, mul, mul_vec};

/// The colour environment of a render: the project's working space and how its pixels arrive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Working {
    pub space: ColorSpace,
    /// Pixels are linear light (Linearize Working Space, or a linear space such as ACEScg).
    pub input_linear: bool,
    /// The grading encoding (the space's curve; sRGB for linear spaces).
    pub curve: Curve,
    /// Luma weights of the space's primaries (the Y row of RGB → XYZ).
    pub luma: [f32; 3],
}

impl Working {
    /// `space` (`None` = unmanaged, treated as sRGB) with `linearized` pixels.
    pub fn new(space: Option<ColorSpace>, linearized: bool) -> Working {
        let space = space.unwrap_or(ColorSpace::Srgb);
        let curve = match space.curve() {
            Curve::Linear => Curve::Srgb,
            c => c,
        };
        let y = space.to_xyz()[1];
        let s = (y[0] + y[1] + y[2]).max(1e-9);
        Working { space, input_linear: linearized || space.is_linear(), curve, luma: [(y[0] / s) as f32, (y[1] / s) as f32, (y[2] / s) as f32] }
    }

    #[inline]
    pub fn decode(&self, v: f32) -> f32 {
        self.curve.decode(v)
    }
    #[inline]
    pub fn encode(&self, v: f32) -> f32 {
        self.curve.encode(v)
    }
    #[inline]
    pub fn y(&self, c: [f32; 3]) -> f32 {
        c[0] * self.luma[0] + c[1] * self.luma[1] + c[2] * self.luma[2]
    }
    /// Curve code for the GPU (`fcl_decode` / `fcl_encode`).
    pub fn curve_code(&self) -> f32 {
        match self.curve {
            Curve::Linear => 0.0,
            Curve::Srgb => 1.0,
            Curve::Gamma24 => 2.0,
            Curve::Pq => 3.0,
            Curve::Hlg => 4.0,
        }
    }
}

impl Default for Working {
    fn default() -> Working {
        Working::new(None, false)
    }
}

#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let d = e1 - e0;
    let t = if d.abs() < 1e-12 { if x < e0 { 0.0 } else { 1.0 } } else { ((x - e0) / d).clamp(0.0, 1.0) };
    t * t * (3.0 - 2.0 * t)
}

#[inline]
pub fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

#[inline]
pub fn mat3(m: &[[f32; 3]; 3], c: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2])
}

/// Sign-preserving power (over-range and negative values keep a continuous shape).
#[inline]
pub fn spow(x: f32, e: f32) -> f32 {
    x.abs().powf(e).copysign(x)
}

/// HSV hue in degrees (0..360) and HSV saturation of RGB (negative channels count as given).
#[inline]
pub fn hue_sat(c: [f32; 3]) -> (f32, f32) {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    let d = mx - mn;
    if d <= 1e-7 {
        return (0.0, 0.0);
    }
    let h = if mx == c[0] {
        60.0 * ((c[1] - c[2]) / d)
    } else if mx == c[1] {
        60.0 * ((c[2] - c[0]) / d + 2.0)
    } else {
        60.0 * ((c[0] - c[1]) / d + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if mx > 1e-7 { (d / mx).min(1.0) } else { 0.0 };
    (h, s)
}

/// Shortest angular distance in degrees (0..180).
#[inline]
pub fn hue_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(360.0);
    if d > 180.0 { 360.0 - d } else { d }
}

/// Rotate `c` around the grey axis by `deg`, keeping its luma `y` (weights `luma`).
#[inline]
pub fn rotate_hue(c: [f32; 3], deg: f32, luma: [f32; 3]) -> [f32; 3] {
    if deg == 0.0 {
        return c;
    }
    let (s, co) = deg.to_radians().sin_cos();
    let k = 1.0 / 3.0f32.sqrt();
    // Rodrigues around k = (1,1,1)/√3.
    let kv = k * (c[0] + c[1] + c[2]);
    let cross = [k * (c[2] - c[1]), k * (c[0] - c[2]), k * (c[1] - c[0])];
    let r = [0, 1, 2].map(|i| c[i] * co + cross[i] * s + k * kv * (1.0 - co));
    let y0 = luma[0] * c[0] + luma[1] * c[1] + luma[2] * c[2];
    let y1 = luma[0] * r[0] + luma[1] * r[1] + luma[2] * r[2];
    let d = y0 - y1;
    [r[0] + d, r[1] + d, r[2] + d]
}

/// The zero-luma direction of hue `deg` (a fully saturated colour minus its luma), scaled so
/// its largest component is 1.
pub fn hue_direction(deg: f32, luma: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = effectcraft_color::hsv_to_rgb((deg.rem_euclid(360.0)) / 360.0, 1.0, 1.0);
    let y = luma[0] * r + luma[1] * g + luma[2] * b;
    let d = [r - y, g - y, b - y];
    let m = d[0].abs().max(d[1].abs()).max(d[2].abs()).max(1e-6);
    d.map(|v| v / m)
}

/// A colour wheel `[x, y, master]` as an RGB offset: the disc position picks the hue (angle,
/// counter-clockwise from red on the right) and amount (radius); master moves all channels.
pub fn wheel_rgb(w: [f32; 3], luma: [f32; 3]) -> [f32; 3] {
    let r = (w[0] * w[0] + w[1] * w[1]).sqrt().min(1.0);
    let dir = if r > 1e-6 { hue_direction(w[1].atan2(w[0]).to_degrees(), luma) } else { [0.0; 3] };
    [0, 1, 2].map(|i| dir[i] * r + w[2])
}

/// CIE daylight chromaticity at correlated colour temperature `t` kelvin (4000..25000).
pub fn daylight_xy(t: f64) -> [f64; 2] {
    let t = t.clamp(4000.0, 25000.0);
    let (t2, t3) = (t * t, t * t * t);
    let x = if t <= 7000.0 { -4.6070e9 / t3 + 2.9678e6 / t2 + 0.09911e3 / t + 0.244063 } else { -2.0064e9 / t3 + 1.9018e6 / t2 + 0.24748e3 / t + 0.237040 };
    [x, -3.0 * x * x + 2.870 * x - 0.275]
}

/// White balance as a linear-RGB matrix in `space`: the scene is treated as lit by a daylight
/// illuminant `temp` (−1..1, mired shift) and `tint` (−1..1, green–magenta) away from
/// 6504 K, and adapted to it (Bradford). 0, 0 is the identity.
pub fn white_balance_matrix(space: ColorSpace, temp: f64, tint: f64) -> [[f32; 3]; 3] {
    let reference = daylight_xy(6504.0);
    let mired = 1e6 / 6504.0 - temp.clamp(-1.0, 1.0) * 80.0;
    let mut src = daylight_xy(1e6 / mired.max(1.0));
    // Positive tint: the light was greener, so the image turns magenta.
    src[1] += tint.clamp(-1.0, 1.0) * 0.012;
    let cat = bradford(src, reference);
    let m = space.to_xyz();
    let r = mul(&invert(&m), &mul(&cat, &m));
    r.map(|row| row.map(|v| v as f32))
}

/// Per-channel gains that make `n` (linear) neutral, keeping its luma.
pub fn neutral_gains(n: [f32; 3], luma: [f32; 3]) -> [f32; 3] {
    let y = luma[0] * n[0] + luma[1] * n[1] + luma[2] * n[2];
    n.map(|v| if v > 1e-4 && y > 1e-4 { (y / v).clamp(0.05, 20.0) } else { 1.0 })
}

/// Linear RGB of `space` → CIE XYZ (f32), for tests and tools that need it.
pub fn to_xyz(space: ColorSpace, c: [f32; 3]) -> [f32; 3] {
    let v = mul_vec(&space.to_xyz(), [c[0] as f64, c[1] as f64, c[2] as f64]);
    v.map(|x| x as f32)
}

// ---- deterministic noise (identical in fx_cinelooks.wgsl) ----

/// 32-bit integer hash (a "lowbias32" mix: xorshift-multiply rounds).
#[inline]
pub fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Uniform value in [−1, 1) for lattice point (`ix`, `iy`) under `seed`.
#[inline]
pub fn lattice(ix: i32, iy: i32, seed: u32) -> f32 {
    let h = hash32((ix as u32) ^ hash32((iy as u32) ^ hash32(seed)));
    (h >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
}

/// Value noise at `(x, y)` (lattice units): bilinear over hashed lattice values with a
/// sharpened smoothstep (`soft` 0 = crisp cells, 1 = smooth).
#[inline]
pub fn value_noise(x: f32, y: f32, seed: u32, soft: f32) -> f32 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ix, iy) = (fx as i32, fy as i32);
    let w = 0.08 + 0.92 * soft.clamp(0.0, 1.0);
    let sh = |t: f32| {
        let u = ((t - 0.5) / w + 0.5).clamp(0.0, 1.0);
        u * u * (3.0 - 2.0 * u)
    };
    let (tx, ty) = (sh(x - fx), sh(y - fy));
    let a = lattice(ix, iy, seed);
    let b = lattice(ix.wrapping_add(1), iy, seed);
    let c = lattice(ix, iy.wrapping_add(1), seed);
    let d = lattice(ix.wrapping_add(1), iy.wrapping_add(1), seed);
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working_luma_and_curves() {
        let w = Working::new(None, false);
        assert!((w.luma[0] - 0.2126).abs() < 1e-3 && (w.luma[1] - 0.7152).abs() < 1e-3);
        assert!(!w.input_linear);
        let a = Working::new(Some(ColorSpace::AcesCg), false);
        assert!(a.input_linear);
        assert_eq!(a.curve, Curve::Srgb);
        assert!((w.encode(w.decode(0.37)) - 0.37).abs() < 1e-5);
    }

    #[test]
    fn white_balance_identity_and_direction() {
        let m = white_balance_matrix(ColorSpace::Srgb, 0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!((m[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-6);
            }
        }
        let warm = mat3(&white_balance_matrix(ColorSpace::Srgb, 1.0, 0.0), [0.5, 0.5, 0.5]);
        assert!(warm[0] > warm[2] + 0.05, "{warm:?}");
        let magenta = mat3(&white_balance_matrix(ColorSpace::Srgb, 0.0, 1.0), [0.5, 0.5, 0.5]);
        assert!(magenta[1] < magenta[0] && magenta[1] < magenta[2], "{magenta:?}");
    }

    #[test]
    fn hue_rotation_keeps_luma() {
        let l = Working::default().luma;
        let c = [0.8, 0.3, 0.1];
        let r = rotate_hue(c, 120.0, l);
        let y = |c: [f32; 3]| l[0] * c[0] + l[1] * c[1] + l[2] * c[2];
        assert!((y(r) - y(c)).abs() < 1e-5);
        let (h0, _) = hue_sat(c);
        let (h1, _) = hue_sat(r);
        assert!(hue_dist(h1, h0 + 120.0) < 25.0, "{h0} {h1}");
    }

    #[test]
    fn noise_is_deterministic_and_bounded() {
        let a = value_noise(10.3, -4.7, 7, 0.5);
        assert_eq!(a, value_noise(10.3, -4.7, 7, 0.5));
        assert_ne!(a, value_noise(10.3, -4.7, 8, 0.5));
        let mut sum = 0.0;
        for i in 0..20000 {
            let v = lattice(i, i * 7, 3);
            assert!((-1.0..1.0).contains(&v));
            sum += v;
        }
        assert!((sum / 20000.0).abs() < 0.02);
    }

    #[test]
    fn wheel_hue_directions() {
        let l = Working::default().luma;
        let red = wheel_rgb([1.0, 0.0, 0.0], l);
        assert!(red[0] > 0.9 && red[1] < 0.0 && red[2] < 0.0);
        let y = l[0] * red[0] + l[1] * red[1] + l[2] * red[2];
        assert!(y.abs() < 1e-5);
        let cyan = wheel_rgb([-1.0, 0.0, 0.0], l);
        assert!(cyan[0] < 0.0 && cyan[1] > 0.0 && cyan[2] > 0.0);
    }
}
