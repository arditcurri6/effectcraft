//! Deterministic synthetic scenes for tests: a smooth multi-frequency texture that can be
//! sampled at any real coordinate, so a warped frame is exact ground truth.

use effectcraft_raster::Image;

use crate::plane::Plane;

/// The texture value at continuous (x, y), 0…1.
pub fn tex(x: f32, y: f32) -> f32 {
    const WAVES: [(f32, f32, f32, f32); 7] = [
        (0.11, 0.20, 0.3, 1.0),
        (0.27, -0.14, 1.1, 0.9),
        (0.05, 0.43, 2.0, 0.8),
        (-0.36, 0.22, 0.7, 0.7),
        (0.52, 0.31, 4.1, 0.5),
        (-0.19, -0.47, 5.3, 0.5),
        (0.63, -0.08, 3.3, 0.35),
    ];
    let mut s = 0.0;
    let mut norm = 0.0;
    for (fx, fy, ph, a) in WAVES {
        s += a * (fx * x + fy * y + ph).sin();
        norm += a;
    }
    0.5 + 0.5 * s / norm
}

/// A plane whose pixel (x, y) is the texture at `map(x, y)`.
pub fn render(w: usize, h: usize, map: impl Fn(f32, f32) -> (f32, f32)) -> Plane {
    let mut p = Plane::new(w, h).unwrap();
    for y in 0..h {
        for x in 0..w {
            let (u, v) = map(x as f32, y as f32);
            p.d[y * w + x] = tex(u, v);
        }
    }
    p
}

/// An opaque colour image whose pixel centre (x + 0.5, y + 0.5) shows the texture at
/// `map(x + 0.5, y + 0.5)`; the three channels are phase-shifted copies of the texture.
pub fn render_image(w: u32, h: u32, map: impl Fn(f32, f32) -> (f32, f32)) -> Image {
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = map(x as f32 + 0.5, y as f32 + 0.5);
            img.set(x, y, [tex(u, v), tex(u + 40.0, v - 25.0), tex(v + 13.0, u + 7.0), 1.0]);
        }
    }
    img
}

/// PSNR (dB) between two images' RGB over the pixels at least `margin` from the border.
pub fn psnr(a: &Image, b: &Image, margin: u32) -> f64 {
    let mut se = 0.0f64;
    let mut n = 0u64;
    for y in margin..a.height.saturating_sub(margin) {
        for x in margin..a.width.saturating_sub(margin) {
            let (p, q) = (a.get(x as i64, y as i64), b.get(x as i64, y as i64));
            for c in 0..3 {
                se += ((p[c] - q[c]) as f64).powi(2);
                n += 1;
            }
        }
    }
    if n == 0 || se == 0.0 { 99.0 } else { -10.0 * (se / n as f64).log10() }
}
