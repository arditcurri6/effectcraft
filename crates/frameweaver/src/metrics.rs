//! Image-quality metrics for tests and benchmarks.

use effectcraft_raster::Image;

/// PSNR (dB, peak 1) of the RGB channels over the pixels at least `margin` from the border;
/// 99 for identical images and for images of different size.
pub fn psnr(a: &Image, b: &Image, margin: u32) -> f64 {
    if a.width != b.width || a.height != b.height {
        return 99.0;
    }
    let (mut se, mut n) = (0.0f64, 0u64);
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

fn luma(p: [f32; 4]) -> f64 {
    (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) as f64
}

/// Mean structural similarity of the luma (Wang et al. 2004) over 8×8 windows on a stride of 4,
/// ignoring a `margin`-pixel border. 1 = identical.
pub fn ssim(a: &Image, b: &Image, margin: u32) -> f64 {
    if a.width != b.width || a.height != b.height {
        return 0.0;
    }
    const C1: f64 = 0.01 * 0.01;
    const C2: f64 = 0.03 * 0.03;
    let (x0, y0) = (margin, margin);
    let (x1, y1) = (a.width.saturating_sub(margin), a.height.saturating_sub(margin));
    let (mut total, mut windows) = (0.0f64, 0u64);
    let mut y = y0;
    while y + 8 <= y1 {
        let mut x = x0;
        while x + 8 <= x1 {
            let (mut ma, mut mb) = (0.0, 0.0);
            for j in 0..8 {
                for i in 0..8 {
                    ma += luma(a.get((x + i) as i64, (y + j) as i64));
                    mb += luma(b.get((x + i) as i64, (y + j) as i64));
                }
            }
            let (ma, mb) = (ma / 64.0, mb / 64.0);
            let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
            for j in 0..8 {
                for i in 0..8 {
                    let (da, db) = (luma(a.get((x + i) as i64, (y + j) as i64)) - ma, luma(b.get((x + i) as i64, (y + j) as i64)) - mb);
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            let (va, vb, cov) = (va / 63.0, vb / 63.0, cov / 63.0);
            total += ((2.0 * ma * mb + C1) * (2.0 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            windows += 1;
            x += 4;
        }
        y += 4;
    }
    if windows == 0 { 1.0 } else { total / windows as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_images_score_perfectly_and_noise_lowers_both() {
        let a = Image::filled(32, 32, [0.5, 0.4, 0.3, 1.0]);
        assert_eq!(psnr(&a, &a, 0), 99.0);
        assert!((ssim(&a, &a, 0) - 1.0).abs() < 1e-9);
        let mut b = a.clone();
        for (i, p) in b.data.iter_mut().enumerate() {
            p[0] += if i % 2 == 0 { 0.1 } else { -0.1 };
        }
        assert!(psnr(&a, &b, 0) < 30.0);
        // Different sizes never panic.
        assert_eq!(psnr(&a, &Image::new(4, 4), 0), 99.0);
        assert_eq!(ssim(&a, &Image::new(4, 4), 0), 0.0);
        assert_eq!(ssim(&Image::new(0, 0), &Image::new(0, 0), 0), 1.0);
    }
}
