//! Writes a PPM montage for visual inspection: `montage OUT.ppm [scene-name]`.
//! Rows: ground truth | frame blend | FrameWeaver forward | FrameWeaver inverse | |forward − truth| × 4.
use effectcraft_frameweaver::pipeline::{Options, interpolate};
use effectcraft_frameweaver::scenes::suite;
use effectcraft_frameweaver::{Quality, Warping, synth};
use effectcraft_raster::Image;

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "montage.ppm".into());
    let want: Vec<String> = std::env::args().skip(2).collect();
    let (w, h) = (320u32, 180u32);
    let mut rows: Vec<Vec<Image>> = vec![];
    for (name, scene) in suite(w, h) {
        if !want.is_empty() && !want.iter().any(|n| n == name) {
            continue;
        }
        let (a, b, truth) = (scene.render(0.0, 0), scene.render(1.0, 1), scene.render(0.5, 2));
        let o = |warping| Options { quality: Quality::Best, warping, ..Options::default() };
        let fwd = interpolate(&a, &b, 0.5, &o(Warping::Forward)).unwrap();
        let inv = interpolate(&a, &b, 0.5, &o(Warping::Inverse)).unwrap();
        let mut err = Image::new(w, h);
        for (e, (p, q)) in err.data.iter_mut().zip(fwd.data.iter().zip(&truth.data)) {
            let d = |c: usize| ((p[c] - q[c]).abs() * 4.0).min(1.0);
            *e = [d(0), d(1), d(2), 1.0];
        }
        rows.push(vec![truth.clone(), synth::blend(&a, &b, 0.5), fwd, inv, err]);
    }
    let (cols, n) = (5u32, rows.len() as u32);
    let mut bytes = format!("P6\n{} {}\n255\n", w * cols, h * n).into_bytes();
    for r in &rows {
        for y in 0..h {
            for img in r {
                for x in 0..w {
                    let p = img.get(x as i64, y as i64);
                    bytes.extend((0..3).map(|c| (p[c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8));
                }
            }
        }
    }
    std::fs::write(&out, bytes).expect("write montage");
}
