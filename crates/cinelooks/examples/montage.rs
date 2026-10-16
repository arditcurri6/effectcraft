//! Visual check: every library look (or the looks named) applied to an image, as a contact sheet
//! (the original first, then the looks in library order, printed with their grid positions).
//!
//! ```text
//! cargo run -p effectcraft-cinelooks --example montage -- <in.png|jpg> <out.png> [--width 480] [--cols 6] [--looks "Noir,Golden Hour"] [--time 1.0]
//! ```
//!
//! The image is treated as an sRGB, unmanaged project (the default), at the given width; grain
//! uses frame `time × 24`.

use effectcraft_cinelooks::presets::builtin;
use effectcraft_cinelooks::{Env, compile, render};
use effectcraft_raster::Image;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (input, out) = match (args.first(), args.get(1)) {
        (Some(i), Some(o)) => (i.clone(), o.clone()),
        _ => return Err("usage: montage <in> <out.png> [--width W] [--cols N] [--looks a,b] [--time s]".into()),
    };
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let width: u32 = opt("--width").and_then(|v| v.parse().ok()).unwrap_or(480);
    let cols: u32 = opt("--cols").and_then(|v| v.parse().ok()).unwrap_or(6);
    let time: f64 = opt("--time").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let names: Option<Vec<String>> = opt("--looks").map(|s| s.split(',').map(|x| x.trim().to_lowercase()).collect());
    let src = image::open(&input).map_err(|e| e.to_string())?.to_rgba32f();
    let h = (src.height() as f64 * width as f64 / src.width() as f64).round() as u32;
    let small = image::imageops::resize(&src, width, h, image::imageops::FilterType::Triangle);
    let mut img = Image::new(width, h);
    for (i, p) in small.pixels().enumerate() {
        let a = p[3];
        img.data[i] = [p[0] * a, p[1] * a, p[2] * a, a];
    }
    let env = Env { layer_size: [width as f64, h as f64], time, ..Env::default() };
    let looks: Vec<_> = builtin().iter().filter(|p| names.as_ref().is_none_or(|n| n.contains(&p.name.to_lowercase()))).collect();
    let n = looks.len() as u32 + 1;
    let rows = n.div_ceil(cols);
    let mut sheet = image::RgbaImage::new(cols * width, rows * h);
    let mut put = |k: u32, im: &Image| {
        let (ox, oy) = ((k % cols) * width, (k / cols) * h);
        for y in 0..h {
            for x in 0..width {
                let p = im.data[(y * width + x) as usize];
                let a = p[3].max(1e-6);
                let c = |v: f32| ((v / a).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                sheet.put_pixel(ox + x, oy + y, image::Rgba([c(p[0]), c(p[1]), c(p[2]), 255]));
            }
        }
    };
    put(0, &img);
    println!("0: (original)");
    for (k, p) in looks.iter().enumerate() {
        let started = std::time::Instant::now();
        let prog = compile(&p.look, &env, 1.0);
        let out = render(&prog, &img);
        println!("{}: {} [{}] {:.1} ms", k + 1, p.name, p.category, started.elapsed().as_secs_f64() * 1000.0);
        put(k as u32 + 1, &out);
    }
    sheet.save(&out).map_err(|e| e.to_string())?;
    println!("wrote {out}");
    Ok(())
}
