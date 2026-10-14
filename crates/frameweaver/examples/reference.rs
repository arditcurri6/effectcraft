//! Score FrameWeaver against ground truth on ffmpeg-generated sequences:
//! `cargo run --release -p effectcraft-frameweaver --example reference`.
//!
//! For each sequence (60 fps, known frames) the even frames are the "footage"; every odd frame
//! is rebuilt from its two neighbours and compared with the real one by PSNR (dB) and SSIM.

#[path = "../tests/common/mod.rs"]
mod common;

use effectcraft_frameweaver::pipeline::{Options, interpolate};
use effectcraft_frameweaver::{Quality, Warping};
use effectcraft_raster::Image;

fn main() {
    if !common::ffmpeg_available() {
        eprintln!("ffmpeg not found: it generates the reference sequences");
        return;
    }
    let methods: Vec<(&str, Box<dyn Fn(&Image, &Image) -> Image>)> = vec![
        ("nearest", Box::new(|a, _| a.clone())),
        ("blend", Box::new(|a, b| effectcraft_frameweaver::synth::blend(a, b, 0.5))),
        ("fw draft", Box::new(|a, b| interpolate(a, b, 0.5, &Options { quality: Quality::Draft, ..Options::default() }).unwrap_or_else(|_| a.clone()))),
        ("fw balanced", Box::new(|a, b| interpolate(a, b, 0.5, &Options::default()).unwrap_or_else(|_| a.clone()))),
        ("fw high", Box::new(|a, b| interpolate(a, b, 0.5, &Options { quality: Quality::High, ..Options::default() }).unwrap_or_else(|_| a.clone()))),
        ("fw best", Box::new(|a, b| interpolate(a, b, 0.5, &Options { quality: Quality::Best, ..Options::default() }).unwrap_or_else(|_| a.clone()))),
        (
            "fw best fwd",
            Box::new(|a, b| {
                interpolate(a, b, 0.5, &Options { quality: Quality::Best, warping: Warping::Forward, ..Options::default() }).unwrap_or_else(|_| a.clone())
            }),
        ),
    ];
    print!("{:10}", "");
    for (name, _) in &methods {
        print!(" | {name:>16}");
    }
    println!(" | {:>16}", "ffmpeg mci");
    for (name, graph, n) in common::sequences() {
        let Some(truth) = common::generate(&graph, n, common::W, common::H) else {
            println!("{name}: generation failed");
            continue;
        };
        let source: Vec<Image> = truth.iter().step_by(2).cloned().collect();
        let targets: Vec<Image> = truth.iter().skip(1).step_by(2).take(source.len() - 1).cloned().collect();
        print!("{name:10}");
        for (_, f) in &methods {
            let made: Vec<Image> = source.windows(2).map(|w| f(&w[0], &w[1])).collect();
            let s = common::score(&made, &targets);
            print!(" | {:6.2} dB {:5.3}", s.psnr, s.ssim);
        }
        match common::ffmpeg_minterpolate(&source) {
            Some(out) => {
                let made: Vec<Image> = out.iter().skip(1).step_by(2).take(targets.len()).cloned().collect();
                let s = common::score(&made, &targets);
                println!(" | {:6.2} dB {:5.3}", s.psnr, s.ssim);
            }
            None => println!(" | {:>16}", "n/a"),
        }
    }
}
