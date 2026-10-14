//! FrameWeaver against ground truth on ffmpeg-generated sequences (fractal zoom, test
//! patterns, crossing textured objects, a noisy pan), with ffmpeg's own motion-compensated
//! interpolator as an external reference. Skipped when ffmpeg is not installed.
//! `cargo run --release -p effectcraft-frameweaver --example reference` prints the table.

mod common;

use effectcraft_frameweaver::pipeline::{Options, interpolate};
use effectcraft_frameweaver::synth::blend;
use effectcraft_frameweaver::{Quality, Warping};
use effectcraft_raster::Image;

#[test]
fn beats_blending_and_keeps_up_with_ffmpegs_motion_interpolation() {
    if !common::ffmpeg_available() {
        eprintln!("ffmpeg not installed: skipping the reference sequences");
        return;
    }
    let o = Options { quality: Quality::Best, warping: Warping::Forward, ..Options::default() };
    let (mut fw_total, mut mci_total, mut n) = (0.0, 0.0, 0.0);
    for (name, graph, frames) in common::sequences() {
        let Some(truth) = common::generate(&graph, frames, common::W, common::H) else { panic!("ffmpeg could not generate {name}") };
        let source: Vec<Image> = truth.iter().step_by(2).cloned().collect();
        let targets: Vec<Image> = truth.iter().skip(1).step_by(2).take(source.len() - 1).cloned().collect();
        let made: Vec<Image> = source.windows(2).map(|w| interpolate(&w[0], &w[1], 0.5, &o).unwrap()).collect();
        let blended: Vec<Image> = source.windows(2).map(|w| blend(&w[0], &w[1], 0.5)).collect();
        let (fw, bl) = (common::score(&made, &targets), common::score(&blended, &targets));
        assert!(fw.psnr >= bl.psnr + 1.5, "{name}: FrameWeaver {:.2} dB vs blend {:.2} dB", fw.psnr, bl.psnr);
        assert!(fw.ssim >= bl.ssim, "{name}: SSIM {:.3} vs blend {:.3}", fw.ssim, bl.ssim);
        let mci = common::ffmpeg_minterpolate(&source).map(|out| {
            let m: Vec<Image> = out.iter().skip(1).step_by(2).take(targets.len()).cloned().collect();
            common::score(&m, &targets)
        });
        eprintln!("{name}: FrameWeaver {:.2} dB / {:.3}, blend {:.2} dB, ffmpeg mci {:?}", fw.psnr, fw.ssim, bl.psnr, mci.as_ref().map(|s| s.psnr));
        if let Some(m) = mci {
            mci_total += m.psnr;
            fw_total += fw.psnr;
            n += 1.0;
        }
    }
    if n > 0.0 {
        assert!(fw_total / n >= mci_total / n - 1.5, "FrameWeaver {:.2} dB vs ffmpeg {:.2} dB on average", fw_total / n, mci_total / n);
    }
}
