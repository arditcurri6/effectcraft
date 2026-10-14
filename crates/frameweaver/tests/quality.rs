//! Objective quality gates on the procedural scene suite (`effectcraft_frameweaver::scenes`):
//! exact ground truth for the flow, for occlusion and for the in-between frame.
//!
//! Scenes: camera pan (sub-pixel), fast pan, zoom + rotation, parallax (a rectangle crossing a
//! moving background), two objects crossing, a small fast object, a noisy pan. Thresholds sit a
//! margin below what the implementation measures, so a regression in flow accuracy, occlusion
//! handling or synthesis fails here; `cargo run --release -p effectcraft-frameweaver --example
//! ablate` prints the full table.

use effectcraft_frameweaver::consistency::FlowPair;
use effectcraft_frameweaver::flow::{FlowParams, Quality};
use effectcraft_frameweaver::metrics::{psnr, ssim};
use effectcraft_frameweaver::pipeline::{Options, interpolate};
use effectcraft_frameweaver::prep::analysis_plane;
use effectcraft_frameweaver::scenes::{Scene, endpoint_error, suite};
use effectcraft_frameweaver::{Warping, synth};
use effectcraft_raster::Image;

const W: u32 = 192;
const H: u32 = 108;

struct Case {
    name: &'static str,
    scene: Scene,
    a: Image,
    b: Image,
    truth: Image,
}

fn cases() -> Vec<Case> {
    suite(W, H).into_iter().map(|(name, scene)| Case { name, a: scene.render(0.0, 0), b: scene.render(1.0, 1), truth: scene.render(0.5, 2), scene }).collect()
}

fn opts(q: Quality, warping: Warping) -> Options {
    Options { quality: q, warping, ..Options::default() }
}

fn epe(c: &Case, q: Quality) -> f32 {
    let p = FlowParams::for_quality(q, 0.7);
    let (pa, pb) = (analysis_plane(&c.a, p.max_long, None, false).unwrap(), analysis_plane(&c.b, p.max_long, None, false).unwrap());
    let pair = FlowPair::estimate(&pa.plane, &pb.plane, &p).unwrap();
    endpoint_error(&c.scene, 0.0, 1.0, &pair.fab, 8)
}

#[test]
fn flow_accuracy_against_the_true_flow() {
    let limits =
        [("pan", 0.06), ("fast pan", 0.05), ("zoom+rotate", 0.06), ("parallax", 0.30), ("crossing", 0.35), ("small fast object", 0.08), ("noisy pan", 0.15)];
    for c in cases() {
        let e = epe(&c, Quality::Best);
        let limit = limits.iter().find(|(n, _)| *n == c.name).map(|(_, l)| *l).unwrap();
        assert!(e <= limit, "{}: endpoint error {e} px (limit {limit})", c.name);
    }
}

#[test]
fn interpolation_beats_blending_on_every_scene_and_by_a_lot_on_average() {
    let (mut fwd, mut inv, mut blend, mut s) = (0.0, 0.0, 0.0, 0.0);
    let all = cases();
    for c in &all {
        let f = interpolate(&c.a, &c.b, 0.5, &opts(Quality::Best, Warping::Forward)).unwrap();
        let i = interpolate(&c.a, &c.b, 0.5, &opts(Quality::Best, Warping::Inverse)).unwrap();
        let bl = synth::blend(&c.a, &c.b, 0.5);
        let (pf, pi, pb) = (psnr(&f, &c.truth, 8).min(60.0), psnr(&i, &c.truth, 8).min(60.0), psnr(&bl, &c.truth, 8).min(60.0));
        assert!(pf >= pb + 3.0, "{}: forward {pf} dB vs blend {pb} dB", c.name);
        assert!(pi >= pb + 3.0 || pi >= 50.0, "{}: inverse {pi} dB vs blend {pb} dB", c.name);
        fwd += pf;
        inv += pi;
        blend += pb;
        s += ssim(&f, &c.truth, 8);
    }
    let n = all.len() as f64;
    assert!(fwd / n >= blend / n + 8.0, "forward {} dB vs blend {} dB", fwd / n, blend / n);
    assert!(inv / n >= blend / n + 6.0, "inverse {} dB vs blend {} dB", inv / n, blend / n);
    assert!(fwd / n >= 38.0 && s / n >= 0.97, "forward {} dB, SSIM {}", fwd / n, s / n);
}

/// Occlusion: on the scenes with relative motion both warps reconstruct the covered and
/// uncovered regions from the one frame that shows them, far better than a blend; the
/// depth-tested forward warp is never meaningfully worse than the inverse one.
#[test]
fn both_warps_handle_occlusion() {
    for c in cases().iter().filter(|c| matches!(c.name, "parallax" | "crossing" | "small fast object")) {
        let f = interpolate(&c.a, &c.b, 0.5, &opts(Quality::Best, Warping::Forward)).unwrap();
        let i = interpolate(&c.a, &c.b, 0.5, &opts(Quality::Best, Warping::Inverse)).unwrap();
        let blend = synth::blend(&c.a, &c.b, 0.5);
        let (pf, pi, pb) = (psnr(&f, &c.truth, 8), psnr(&i, &c.truth, 8), psnr(&blend, &c.truth, 8));
        assert!(pf >= pb + 5.0, "{}: forward {pf} dB vs blend {pb} dB", c.name);
        assert!(pi >= pb + 4.0, "{}: inverse {pi} dB vs blend {pb} dB", c.name);
        assert!(pf >= pi - 1.0, "{}: forward {pf} dB vs inverse {pi} dB", c.name);
    }
}

/// Each preset does more work and delivers more: flow error falls and PSNR rises (within noise)
/// from Draft to Best.
#[test]
fn quality_presets_are_ordered() {
    let all = cases();
    let mut prev: Option<(f32, f64)> = None;
    for q in Quality::ALL {
        let e = all.iter().map(|c| epe(c, q)).sum::<f32>() / all.len() as f32;
        let p =
            all.iter().map(|c| psnr(&interpolate(&c.a, &c.b, 0.5, &opts(q, Warping::Forward)).unwrap(), &c.truth, 8).min(60.0)).sum::<f64>() / all.len() as f64;
        if let Some((pe, pp)) = prev {
            assert!(e <= pe + 0.005, "{q:?}: endpoint error {e} px after {pe} px");
            assert!(p >= pp - 0.1, "{q:?}: {p} dB after {pp} dB");
        }
        prev = Some((e, p));
    }
    let (best_e, best_p) = prev.unwrap();
    assert!(best_e < 0.1 && best_p > 40.0, "best: {best_e} px, {best_p} dB");
}

/// Interpolating from the other end gives the same frame: A → B at w equals B → A at 1 − w.
#[test]
fn interpolation_is_symmetric_in_time() {
    for c in cases().iter().filter(|c| matches!(c.name, "pan" | "zoom+rotate" | "noisy pan")) {
        let o = opts(Quality::Balanced, Warping::Forward);
        let ab = interpolate(&c.a, &c.b, 0.3, &o).unwrap();
        let ba = interpolate(&c.b, &c.a, 0.7, &o).unwrap();
        assert!(psnr(&ab, &ba, 8) >= 35.0, "{}: {} dB", c.name, psnr(&ab, &ba, 8));
    }
}

/// Interpolating at other fractions: a 4× slow-down sequence (w = ¼, ½, ¾) tracks the truth.
#[test]
fn several_fractions_between_the_same_pair() {
    let c = cases().into_iter().find(|c| c.name == "zoom+rotate").unwrap();
    let (pair, _) = effectcraft_frameweaver::pipeline::estimate_pair(&c.a, &c.b, &opts(Quality::Best, Warping::Forward)).unwrap();
    for w in [0.25f32, 0.5, 0.75] {
        let truth = c.scene.render(w, 3);
        let tf = effectcraft_frameweaver::pipeline::time_field(&pair, &c.a, &c.b, w, &opts(Quality::Best, Warping::Forward)).unwrap();
        let out = synth::synthesize(&c.a, &c.b, &tf, w, true);
        assert!(psnr(&out, &truth, 8) >= 45.0, "w={w}: {} dB", psnr(&out, &truth, 8));
    }
}
