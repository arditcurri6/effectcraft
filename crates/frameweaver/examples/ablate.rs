//! Tuning tool: score variants of the estimator on the procedural scene suite (exact ground
//! truth for flow and for the in-between frame). `cargo run --release -p effectcraft-frameweaver
//! --example ablate [width]`.

use effectcraft_frameweaver::consistency::FlowPair;
use effectcraft_frameweaver::flow::{FlowParams, Quality};
use effectcraft_frameweaver::metrics::{psnr, ssim};
use effectcraft_frameweaver::prep::analysis_plane;
use effectcraft_frameweaver::scenes::{endpoint_error, suite};
use effectcraft_frameweaver::{field, synth};

fn main() {
    let width: u32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(256);
    let height = width * 9 / 16;
    let scenes = suite(width, height);
    let frames: Vec<_> = scenes.iter().map(|(_, s)| (s.render(0.0, 0), s.render(1.0, 1), s.render(0.5, 2))).collect();
    let base = FlowParams::for_quality(Quality::Best, 0.7);
    let tweak = |name: &str, f: &dyn Fn(&mut FlowParams)| {
        let mut p = base;
        f(&mut p);
        (name.to_string(), p)
    };
    let variants: Vec<(String, FlowParams)> = vec![
        ("draft".into(), FlowParams::for_quality(Quality::Draft, 0.7)),
        ("balanced".into(), FlowParams::for_quality(Quality::Balanced, 0.7)),
        ("high".into(), FlowParams::for_quality(Quality::High, 0.7)),
        ("best".into(), base),
        tweak("best no refine", &|p| p.refine_outer = 0),
        tweak("best alpha x10", &|p| p.alpha *= 10.0),
        tweak("best alpha x30", &|p| p.alpha *= 30.0),
        tweak("best alpha x100", &|p| p.alpha *= 100.0),
    ];
    print!("{:20}", "variant");
    for (name, _) in &scenes {
        print!(" | {:>14}", name);
    }
    println!(" | {:>8} {:>6} {:>7}", "PSNR", "SSIM", "EPE px");
    for (vname, p) in &variants {
        for forward in [false, true] {
            print!("{:20}", format!("{vname}{}", if forward { " fwd" } else { "" }));
            let (mut tp, mut ts, mut te) = (0.0, 0.0, 0.0);
            for ((_, scene), (a, b, truth)) in scenes.iter().zip(&frames) {
                let (pa, pb) = (analysis_plane(a, p.max_long, None, false).unwrap(), analysis_plane(b, p.max_long, None, false).unwrap());
                let pair = FlowPair::estimate(&pa.plane, &pb.plane, p).unwrap();
                let tf = if forward {
                    field::forward(&pair, &pa.plane, &pb.plane, 0.5, (width, height)).unwrap()
                } else {
                    field::inverse(&pair, &pa.plane, &pb.plane, 0.5, (width, height)).unwrap()
                };
                let out = synth::synthesize(a, b, &tf, 0.5, true);
                let (ps, ss, ee) = (psnr(&out, truth, 8).min(60.0), ssim(&out, truth, 8), endpoint_error(scene, 0.0, 1.0, &pair.fab, 8));
                print!(" | {:5.1}dB {:4.2}e", ps, ee);
                tp += ps;
                ts += ss;
                te += ee as f64;
            }
            let n = scenes.len() as f64;
            println!(" | {:6.2}dB {:6.3} {:7.3}", tp / n, ts / n, te / n);
        }
    }
    // The references.
    for (name, f) in [("nearest", 0), ("blend", 1)] {
        print!("{name:20}");
        let (mut tp, mut ts) = (0.0, 0.0);
        for (_, (a, b, truth)) in scenes.iter().zip(&frames) {
            let out = if f == 0 { a.clone() } else { synth::blend(a, b, 0.5) };
            let (ps, ss) = (psnr(&out, truth, 8).min(60.0), ssim(&out, truth, 8));
            print!(" | {:5.1}dB       ", ps);
            tp += ps;
            ts += ss;
        }
        println!(" | {:6.2}dB {:6.3}", tp / scenes.len() as f64, ts / scenes.len() as f64);
    }
}
