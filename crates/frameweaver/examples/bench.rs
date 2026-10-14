//! Performance: `cargo run --release -p effectcraft-frameweaver --example bench [720|1080|2160 ...]`.
//!
//! Renders the "parallax" scene (fractal texture, a moving occluder) at each size, then times
//! every stage of one in-between frame per quality preset: analysis planes, the two flows,
//! the consistency analysis and repair, the time field (forward and inverse) and the synthesis.
//! A second run of the same pair shows the flow cache. Prints a Markdown table and the
//! process's peak resident memory (Linux).

use std::time::Instant;

use effectcraft_frameweaver::consistency::FlowPair;
use effectcraft_frameweaver::flow::{self, FlowParams, Quality};
use effectcraft_frameweaver::prep::analysis_plane;
use effectcraft_frameweaver::scenes::suite;
use effectcraft_frameweaver::{cache, field, synth};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn peak_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1024.0)
}

fn main() {
    let heights: Vec<u32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let heights = if heights.is_empty() { vec![720, 1080] } else { heights };
    println!("threads: {}", std::thread::available_parallelism().map_or(1, |n| n.get()));
    println!(
        "| frame | quality | analysis | flow A→B + B→A | consistency + repair | field fwd | field inv | synthesis | total (fwd) | cached re-render | peak RSS |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for h in heights {
        let w = h * 16 / 9;
        let scene = suite(w, h).into_iter().find(|(n, _)| *n == "parallax").map(|(_, s)| s);
        let Some(scene) = scene else { continue };
        let (a, b) = (scene.render(0.0, 0), scene.render(1.0, 1));
        for q in Quality::ALL {
            let p = FlowParams::for_quality(q, 0.7);
            cache::clear();
            let t = Instant::now();
            let (pa, pb) = (analysis_plane(&a, p.max_long, None, false), analysis_plane(&b, p.max_long, None, false));
            let (Some(pa), Some(pb)) = (pa, pb) else { continue };
            let t_analysis = ms(t);
            let t = Instant::now();
            let (fab, fba) = rayon::join(|| flow::estimate(&pa.plane, &pb.plane, &p), || flow::estimate(&pb.plane, &pa.plane, &p));
            let t_flow = ms(t);
            let (Ok(fab), Ok(fba)) = (fab, fba) else { continue };
            let t = Instant::now();
            let Some(pair) = FlowPair::from_flows(fab, fba, &p) else { continue };
            let t_cons = ms(t);
            let t = Instant::now();
            let tf = field::forward(&pair, &pa.plane, &pb.plane, 0.5, (w, h));
            let t_fwd = ms(t);
            let t = Instant::now();
            let _ = field::inverse(&pair, &pa.plane, &pb.plane, 0.5, (w, h));
            let t_inv = ms(t);
            let Some(tf) = tf else { continue };
            let t = Instant::now();
            let _ = synth::synthesize(&a, &b, &tf, 0.5, true);
            let t_syn = ms(t);
            // The effect's cached path: the second request for the same pair finds the flow.
            let key = cache::key(&pa.plane, &pb.plane, &p, &[]);
            let _ = cache::get_or_compute(key, || FlowPair::estimate(&pa.plane, &pb.plane, &p).ok());
            let t = Instant::now();
            let hit = cache::get_or_compute(key, || None);
            let (Some(_), Some(tf2)) = (hit, field::forward(&pair, &pa.plane, &pb.plane, 0.6, (w, h))) else { continue };
            let _ = synth::synthesize(&a, &b, &tf2, 0.6, true);
            let t_cached = ms(t);
            println!(
                "| {w}×{h} | {q:?} | {t_analysis:.0} ms | {t_flow:.0} ms | {t_cons:.0} ms | {t_fwd:.0} ms | {t_inv:.0} ms | {t_syn:.0} ms | {:.0} ms | {t_cached:.0} ms | {:.0} MiB |",
                t_analysis + t_flow + t_cons + t_fwd + t_syn,
                peak_mib()
            );
        }
    }
}
