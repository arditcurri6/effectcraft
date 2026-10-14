//! Where the flow time goes: `cargo run --release -p effectcraft-frameweaver --example profile_flow [height]`.
use std::time::Instant;

use effectcraft_frameweaver::flow::{self, FlowParams, Quality};
use effectcraft_frameweaver::prep::analysis_plane;
use effectcraft_frameweaver::scenes::suite;

fn main() {
    let h: u32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(1080);
    let w = h * 16 / 9;
    let scene = suite(w, h).into_iter().find(|(n, _)| *n == "parallax").map(|(_, s)| s).unwrap();
    let (a, b) = (scene.render(0.0, 0), scene.render(1.0, 1));
    let base = FlowParams::for_quality(Quality::Best, 0.7);
    let (pa, pb) = (analysis_plane(&a, base.max_long, None, false).unwrap(), analysis_plane(&b, base.max_long, None, false).unwrap());
    let run = |name: &str, p: &FlowParams| {
        let t = Instant::now();
        let _ = flow::estimate(&pa.plane, &pb.plane, p);
        println!("{name:34} {:7.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    };
    run("best (one direction)", &base);
    run("  no variational refinement", &FlowParams { refine_outer: 0, ..base });
    run("  no refinement, no median", &FlowParams { refine_outer: 0, median: 0, ..base });
    run("  refinement only 1 outer x 10", &FlowParams { refine_outer: 1, ..base });
    run("  LK iterations 4", &FlowParams { iters: 4, ..base });
    run("  overlap 0.3", &FlowParams { overlap: 0.3, ..base });
}
