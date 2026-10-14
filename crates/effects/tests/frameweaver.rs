//! FrameWeaver through the effect API, with a mock host that serves a moving scene at
//! discrete source frames (like footage): behaviour, accuracy against analytic ground truth,
//! time remapping, cuts, boundaries, separation, guides, external vectors, caching and
//! hostile input. Everything runs on the CPU.

use std::sync::{Mutex, MutexGuard, PoisonError};

use effectcraft_effects::{Buf, EffectCtx, EffectEnv, EffectHost, LayerPixels, Params, apply, default_value, find, flatten_params, instantiate};
use effectcraft_keyframe::Value;
use effectcraft_project::build::Ids;
use effectcraft_raster::Image;

const W: u32 = 128;
const H: u32 = 96;
const COMP_FPS: f64 = 30.0;
const ID: &str = "ec.time.frameweaver";

const MATTE: u64 = 20;
const VECTORS: u64 = 30;
const MOTION: u64 = 40;

/// The cache and its counters are process-wide: tests take turns.
fn serial() -> MutexGuard<'static, ()> {
    static L: Mutex<()> = Mutex::new(());
    L.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A smooth multi-frequency texture, sampled at any real coordinate (exact ground truth).
fn tex(x: f32, y: f32) -> f32 {
    const WAVES: [(f32, f32, f32, f32); 7] = [
        (0.11, 0.20, 0.3, 1.0),
        (0.27, -0.14, 1.1, 0.9),
        (0.05, 0.43, 2.0, 0.8),
        (-0.36, 0.22, 0.7, 0.7),
        (0.52, 0.31, 4.1, 0.5),
        (-0.19, -0.47, 5.3, 0.5),
        (0.63, -0.08, 3.3, 0.35),
    ];
    let (mut s, mut norm) = (0.0, 0.0);
    for (fx, fy, ph, a) in WAVES {
        s += a * (fx * x + fy * y + ph).sin();
        norm += a;
    }
    0.5 + 0.5 * s / norm
}

fn paint(map: impl Fn(f32, f32) -> (f32, f32)) -> Image {
    let mut img = Image::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let (u, v) = map(x as f32 + 0.5, y as f32 + 0.5);
            img.set(x, y, [tex(u, v), tex(u + 40.0, v - 25.0), tex(v + 13.0, u + 7.0), 1.0]);
        }
    }
    img
}

/// A different scene altogether: a dark blue palette.
fn paint_other(map: impl Fn(f32, f32) -> (f32, f32)) -> Image {
    let mut img = Image::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let (u, v) = map(x as f32 + 0.5, y as f32 + 0.5);
            img.set(x, y, [0.1 * tex(u, v), 0.2 * tex(v, u), 0.6 + 0.4 * tex(u + 9.0, v), 1.0]);
        }
    }
    img
}

fn psnr(a: &Image, b: &Image, margin: u32) -> f64 {
    let (mut se, mut n) = (0.0f64, 0u64);
    for y in margin..a.height - margin {
        for x in margin..a.width - margin {
            let (p, q) = (a.get(x as i64, y as i64), b.get(x as i64, y as i64));
            for c in 0..3 {
                se += ((p[c] - q[c]) as f64).powi(2);
                n += 1;
            }
        }
    }
    if se == 0.0 { 99.0 } else { -10.0 * (se / n as f64).log10() }
}

#[derive(Clone, Copy)]
enum Scene {
    /// The whole texture moves by (dx, dy) px per source frame.
    Pan(f32, f32),
    /// Like Pan, with an unrelated scene from source frame `cut` on.
    Cut(f32, f32, i64),
    /// A background moving left and a square foreground moving right.
    Sep,
}

impl Scene {
    /// The frame at (fractional) source frame `k`.
    fn at(&self, k: f32) -> Image {
        match *self {
            Scene::Pan(dx, dy) => paint(move |x, y| (x - dx * k, y - dy * k)),
            Scene::Cut(dx, dy, cut) => {
                if k >= cut as f32 {
                    paint_other(move |x, y| (y * 0.7 + 300.0 - dx * k, x * 0.9 + 123.0 - dy * k))
                } else {
                    paint(move |x, y| (x - dx * k, y - dy * k))
                }
            }
            Scene::Sep => {
                let sx = 20.0 + 5.0 * k;
                paint(move |x, y| if x >= sx && x < sx + 44.0 && (28.0..72.0).contains(&y) { (x - sx + 500.0, y + 300.0) } else { (x + 3.0 * k, y) })
            }
        }
    }

    /// The foreground matte (the square) at source frame `k`.
    fn matte(&self, k: f32) -> Image {
        let sx = 20.0 + 5.0 * k;
        let mut m = Image::new(W, H);
        for y in 0..H {
            for x in 0..W {
                let a = if (x as f32 + 0.5) >= sx && (x as f32 + 0.5) < sx + 44.0 && (28.0..72.0).contains(&(y as f32 + 0.5)) { 1.0 } else { 0.0 };
                m.set(x, y, [a, a, a, a]);
            }
        }
        m
    }
}

type Curve = Box<dyn Fn(f64) -> f64 + Sync>;

struct Mock {
    scene: Scene,
    /// Source frame rate of the footage.
    fps_in: f64,
    /// Source frames that exist.
    frames: std::ops::Range<i64>,
    /// Keyframed values (layer time → value) overriding the static parameter.
    speed: Option<Curve>,
    marker: Option<Curve>,
    /// Guide 1's point (x, y) at layer time.
    guide: Option<Box<dyn Fn(f64) -> [f64; 2] + Sync>>,
    /// Vector layer content (encoded motion), if any.
    vectors: Option<Image>,
    base: Params,
}

impl Mock {
    fn new(scene: Scene, vals: &[(&str, Value)]) -> Mock {
        let spec = find(ID).unwrap();
        let size = [W as f64, H as f64];
        let mut base = Params { values: spec.params.iter().map(|p| (p.id.to_string(), default_value(p, size))).collect() };
        for (k, v) in vals {
            assert!(base.values.contains_key(*k), "unknown param {k}");
            base.values.insert(k.to_string(), v.clone());
        }
        Mock { scene, fps_in: COMP_FPS, frames: 0..1000, speed: None, marker: None, guide: None, vectors: None, base }
    }

    fn frame_index(&self, t: f64) -> Option<i64> {
        let k = (t * self.fps_in).round();
        ((t * self.fps_in - k).abs() < 1e-3 && self.frames.contains(&(k as i64))).then_some(k as i64)
    }
}

impl EffectHost for Mock {
    fn layer(&self, _: u64, _: bool) -> Option<LayerPixels> {
        None
    }
    fn audio(&self, _: u64, _: f64, _: usize, _: u32) -> Option<Vec<f32>> {
        None
    }
    fn self_at(&self, t: f64, _: usize) -> Option<Buf> {
        let k = self.frame_index(t)?;
        Some(Buf { img: self.scene.at(k as f32), offset: [0.0; 2], scale: 1.0 })
    }
    fn layer_at(&self, id: u64, t: f64, _: bool) -> Option<LayerPixels> {
        let k = (t * self.fps_in).round() as f32;
        let img = match id {
            MATTE => self.scene.matte(k),
            VECTORS => self.vectors.clone()?,
            MOTION => self.scene.at(k),
            _ => return None,
        };
        Some(LayerPixels { buf: Buf { img, offset: [0.0; 2], scale: 1.0 }, size: [W as f64, H as f64] })
    }
    fn params_at(&self, t: f64) -> Option<Params> {
        let mut p = self.base.clone();
        if let Some(f) = &self.speed {
            p.values.insert("timeRemap/speed".into(), Value::Scalar(f(t)));
        }
        if let Some(f) = &self.marker {
            p.values.insert("segments/marker".into(), Value::Scalar(f(t)));
        }
        if let Some(g) = &self.guide {
            p.values.insert("guidance/guides/guide1/point".into(), Value::Vec2(g(t)));
        }
        Some(p)
    }
}

fn run(host: &Mock, t: f64) -> Image {
    let spec = find(ID).unwrap();
    let mut params = host.base.clone();
    if let Some(f) = &host.speed {
        params.values.insert("timeRemap/speed".into(), Value::Scalar(f(t)));
    }
    let env = EffectEnv { host: Some(host), frame_rate: COMP_FPS, comp_time: t, ..Default::default() };
    let ctx = EffectCtx { params: &params, time: t, layer_size: [W as f64, H as f64], seed: 1, adjustment: false, env };
    apply(spec, &ctx, Buf { img: Image::filled(W, H, [0.9, 0.1, 0.1, 1.0]), offset: [0.0; 2], scale: 1.0 }).img
}

fn n(v: f64) -> Value {
    Value::Scalar(v)
}
fn e(v: u32) -> Value {
    Value::Enum(v)
}
fn on(b: bool) -> Value {
    Value::Bool(b)
}

/// Output time at which source frame `x` is shown at Speed 50 % and 30 fps footage.
fn t_at_half_speed(x: f64) -> f64 {
    2.0 * x / COMP_FPS
}

const BEST: u32 = 3;

// ---------------------------------------------------------------- interpolation modes

#[test]
fn static_scene_stays_static_in_every_mode() {
    let _g = serial();
    for mode in 0..3 {
        let host = Mock::new(Scene::Pan(0.0, 0.0), &[("interp/mode", e(mode))]);
        let out = run(&host, t_at_half_speed(3.5));
        assert!(psnr(&out, &host.scene.at(3.5), 4) > 55.0, "mode {mode}");
    }
}

#[test]
fn modes_are_genuinely_different_and_motion_wins_on_constant_velocity() {
    let _g = serial();
    let truth = Scene::Pan(8.0, 2.0).at(3.5);
    let t = t_at_half_speed(3.5);
    let render = |mode: u32| run(&Mock::new(Scene::Pan(8.0, 2.0), &[("interp/mode", e(mode)), ("motion/quality", e(BEST))]), t);
    let (nearest, blend, motion) = (render(0), render(1), render(2));
    // Nearest is exactly a source frame; Blend is exactly the cross-fade.
    assert_eq!(nearest, Scene::Pan(8.0, 2.0).at(4.0));
    let (a, b) = (Scene::Pan(8.0, 2.0).at(3.0), Scene::Pan(8.0, 2.0).at(4.0));
    for i in [10usize, 5000, 9000] {
        for c in 0..4 {
            assert!((blend.data[i][c] - 0.5 * (a.data[i][c] + b.data[i][c])).abs() < 1e-6);
        }
    }
    let (pm, pb, pn) = (psnr(&motion, &truth, 16), psnr(&blend, &truth, 16), psnr(&nearest, &truth, 16));
    assert!(pm >= 35.0, "motion {pm} dB");
    assert!(pm >= pb + 8.0, "motion {pm} dB vs blend {pb} dB");
    assert!(pm >= pn + 8.0, "motion {pm} dB vs nearest {pn} dB");
    assert!(motion != blend && motion != nearest && blend != nearest);
}

#[test]
fn fast_motion_and_camera_pan_are_followed() {
    let _g = serial();
    // 22 px per frame on a 128 px wide frame.
    let host = Mock::new(Scene::Pan(22.0, -5.0), &[("motion/quality", e(BEST))]);
    let out = run(&host, t_at_half_speed(5.5));
    let truth = host.scene.at(5.5);
    let blend = run(&Mock::new(Scene::Pan(22.0, -5.0), &[("interp/mode", e(1))]), t_at_half_speed(5.5));
    let (pm, pb) = (psnr(&out, &truth, 28), psnr(&blend, &truth, 28));
    assert!(pm >= 28.0 && pm >= pb + 6.0, "fast: motion {pm} dB vs blend {pb} dB");
    // Every quality preset interpolates a slow pan correctly and all four differ in work done.
    let truth = Scene::Pan(4.0, 1.0).at(2.25);
    for q in 0..4 {
        let host = Mock::new(Scene::Pan(4.0, 1.0), &[("motion/quality", e(q)), ("timeRemap/speed", n(25.0))]);
        let out = run(&host, 2.0 * 2.25 / COMP_FPS * 2.0);
        assert!(psnr(&out, &truth, 12) >= 30.0, "quality {q}: {} dB", psnr(&out, &truth, 12));
    }
}

#[test]
fn warping_modes_agree_on_rigid_motion() {
    let _g = serial();
    let t = t_at_half_speed(3.5);
    let inv = run(&Mock::new(Scene::Pan(7.0, -2.0), &[("motion/quality", e(BEST))]), t);
    let fwd = run(&Mock::new(Scene::Pan(7.0, -2.0), &[("motion/quality", e(BEST)), ("interp/warping", e(1))]), t);
    assert!(psnr(&inv, &fwd, 16) >= 35.0, "{} dB", psnr(&inv, &fwd, 16));
    assert!(inv != fwd);
}

// ---------------------------------------------------------------- time remapping

#[test]
fn reverse_playback_and_exact_frames() {
    let _g = serial();
    // Speed −100 %, starting at frame 8: output frame j shows source frame 8 − j exactly.
    let host = Mock::new(Scene::Pan(5.0, 0.0), &[("timeRemap/speed", n(-100.0)), ("timeRemap/startFrame", n(8.0))]);
    for j in 0..4 {
        assert_eq!(run(&host, j as f64 / COMP_FPS), host.scene.at((8 - j) as f32), "frame {j}");
    }
    // −50 %: halfway between 7 and 8 at the first output frame.
    let host = Mock::new(Scene::Pan(5.0, 0.0), &[("timeRemap/speed", n(-50.0)), ("timeRemap/startFrame", n(8.0)), ("motion/quality", e(BEST))]);
    assert!(psnr(&run(&host, 1.0 / COMP_FPS), &host.scene.at(7.5), 12) >= 30.0);
}

#[test]
fn freeze_frame_and_frame_number_mode() {
    let _g = serial();
    let host = Mock::new(Scene::Pan(6.0, 1.0), &[("timeRemap/mode", e(1)), ("timeRemap/frameNum", n(3.5)), ("motion/quality", e(BEST))]);
    let (a, b, c) = (run(&host, 0.0), run(&host, 0.37), run(&host, 1.9));
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert!(psnr(&a, &host.scene.at(3.5), 12) >= 32.0);
    // An integer frame number is exactly that frame.
    let host = Mock::new(Scene::Pan(6.0, 1.0), &[("timeRemap/mode", e(1)), ("timeRemap/frameNum", n(5.0))]);
    assert_eq!(run(&host, 0.2), host.scene.at(5.0));
}

#[test]
fn speed_ramp_follows_the_integral_of_the_speed() {
    let _g = serial();
    // Speed ramps 0 → 200 % over 8 frames, then holds 200 %. s(t) is the area under it.
    let ramp = |t: f64| (t / (8.0 / COMP_FPS)).min(1.0) * 200.0;
    let source = |t: f64| -> f64 {
        let steps = 4000;
        let h = t / steps as f64;
        (0..steps).map(|i| 0.5 * (ramp(i as f64 * h) + ramp((i + 1) as f64 * h)) * h).sum::<f64>() / 100.0
    };
    let mut host = Mock::new(Scene::Pan(5.0, 1.0), &[("motion/quality", e(BEST))]);
    host.speed = Some(Box::new(ramp));
    for j in [3, 6, 11, 14] {
        let t = j as f64 / COMP_FPS;
        let s = source(t) * COMP_FPS;
        let out = run(&host, t);
        let p = psnr(&out, &host.scene.at(s as f32), 12);
        assert!(p >= 28.0, "frame {j}: source frame {s:.3}, {p} dB");
    }
    // The source time is continuous across the end of the ramp (no jump in what is shown).
    let before = run(&host, 7.9 / COMP_FPS);
    let after = run(&host, 8.1 / COMP_FPS);
    assert!(psnr(&before, &after, 12) > 18.0);
}

#[test]
fn different_source_and_output_frame_rates() {
    let _g = serial();
    // 24 fps footage in a 30 fps comp at 100 %: output frame 1 shows source frame 0.8.
    let mut host = Mock::new(Scene::Pan(6.0, 0.0), &[("timeRemap/speed", n(100.0)), ("timeRemap/inputFrameRate", n(24.0)), ("motion/quality", e(BEST))]);
    host.fps_in = 24.0;
    let out = run(&host, 1.0 / 30.0);
    assert!(psnr(&out, &host.scene.at(0.8), 12) >= 32.0, "{} dB", psnr(&out, &host.scene.at(0.8), 12));
    // Frame 5 of the comp = source frame 4 exactly (5 · 24/30).
    assert_eq!(run(&host, 5.0 / 30.0), host.scene.at(4.0));
}

#[test]
fn clip_boundaries_hold_the_available_frame_and_outside_is_transparent() {
    let _g = serial();
    let mut host = Mock::new(Scene::Pan(5.0, 0.0), &[("timeRemap/speed", n(100.0)), ("timeRemap/startFrame", n(4.5))]);
    host.frames = 0..6;
    // Source frame 5.5 needs frame 6, which does not exist: frame 5 is held.
    assert_eq!(run(&host, 1.0 / COMP_FPS), host.scene.at(5.0));
    // Before the start: frame −1 does not exist, frame 0 is held.
    let mut host = Mock::new(Scene::Pan(5.0, 0.0), &[("timeRemap/speed", n(100.0)), ("timeRemap/startFrame", n(-0.5))]);
    host.frames = 0..6;
    assert_eq!(run(&host, 0.0), host.scene.at(0.0));
    // Far outside: nothing to show.
    let mut host = Mock::new(Scene::Pan(5.0, 0.0), &[("timeRemap/startFrame", n(50.0)), ("timeRemap/speed", n(100.0))]);
    host.frames = 0..6;
    let out = run(&host, 0.0);
    assert!(out.data.iter().all(|p| p[3] == 0.0));
}

// ---------------------------------------------------------------- cuts

#[test]
fn hard_cuts_are_not_interpolated_across() {
    let _g = serial();
    let host = Mock::new(Scene::Cut(4.0, 1.0, 5), &[("motion/quality", e(1))]);
    // Between frames 4 and 5 (the cut): the nearest frame is held, no mixing.
    let early = run(&host, t_at_half_speed(4.25));
    let late = run(&host, t_at_half_speed(4.75));
    assert_eq!(early, host.scene.at(4.0));
    assert_eq!(late, host.scene.at(5.0));
    // Away from the cut the same footage is interpolated normally.
    let inside = run(&host, t_at_half_speed(2.5));
    assert!(psnr(&inside, &host.scene.at(2.5), 12) >= 30.0);
    // A big pan (20 px) is not mistaken for a cut.
    let pan = Mock::new(Scene::Pan(20.0, 0.0), &[("motion/quality", e(1))]);
    let out = run(&pan, t_at_half_speed(2.5));
    assert_ne!(out, pan.scene.at(2.0));
    assert_ne!(out, pan.scene.at(3.0));
}

#[test]
fn segment_marker_and_disabling_auto_detect() {
    let _g = serial();
    // A marker that changes between source frames 2 and 3 splits the segment without any
    // visual cut.
    let mut host = Mock::new(Scene::Pan(5.0, 0.0), &[("segments/autoDetect", on(false))]);
    host.marker = Some(Box::new(|t| if t * COMP_FPS > 2.5 { 1.0 } else { 0.0 }));
    assert_eq!(run(&host, t_at_half_speed(2.25)), host.scene.at(2.0));
    assert_eq!(run(&host, t_at_half_speed(2.75)), host.scene.at(3.0));
    // With Auto-Detect off a real cut is interpolated through (garbage in between, not a hold).
    let host = Mock::new(Scene::Cut(4.0, 1.0, 5), &[("segments/autoDetect", on(false))]);
    let mid = run(&host, t_at_half_speed(4.5));
    assert!(mid != host.scene.at(4.0) && mid != host.scene.at(5.0));
}

// ---------------------------------------------------------------- guidance

#[test]
fn separation_keeps_both_objects_sharp() {
    let _g = serial();
    let t = t_at_half_speed(2.5);
    let truth = Scene::Sep.at(2.5);
    let plain = run(&Mock::new(Scene::Sep, &[("motion/quality", e(BEST))]), t);
    let sep = run(&Mock::new(Scene::Sep, &[("motion/quality", e(BEST)), ("guidance/separation/matteLayer", Value::Layer(Some(MATTE)))]), t);
    let (ps, pp) = (psnr(&sep, &truth, 8), psnr(&plain, &truth, 8));
    assert!(ps >= 28.0, "separated {ps} dB");
    assert!(ps >= pp - 0.5, "separated {ps} dB vs plain {pp} dB");
    // The pieces are available as displays: FG only has the square, BG only has a plate.
    let fg = run(&Mock::new(Scene::Sep, &[("guidance/separation/matteLayer", Value::Layer(Some(MATTE))), ("diagnostics/display", e(6))]), t);
    let bg = run(&Mock::new(Scene::Sep, &[("guidance/separation/matteLayer", Value::Layer(Some(MATTE))), ("diagnostics/display", e(7))]), t);
    let matte = run(&Mock::new(Scene::Sep, &[("guidance/separation/matteLayer", Value::Layer(Some(MATTE))), ("diagnostics/display", e(5))]), t);
    let inside = (W as usize / 2 + 3) + (H as usize / 2) * W as usize; // inside the square at t
    let outside = 3 + (H as usize / 2) * W as usize;
    assert!(fg.data[inside][3] > 0.9 && fg.data[outside][3] < 0.1, "fg alpha {} / {}", fg.data[inside][3], fg.data[outside][3]);
    assert!(bg.data[inside][3] > 0.9 && bg.data[outside][3] > 0.9);
    assert!(matte.data[inside][0] > 0.9 && matte.data[outside][0] < 0.1);
    for name in [BG_FILL_NEIGHBOUR, BG_FILL_INPAINT] {
        let out = run(&Mock::new(Scene::Sep, &[("guidance/separation/matteLayer", Value::Layer(Some(MATTE))), ("guidance/separation/bgFill", e(name))]), t);
        assert!(psnr(&out, &truth, 8) >= 24.0);
    }
}
const BG_FILL_NEIGHBOUR: u32 = 0;
const BG_FILL_INPAINT: u32 = 1;

#[test]
fn guides_change_the_motion_and_a_pinned_guide_holds_still() {
    let _g = serial();
    let t = t_at_half_speed(3.5);
    let base = Mock::new(Scene::Pan(8.0, 0.0), &[("motion/quality", e(1))]);
    let without = run(&base, t);
    // Guide 1 enabled at the middle of the frame, moving nowhere: the neighbourhood is pinned
    // to zero motion, which contradicts the footage (8 px/frame) and so changes the output.
    let vals = [
        ("motion/quality", e(1)),
        ("guidance/guides/guide1/enable", on(true)),
        ("guidance/guides/radius", n(30.0)),
        ("guidance/guides/guide1/point", Value::Vec2([64.0, 48.0])),
    ];
    let mut pinned = Mock::new(Scene::Pan(8.0, 0.0), &vals);
    pinned.guide = Some(Box::new(|_| [64.0, 48.0]));
    let with = run(&pinned, t);
    let centre = |im: &Image| im.get(64, 48);
    assert_ne!(centre(&with), centre(&without));
    // Far from the guide (3σ = 45 px away) the output is unchanged.
    assert!(psnr(&with, &without, 0) > 18.0);
    let corner = (6usize, 6usize);
    let (a, b) = (with.get(corner.0 as i64, corner.1 as i64), without.get(corner.0 as i64, corner.1 as i64));
    assert!((0..3).all(|c| (a[c] - b[c]).abs() < 0.05));
    // A guide that follows the true motion (8 px per frame to the right) leaves the result good.
    let mut right = Mock::new(Scene::Pan(8.0, 0.0), &vals);
    right.guide = Some(Box::new(|t| [64.0 + 8.0 * t * COMP_FPS, 48.0]));
    let good = run(&right, t);
    assert!(psnr(&good, &Scene::Pan(8.0, 0.0).at(3.5), 12) >= 26.0, "{} dB", psnr(&good, &Scene::Pan(8.0, 0.0).at(3.5), 12));
    // The Guides display draws them.
    let mut shown = Mock::new(Scene::Pan(8.0, 0.0), &[("guidance/guides/guide1/enable", on(true)), ("diagnostics/display", e(8))]);
    shown.guide = Some(Box::new(|_| [64.0, 48.0]));
    let img = run(&shown, t);
    assert!(img.data.contains(&[1.0, 0.85, 0.1, 1.0]));
}

#[test]
fn external_vectors_drive_the_warp_and_motion_source_supplies_the_estimate() {
    let _g = serial();
    let t = t_at_half_speed(3.5);
    let truth = Scene::Pan(8.0, 0.0).at(3.5);
    // A vector layer holding the true motion (+8, 0) with scale 16: signed 0.75, 0.5.
    let mut host = Mock::new(
        Scene::Pan(8.0, 0.0),
        &[
            ("motion/vectors", e(2)),
            ("guidance/externalVectors/layer", Value::Layer(Some(VECTORS))),
            ("guidance/externalVectors/scale", n(16.0)),
            ("motion/quality", e(0)),
        ],
    );
    host.vectors = Some(Image::filled(W, H, [0.75, 0.5, 0.0, 1.0]));
    let good = run(&host, t);
    assert!(psnr(&good, &truth, 14) >= 33.0, "{} dB", psnr(&good, &truth, 14));
    // The same layer flipped / pointing the wrong way produces a clearly worse frame.
    host.vectors = Some(Image::filled(W, H, [0.25, 0.5, 0.0, 1.0]));
    let wrong = run(&host, t);
    assert!(psnr(&wrong, &truth, 14) < psnr(&good, &truth, 14) - 6.0);
    // A layer that cannot be read falls back to estimating.
    let mut missing = Mock::new(
        Scene::Pan(8.0, 0.0),
        &[("motion/vectors", e(2)), ("guidance/externalVectors/layer", Value::Layer(Some(VECTORS))), ("motion/quality", e(BEST))],
    );
    missing.vectors = None;
    assert!(psnr(&run(&missing, t), &truth, 14) >= 33.0);
    // Motion Source: the estimate comes from that layer (the same footage) and the effect's
    // own frames are warped by it.
    let src = Mock::new(Scene::Pan(8.0, 0.0), &[("motion/sourceLayer", Value::Layer(Some(MOTION))), ("motion/quality", e(BEST))]);
    assert!(psnr(&run(&src, t), &truth, 14) >= 33.0);
}

// ---------------------------------------------------------------- diagnostics, blur

#[test]
fn every_display_mode_renders_and_differs_from_the_output() {
    let _g = serial();
    let t = t_at_half_speed(3.5);
    let out = run(&Mock::new(Scene::Sep, &[("guidance/separation/matteLayer", Value::Layer(Some(MATTE)))]), t);
    for d in 1..9 {
        let img = run(&Mock::new(Scene::Sep, &[("guidance/separation/matteLayer", Value::Layer(Some(MATTE))), ("diagnostics/display", e(d))]), t);
        assert_eq!((img.width, img.height), (W, H), "display {d}");
        assert!(img.data.iter().all(|p| p.iter().all(|v| v.is_finite())));
        if d != 1 && d != 8 {
            assert_ne!(img, out, "display {d} shows the plain output");
        }
    }
    // Source = the nearest source frame, exactly.
    let src = run(&Mock::new(Scene::Pan(6.0, 0.0), &[("diagnostics/display", e(1))]), t);
    assert_eq!(src, Scene::Pan(6.0, 0.0).at(4.0));
}

#[test]
fn motion_blur_adds_blur_along_the_motion_when_speeding_up() {
    let _g = serial();
    let t = 3.0 / COMP_FPS;
    let sharp = run(&Mock::new(Scene::Pan(5.0, 0.0), &[("timeRemap/speed", n(250.0)), ("motion/quality", e(1))]), t);
    let blurred = run(
        &Mock::new(
            Scene::Pan(5.0, 0.0),
            &[("timeRemap/speed", n(250.0)), ("motion/quality", e(1)), ("motionBlur/enable", on(true)), ("motionBlur/shutterAngle", n(360.0))],
        ),
        t,
    );
    let detail = |im: &Image| (20..W - 21).map(|x| (im.get(x as i64 + 1, 40)[0] - im.get(x as i64, 40)[0]).abs()).sum::<f32>();
    assert!(detail(&blurred) < detail(&sharp) * 0.9, "{} vs {}", detail(&blurred), detail(&sharp));
    // A zero shutter is a no-op.
    let none = run(
        &Mock::new(
            Scene::Pan(5.0, 0.0),
            &[("timeRemap/speed", n(250.0)), ("motion/quality", e(1)), ("motionBlur/enable", on(true)), ("motionBlur/shutterAngle", n(0.0))],
        ),
        t,
    );
    assert_eq!(none, sharp);
}

// ---------------------------------------------------------------- caching, serialisation, robustness

#[test]
fn flows_are_cached_by_content() {
    let _g = serial();
    effectcraft_frameweaver::cache::clear();
    let host = Mock::new(Scene::Pan(5.0, 0.0), &[("motion/quality", e(0))]);
    let t0 = t_at_half_speed(3.25);
    let first = run(&host, t0);
    let s1 = effectcraft_frameweaver::cache::stats();
    assert_eq!(s1.misses, 1);
    // Another output frame between the same two source frames reuses the flow.
    let second = run(&host, t_at_half_speed(3.75));
    let s2 = effectcraft_frameweaver::cache::stats();
    assert_eq!((s2.hits, s2.misses), (1, 1));
    assert_ne!(first, second);
    // Different footage (same times) does not hit.
    let other = Mock::new(Scene::Pan(6.0, 0.0), &[("motion/quality", e(0))]);
    run(&other, t0);
    assert_eq!(effectcraft_frameweaver::cache::stats().misses, 2);
    // Neither does a different quality.
    run(&Mock::new(Scene::Pan(5.0, 0.0), &[("motion/quality", e(1))]), t0);
    assert_eq!(effectcraft_frameweaver::cache::stats().misses, 3);
    // Rendering twice is deterministic.
    assert_eq!(run(&host, t0), first);
}

#[test]
fn parameters_survive_a_project_round_trip() {
    let spec = find(ID).unwrap();
    let mut next = 0;
    let mut g = instantiate(spec, &mut Ids(&mut next), "FrameWeaver", [W as f64, H as f64]);
    g.prop_mut("timeRemap/speed").unwrap().value = Value::Scalar(-37.5);
    g.prop_mut("interp/warping").unwrap().value = Value::Enum(1);
    g.prop_mut("guidance/guides/guide3/enable").unwrap().value = Value::Bool(true);
    let json = serde_json::to_string(&g).unwrap();
    let back: effectcraft_project::PropGroup = serde_json::from_str(&json).unwrap();
    let eval = |g: &effectcraft_project::PropGroup| flatten_params(g, &mut |p| p.value.clone());
    let (a, b) = (eval(&g), eval(&back));
    assert_eq!(a.values.len(), b.values.len());
    for (k, v) in &a.values {
        assert_eq!(b.values.get(k), Some(v), "{k}");
    }
    assert_eq!(b.f("timeRemap/speed"), -37.5);
    assert!(b.b("guidance/guides/guide3/enable"));
    // Every control the spec declares exists on the instance, with the declared default.
    for ps in &spec.params {
        assert!(a.values.contains_key(ps.id), "{}", ps.id);
    }
    // Eight guides, in two rows of four, off by default.
    for i in 1..=8 {
        assert!(!find(ID).unwrap().params.iter().any(|p| p.id == format!("guidance/guides/guide{i}/enable") && p.default != Value::Bool(false)));
    }
    let p = |i: usize| match spec.params.iter().find(|p| p.id == format!("guidance/guides/guide{i}/point")).map(|p| &p.default) {
        Some(Value::Vec2(v)) => *v,
        other => panic!("guide {i}: {other:?}"),
    };
    assert_eq!(p(1), [0.2, 0.35]);
    assert_eq!(p(4), [0.8, 0.35]);
    assert_eq!(p(5), [0.2, 0.65]);
    assert_eq!(p(8), [0.8, 0.65]);
}

#[test]
fn without_a_host_the_layer_passes_through() {
    let spec = find(ID).unwrap();
    let size = [W as f64, H as f64];
    let params = Params { values: spec.params.iter().map(|p| (p.id.to_string(), default_value(p, size))).collect() };
    let ctx = EffectCtx { params: &params, time: 0.3, layer_size: size, seed: 1, adjustment: false, env: EffectEnv::default() };
    let input = Image::filled(W, H, [0.2, 0.3, 0.4, 1.0]);
    let out = apply(spec, &ctx, Buf { img: input.clone(), offset: [0.0; 2], scale: 1.0 });
    assert_eq!(out.img, input);
}

#[test]
fn hostile_parameters_and_frames_never_panic() {
    let _g = serial();
    let hostile: &[(&str, Value)] = &[
        ("timeRemap/speed", n(f64::NAN)),
        ("timeRemap/startFrame", n(f64::INFINITY)),
        ("timeRemap/inputFrameRate", n(-5.0)),
        ("motion/sensitivity", n(f64::NEG_INFINITY)),
        ("motionBlur/enable", on(true)),
        ("motionBlur/samples", n(1e12)),
        ("motionBlur/shutterAngle", n(f64::NAN)),
        ("guidance/guides/radius", n(f64::NAN)),
        ("guidance/guides/guide1/enable", on(true)),
        ("guidance/guides/guide1/point", Value::Vec2([f64::NAN, 1e300])),
        ("motion/quality", e(1)),
    ];
    let host = Mock::new(Scene::Pan(5.0, 0.0), hostile);
    for t in [0.0, 0.123, 1e7, -3.0] {
        let out = run(&host, t);
        assert!(out.data.iter().all(|p| p.iter().all(|v| v.is_finite())), "t={t}");
    }
    // Huge, absurd speeds and Frame Number values.
    for (speed, frame) in [(10000.0, 100000.0), (-10000.0, -100000.0)] {
        let host = Mock::new(Scene::Pan(5.0, 0.0), &[("timeRemap/speed", n(speed)), ("timeRemap/frameNum", n(frame)), ("timeRemap/mode", e(1))]);
        run(&host, 0.5);
    }
    // Degenerate frame sizes: a host that serves 1×1 and 0×0 frames.
    struct Tiny(u32, u32);
    impl EffectHost for Tiny {
        fn layer(&self, _: u64, _: bool) -> Option<LayerPixels> {
            None
        }
        fn audio(&self, _: u64, _: f64, _: usize, _: u32) -> Option<Vec<f32>> {
            None
        }
        fn self_at(&self, t: f64, _: usize) -> Option<Buf> {
            let mut img = Image::filled(self.0, self.1, [0.4, 0.4, 0.4, 1.0]);
            if let Some(p) = img.data.first_mut() {
                p[0] = (t * 30.0).fract() as f32;
            }
            Some(Buf { img, offset: [0.0; 2], scale: 1.0 })
        }
    }
    let spec = find(ID).unwrap();
    for (w, h) in [(1, 1), (1, 300), (300, 1), (2, 2)] {
        let size = [w as f64, h as f64];
        let params = Params { values: spec.params.iter().map(|p| (p.id.to_string(), default_value(p, size))).collect() };
        let host = Tiny(w, h);
        let env = EffectEnv { host: Some(&host), frame_rate: 30.0, comp_time: 0.05, ..Default::default() };
        let ctx = EffectCtx { params: &params, time: 0.05, layer_size: size, seed: 1, adjustment: false, env };
        let out = apply(spec, &ctx, Buf { img: Image::new(w, h), offset: [0.0; 2], scale: 1.0 });
        assert!(out.img.data.iter().all(|p| p.iter().all(|v| v.is_finite())), "{w}x{h}");
    }
}
