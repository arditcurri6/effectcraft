//! FrameWeaver (Effect > Time > FrameWeaver): optical-flow retiming.
//!
//! The effect asks the host for the layer at the two source frames around the remapped source
//! time ([`time_fx::time_frames_avail`]), estimates dense motion between them
//! ([`effectcraft_frameweaver`]) and builds the in-between frame along it. The work is split in
//! two so the GPU compositor can share it: [`frameweaver_plan`] does everything data-dependent
//! on the CPU (frame fetching, cuts, flow, guides, time fields) and returns a [`FwPlan`];
//! [`FwPlan::render_cpu`] is the CPU effect and effectcraft-gpu renders the same plan with its
//! kernels.
//!
//! Source time. In Speed mode the source time is the *integral* of the speed over layer time,
//! `s(t) = start/fps_in + ∫₀ᵗ speed(τ)/100 dτ`, so a keyframed speed ramps smoothly and never
//! jumps (the integral is taken numerically on the composition's frame grid through
//! [`EffectHost::params_at`]; with a constant speed it is exactly `t·speed/100`). In Frame
//! Number mode `s(t) = frameNum(t)/fps_in`, which gives exact control for freezes, reverses and
//! arbitrary curves. Source frame `k` is at layer time `k/fps_in`.
//!
//! Outside the footage (no frame from the host) the output is transparent; with only one of
//! the two frames available the available one is held.

use std::sync::Arc;

use effectcraft_frameweaver::{self as fw, FlowBackend};
use effectcraft_keyframe::Value;
use effectcraft_project::ParamUi;
use effectcraft_raster::Image;

use crate::time_fx;
use crate::{Buf, EffectCtx, EffectSpec, Params, num, p, popup, slider};

pub const ID: &str = "ec.time.frameweaver";

const INTERP: [&str; 3] = ["Nearest Frame", "Frame Blend", "Motion Weighted Blend"];
const QUALITY: [&str; 4] = ["Draft", "Balanced", "High", "Best"];
const VECTORS: [&str; 3] = ["Off", "Estimate", "External Vector Layer"];
const MATTE_CHANNEL: [&str; 4] = ["Alpha", "Luminance", "Inverted Alpha", "Inverted Luminance"];
const DISPLAY: [&str; 9] = ["Output", "Source", "Motion Vectors", "Occlusion", "Confidence", "FG Matte", "FG Only", "BG Only", "Guides"];

/// Guide points per instance.
pub const GUIDES: usize = 8;

/// The most speed samples one integration takes (the step grows beyond that).
const MAX_RAMP_STEPS: usize = 16384;

pub fn specs() -> Vec<EffectSpec> {
    let mut params = vec![
        p("timeRemap/inputFrameRate", "Input Frame Rate", num(0.0), slider(0.0, 240.0, 0.0, 120.0, 3)),
        p("timeRemap/mode", "Time Remap Mode", Value::Enum(0), popup(&["Speed", "Frame Number"])),
        p("timeRemap/speed", "Speed %", num(50.0), slider(-10000.0, 10000.0, -400.0, 400.0, 1)),
        p("timeRemap/startFrame", "Start Frame", num(0.0), slider(-100000.0, 100000.0, 0.0, 300.0, 2)),
        p("timeRemap/frameNum", "Frame Number", num(0.0), slider(-100000.0, 100000.0, 0.0, 300.0, 2)),
        p("motion/vectors", "Motion Vectors", Value::Enum(1), popup(&VECTORS)),
        p("motion/quality", "Quality", Value::Enum(1), popup(&QUALITY)),
        p("motion/sensitivity", "Motion Sensitivity", num(70.0), slider(0.0, 100.0, 0.0, 100.0, 0)),
        p("motion/imagePrep", "Image Prep", Value::Enum(0), popup(&["None", "Contrast/Edge Enhance"])),
        p("motion/sourceLayer", "Motion Source Layer", Value::Layer(None), ParamUi::Layer),
        p("interp/mode", "Frame Interpolation", Value::Enum(2), popup(&INTERP)),
        p("interp/warping", "Warping", Value::Enum(0), popup(&["Inverse", "Forward"])),
        p("interp/smartBlend", "Smart Blend", Value::Bool(true), ParamUi::Checkbox),
        p("motionBlur/enable", "Enable Motion Blur", Value::Bool(false), ParamUi::Checkbox),
        p("motionBlur/shutterAngle", "Shutter Angle", num(180.0), slider(0.0, 720.0, 0.0, 360.0, 1)),
        p("motionBlur/samples", "Samples", num(8.0), slider(1.0, 32.0, 1.0, 32.0, 0)),
        p("segments/autoDetect", "Auto-Detect Cuts", Value::Bool(true), ParamUi::Checkbox),
        p("segments/sensitivity", "Cut Sensitivity", num(50.0), slider(0.0, 100.0, 0.0, 100.0, 0)),
        p("segments/marker", "Segment Marker", num(0.0), slider(-1000.0, 1000.0, -100.0, 100.0, 0)),
        p("guidance/separation/matteLayer", "FG Matte Layer", Value::Layer(None), ParamUi::Layer),
        p("guidance/separation/matteChannel", "Matte Channel", Value::Enum(0), popup(&MATTE_CHANNEL)),
        p("guidance/separation/bgFill", "BG Fill", Value::Enum(0), popup(&["Neighbour Frames", "Inpaint"])),
        p("guidance/externalVectors/layer", "Vector Layer", Value::Layer(None), ParamUi::Layer),
        p("guidance/externalVectors/encoding", "Encoding", Value::Enum(0), popup(&["Signed RG (0.5 = still)", "Float RG"])),
        p("guidance/externalVectors/scale", "Vector Scale", num(16.0), slider(0.0, 10000.0, 0.0, 512.0, 2)),
        p("guidance/externalVectors/direction", "Direction", Value::Enum(0), popup(&["To Next Frame", "To Previous Frame"])),
        p("guidance/externalVectors/flipY", "Flip Y", Value::Bool(false), ParamUi::Checkbox),
        p("guidance/guides/radius", "Guide Radius", num(60.0), slider(1.0, 2000.0, 1.0, 400.0, 0)),
    ];
    // Eight guide points in two rows of four, off by default.
    const IDS: [(&str, &str, &str); GUIDES] = [
        ("guidance/guides/guide1/enable", "guidance/guides/guide1/point", "Guide 1"),
        ("guidance/guides/guide2/enable", "guidance/guides/guide2/point", "Guide 2"),
        ("guidance/guides/guide3/enable", "guidance/guides/guide3/point", "Guide 3"),
        ("guidance/guides/guide4/enable", "guidance/guides/guide4/point", "Guide 4"),
        ("guidance/guides/guide5/enable", "guidance/guides/guide5/point", "Guide 5"),
        ("guidance/guides/guide6/enable", "guidance/guides/guide6/point", "Guide 6"),
        ("guidance/guides/guide7/enable", "guidance/guides/guide7/point", "Guide 7"),
        ("guidance/guides/guide8/enable", "guidance/guides/guide8/point", "Guide 8"),
    ];
    for (i, (en, pt, _)) in IDS.iter().enumerate() {
        let (x, y) = (0.2 + 0.2 * (i % 4) as f64, if i < 4 { 0.35 } else { 0.65 });
        params.push(p(en, "Enable", Value::Bool(false), ParamUi::Checkbox));
        params.push(p(pt, "Point", Value::Vec2([x, y]), ParamUi::Point));
    }
    params.push(p("diagnostics/display", "Display", Value::Enum(0), popup(&DISPLAY)));
    vec![EffectSpec { id: ID, name: "FrameWeaver", category: "Time", params, render: frameweaver, gpu: false, float: true }]
}

/// Display names of the groups FrameWeaver adds (for the Effect Controls tree).
pub const GROUPS: &[(&str, &str)] = &[
    ("timeRemap", "Time Remapping"),
    ("motion", "Motion Estimation"),
    ("interp", "Interpolation"),
    ("segments", "Segments"),
    ("guidance", "Advanced Guidance"),
    ("separation", "Foreground / Background Separation"),
    ("externalVectors", "External Vectors"),
    ("guides", "Guide Points"),
    ("guide1", "Guide 1"),
    ("guide2", "Guide 2"),
    ("guide3", "Guide 3"),
    ("guide4", "Guide 4"),
    ("guide5", "Guide 5"),
    ("guide6", "Guide 6"),
    ("guide7", "Guide 7"),
    ("guide8", "Guide 8"),
    ("diagnostics", "Diagnostics"),
];

// ---------------------------------------------------------------- settings

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Nearest,
    Blend,
    Motion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Display {
    Output,
    Source,
    Vectors,
    Occlusion,
    Confidence,
    FgMatte,
    FgOnly,
    BgOnly,
    Guides,
}

/// What the Layered step shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerShow {
    Output,
    FgOnly,
    BgOnly,
    Matte,
}

/// Every control, validated (non-finite values take the default, ranges are clamped).
#[derive(Clone, Debug)]
pub struct Settings {
    pub input_fps: f64,
    pub by_frame: bool,
    pub speed: f64,
    pub start_frame: f64,
    pub frame_num: f64,
    pub vectors: u32,
    pub quality: fw::Quality,
    pub sensitivity: f32,
    pub prep: bool,
    pub interp: Interp,
    pub warping: fw::Warping,
    pub smart_blend: bool,
    pub blur: bool,
    pub shutter: f64,
    pub samples: u32,
    pub auto_cuts: bool,
    pub cut_sensitivity: f32,
    pub display: Display,
}

fn clamped(p: &Params, id: &str, default: f64, lo: f64, hi: f64) -> f64 {
    match p.get(id).map(Value::as_f64) {
        Some(v) if v.is_finite() => v.clamp(lo, hi),
        _ => default,
    }
}

impl Settings {
    pub fn read(p: &Params) -> Settings {
        Settings {
            input_fps: clamped(p, "timeRemap/inputFrameRate", 0.0, 0.0, 240.0),
            by_frame: p.e("timeRemap/mode") == 1,
            speed: clamped(p, "timeRemap/speed", 50.0, -10000.0, 10000.0),
            start_frame: clamped(p, "timeRemap/startFrame", 0.0, -100000.0, 100000.0),
            frame_num: clamped(p, "timeRemap/frameNum", 0.0, -100000.0, 100000.0),
            vectors: p.e("motion/vectors").min(2),
            quality: fw::Quality::from_index(p.e("motion/quality")),
            sensitivity: (clamped(p, "motion/sensitivity", 70.0, 0.0, 100.0) / 100.0) as f32,
            prep: p.e("motion/imagePrep") == 1,
            interp: match p.e("interp/mode") {
                0 => Interp::Nearest,
                1 => Interp::Blend,
                _ => Interp::Motion,
            },
            warping: if p.e("interp/warping") == 1 { fw::Warping::Forward } else { fw::Warping::Inverse },
            smart_blend: p.b("interp/smartBlend"),
            blur: p.b("motionBlur/enable"),
            shutter: clamped(p, "motionBlur/shutterAngle", 180.0, 0.0, 720.0),
            samples: clamped(p, "motionBlur/samples", 8.0, 1.0, 32.0).round() as u32,
            auto_cuts: p.b("segments/autoDetect"),
            cut_sensitivity: (clamped(p, "segments/sensitivity", 50.0, 0.0, 100.0) / 100.0) as f32,
            display: match p.e("diagnostics/display") {
                1 => Display::Source,
                2 => Display::Vectors,
                3 => Display::Occlusion,
                4 => Display::Confidence,
                5 => Display::FgMatte,
                6 => Display::FgOnly,
                7 => Display::BgOnly,
                8 => Display::Guides,
                _ => Display::Output,
            },
        }
    }

    /// The input frame rate, falling back to the composition's.
    pub fn fps_in(&self, comp_fps: f64) -> f64 {
        if self.input_fps > 0.0 { self.input_fps } else { comp_fps }
    }
}

// ---------------------------------------------------------------- time maths

/// `∫₀ᵗ speed(τ)/100 dτ` by the trapezoid rule on a grid of at most `step` seconds (at most
/// [`MAX_RAMP_STEPS`] steps). `speed_at` returns percent; non-finite speeds count as 0.
pub fn integrate_speed(t: f64, step: f64, speed_at: &dyn Fn(f64) -> f64) -> f64 {
    if !t.is_finite() || t == 0.0 {
        return 0.0;
    }
    let f = |tau: f64| {
        let v = speed_at(tau);
        if v.is_finite() { v } else { 0.0 }
    };
    let n = ((t.abs() / step.max(1e-6)).ceil() as usize).clamp(1, MAX_RAMP_STEPS);
    let h = t / n as f64;
    let mut acc = 0.0;
    let mut prev = f(0.0);
    for i in 1..=n {
        let cur = f(h * i as f64);
        acc += 0.5 * (prev + cur) * h;
        prev = cur;
    }
    acc / 100.0
}

/// The source layer time (seconds) shown at layer time `t`.
pub fn source_time(ctx: &EffectCtx, st: &Settings, fps_in: f64) -> f64 {
    if st.by_frame {
        return st.frame_num / fps_in;
    }
    let start = st.start_frame / fps_in;
    let integral = match ctx.env.host {
        Some(host) => integrate_speed(ctx.time, 1.0 / ctx.fps(), &|tau| {
            host.params_at(tau).map_or(st.speed, |p| clamped(&p, "timeRemap/speed", st.speed, -10000.0, 10000.0))
        }),
        None => ctx.time * st.speed / 100.0,
    };
    start + integral
}

/// Source frames advanced per output frame (the extent of a full-shutter blur).
fn frames_per_output(ctx: &EffectCtx, st: &Settings, fps_in: f64) -> f64 {
    let comp = ctx.fps();
    if st.by_frame {
        let next = ctx
            .env
            .host
            .and_then(|h| h.params_at(ctx.time + 1.0 / comp))
            .map_or(st.frame_num, |p| clamped(&p, "timeRemap/frameNum", st.frame_num, -100000.0, 100000.0));
        (next - st.frame_num).abs()
    } else {
        st.speed.abs() / 100.0 * fps_in / comp
    }
}

/// The source frame at or before `x` (in frames) and the fraction to the next.
pub fn split_frame(x: f64) -> (i64, f32) {
    let a = (x + 1e-6).floor();
    (a as i64, ((x - a) as f32).max(0.0))
}

// ---------------------------------------------------------------- plan

/// One output frame, as steps over [`FwPlan::frames`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FwStep {
    /// A source frame as it is (Nearest, an exact frame, a cut, a missing neighbour).
    Hold(usize),
    /// A cross-fade of two frames.
    Mix { a: usize, b: usize, w: f32 },
    /// The in-between frame along time field `field`.
    Motion { a: usize, b: usize, w: f32, field: usize },
    /// Foreground and background each built on their own motion; `mt` are the matte images
    /// (grey, opaque) warped along the foreground's field for Display ▸ FG Matte.
    Layered { fg: [usize; 3], bg: [usize; 3], mt: [usize; 2], w: f32, show: LayerShow },
    /// A finished image (diagnostic displays).
    Image(usize),
    /// No footage at the source time at all: transparent, at the input's size.
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionBlur {
    /// The shutter's extent in source frames.
    pub span: f32,
    pub samples: u32,
}

/// FrameWeaver prepared for the GPU compositor: everything data-dependent is done, what is left
/// is sampling.
pub struct FwPlan {
    pub grid: Buf,
    pub frames: Vec<Image>,
    pub fields: Vec<fw::TimeField>,
    pub step: FwStep,
    pub smart_blend: bool,
    pub mblur: Option<MotionBlur>,
}

impl FwPlan {
    fn frame(&self, i: usize) -> Image {
        self.frames.get(i).cloned().unwrap_or_default()
    }

    fn synth(&self, a: usize, b: usize, w: f32, field: usize, blur: bool) -> Image {
        let (Some(fa), Some(fb), Some(tf)) = (self.frames.get(a), self.frames.get(b), self.fields.get(field)) else {
            return self.frame(a);
        };
        let img = fw::synth::synthesize(fa, fb, tf, w, self.smart_blend);
        match self.mblur.filter(|_| blur) {
            Some(m) => fw::synth::motion_blur(&img, tf, m.span, m.samples),
            None => img,
        }
    }

    /// The effect on the CPU.
    /// Is there no footage to show (the output is transparent)?
    pub fn is_empty(&self) -> bool {
        self.step == FwStep::Empty
    }

    pub fn render_cpu(&self) -> Buf {
        let img = match self.step {
            FwStep::Empty => Image::default(),
            FwStep::Hold(i) | FwStep::Image(i) => self.frame(i),
            FwStep::Mix { a, b, w } => match (self.frames.get(a), self.frames.get(b)) {
                (Some(x), Some(y)) => fw::synth::blend(x, y, w),
                _ => self.frame(a),
            },
            FwStep::Motion { a, b, w, field } => self.synth(a, b, w, field, true),
            FwStep::Layered { fg, bg, mt, w, show } => match show {
                LayerShow::FgOnly => self.synth(fg[0], fg[1], w, fg[2], true),
                LayerShow::BgOnly => self.synth(bg[0], bg[1], w, bg[2], true),
                LayerShow::Matte => self.synth(mt[0], mt[1], w, fg[2], false),
                LayerShow::Output => {
                    let f = self.synth(fg[0], fg[1], w, fg[2], true);
                    let g = self.synth(bg[0], bg[1], w, bg[2], true);
                    fw::synth::over(&f, &g)
                }
            },
        };
        Buf { img, ..self.grid.clone() }
    }
}

/// Source layer times FrameWeaver needs for a pair, plus the neighbours Neighbour-Frames
/// background fill reads.
struct Times {
    ta: f64,
    tb: f64,
    /// a − 1, b + 1, a − 2, b + 2.
    extra: [f64; 4],
}

fn is_cut(ctx: &EffectCtx, st: &Settings, fa: &Image, fb: &Image, ta: f64, tb: f64) -> bool {
    if st.auto_cuts {
        let threshold = 0.6 - 0.5 * st.cut_sensitivity;
        let d = effectcraft_raster::cuts::dissimilarity(&effectcraft_raster::cuts::signature(fa), &effectcraft_raster::cuts::signature(fb));
        if d > threshold {
            return true;
        }
    }
    // Segment Marker: any change of its value between the two frames starts a new segment.
    if let Some(host) = ctx.env.host
        && let (Some(pa), Some(pb)) = (host.params_at(ta), host.params_at(tb))
    {
        return pa.f("segments/marker").round() != pb.f("segments/marker").round();
    }
    false
}

fn empty_plan(st: &Settings) -> FwPlan {
    FwPlan { grid: Buf::default(), frames: vec![], fields: vec![], step: FwStep::Empty, smart_blend: st.smart_blend, mblur: None }
}

fn hold_plan(grid: Buf, frames: Vec<Image>, i: usize, st: &Settings) -> FwPlan {
    FwPlan { grid, frames, fields: vec![], step: FwStep::Hold(i), smart_blend: st.smart_blend, mblur: None }
}

/// Matte channel of the popup → `time_fx::matte_of`'s channel numbering.
fn matte_channel(p: &Params) -> u32 {
    match p.e("guidance/separation/matteChannel") {
        1 => 2,
        2 => 1,
        3 => 3,
        _ => 0,
    }
}

/// The guide points enabled on the effect, at source times `ta` and `tb`.
fn guides_at(ctx: &EffectCtx, ta: f64, tb: f64) -> Vec<([f64; 2], [f64; 2])> {
    let host = ctx.env.host;
    let (pa, pb) = (host.and_then(|h| h.params_at(ta)), host.and_then(|h| h.params_at(tb)));
    let (pa, pb) = (pa.as_ref().unwrap_or(ctx.params), pb.as_ref().unwrap_or(ctx.params));
    (1..=GUIDES)
        .filter(|i| ctx.params.b(&format!("guidance/guides/guide{i}/enable")))
        .map(|i| {
            let id = format!("guidance/guides/guide{i}/point");
            (pa.v2(&id), pb.v2(&id))
        })
        .collect()
}

/// The background plate of a frame whose foreground (`matte` > 0.01) is removed: the hole is
/// filled from the neighbouring frames' pixels that are background there, then by membrane
/// interpolation of what remains (Inpaint: membrane interpolation only).
fn bg_plate(frame: &Image, matte: &[f32], neighbours: &[(Image, Vec<f32>)]) -> Image {
    let n = frame.data.len();
    if matte.len() != n {
        return frame.clone();
    }
    let mut out = frame.clone();
    let mut missing = vec![false; n];
    for i in 0..n {
        if matte[i] <= 0.01 {
            continue;
        }
        let from = neighbours.iter().find(|(img, m)| m.get(i).is_some_and(|&v| v < 0.01) && img.data.len() == n);
        match from {
            Some((img, _)) => out.data[i] = img.data[i],
            None => missing[i] = true,
        }
    }
    if missing.iter().any(|&m| m) { effectcraft_raster::inpaint::membrane_fill(&out, &missing) } else { out }
}

/// FrameWeaver's [`FwPlan`] for the instance at `ctx.time`. `None` without the effect host
/// (thumbnails, tests): the layer passes through.
pub fn frameweaver_plan(ctx: &EffectCtx) -> Option<FwPlan> {
    ctx.env.host?;
    let st = Settings::read(ctx.params);
    let fps_in = st.fps_in(ctx.fps());
    let s = source_time(ctx, &st, fps_in);
    let (a, w) = split_frame(s * fps_in);
    let times = Times {
        ta: a as f64 / fps_in,
        tb: (a + 1) as f64 / fps_in,
        extra: [(a - 1) as f64 / fps_in, (a + 2) as f64 / fps_in, (a - 2) as f64 / fps_in, (a + 3) as f64 / fps_in],
    };
    // A whole source frame: exactly on one, Nearest, or the Source display.
    if w < 1e-4 || st.interp == Interp::Nearest || st.display == Display::Source {
        let pick = if w >= 0.5 { times.tb } else { times.ta };
        let Some((grid, frames, avail)) = time_fx::time_frames_avail(ctx, &[pick]) else { return Some(empty_plan(&st)) };
        if !avail.first().copied().unwrap_or(false) {
            return Some(empty_plan(&st));
        }
        return Some(hold_plan(grid, frames, 0, &st));
    }
    let matte_id = "guidance/separation/matteLayer";
    let separating = ctx.params.get(matte_id).and_then(Value::as_layer).is_some();
    let neighbour_fill = separating && ctx.params.e("guidance/separation/bgFill") == 0 && st.interp == Interp::Motion && st.vectors > 0;
    let mut want = vec![times.ta, times.tb];
    if neighbour_fill {
        want.extend(times.extra);
    }
    let Some((grid, mut frames, avail)) = time_fx::time_frames_avail(ctx, &want) else { return Some(empty_plan(&st)) };
    match (avail.first().copied().unwrap_or(false), avail.get(1).copied().unwrap_or(false)) {
        (true, true) => {}
        (true, false) => return Some(hold_plan(grid, frames, 0, &st)),
        (false, true) => return Some(hold_plan(grid, frames, 1, &st)),
        (false, false) => return Some(empty_plan(&st)),
    }
    let nearest = if w >= 0.5 { 1 } else { 0 };
    if st.interp == Interp::Blend || st.vectors == 0 {
        return Some(FwPlan { grid, frames, fields: vec![], step: FwStep::Mix { a: 0, b: 1, w }, smart_blend: st.smart_blend, mblur: None });
    }
    if is_cut(ctx, &st, &frames[0], &frames[1], times.ta, times.tb) {
        return Some(hold_plan(grid, frames, nearest, &st));
    }
    // ---- motion
    let params = fw::FlowParams::for_quality(st.quality, st.sensitivity);
    let motion_src = time_fx::fetch_layer(ctx, "motion/sourceLayer", &[times.ta, times.tb], &grid).filter(|v| v.len() == 2);
    let matte_ch = matte_channel(ctx.params);
    let mattes = if separating { time_fx::fetch_layer(ctx, matte_id, &want, &grid) } else { None };
    let sep = mattes.as_ref().filter(|m| m.len() >= 2).map(|m| m.iter().map(|img| time_fx::matte_of(img, matte_ch)).collect::<Vec<_>>());
    let analysis = |img: &Image, m: Option<&[f32]>| fw::prep::analysis_plane(img, params.max_long, m, st.prep);
    let vec_layer = if st.vectors == 2 {
        let dir_prev = ctx.params.e("guidance/externalVectors/direction") == 1;
        time_fx::fetch_layer(ctx, "guidance/externalVectors/layer", &[if dir_prev { times.tb } else { times.ta }], &grid)
            .and_then(|mut v| v.pop())
            .filter(|img| img.data.iter().any(|p| p[3] > 1e-6))
    } else {
        None
    };
    let vopts = fw::vectors::VectorOpts {
        encoding: if ctx.params.e("guidance/externalVectors/encoding") == 1 { fw::vectors::Encoding::FloatRg } else { fw::vectors::Encoding::SignedRg },
        scale: clamped(ctx.params, "guidance/externalVectors/scale", 16.0, 0.0, 10000.0) as f32,
        direction: if ctx.params.e("guidance/externalVectors/direction") == 1 { fw::vectors::Direction::ToPrevious } else { fw::vectors::Direction::ToNext },
        flip_y: ctx.params.b("guidance/externalVectors/flipY"),
    };
    // One flow pair: from the vector layer, or estimated on the analysis planes (cached).
    let pair_of = |ia: &Image, ib: &Image, ma: Option<&[f32]>, mb: Option<&[f32]>, use_vectors: bool| -> Option<(Arc<fw::FlowPair>, f32)> {
        let (pa, pb) = (analysis(ia, ma)?, analysis(ib, mb)?);
        if use_vectors && let Some(layer) = &vec_layer {
            let mut h = fw::cache::Hasher::new();
            h.word(0xEE_C7_0B);
            h.word(layer.width);
            h.word(layer.height);
            for px in &layer.data {
                px.iter().for_each(|&v| h.f(v));
            }
            h.params(&params);
            h.f(vopts.scale);
            h.word(vopts.encoding as u32 + 2 * vopts.direction as u32 + 4 * vopts.flip_y as u32);
            h.word(pa.plane.w as u32);
            h.word(pa.plane.h as u32);
            if let Some(pair) = fw::cache::get_or_compute(h.finish(), || fw::vectors::pair_from_layer(layer, &vopts, pa.plane.w, pa.plane.h, &params)) {
                return Some((pair, pa.scale));
            }
        }
        let key = fw::cache::key(&pa.plane, &pb.plane, &params, &[]);
        let pair = fw::cache::get_or_compute(key, || fw::ClassicDis.estimate(&pa.plane, &pb.plane, &params).ok())?;
        Some((pair, pa.scale))
    };
    let (own_a, own_b) = (frames[0].clone(), frames[1].clone());
    let (mot_a, mot_b) = match &motion_src {
        Some(m) => (m[0].clone(), m[1].clone()),
        None => (own_a.clone(), own_b.clone()),
    };
    let guide_pts = guides_at(ctx, times.ta, times.tb);
    let radius_px = clamped(ctx.params, "guidance/guides/radius", 60.0, 1.0, 2000.0) as f32;
    let steer = |pair: &Arc<fw::FlowPair>, scale: f32, keep: &dyn Fn([f32; 2]) -> bool| -> Arc<fw::FlowPair> {
        let guides: Vec<fw::guides::Guide> = guide_pts
            .iter()
            .filter_map(|(pa, pb)| {
                let (a, b) = (grid.to_px(*pa), grid.to_px(*pb));
                let (fa, fb) = ([a.0 as f32, a.1 as f32], [b.0 as f32, b.1 as f32]);
                keep(fa).then_some(fw::guides::Guide { from: [fa[0] / scale, fa[1] / scale], to: [fb[0] / scale, fb[1] / scale] })
            })
            .collect();
        if guides.is_empty() {
            return pair.clone();
        }
        let mut owned = (**pair).clone();
        fw::guides::apply(&mut owned, &guides, radius_px / scale);
        Arc::new(owned)
    };
    let frame_dims = (grid.img.width, grid.img.height);
    let build_field = |pair: &fw::FlowPair, ia: &Image, ib: &Image, ma: Option<&[f32]>, mb: Option<&[f32]>| -> Option<fw::TimeField> {
        match st.warping {
            fw::Warping::Inverse => fw::field::inverse(pair, w, frame_dims),
            fw::Warping::Forward => {
                let (pa, pb) = (analysis(ia, ma)?, analysis(ib, mb)?);
                fw::field::forward(pair, &pa.plane, &pb.plane, w, frame_dims)
            }
        }
    };
    let blur = st
        .blur
        .then(|| MotionBlur { span: (st.shutter / 360.0 * frames_per_output(ctx, &st, fps_in)) as f32, samples: st.samples })
        .filter(|m| m.span > 1e-6 && m.samples > 1);
    let smart_blend = st.smart_blend;
    let finish = |grid: Buf, frames: Vec<Image>, fields: Vec<fw::TimeField>, step: FwStep, mblur: Option<MotionBlur>| FwPlan {
        grid,
        frames,
        fields,
        step,
        smart_blend,
        mblur,
    };
    let diag_image = |plan: FwPlan, img: Image| -> FwPlan {
        let mut frames = plan.frames;
        frames.push(img);
        FwPlan { frames: frames.clone(), step: FwStep::Image(frames.len() - 1), fields: vec![], mblur: None, ..plan }
    };

    let Some(m) = sep else {
        // ---- one layer of motion
        let Some((pair, scale)) = pair_of(&mot_a, &mot_b, None, None, true) else {
            return Some(finish(grid, frames, vec![], FwStep::Mix { a: 0, b: 1, w }, None));
        };
        let pair = steer(&pair, scale, &|_| true);
        let Some(field) = build_field(&pair, &mot_a, &mot_b, None, None) else {
            return Some(finish(grid, frames, vec![], FwStep::Mix { a: 0, b: 1, w }, None));
        };
        let plan = finish(grid, frames, vec![field], FwStep::Motion { a: 0, b: 1, w, field: 0 }, blur);
        return Some(match st.display {
            Display::Vectors => {
                let img = fw::display::flow_wheel(&pair.fab, frame_dims);
                diag_image(plan, img)
            }
            Display::Occlusion => {
                let img = fw::display::field_view(&plan.fields[0], frame_dims, fw::display::FieldView::Occlusion);
                diag_image(plan, img)
            }
            Display::Confidence => {
                let img = fw::display::field_view(&plan.fields[0], frame_dims, fw::display::FieldView::Confidence);
                diag_image(plan, img)
            }
            Display::Guides => draw_guides(plan, &guide_pts, w),
            _ => plan,
        });
    };

    // ---- foreground / background separation
    let (ma, mb) = (&m[0], &m[1]);
    let neighbours: Vec<(Image, Vec<f32>)> = if neighbour_fill {
        (0..4).filter(|k| avail.get(2 + k).copied().unwrap_or(false)).filter_map(|k| Some((frames.get(2 + k)?.clone(), m.get(2 + k)?.clone()))).collect()
    } else {
        vec![]
    };
    let (bg_a, bg_b) = (bg_plate(&own_a, ma, &neighbours), bg_plate(&own_b, mb, &neighbours));
    let (fg_a, fg_b) = (time_fx::times_mask(&own_a, ma, false), time_fx::times_mask(&own_b, mb, false));
    let (mt_a, mt_b) = (time_fx::matte_image(ma, &own_a), time_fx::matte_image(mb, &own_b));
    // Motion is measured on the motion source (when set) with the same matte.
    let (fmot_a, fmot_b) = (time_fx::times_mask(&mot_a, ma, false), time_fx::times_mask(&mot_b, mb, false));
    let (bmot_a, bmot_b) =
        if motion_src.is_some() { (bg_plate(&mot_a, ma, &neighbours), bg_plate(&mot_b, mb, &neighbours)) } else { (bg_a.clone(), bg_b.clone()) };
    let matte_at = |p: [f32; 2]| -> bool {
        let (x, y) = (p[0].floor() as i64, p[1].floor() as i64);
        x >= 0
            && y >= 0
            && x < frame_dims.0 as i64
            && y < frame_dims.1 as i64
            && ma.get(y as usize * frame_dims.0 as usize + x as usize).is_some_and(|&v| v >= 0.5)
    };
    let (Some((fg_pair, fscale)), Some((bg_pair, bscale))) =
        (pair_of(&fmot_a, &fmot_b, Some(ma), Some(mb), false), pair_of(&bmot_a, &bmot_b, None, None, true))
    else {
        return Some(finish(grid, frames, vec![], FwStep::Mix { a: 0, b: 1, w }, None));
    };
    let fg_pair = steer(&fg_pair, fscale, &|p| matte_at(p));
    let bg_pair = steer(&bg_pair, bscale, &|p| !matte_at(p));
    let (Some(ff), Some(bf)) = (build_field(&fg_pair, &fmot_a, &fmot_b, Some(ma), Some(mb)), build_field(&bg_pair, &bmot_a, &bmot_b, None, None)) else {
        return Some(finish(grid, frames, vec![], FwStep::Mix { a: 0, b: 1, w }, None));
    };
    let base = frames.len();
    frames.extend([fg_a, fg_b, bg_a, bg_b, mt_a, mt_b]);
    let show = match st.display {
        Display::FgOnly => LayerShow::FgOnly,
        Display::BgOnly => LayerShow::BgOnly,
        Display::FgMatte => LayerShow::Matte,
        _ => LayerShow::Output,
    };
    let step = FwStep::Layered { fg: [base, base + 1, 0], bg: [base + 2, base + 3, 1], mt: [base + 4, base + 5], w, show };
    let plan = finish(grid, frames, vec![ff, bf], step, blur);
    // Diagnostics show the foreground's view over the background's.
    let layered_diag = |fg: Image, bg: Image| -> Image {
        let masked = time_fx::times_mask(&fg, ma, false);
        fw::synth::over(&masked, &bg)
    };
    Some(match st.display {
        Display::Vectors => {
            let img = layered_diag(fw::display::flow_wheel(&fg_pair.fab, frame_dims), fw::display::flow_wheel(&bg_pair.fab, frame_dims));
            diag_image(plan, img)
        }
        Display::Occlusion | Display::Confidence => {
            let view = if st.display == Display::Occlusion { fw::display::FieldView::Occlusion } else { fw::display::FieldView::Confidence };
            let img = layered_diag(fw::display::field_view(&plan.fields[0], frame_dims, view), fw::display::field_view(&plan.fields[1], frame_dims, view));
            diag_image(plan, img)
        }
        Display::Guides => draw_guides(plan, &guide_pts, w),
        _ => plan,
    })
}

/// The Guides display: the output with each enabled guide drawn as a dot at its position at the
/// output time and a line to where it goes in B.
fn draw_guides(plan: FwPlan, guides: &[([f64; 2], [f64; 2])], w: f32) -> FwPlan {
    let mut img = plan.render_cpu().img;
    for (a, b) in guides {
        let (pa, pb) = (plan.grid.to_px(*a), plan.grid.to_px(*b));
        let (fa, fb) = ([pa.0 as f32, pa.1 as f32], [pb.0 as f32, pb.1 as f32]);
        let here = [fa[0] + (fb[0] - fa[0]) * w, fa[1] + (fb[1] - fa[1]) * w];
        fw::display::draw_guide(&mut img, here, fb, [1.0, 0.85, 0.1]);
    }
    let mut frames = plan.frames;
    frames.push(img);
    FwPlan { step: FwStep::Image(frames.len() - 1), frames, fields: vec![], mblur: None, grid: plan.grid, smart_blend: plan.smart_blend }
}

/// FrameWeaver. Without the effect host the layer passes through; with no footage at the
/// source time the output is transparent.
fn frameweaver(ctx: &EffectCtx, b: Buf) -> Buf {
    match frameweaver_plan(ctx) {
        Some(plan) if plan.is_empty() => Buf { img: Image::new(b.img.width, b.img.height), ..b },
        Some(plan) => plan.render_cpu(),
        None => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_speed_integrates_exactly() {
        let s = integrate_speed(2.5, 1.0 / 30.0, &|_| 50.0);
        assert!((s - 1.25).abs() < 1e-12, "{s}");
        assert!((integrate_speed(-2.0, 1.0 / 30.0, &|_| 100.0) + 2.0).abs() < 1e-12);
        assert_eq!(integrate_speed(0.0, 1.0 / 30.0, &|_| 100.0), 0.0);
        assert_eq!(integrate_speed(f64::NAN, 1.0 / 30.0, &|_| 100.0), 0.0);
    }

    #[test]
    fn linear_ramp_is_the_area_under_the_curve_and_continuous() {
        // Speed 0 → 200 % over 2 s: the source time is t²/2 · 100/100 = t² · 0.5 · (200/2)/100.
        let ramp = |t: f64| (t * 100.0).min(200.0);
        let at = |t: f64| integrate_speed(t, 1.0 / 30.0, &ramp);
        assert!((at(2.0) - 2.0).abs() < 1e-9, "{}", at(2.0));
        assert!((at(1.0) - 0.5).abs() < 1e-9);
        // After the ramp it continues at 2×; the source time never jumps.
        assert!((at(3.0) - 4.0).abs() < 1e-9);
        let (before, after) = (at(2.0 - 1e-6), at(2.0 + 1e-6));
        assert!((after - before).abs() < 1e-5);
    }

    #[test]
    fn nonfinite_speed_is_zero_and_steps_are_capped() {
        assert_eq!(integrate_speed(1.0, 1.0 / 30.0, &|_| f64::NAN), 0.0);
        let calls = std::cell::Cell::new(0usize);
        let _ = integrate_speed(1.0e9, 1.0 / 30.0, &|_| {
            calls.set(calls.get() + 1);
            100.0
        });
        assert!(calls.get() <= MAX_RAMP_STEPS + 1);
    }

    #[test]
    fn frame_split_is_stable_at_frame_boundaries() {
        assert_eq!(split_frame(3.0), (3, 0.0));
        assert_eq!(split_frame(2.9999999), (3, 0.0));
        let (a, w) = split_frame(-0.25);
        assert_eq!(a, -1);
        assert!((w - 0.75).abs() < 1e-6);
        let (a, w) = split_frame(7.5);
        assert_eq!((a, w), (7, 0.5));
    }

    #[test]
    fn settings_clamp_and_default() {
        let mut p = Params::default();
        p.values.insert("timeRemap/speed".into(), Value::Scalar(f64::NAN));
        p.values.insert("motion/sensitivity".into(), Value::Scalar(1e9));
        p.values.insert("motionBlur/samples".into(), Value::Scalar(-5.0));
        p.values.insert("motion/quality".into(), Value::Enum(77));
        let s = Settings::read(&p);
        assert_eq!(s.speed, 50.0);
        assert_eq!(s.sensitivity, 1.0);
        assert_eq!(s.samples, 1);
        assert_eq!(s.quality, fw::Quality::Best);
    }
}
