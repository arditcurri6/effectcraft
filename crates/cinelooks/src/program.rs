//! Compile a [`Look`] for one render ([`Env`]: layer geometry, time, working space) into a
//! [`Program`] of stages, and run it on the CPU. The GPU runs the same program.
//!
//! Compilation decides everything that doesn't depend on the pixel: matrices, tables, wheel
//! offsets, grain seeds and frame, blur plans, pixel radii. Consecutive per-pixel tools share a
//! pixel stage; domain conversions are inserted only where the next tool needs the other domain,
//! and once at the end back to the project's.

use std::sync::Arc;

use effectcraft_color::{ColorSpace, Conversion};
use effectcraft_raster::Image;
use rayon::prelude::*;

use crate::blur::Plan;
use crate::cmath::{Working, hash32, neutral_gains, wheel_rgb, white_balance_matrix};
use crate::curve::Points;
use crate::look::{Look, Node};
use crate::ops::{PixelOp, Qualifier, Shape, run};
use crate::spatial::{Buffer, SpatialOp};
use crate::tools::{Domain, ToolKind};

/// Everything a render knows besides the look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Env {
    /// Layer size in layer pixels.
    pub layer_size: [f64; 2],
    /// Buffer mapping: layer point `p` is at pixel `p * scale + offset`.
    pub offset: [f64; 2],
    pub scale: f64,
    /// Layer time (seconds) and frame rate (grain frames).
    pub time: f64,
    pub fps: f64,
    /// Effect instance seed.
    pub seed: u32,
    pub working: Working,
}

impl Default for Env {
    fn default() -> Env {
        Env { layer_size: [1920.0, 1080.0], offset: [0.0; 2], scale: 1.0, time: 0.0, fps: 24.0, seed: 0, working: Working::default() }
    }
}

impl Env {
    /// The frame index grain uses at `animation` (0 = static, 1 = new grain every frame).
    pub fn grain_frame(&self, animation: f64) -> u32 {
        let fps = if self.fps > 0.0 && self.fps.is_finite() { self.fps } else { 24.0 };
        let f = (self.time * fps * animation.clamp(0.0, 1.0) + 1e-3).floor();
        if f.is_finite() { f.rem_euclid(4_294_967_296.0) as u32 } else { 0 }
    }
    fn h(&self) -> f32 {
        let h = self.layer_size[1];
        if h.is_finite() && h > 0.0 { h as f32 } else { 1080.0 }
    }
    fn w(&self) -> f32 {
        let w = self.layer_size[0];
        if w.is_finite() && w > 0.0 { w as f32 } else { 1920.0 }
    }
    fn s(&self) -> f32 {
        if self.scale.is_finite() && self.scale > 0.0 { self.scale as f32 } else { 1.0 }
    }
    /// Layer point (fractions of the layer size) → layer pixels.
    fn lp(&self, f: [f64; 2]) -> [f32; 2] {
        [(f[0] * self.w() as f64) as f32, (f[1] * self.h() as f64) as f32]
    }
    /// Layer pixels → buffer pixels.
    fn px(&self, p: [f32; 2]) -> [f32; 2] {
        [p[0] * self.s() + self.offset[0] as f32, p[1] * self.s() + self.offset[1] as f32]
    }
}

#[derive(Clone, Debug)]
pub enum Stage {
    Pixel(Vec<PixelOp>),
    Spatial(SpatialOp),
}

#[derive(Clone, Debug)]
pub struct Program {
    pub stages: Vec<Stage>,
    pub working: Working,
    /// Pixel → layer mapping for positional ops: `p = (px − offset) / scale`.
    pub offset: [f32; 2],
    pub scale: f32,
    /// Look Strength (0..1): the final mix with the input.
    pub strength: f32,
    /// Problems the user should hear about (a missing LUT…); the look renders without them.
    pub warnings: Vec<String>,
}

impl Program {
    /// Nothing to do: the output is the input.
    pub fn is_identity(&self) -> bool {
        self.stages.is_empty() || self.strength <= 0.0
    }
}

enum Compiled {
    Pixel(Vec<PixelOp>, Domain),
    Spatial(SpatialOp, Domain),
}

/// Compile `look` for `env`, mixed with the input by `strength` (0..1).
pub fn compile(look: &Look, env: &Env, strength: f32) -> Program {
    let w = env.working;
    let input = if w.input_linear { Domain::Linear } else { Domain::Encoded };
    let mut cur = input;
    let mut stages: Vec<Stage> = vec![];
    // Per-pixel ops waiting to become a pixel stage (they fuse until a spatial tool comes).
    let mut pending: Vec<PixelOp> = vec![];
    let mut warnings = vec![];
    let mut grain_index = 0u32;
    for n in look.nodes.iter().take(crate::look::MAX_NODES) {
        if !n.enabled || n.mix() <= 0.0 {
            continue;
        }
        let c = match compile_node(n, env, &mut grain_index) {
            Ok(Some(c)) => c,
            Ok(None) => continue,
            Err(e) => {
                warnings.push(format!("{}: {e}", n.name()));
                continue;
            }
        };
        match c {
            Compiled::Pixel(ops, d) => {
                convert(&mut pending, &mut cur, d);
                let s = n.mix();
                if s < 1.0 {
                    pending.push(PixelOp::Begin);
                    pending.extend(ops);
                    pending.push(PixelOp::End(s));
                } else {
                    pending.extend(ops);
                }
            }
            Compiled::Spatial(op, d) => {
                if d != Domain::Any {
                    convert(&mut pending, &mut cur, d);
                }
                if !pending.is_empty() {
                    stages.push(Stage::Pixel(std::mem::take(&mut pending)));
                }
                stages.push(Stage::Spatial(op));
            }
        }
    }
    convert(&mut pending, &mut cur, input);
    if !pending.is_empty() {
        stages.push(Stage::Pixel(pending));
    }
    let strength = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 1.0 };
    let s = env.s();
    Program { stages, working: w, offset: [env.offset[0] as f32, env.offset[1] as f32], scale: s, strength, warnings }
}

fn convert(ops: &mut Vec<PixelOp>, cur: &mut Domain, want: Domain) {
    match (*cur, want) {
        (Domain::Encoded, Domain::Linear) => {
            // Undo a preceding Encode instead of stacking Decode on it.
            if matches!(ops.last(), Some(PixelOp::Encode)) {
                ops.pop();
            } else {
                ops.push(PixelOp::Decode);
            }
            *cur = Domain::Linear;
        }
        (Domain::Linear, Domain::Encoded) => {
            if matches!(ops.last(), Some(PixelOp::Decode)) {
                ops.pop();
            } else {
                ops.push(PixelOp::Encode);
            }
            *cur = Domain::Encoded;
        }
        _ => {}
    }
}

fn exp2(v: f32) -> f32 {
    v.exp2()
}

/// Tint gains for a picked tint colour (encoded): normalised so its luma is 1, then mixed in.
fn tint_gains(w: &Working, encoded: [f32; 3], amount: f32, linear: bool) -> [f32; 3] {
    let c = if linear { encoded.map(|v| w.decode(v)) } else { encoded };
    let y = w.y(c).max(1e-4);
    c.map(|v| 1.0 + ((v / y).clamp(0.0, 8.0) - 1.0) * amount.clamp(0.0, 1.0))
}

/// The LUT tool's conversion into the space its table expects (and back), and the domain the
/// chain must be in for it.
fn lut_spaces(w: &Working, input: u32) -> (Domain, Option<Conversion>, Option<Conversion>) {
    let named = match input {
        0 => return (Domain::Encoded, None, None),
        1 => return (Domain::Linear, None, None),
        2 => (ColorSpace::Srgb, false),
        3 => (ColorSpace::Rec709, false),
        4 => (ColorSpace::Rec2020, false),
        5 => (ColorSpace::DisplayP3, false),
        _ => (ColorSpace::AcesCg, true),
    };
    let pre = Conversion::new(w.space, true, named.0, named.1);
    let post = Conversion::new(named.0, named.1, w.space, true);
    (Domain::Linear, pre, post)
}

fn compile_node(n: &Node, env: &Env, grain_index: &mut u32) -> Result<Option<Compiled>, String> {
    let w = &env.working;
    let l = w.luma;
    let d = n.tool.def().domain;
    let px_h = env.h() * env.s();
    let pixel = |ops: Vec<PixelOp>| Ok(Some(Compiled::Pixel(ops, d)));
    match n.tool {
        ToolKind::Exposure => {
            let protect = n.frac("highlightProtect").clamp(0.0, 1.0);
            let comp = n.f("compensation").clamp(-2.4, 2.4);
            // y = x^g maps 0.18 to 0.18·2^comp and keeps 0 and 1.
            let gamma = ((0.18f32 * comp.exp2()).ln() / 0.18f32.ln()).clamp(0.05, 20.0);
            let gain = exp2(n.f("exposure"));
            if gain == 1.0 && comp == 0.0 && protect == 0.0 {
                return Ok(None);
            }
            pixel(vec![PixelOp::Exposure { gain, gamma: if comp == 0.0 { 1.0 } else { gamma }, protect, knee: 1.0 - 0.5 * protect }])
        }
        ToolKind::Contrast => {
            let k = exp2(n.frac("contrast") * 1.2);
            if k == 1.0 {
                return Ok(None);
            }
            pixel(vec![PixelOp::Contrast { k, pivot: n.f("pivot").clamp(0.05, 0.95), filmic: n.frac("filmic") }])
        }
        ToolKind::WhiteBalance => {
            let mut m = white_balance_matrix(w.space, n.num("temperature") / 100.0, n.num("tint") / 100.0);
            let amt = n.frac("neutralAmount");
            if amt > 0.0 {
                let neutral = n.color("neutral").map(|v| w.decode(v));
                let g = neutral_gains(crate::cmath::mat3(&m, neutral), l);
                for (i, row) in m.iter_mut().enumerate() {
                    let gi = 1.0 + (g[i] - 1.0) * amt;
                    for v in row.iter_mut() {
                        *v *= gi;
                    }
                }
            }
            pixel(vec![PixelOp::Matrix(m)])
        }
        ToolKind::Wheels => {
            let wr = |id: &str| wheel_rgb(n.wheel(id), l);
            let offset = wr("offset").map(|v| v * 0.25);
            let lift = wr("lift").map(|v| v * 0.4);
            let gexp = wr("gamma").map(|v| (-0.8 * v).exp2());
            let gain = wr("gain").map(exp2);
            let shadows = wr("shadows").map(|v| v * 0.25);
            let mids = wr("midtones").map(|v| v * 0.2);
            let highs = wr("highlights").map(|v| v * 0.25);
            if ["lift", "gamma", "gain", "offset", "shadows", "midtones", "highlights"].iter().all(|id| n.wheel(id) == [0.0; 3]) {
                return Ok(None);
            }
            pixel(vec![PixelOp::Wheels { offset, lift, gexp, gain, shadows, mids, highs }])
        }
        ToolKind::Curves => {
            let t = ["master", "red", "green", "blue"].map(|id| {
                let p = Points::parse(&n.text(id));
                (!p.is_identity()).then(|| p.bake())
            });
            if t.iter().all(Option::is_none) {
                return Ok(None);
            }
            pixel(vec![PixelOp::Curves(Arc::new(t))])
        }
        ToolKind::Hsl => {
            let ranges = crate::tools::HSL_RANGES
                .map(|(id, _, _)| [n.f(&format!("{id}Hue")) / 100.0 * 30.0, n.f(&format!("{id}Sat")) / 100.0, n.f(&format!("{id}Lum")) / 100.0]);
            let rot = n.f("hue");
            let sat = 1.0 + n.frac("saturation");
            let light = n.frac("lightness");
            if rot == 0.0 && sat == 1.0 && light == 0.0 && ranges.iter().all(|r| *r == [0.0; 3]) {
                return Ok(None);
            }
            pixel(vec![PixelOp::Hsl { rot, sat, light, ranges }])
        }
        ToolKind::ChannelMixer => {
            if n.flag("monochrome") {
                return pixel(vec![PixelOp::Mono([n.frac("monoR"), n.frac("monoG"), n.frac("monoB")])]);
            }
            let m = [[n.frac("rr"), n.frac("rg"), n.frac("rb")], [n.frac("gr"), n.frac("gg"), n.frac("gb")], [n.frac("br"), n.frac("bg"), n.frac("bb")]];
            if m == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                return Ok(None);
            }
            pixel(vec![PixelOp::Matrix(m)])
        }
        ToolKind::HueRemap => {
            let (shift, sat, lum) = (n.f("shift"), n.frac("saturation"), n.frac("luminance"));
            if shift == 0.0 && sat == 0.0 && lum == 0.0 {
                return Ok(None);
            }
            pixel(vec![PixelOp::HueRemap { center: n.f("center").rem_euclid(360.0), range: n.f("range"), falloff: n.f("falloff"), shift, sat, lum }])
        }
        ToolKind::FilmResponse => {
            let k = film_slope(w) * exp2(n.frac("contrast") * 0.6) * 6.0 / (6.0 + n.f("latitude"));
            pixel(vec![PixelOp::Film {
                pre_gain: exp2(n.f("exposure")),
                matrix: None,
                offsets: [0.0; 3],
                k,
                grey: w.encode(0.18),
                p_lo: 5.0 + (1.2 - 5.0) * n.frac("toe"),
                p_hi: 5.0 + (1.0 - 5.0) * n.frac("shoulder"),
                sat: 1.0 + 0.5 * n.frac("color"),
            }])
        }
        ToolKind::FilmPrint => {
            let s = print_style(n.choice("style"));
            let toe = (s.toe + 0.35 * n.frac("shadows")).clamp(0.0, 1.0);
            let shoulder = (s.shoulder + 0.35 * n.frac("highlights")).clamp(0.0, 1.0);
            pixel(vec![PixelOp::Film {
                pre_gain: exp2(n.f("density")),
                matrix: Some(s.matrix),
                offsets: s.offsets,
                k: film_slope(w) * exp2(s.contrast),
                grey: w.encode(0.18),
                p_lo: 5.0 + (1.2 - 5.0) * toe,
                p_hi: 5.0 + (1.0 - 5.0) * shoulder,
                sat: s.sat * n.frac("saturation"),
            }])
        }
        ToolKind::BleachBypass => pixel(vec![PixelOp::Bleach { contrast: n.frac("contrast"), desat: n.frac("desaturation") }]),
        ToolKind::FilmFade => {
            let st = n.color("shadowTint");
            let ht = n.color("highlightTint");
            let (ys, yh) = (w.y(st), w.y(ht));
            pixel(vec![PixelOp::Fade {
                contrast: n.frac("contrast"),
                blacks: n.frac("blacks"),
                sat: n.frac("saturation"),
                shadow: st.map(|v| v - ys),
                high: ht.map(|v| v - yh),
                amount: n.frac("tintAmount"),
            }])
        }
        ToolKind::FilmGrain => {
            let amount = n.frac("amount");
            if amount <= 0.0 {
                return Ok(None);
            }
            let ty = [1.0, 1.7, 2.8][(n.choice("type") as usize).min(2)];
            // Grain cells in layer pixels, sized for the layer's height (1 cell ≈ 1 px at 1080p).
            let cell = (n.f("size") * ty * env.h() / 1080.0).max(0.05);
            let gi = *grain_index;
            *grain_index += 1;
            let seed =
                hash32(env.seed ^ hash32(gi ^ hash32(n.num("seed") as u32 ^ 0x5eed))) ^ hash32(env.grain_frame(n.num("animation")).wrapping_add(0x9e37_79b9));
            pixel(vec![PixelOp::Grain {
                amount,
                cell,
                soft: n.frac("softness"),
                luma_resp: n.frac("lumaResponse"),
                colored: n.flag("colored"),
                color_amt: n.frac("colorAmount"),
                seed,
            }])
        }
        ToolKind::Vignette => {
            let amount = n.frac("amount");
            if amount == 0.0 {
                return Ok(None);
            }
            let r = n.f("radius") / 100.0;
            let a = n.f("aspect").clamp(0.25, 4.0).sqrt();
            let (rx, ry) = ((env.w() * 0.5 * r * a).max(1e-3), (env.h() * 0.5 * r / a).max(1e-3));
            pixel(vec![PixelOp::Vignette {
                c: env.lp(n.point("center")),
                inv_r: [1.0 / rx, 1.0 / ry],
                feather: n.frac("feather").clamp(0.01, 1.0),
                stops: amount * 2.5,
                keep: n.frac("highlights"),
            }])
        }
        ToolKind::GradientExposure => pixel(vec![PixelOp::Masked {
            shape: Shape::Linear { a: env.lp(n.point("start")), b: env.lp(n.point("end")), feather: n.frac("feather") },
            invert: n.flag("invert"),
            opacity: n.frac("opacity"),
            gain: exp2(n.f("exposure")),
            sat: n.frac("saturation"),
            tint: tint_gains(w, n.color("tint"), n.frac("tintAmount"), true),
        }]),
        ToolKind::RadialExposure => {
            let (s, c) = n.f("rotation").to_radians().sin_cos();
            pixel(vec![PixelOp::Masked {
                shape: Shape::Ellipse {
                    c: env.lp(n.point("center")),
                    r: [n.f("width") / 200.0 * env.h(), n.f("height") / 200.0 * env.h()],
                    rot: [c, s],
                    feather: n.frac("feather"),
                },
                invert: n.flag("invert"),
                opacity: n.frac("opacity"),
                gain: exp2(n.f("exposure")),
                sat: n.frac("saturation"),
                tint: tint_gains(w, n.color("tint"), n.frac("tintAmount"), true),
            }])
        }
        ToolKind::SpotExposure => {
            let r = n.f("size") / 200.0 * env.h();
            pixel(vec![PixelOp::Masked {
                shape: Shape::Ellipse { c: env.lp(n.point("center")), r: [r, r], rot: [1.0, 0.0], feather: n.frac("feather") },
                invert: n.flag("invert"),
                opacity: n.frac("opacity"),
                gain: exp2(n.f("exposure")),
                sat: 0.0,
                tint: tint_gains(w, n.color("tint"), n.frac("tintAmount"), true),
            }])
        }
        ToolKind::SelectiveColor => {
            let soft = n.frac("softness") * 50.0;
            let (lo_s, hi_s) = (n.f("satLow").min(n.f("satHigh")), n.f("satHigh").max(n.f("satLow")));
            let (lo_l, hi_l) = (n.f("lumLow").min(n.f("lumHigh")), n.f("lumHigh").max(n.f("lumLow")));
            pixel(vec![PixelOp::Qualifier(Box::new(Qualifier {
                hue: n.f("hue").rem_euclid(360.0),
                hue_width: n.f("hueWidth"),
                hue_soft: n.f("hueSoft"),
                sat: [lo_s, hi_s],
                lum: [lo_l, hi_l],
                soft,
                invert: n.flag("invert"),
                show_mask: n.flag("showMask"),
                opacity: n.frac("opacity"),
                shift: n.f("hueShift"),
                saturation: n.frac("saturation"),
                gain: exp2(n.f("exposure")),
                tint: tint_gains(w, n.color("tint"), n.frac("tintAmount"), false),
            }))])
        }
        ToolKind::Lut => {
            let lut = crate::lut::load(&n.text("file"), &n.text("data"))?;
            let (dom, pre, post) = lut_spaces(w, n.choice("input"));
            Ok(Some(Compiled::Pixel(vec![PixelOp::Lut { lut, trilinear: n.choice("interpolation") == 1, pre, post }], dom)))
        }
        ToolKind::Halation => {
            let sigma = n.f("radius") / 100.0 * px_h * 0.5;
            Ok(Some(Compiled::Spatial(
                SpatialOp::Halation {
                    thr: w.decode(n.f("threshold")),
                    tight: Plan::new(sigma * 0.3),
                    wide: Plan::new(sigma),
                    falloff: n.frac("falloff"),
                    strength: n.frac("amount") * 1.5,
                    color: n.color("color").map(|v| w.decode(v)),
                    mix: n.mix(),
                },
                d,
            )))
        }
        ToolKind::Diffusion => {
            let sigma = n.f("radius") / 100.0 * px_h * 0.5;
            Ok(Some(Compiled::Spatial(
                SpatialOp::Diffusion {
                    thr: w.decode(n.f("threshold")),
                    bloom: Plan::new(sigma),
                    mist: Plan::new(sigma * 0.35),
                    intensity: n.frac("intensity"),
                    softness: n.frac("softness"),
                    color: n.color("color").map(|v| w.decode(v)),
                    mix: n.mix(),
                },
                d,
            )))
        }
        ToolKind::AnamorphicGlow => {
            let len = (n.f("length") / 100.0 * env.w() * env.s()).max(0.5);
            Ok(Some(Compiled::Spatial(
                SpatialOp::Streak {
                    thr: w.decode(n.f("threshold")),
                    decay: (-3.0 / len).exp(),
                    intensity: n.frac("intensity"),
                    color: n.color("color").map(|v| w.decode(v)),
                    mix: n.mix(),
                },
                d,
            )))
        }
        ToolKind::ChromaticAberration => {
            let k = n.frac("amount") * env.h() / 1080.0 * env.s();
            let (rc, by) = (n.f("redCyan") * k, n.f("blueYellow") * k);
            if rc == 0.0 && by == 0.0 {
                return Ok(None);
            }
            let half_diag = (env.w() * env.w() + env.h() * env.h()).sqrt() * 0.5 * env.s();
            Ok(Some(Compiled::Spatial(
                SpatialOp::Chroma {
                    rc,
                    by,
                    radial: n.frac("radial"),
                    center: env.px(env.lp(n.point("center"))),
                    inv_ref: 1.0 / half_diag.max(1.0),
                    mix: n.mix(),
                },
                d,
            )))
        }
        ToolKind::LensDistortion => {
            let k = n.frac("amount") * 0.35;
            let zoom = n.f("zoom") / 100.0;
            if k == 0.0 && zoom == 1.0 {
                return Ok(None);
            }
            let half_diag = (env.w() * env.w() + env.h() * env.h()).sqrt() * 0.5 * env.s();
            Ok(Some(Compiled::Spatial(
                SpatialOp::Distort {
                    k,
                    inv_zoom: 1.0 / zoom.max(0.01),
                    center: env.px(env.lp(n.point("center"))),
                    ref_r: half_diag.max(1.0),
                    extend: n.choice("edges") == 1,
                    mix: n.mix(),
                },
                d,
            )))
        }
        ToolKind::EdgeSoftness => {
            let sigma = n.f("radius") / 100.0 * px_h;
            if sigma < 0.05 {
                return Ok(None);
            }
            let a = env.px([0.0, 0.0]);
            let b = env.px([env.w(), env.h()]);
            Ok(Some(Compiled::Spatial(
                SpatialOp::EdgeSoft { blur: Plan::new(sigma), rect: [a[0], a[1], b[0], b[1]], size: n.frac("size"), feather: n.frac("feather"), mix: n.mix() },
                d,
            )))
        }
        ToolKind::ShadowHighlight => {
            let (sh, hi) = (n.frac("shadows"), n.frac("highlights"));
            if sh == 0.0 && hi == 0.0 {
                return Ok(None);
            }
            let sigma = n.f("radius") / 100.0 * px_h * 0.5;
            Ok(Some(Compiled::Spatial(SpatialOp::Local { blur: Plan::new(sigma), shadows: sh, highlights: hi, width: n.frac("width"), mix: n.mix() }, d)))
        }
    }
}

/// Display-domain slope per stop of the grading encoding at mid grey (Film Response's
/// contrast 0 matches the plain encoding there).
fn film_slope(w: &Working) -> f32 {
    let e = |s: f32| w.encode(0.18 * s.exp2());
    ((e(0.05) - e(-0.05)) / 0.1).clamp(0.02, 1.0)
}

/// An original print style: log-space dye crosstalk and offsets (stops), contrast (log2 of
/// the slope factor), toe / shoulder softness and saturation.
pub struct PrintStyle {
    pub matrix: [[f32; 3]; 3],
    pub offsets: [f32; 3],
    pub contrast: f32,
    pub toe: f32,
    pub shoulder: f32,
    pub sat: f32,
}

pub fn print_style(i: u32) -> PrintStyle {
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    match i {
        1 => PrintStyle {
            matrix: [[1.05, -0.03, -0.02], [-0.02, 1.04, -0.02], [-0.01, -0.05, 1.06]],
            offsets: [0.10, 0.02, -0.14],
            contrast: 0.2,
            toe: 0.45,
            shoulder: 0.5,
            sat: 1.05,
        },
        2 => PrintStyle {
            matrix: [[1.04, -0.02, -0.02], [-0.03, 1.05, -0.02], [-0.02, -0.02, 1.04]],
            offsets: [-0.12, 0.0, 0.12],
            contrast: 0.2,
            toe: 0.45,
            shoulder: 0.5,
            sat: 0.97,
        },
        3 => PrintStyle {
            matrix: [[1.06, -0.03, -0.03], [-0.03, 1.06, -0.03], [-0.03, -0.03, 1.06]],
            offsets: [0.02, -0.02, 0.03],
            contrast: 0.4,
            toe: 0.2,
            shoulder: 0.35,
            sat: 1.0,
        },
        4 => PrintStyle { matrix: id, offsets: [0.03, 0.01, -0.02], contrast: -0.2, toe: 0.75, shoulder: 0.85, sat: 0.92 },
        _ => PrintStyle {
            matrix: [[1.03, -0.015, -0.015], [-0.015, 1.03, -0.015], [-0.015, -0.015, 1.03]],
            offsets: [0.0; 3],
            contrast: 0.25,
            toe: 0.5,
            shoulder: 0.45,
            sat: 1.0,
        },
    }
}

/// Run `prog` on a premultiplied image whose pixel `(x, y)` maps to layer point
/// `((x + .5 − offset) / scale)`; premultiplied out, same size.
pub fn render(prog: &Program, img: &Image) -> Image {
    if prog.is_identity() || img.is_empty() {
        return img.clone();
    }
    let (w, h) = (img.width as usize, img.height as usize);
    let mut buf = Buffer {
        w,
        h,
        px: img
            .data
            .par_iter()
            .map(|p| {
                let a = p[3];
                if a > 1e-6 { [p[0] / a, p[1] / a, p[2] / a, a] } else { [0.0, 0.0, 0.0, a.max(0.0)] }
            })
            .collect(),
    };
    let wk = prog.working;
    let (ox, oy, s) = (prog.offset[0], prog.offset[1], prog.scale.max(1e-6));
    for st in &prog.stages {
        match st {
            Stage::Pixel(ops) => {
                if ops.is_empty() {
                    continue;
                }
                buf.px.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
                    let ly = (y as f32 + 0.5 - oy) / s;
                    for (x, px) in row.iter_mut().enumerate() {
                        let lp = [(x as f32 + 0.5 - ox) / s, ly];
                        let c = run(ops, [px[0], px[1], px[2]], lp, &wk);
                        *px = [c[0], c[1], c[2], px[3]];
                    }
                });
            }
            Stage::Spatial(op) => {
                let out = op.run(&buf, &wk);
                if out.len() == buf.px.len() {
                    buf.px = out.into_iter().map(|p| p.map(|v| if v.is_finite() { v } else { 0.0 })).collect();
                }
            }
        }
    }
    let k = prog.strength;
    let data = img
        .data
        .par_iter()
        .zip(buf.px.par_iter())
        .map(|(src, o)| {
            let a = o[3];
            let r = [o[0] * a, o[1] * a, o[2] * a, a];
            [0, 1, 2, 3].map(|i| src[i] + (r[i] - src[i]) * k)
        })
        .collect();
    Image { width: img.width, height: img.height, data }
}

#[cfg(test)]
mod tests;
