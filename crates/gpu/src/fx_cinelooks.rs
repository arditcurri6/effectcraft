//! GPU effect, CineLooks (kernels in `shaders/fx_cinelooks.wgsl`, every entry point prefixed
//! `fcl_`).
//!
//! The look compiles on the CPU exactly as for the CPU effect
//! (`effectcraft_effects::cinelooks::program`): matrices, curve tables, grain seeds, blur plans
//! and pixel radii are decided there. Each pixel stage becomes one op stream (curve tables and
//! LUT lattices appended) interpreted per pixel by `fcl_pixel`; the first stage unpremultiplies,
//! the last premultiplies and mixes with the input by the look strength. Spatial stages run the
//! blur pyramid (`fcl_extract`, `fcl_down`, `fcl_blur`) and a composite kernel that samples the
//! blurred level with the CPU's bilinear upsample. Between stages the image is straight colour,
//! as on the CPU, so the two agree to float rounding.
//!
//! Anything the kernels don't cover (a transfer curve other than sRGB / gamma 2.4, a LUT too big
//! for one buffer) returns `None` and the compositor's CPU fallback renders the effect.

use effectcraft_cinelooks::blur::Plan;
use effectcraft_cinelooks::cmath::Working;
use effectcraft_cinelooks::lut::Lut;
use effectcraft_cinelooks::ops::{PixelOp, Shape};
use effectcraft_cinelooks::spatial::SpatialOp;
use effectcraft_cinelooks::{Program, Stage};
use effectcraft_color::space::Curve;
use effectcraft_color::{ColorSpace, Conversion};
use effectcraft_effects::EffectCtx;

use crate::context::{Enc, GpuImage, Params};
use crate::effects::GBuf;

/// Compute entry points in `fx_cinelooks.wgsl`.
pub(crate) const KERNELS: &[&str] = &[
    "fcl_pixel",
    "fcl_extract",
    "fcl_down",
    "fcl_blur",
    "fcl_addglow",
    "fcl_mist",
    "fcl_streak_fwd",
    "fcl_streak_bwd",
    "fcl_chroma",
    "fcl_distort",
    "fcl_edgesoft",
    "fcl_local",
];

/// Effect ids implemented here.
pub(crate) const IDS: &[&str] = &[effectcraft_effects::cinelooks::ID];

/// Largest op stream (floats) uploaded; bigger looks (huge LUTs) render on the CPU.
const MAX_FLOATS: usize = 1 << 24;

pub(crate) fn apply(e: &mut Enc, _id: &str, ctx: &EffectCtx, b: GBuf) -> Option<GBuf> {
    let Some(prog) = effectcraft_effects::cinelooks::program(ctx, b.offset, b.scale) else { return Some(b) };
    let img = run(e, &prog, &b.img)?;
    Some(GBuf { img, ..b })
}

/// Run a compiled program on a premultiplied GPU image; `None` = render on the CPU.
pub(crate) fn run(e: &mut Enc, prog: &Program, input: &GpuImage) -> Option<GpuImage> {
    let code = curve_code(prog.working.curve)?;
    let empty = Stage::Pixel(vec![]);
    // Start and end with a pixel stage (they unpremultiply and premultiply).
    let mut list: Vec<&Stage> = vec![];
    if !matches!(prog.stages.first(), Some(Stage::Pixel(_))) {
        list.push(&empty);
    }
    list.extend(prog.stages.iter());
    if !matches!(list.last(), Some(Stage::Pixel(_))) {
        list.push(&empty);
    }
    let n = list.len();
    let mut cur = input.clone();
    for (i, st) in list.into_iter().enumerate() {
        cur = match st {
            Stage::Pixel(ops) => pixel_stage(e, prog, ops, &cur, input, i == 0, i + 1 == n, code)?,
            Stage::Spatial(op) => spatial(e, &prog.working, op, &cur)?,
        };
    }
    Some(cur)
}

fn curve_code(c: Curve) -> Option<u32> {
    match c {
        Curve::Srgb => Some(1),
        Curve::Gamma24 => Some(2),
        Curve::Linear => Some(0),
        Curve::Pq | Curve::Hlg => None,
    }
}

fn space_code(s: ColorSpace) -> Option<f32> {
    curve_code(s.curve()).map(|c| c as f32)
}

#[allow(clippy::too_many_arguments)]
fn pixel_stage(e: &mut Enc, prog: &Program, ops: &[PixelOp], src: &GpuImage, orig: &GpuImage, first: bool, last: bool, code: u32) -> Option<GpuImage> {
    let stream = encode(ops)?;
    let buf = e.data(&stream);
    let mut p = Params::default();
    p.u[0] = [first as u32, last as u32, code, 0];
    p.f[0] = [prog.offset[0], prog.offset[1], prog.scale, prog.strength];
    let l = prog.working.luma;
    p.f[1] = [l[0], l[1], l[2], 0.0];
    let out = e.scratch(src.width, src.height);
    e.pixels("fcl_pixel", &p, src, Some(orig), &out, Some(&buf));
    Some(out)
}

/// Push a table ([`effectcraft_cinelooks::curve::Table`]): `[n, slope_lo, slope_hi, v…]`.
fn push_table(d: &mut Vec<f32>, t: &effectcraft_cinelooks::curve::Table) -> f32 {
    let at = d.len() as f32;
    d.extend([t.v.len() as f32, t.slope_lo, t.slope_hi]);
    d.extend(&t.v);
    at
}

fn push_lut(d: &mut Vec<f32>, l: &Lut) -> f32 {
    let at = d.len() as f32;
    d.extend([l.n1 as f32, l.n3 as f32]);
    d.extend(l.min1);
    d.extend(l.max1);
    d.extend(l.min3);
    d.extend(l.max3);
    for v in l.d1.iter().chain(&l.d3) {
        d.extend(v);
    }
    at
}

/// A conversion: `[has, decode code, encode code, has matrix, m (9)]` (13 floats).
fn push_conv(prog: &mut Vec<f32>, c: &Option<Conversion>) -> Option<()> {
    let Some(c) = c else {
        prog.extend([0.0; 13]);
        return Some(());
    };
    prog.push(1.0);
    prog.push(match c.decode {
        Some(s) => space_code(s)?,
        None => 0.0,
    });
    prog.push(match c.encode {
        Some(s) => space_code(s)?,
        None => 0.0,
    });
    match &c.matrix {
        Some(m) => {
            prog.push(1.0);
            for r in m {
                prog.extend(r);
            }
        }
        None => prog.extend([0.0; 10]),
    }
    Some(())
}

/// The op stream of a pixel stage (see `fcl_run`), ending in 0, then its tables. `None` when a
/// part can't be expressed (it renders on the CPU).
fn encode(ops: &[PixelOp]) -> Option<Vec<f32>> {
    let mut prog: Vec<f32> = vec![];
    // Tables, appended after the program; their offsets are patched in once its length is known.
    let mut tables: Vec<f32> = vec![];
    let mut patches: Vec<(usize, f32)> = vec![];
    for op in ops {
        match op {
            PixelOp::Decode => prog.push(1.0),
            PixelOp::Encode => prog.push(2.0),
            PixelOp::Begin => prog.push(3.0),
            PixelOp::End(s) => prog.extend([4.0, *s]),
            PixelOp::Exposure { gain, gamma, protect, knee } => prog.extend([5.0, *gain, *gamma, *protect, *knee]),
            PixelOp::Contrast { k, pivot, filmic } => prog.extend([6.0, *k, *pivot, *filmic]),
            PixelOp::Matrix(m) => {
                prog.push(7.0);
                for r in m {
                    prog.extend(r);
                }
            }
            PixelOp::Mono(m) => prog.extend([8.0, m[0], m[1], m[2]]),
            PixelOp::Wheels { offset, lift, gexp, gain, shadows, mids, highs } => {
                prog.push(9.0);
                for v in [offset, lift, gexp, gain, shadows, mids, highs] {
                    prog.extend(v);
                }
            }
            PixelOp::Curves(t) => {
                prog.push(10.0);
                for tb in t.iter() {
                    match tb {
                        Some(tb) => {
                            patches.push((prog.len(), push_table(&mut tables, tb)));
                            prog.push(0.0);
                        }
                        None => prog.push(-1.0),
                    }
                }
            }
            PixelOp::Hsl { rot, sat, light, ranges } => {
                prog.extend([11.0, *rot, *sat, *light]);
                for r in ranges {
                    prog.extend(r);
                }
            }
            PixelOp::HueRemap { center, range, falloff, shift, sat, lum } => prog.extend([12.0, *center, *range, *falloff, *shift, *sat, *lum]),
            PixelOp::Film { pre_gain, matrix, offsets, k, grey, p_lo, p_hi, sat } => {
                prog.extend([13.0, *pre_gain]);
                match matrix {
                    Some(m) => {
                        prog.push(1.0);
                        for r in m {
                            prog.extend(r);
                        }
                    }
                    None => prog.extend([0.0; 10]),
                }
                prog.extend(offsets);
                prog.extend([*k, *grey, *p_lo, *p_hi, *sat]);
            }
            PixelOp::Bleach { contrast, desat } => prog.extend([14.0, *contrast, *desat]),
            PixelOp::Fade { contrast, blacks, sat, shadow, high, amount } => {
                prog.extend([15.0, *contrast, *blacks, *sat]);
                prog.extend(shadow);
                prog.extend(high);
                prog.push(*amount);
            }
            PixelOp::Grain { amount, cell, soft, luma_resp, colored, color_amt, seed } => {
                prog.extend([16.0, *amount, *cell, *soft, *luma_resp, *colored as u32 as f32, *color_amt]);
                prog.extend([(seed >> 16) as f32, (seed & 0xffff) as f32]);
            }
            PixelOp::Vignette { c, inv_r, feather, stops, keep } => prog.extend([17.0, c[0], c[1], inv_r[0], inv_r[1], *feather, *stops, *keep]),
            PixelOp::Masked { shape, invert, opacity, gain, sat, tint } => {
                prog.push(18.0);
                match shape {
                    Shape::Linear { a, b, feather } => prog.extend([0.0, a[0], a[1], b[0], b[1], 0.0, 0.0, *feather]),
                    Shape::Ellipse { c, r, rot, feather } => prog.extend([1.0, c[0], c[1], r[0], r[1], rot[0], rot[1], *feather]),
                }
                prog.extend([*invert as u32 as f32, *opacity, *gain, *sat]);
                prog.extend(tint);
            }
            PixelOp::Qualifier(q) => {
                prog.extend([19.0, q.hue, q.hue_width, q.hue_soft, q.sat[0], q.sat[1], q.lum[0], q.lum[1], q.soft]);
                prog.extend([q.invert as u32 as f32, q.show_mask as u32 as f32, q.opacity, q.shift, q.saturation, q.gain]);
                prog.extend(q.tint);
            }
            PixelOp::Lut { lut, trilinear, pre, post } => {
                prog.push(20.0);
                patches.push((prog.len(), push_lut(&mut tables, lut)));
                prog.extend([0.0, *trilinear as u32 as f32]);
                push_conv(&mut prog, pre)?;
                push_conv(&mut prog, post)?;
            }
        }
        if prog.len() + tables.len() > MAX_FLOATS {
            return None;
        }
    }
    prog.push(0.0);
    let base = prog.len() as f32;
    for (at, off) in patches {
        *prog.get_mut(at)? = base + off;
    }
    prog.extend(tables);
    Some(prog)
}

fn kernel(e: &mut Enc, entry: &str, p: &Params, src: &GpuImage, aux: Option<&GpuImage>) -> GpuImage {
    let out = e.scratch(src.width, src.height);
    e.pixels(entry, p, src, aux, &out, None);
    out
}

/// A glow source (`fcl_extract` mode 0 excess luma, 1 excess RGB, 2 premultiplied, 3 luma).
fn extract(e: &mut Enc, w: &Working, src: &GpuImage, mode: u32, thr: f32) -> GpuImage {
    let mut p = Params::default();
    p.u[0][0] = mode;
    p.f[0][0] = thr;
    p.f[1] = [w.luma[0], w.luma[1], w.luma[2], 0.0];
    kernel(e, "fcl_extract", &p, src, None)
}

/// [`effectcraft_cinelooks::blur::blur`]: the blurred working level (sampled by the composite
/// kernels with the CPU's upsample).
fn blur(e: &mut Enc, src: &GpuImage, plan: &Plan) -> GpuImage {
    if plan.is_identity() {
        return src.clone();
    }
    let mut cur = src.clone();
    for _ in 0..plan.levels {
        let out = e.scratch(cur.width.div_ceil(2).max(1), cur.height.div_ceil(2).max(1));
        e.pixels("fcl_down", &Params::default(), &cur, None, &out, None);
        cur = out;
    }
    if plan.weights.len() > 1 {
        let wb = e.data(&plan.weights);
        for horizontal in [true, false] {
            let mut p = Params::default();
            p.u[0] = [horizontal as u32, (plan.weights.len() - 1) as u32, 0, 0];
            let out = e.scratch(cur.width, cur.height);
            e.pixels("fcl_blur", &p, &cur, None, &out, Some(&wb));
            cur = out;
        }
    }
    cur
}

/// Add a blurred level times `k` (per channel) to `src`; `rgb`: the level holds colour.
fn add_glow(e: &mut Enc, src: &GpuImage, level: &GpuImage, k: [f32; 3], rgb: bool) -> GpuImage {
    let mut p = Params::default();
    p.u[0][0] = rgb as u32;
    p.f[0] = [k[0], k[1], k[2], 0.0];
    kernel(e, "fcl_addglow", &p, src, Some(level))
}

fn spatial(e: &mut Enc, w: &Working, op: &SpatialOp, src: &GpuImage) -> Option<GpuImage> {
    let scaled = |c: &[f32; 3], k: f32| [c[0] * k, c[1] * k, c[2] * k];
    Some(match op {
        SpatialOp::Halation { thr, tight, wide, falloff, strength, color, mix } => {
            let ex = extract(e, w, src, 0, *thr);
            let t = blur(e, &ex, tight);
            let wd = blur(e, &ex, wide);
            let a = add_glow(e, src, &t, scaled(color, mix * strength * (1.0 - falloff)), false);
            add_glow(e, &a, &wd, scaled(color, mix * strength * falloff), false)
        }
        SpatialOp::Diffusion { thr, bloom, mist, intensity, softness, color, mix } => {
            let ex = extract(e, w, src, 1, *thr);
            let bl = blur(e, &ex, bloom);
            let pre = extract(e, w, src, 2, 0.0);
            let ms = blur(e, &pre, mist);
            let mut p = Params::default();
            p.f[0][0] = softness * intensity * mix;
            let base = kernel(e, "fcl_mist", &p, src, Some(&ms));
            add_glow(e, &base, &bl, scaled(color, mix * intensity), true)
        }
        SpatialOp::Streak { thr, decay, intensity, color, mix } => {
            let ex = extract(e, w, src, 0, *thr);
            let mut p = Params::default();
            p.f[0][0] = *decay;
            let groups = (1, src.height.div_ceil(16));
            let fwd = e.scratch(src.width, src.height);
            e.dispatch("fcl_streak_fwd", &p, &ex, None, &fwd, None, groups);
            let st = e.scratch(src.width, src.height);
            e.dispatch("fcl_streak_bwd", &p, &ex, Some(&fwd), &st, None, groups);
            add_glow(e, src, &st, scaled(color, mix * intensity * 3.0), false)
        }
        SpatialOp::Chroma { rc, by, radial, center, inv_ref, mix } => {
            let mut p = Params::default();
            p.f[0] = [*rc, *by, *radial, *inv_ref];
            p.f[1] = [center[0], center[1], *mix, 0.0];
            kernel(e, "fcl_chroma", &p, src, None)
        }
        SpatialOp::Distort { k, inv_zoom, center, ref_r, extend, mix } => {
            let mut p = Params::default();
            p.f[0] = [*k, *inv_zoom, center[0], center[1]];
            p.f[1] = [*ref_r, *extend as u32 as f32, *mix, 0.0];
            kernel(e, "fcl_distort", &p, src, None)
        }
        SpatialOp::EdgeSoft { blur: plan, rect, size, feather, mix } => {
            let pre = extract(e, w, src, 2, 0.0);
            let bl = blur(e, &pre, plan);
            let mut p = Params::default();
            p.f[0] = *rect;
            p.f[1] = [*size, *feather, *mix, 0.0];
            kernel(e, "fcl_edgesoft", &p, src, Some(&bl))
        }
        SpatialOp::Local { blur: plan, shadows, highlights, width, mix } => {
            let ly = extract(e, w, src, 3, 0.0);
            let bl = blur(e, &ly, plan);
            let mut p = Params::default();
            p.f[0] = [*shadows, *highlights, *width, *mix];
            kernel(e, "fcl_local", &p, src, Some(&bl))
        }
    })
}
