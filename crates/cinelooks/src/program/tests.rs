use super::*;
use crate::look::PV;
use crate::tools::{Def, Kind, TOOLS};

/// A test frame: a luminance ramp across, a hue sweep down, a few hot highlights, partly
/// transparent on the right.
pub(crate) fn frame(w: u32, h: u32) -> Image {
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / (w - 1) as f32;
            let hue = y as f32 / h as f32;
            let sat = 0.15 + 0.85 * ((x * 13 + y * 7) % 10) as f32 / 9.0;
            let (r, g, b) = effectcraft_color::hsv_to_rgb(hue, sat, 0.15 + 0.8 * t);
            let mut c = [r, g, b];
            if (x + 3 * y) % 23 == 0 {
                c = [2.5, 2.2, 1.8];
            }
            let a = if x > w * 7 / 8 { 0.5 } else { 1.0 };
            img.set(x, y, [c[0] * a, c[1] * a, c[2] * a, a]);
        }
    }
    img
}

fn env_for(img: &Image) -> Env {
    Env { layer_size: [img.width as f64, img.height as f64], time: 1.37, ..Env::default() }
}

fn grade(nodes: Vec<Node>, img: &Image) -> Image {
    crate::grade(&Look::new("t", nodes), &env_for(img), 1.0, img).0
}

fn max_diff(a: &Image, b: &Image) -> f32 {
    a.data.iter().zip(&b.data).map(|(p, q)| (0..4).map(|i| (p[i] - q[i]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max)
}

/// Settings that make every parameter of `k` observable (its dependencies switched on).
fn base(k: ToolKind) -> Node {
    let n = Node::new(k);
    match k {
        ToolKind::Exposure => n.with("exposure", 0.5).with("compensation", 0.3).with("highlightProtect", 30.0),
        ToolKind::Contrast => n.with("contrast", 30.0),
        ToolKind::WhiteBalance => n.with("neutralAmount", 50.0).with("neutral", [0.6, 0.5, 0.4, 1.0]),
        ToolKind::ChannelMixer => n,
        ToolKind::HueRemap => n.with("shift", 20.0).with("saturation", 20.0).with("luminance", 10.0),
        ToolKind::FilmGrain => n.with("colored", true),
        ToolKind::Vignette => n,
        ToolKind::GradientExposure | ToolKind::RadialExposure | ToolKind::SpotExposure => n.with("tintAmount", 50.0).with("tint", [1.0, 0.7, 0.4, 1.0]),
        ToolKind::SelectiveColor => n
            .with("hueWidth", 40.0)
            .with("satLow", 40.0)
            .with("hueShift", 10.0)
            .with("tintAmount", 40.0)
            .with("tint", [1.0, 0.8, 0.6, 1.0])
            .with("exposure", 0.5)
            .with("saturation", 20.0),
        ToolKind::LensDistortion => n.with("amount", 40.0),
        ToolKind::Lut => n.with("data", identity_cube_text()).with("input", 2.0),
        _ => n,
    }
}

fn identity_cube_text() -> String {
    // A non-identity lattice so the LUT's controls show: swap red and blue, darken.
    let n = 9;
    let mut s = format!("LUT_3D_SIZE {n}\n");
    let k = (n - 1) as f32;
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let (r, g, b) = (r as f32 / k, g as f32 / k, b as f32 / k);
                s.push_str(&format!("{} {} {}\n", b.powf(1.3) * 0.8 + 0.2 * g * r, g * 0.9 * (1.0 - 0.3 * b), r.sqrt() * (0.7 + 0.3 * g)));
            }
        }
    }
    s
}

/// A value different enough from `v` to show in the result.
fn other(p: &crate::tools::ParamDef, v: &PV) -> Option<PV> {
    Some(match (p.kind, v) {
        (Kind::Num { min, max, .. }, PV::Num(x)) => {
            let span = max - min;
            let y = if x + span * 0.3 <= max { x + span * 0.3 } else { x - span * 0.3 };
            PV::Num(y)
        }
        (Kind::Angle, PV::Num(x)) => PV::Num(x + 60.0),
        (Kind::Choice(o), PV::Num(x)) => PV::Num(((*x as usize + 1) % o.len()) as f64),
        (Kind::Toggle, PV::Bool(b)) => PV::Bool(!b),
        (Kind::Color, _) => PV::List(vec![0.2, 0.9, 0.5, 1.0]),
        (Kind::Wheel, _) => PV::List(vec![0.6, -0.3, 0.2]),
        (Kind::Point, _) => PV::List(vec![0.3, 0.7]),
        (Kind::Curve, _) => PV::Text("0,0 0.4,0.6 1,1".into()),
        _ => return None,
    })
}

#[test]
fn empty_and_disabled_looks_are_identity() {
    let img = frame(48, 32);
    assert_eq!(grade(vec![], &img).data, img.data);
    let off = Node { enabled: false, ..Node::new(ToolKind::FilmGrain) };
    assert_eq!(grade(vec![off, Node::new(ToolKind::Exposure).strength(0.0).with("exposure", 2.0)], &img).data, img.data);
}

#[test]
fn neutral_tools_at_defaults_change_nothing() {
    let img = frame(48, 32);
    for k in [
        ToolKind::Exposure,
        ToolKind::Contrast,
        ToolKind::WhiteBalance,
        ToolKind::Wheels,
        ToolKind::Curves,
        ToolKind::Hsl,
        ToolKind::ChannelMixer,
        ToolKind::HueRemap,
        ToolKind::LensDistortion,
    ] {
        let out = grade(vec![Node::new(k)], &img);
        assert!(max_diff(&out, &img) < 2e-5, "{k:?} at defaults changed the image by {}", max_diff(&out, &img));
    }
}

#[test]
fn every_control_changes_the_image() {
    let img = frame(192, 108);
    let mut silent = vec![];
    for t in TOOLS {
        let base_node = base(t.kind);
        let reference = grade(vec![base_node.clone()], &img);
        for p in t.params {
            if matches!(p.kind, Kind::Data | Kind::File(_)) {
                continue;
            }
            let cur = base_node.params.get(p.id).cloned().unwrap_or_else(|| PV::from_def(p.def));
            let Some(v) = other(p, &cur) else { continue };
            // Mono weights act in monochrome mode only; colour variation with coloured grain.
            let mut n = base_node.clone().with(p.id, v);
            if t.kind == ToolKind::ChannelMixer && p.id.starts_with("mono") && p.id != "monochrome" {
                n = n.with("monochrome", true);
                let r = grade(vec![base_node.clone().with("monochrome", true)], &img);
                if max_diff(&grade(vec![n], &img), &r) < 1e-4 {
                    silent.push(format!("{}.{}", t.name, p.id));
                }
                continue;
            }
            if max_diff(&grade(vec![n], &img), &reference) < 1e-4 {
                silent.push(format!("{}.{}", t.name, p.id));
            }
        }
        // Strength acts on every tool that changes the image.
        if max_diff(&reference, &img) > 1e-4 && max_diff(&grade(vec![base_node.clone().strength(40.0)], &img), &reference) < 1e-4 {
            silent.push(format!("{}.strength", t.name));
        }
    }
    assert!(silent.is_empty(), "controls with no effect: {silent:?}");
}

#[test]
fn every_tool_is_finite_and_keeps_alpha() {
    let img = frame(40, 30);
    for t in TOOLS {
        let out = grade(vec![base(t.kind)], &img);
        assert!(out.data.iter().all(|p| p.iter().all(|v| v.is_finite())), "{}", t.name);
        if !matches!(t.kind, ToolKind::LensDistortion | ToolKind::ChromaticAberration) {
            for (a, b) in out.data.iter().zip(&img.data) {
                assert!((a[3] - b[3]).abs() < 1e-6, "{} changed alpha", t.name);
            }
        }
        // Fully transparent pixels stay transparent.
        let mut clear = img.clone();
        clear.data.iter_mut().for_each(|p| *p = [0.0; 4]);
        let o = grade(vec![base(t.kind)], &clear);
        assert!(o.data.iter().all(|p| p[3] == 0.0 && p[0].abs() < 1e-6), "{} on a transparent layer", t.name);
    }
}

#[test]
fn hostile_parameters_render_finite() {
    let img = frame(24, 16);
    for t in TOOLS {
        let mut n = Node::new(t.kind);
        for p in t.params {
            let v = match p.def {
                Def::Num(_) | Def::Choice(_) => PV::Num(f64::NAN),
                Def::Vec3(_) | Def::Point(_) | Def::Color(_) => PV::List(vec![f64::INFINITY, -1e300, f64::NAN, 9.0]),
                Def::Bool(_) => PV::Bool(true),
                Def::Text(_) => PV::Text("nan,nan 0.5,inf garbage".into()),
            };
            n = n.with(p.id, v);
        }
        n.strength = f64::NAN;
        let out = grade(vec![n], &img);
        assert!(out.data.iter().all(|p| p.iter().all(|v| v.is_finite())), "{}", t.name);
    }
}

#[test]
fn exposure_is_in_linear_light() {
    let img = Image::filled(4, 4, [0.5, 0.25, 0.75, 1.0]);
    let w = Working::default();
    let out = grade(vec![Node::new(ToolKind::Exposure).with("exposure", 1.0)], &img);
    for i in 0..3 {
        let want = w.encode(2.0 * w.decode(img.data[0][i]));
        assert!((out.data[0][i] - want).abs() < 1e-5);
    }
    // In a linearised project the pixels are linear already.
    let env = Env { working: Working::new(None, true), ..env_for(&img) };
    let (o, _) = crate::grade(&Look::new("", vec![Node::new(ToolKind::Exposure).with("exposure", 1.0)]), &env, 1.0, &img);
    assert!((o.data[0][0] - 1.0).abs() < 1e-6);
}

#[test]
fn domain_conversions_are_minimal() {
    let look = Look::new(
        "",
        vec![
            Node::new(ToolKind::Exposure).with("exposure", 0.5),
            Node::new(ToolKind::WhiteBalance).with("temperature", 20.0),
            Node::new(ToolKind::Contrast).with("contrast", 20.0),
            Node::new(ToolKind::Curves).with("master", "0,0 0.5,0.6 1,1"),
        ],
    );
    let p = compile(&look, &Env::default(), 1.0);
    assert_eq!(p.stages.len(), 1);
    let Stage::Pixel(ops) = &p.stages[0] else { panic!() };
    let dec = ops.iter().filter(|o| matches!(o, PixelOp::Decode)).count();
    let enc = ops.iter().filter(|o| matches!(o, PixelOp::Encode)).count();
    assert_eq!((dec, enc), (1, 1), "{ops:?}");
    // Two linear tools in a linear project: no conversion at all.
    let env = Env { working: Working::new(None, true), ..Env::default() };
    let p = compile(&Look::new("", look.nodes[..2].to_vec()), &env, 1.0);
    let Stage::Pixel(ops) = &p.stages[0] else { panic!() };
    assert!(!ops.iter().any(|o| matches!(o, PixelOp::Decode | PixelOp::Encode)));
}

#[test]
fn strength_mixes() {
    let img = frame(32, 20);
    let full = grade(vec![Node::new(ToolKind::Curves).with("master", "0,0 0.5,0.8 1,1")], &img);
    let half = grade(vec![Node::new(ToolKind::Curves).with("master", "0,0 0.5,0.8 1,1").strength(50.0)], &img);
    let i = 5 * 32 + 10;
    for c in 0..3 {
        let want = (img.data[i][c] + full.data[i][c]) * 0.5;
        assert!((half.data[i][c] - want).abs() < 1e-5);
    }
    let (look_half, _) = crate::grade(&Look::new("", vec![Node::new(ToolKind::Curves).with("master", "0,0 0.5,0.8 1,1")]), &env_for(&img), 0.5, &img);
    assert!((look_half.data[i][0] - (img.data[i][0] + full.data[i][0]) * 0.5).abs() < 1e-5);
}

#[test]
fn over_range_values_survive_colour_tools() {
    let img = Image::filled(2, 2, [3.0, 2.0, 1.5, 1.0]);
    for n in [
        Node::new(ToolKind::Curves).with("master", "0,0 0.5,0.55 1,1"),
        Node::new(ToolKind::Contrast).with("contrast", 20.0),
        Node::new(ToolKind::Wheels).with("gain", [0.0, 0.0, 0.1]),
        Node::new(ToolKind::Hsl).with("saturation", 10.0),
    ] {
        let out = grade(vec![n.clone()], &img);
        assert!(out.data[0][0] > 1.5, "{:?} clipped: {:?}", n.tool, out.data[0]);
    }
}

#[test]
fn grain_is_stable_per_frame_zero_mean_and_resolution_independent() {
    let img = Image::filled(128, 72, [0.5, 0.5, 0.5, 1.0]);
    let look = Look::new("", vec![Node::new(ToolKind::FilmGrain).with("amount", 50.0)]);
    let at = |t: f64| crate::grade(&look, &Env { time: t, ..env_for(&img) }, 1.0, &img).0;
    let a = at(1.0);
    assert_eq!(a.data, at(1.0).data, "same frame, same grain");
    assert_ne!(a.data, at(1.0 + 1.0 / 24.0).data, "next frame, new grain");
    // Within one frame's duration the grain holds (sub-frame times, motion blur samples).
    assert_eq!(a.data, at(1.0 + 0.4 / 24.0).data);
    let mean: f32 = a.data.iter().map(|p| p[0] - 0.5).sum::<f32>() / a.data.len() as f32;
    assert!(mean.abs() < 0.01, "grain shifts brightness: {mean}");
    let sd = (a.data.iter().map(|p| (p[0] - 0.5).powi(2)).sum::<f32>() / a.data.len() as f32).sqrt();
    assert!(sd > 0.01, "{sd}");
    // Static grain doesn't change between frames.
    let stat = Look::new("", vec![Node::new(ToolKind::FilmGrain).with("animation", 0.0)]);
    let s = |t: f64| crate::grade(&stat, &Env { time: t, ..env_for(&img) }, 1.0, &img).0;
    assert_eq!(s(0.0).data, s(2.0).data);
}

#[test]
fn grain_looks_the_same_at_twice_the_resolution() {
    // Same layer content at 480×270 and 960×540 (Medium cells of 1.7 and 3.4 px): the larger one,
    // averaged down, has the smaller one's grain statistics.
    let sd_at = |w: u32, h: u32| {
        let img = Image::filled(w, h, [0.5, 0.5, 0.5, 1.0]);
        let look = Look::new("", vec![Node::new(ToolKind::FilmGrain).with("amount", 60.0).with("size", 4.0).with("type", 1.0)]);
        let out = crate::grade(&look, &Env { time: 1.0, ..env_for(&img) }, 1.0, &img).0;
        let out = effectcraft_raster::resample(&out, 480, 270);
        (out.data.iter().map(|p| (p[0] - 0.5).powi(2)).sum::<f32>() / out.data.len() as f32).sqrt()
    };
    let (a, b) = (sd_at(480, 270), sd_at(960, 540));
    assert!((a - b).abs() / a < 0.25, "{a} vs {b}");
}

/// Smooth content (no single-pixel detail): gradients and soft highlight blobs.
fn smooth_frame(w: u32, h: u32) -> Image {
    let mut img = Image::new(w, h);
    let blobs = [(0.3f32, 0.3f32), (0.7, 0.6), (0.5, 0.8)];
    for y in 0..h {
        for x in 0..w {
            let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let mut c = [0.2 + 0.5 * u, 0.25 + 0.3 * v, 0.5 - 0.3 * u];
            for (bx, by) in blobs {
                let d2 = ((u - bx) * w as f32).powi(2) + ((v - by) * h as f32).powi(2);
                let k = 2.0 * (-d2 / (2.0 * (0.05 * h as f32).powi(2))).exp();
                c = c.map(|q| q + k);
            }
            img.set(x, y, [c[0], c[1], c[2], 1.0]);
        }
    }
    img
}

#[test]
fn half_resolution_matches_full_resolution() {
    // Each spatial or positional tool rendered at scale 0.5 matches the full render averaged
    // down (the proxy / half-resolution preview shows the export's look).
    let (w, h) = (320, 180);
    let img = smooth_frame(w, h);
    let half_in = effectcraft_raster::resample(&img, w / 2, h / 2);
    for k in [
        ToolKind::Halation,
        ToolKind::Diffusion,
        ToolKind::AnamorphicGlow,
        ToolKind::Vignette,
        ToolKind::RadialExposure,
        ToolKind::GradientExposure,
        ToolKind::ChromaticAberration,
        ToolKind::EdgeSoftness,
        ToolKind::ShadowHighlight,
        ToolKind::LensDistortion,
    ] {
        let look = Look::new("", vec![base(k)]);
        let full = crate::grade(&look, &env_for(&img), 1.0, &img).0;
        let env = Env { layer_size: [w as f64, h as f64], scale: 0.5, ..env_for(&img) };
        let half = crate::grade(&look, &env, 1.0, &half_in).0;
        let full_down = effectcraft_raster::resample(&full, w / 2, h / 2);
        // Mean absolute difference per channel.
        let mad =
            half.data.iter().zip(&full_down.data).map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).abs()).sum::<f32>() / 3.0).sum::<f32>() / half.data.len() as f32;
        assert!(mad < 0.02, "{k:?}: half resolution differs by {mad}");
    }
}

#[test]
fn missing_lut_warns_and_passes_through() {
    let img = frame(16, 8);
    let look = Look::new("", vec![Node::new(ToolKind::Lut).with("file", "/nope/missing.cube")]);
    let (out, warnings) = crate::grade(&look, &env_for(&img), 1.0, &img);
    assert_eq!(out.data, img.data);
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("not found"), "{warnings:?}");
}

#[test]
fn white_balance_neutralises_the_picked_colour() {
    let w = Working::default();
    let cast = [0.6f32, 0.5, 0.38];
    let img = Image::filled(4, 4, [cast[0], cast[1], cast[2], 1.0]);
    let out = grade(vec![Node::new(ToolKind::WhiteBalance).with("neutral", [0.6, 0.5, 0.38, 1.0]).with("neutralAmount", 100.0)], &img);
    let lin = [0, 1, 2].map(|i| w.decode(out.data[0][i]));
    assert!((lin[0] - lin[1]).abs() < 1e-3 && (lin[1] - lin[2]).abs() < 1e-3, "{lin:?}");
}
