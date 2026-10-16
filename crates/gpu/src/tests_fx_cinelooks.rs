//! GPU CineLooks vs the CPU effect (the oracle): every tool alone, a long chained look with
//! partial strengths, the look strength, linear and gamma-2.4 working spaces, full and half
//! resolution with an offset buffer, and the composited frame at 8 and 32 bpc.

use effectcraft_cinelooks::look::{Look, Node};
use effectcraft_cinelooks::tools::ToolKind;
use effectcraft_color::ColorSpace;
use effectcraft_effects::cinelooks::{ID, NAME, merge_look};
use effectcraft_effects::{Buf, EffectCtx, EffectEnv, Params, flatten_params};
use effectcraft_keyframe::Value;
use effectcraft_project::build::Ids;
use effectcraft_project::{BitDepth, PropGroup};
use effectcraft_render::RenderOpts;
use effectcraft_time::Tick;

use crate::tests::{GL_BOUNDARY_FLIPS, Scene, check, compare_at, diff, gpu, on_gl, opts, pattern, set};

fn instance(look: &Look, size: [f64; 2], next: &mut u64) -> PropGroup {
    let spec = effectcraft_effects::find(ID).unwrap();
    let mut g = effectcraft_effects::instantiate(spec, &mut Ids(next), NAME, size);
    merge_look(&mut g, look, &mut Ids(next), size, Tick::ZERO);
    g
}

/// A hot-spot pattern: the shared test footage plus bright highlights (glows need them).
fn footage(w: u32, h: u32) -> effectcraft_raster::Image {
    let mut img = pattern(7, w, h);
    for (i, p) in img.data.iter_mut().enumerate() {
        let (x, y) = ((i as u32 % w) as f32 / w as f32, (i as u32 / w) as f32 / h as f32);
        let d = ((x - 0.35).powi(2) + (y - 0.4).powi(2)).sqrt();
        if d < 0.06 {
            let a = p[3];
            *p = [2.2 * a, 1.9 * a, 1.6 * a, a];
        }
    }
    img
}

/// The effect alone on buffers at full and half resolution (the latter with an offset).
fn direct(label: &str, look: &Look, strength: f64, env: EffectEnv) {
    let Some(g) = gpu() else { return };
    let spec = effectcraft_effects::find(ID).unwrap();
    let size = [160.0, 96.0];
    let mut next = 1;
    let mut inst = instance(look, size, &mut next);
    inst.get_mut("lookStrength").unwrap().value = Value::Scalar(strength);
    let params: Params = flatten_params(&inst, &mut |p| p.value.clone());
    for (scale, offset) in [(1.0, [0.0, 0.0]), (0.5, [3.0, 2.0])] {
        let mut img = footage((size[0] * scale) as u32 + 6, (size[1] * scale) as u32 + 4);
        if scale == 1.0 {
            img = footage(size[0] as u32, size[1] as u32);
        }
        let buf = Buf { img, offset, scale };
        let ctx = || EffectCtx { params: &params, time: 0.75, layer_size: size, seed: 11, adjustment: false, env };
        let cpu = (spec.render)(&ctx(), buf.clone());
        let out = effectcraft_render::Accelerator::effects(g, &[effectcraft_render::FxStep { spec, ctx: ctx() }], &buf, None).expect("the GPU runs the effect");
        assert_eq!((out.offset, out.scale), (cpu.offset, cpu.scale), "{label}: geometry");
        let d = diff(&cpu.img, &out.img, 1e-3);
        let allow = if on_gl() { (d.total as f64 * GL_BOUNDARY_FLIPS) as usize } else { 0 };
        assert!(d.over <= allow, "{label} scale {scale}: {} of {} pixels over 1e-3 (max {}); worst {:?}", d.over, d.total, d.max, d.worst);
    }
}

fn tool_case(n: Node) -> Node {
    match n.tool {
        ToolKind::Exposure => n.with("exposure", 0.7).with("compensation", 0.4).with("highlightProtect", 40.0),
        ToolKind::Contrast => n.with("contrast", 35.0).with("pivot", 0.4),
        ToolKind::WhiteBalance => n.with("temperature", 30.0).with("tint", -20.0).with("neutralAmount", 50.0).with("neutral", [0.6, 0.5, 0.4, 1.0]),
        ToolKind::Wheels => n
            .with("lift", [0.3, -0.2, -0.1])
            .with("gamma", [-0.2, 0.4, 0.1])
            .with("gain", [0.1, 0.2, -0.1])
            .with("offset", [0.1, 0.1, 0.0])
            .with("shadows", [-0.4, 0.3, 0.0])
            .with("midtones", [0.2, 0.0, 0.1])
            .with("highlights", [0.5, 0.2, 0.0]),
        ToolKind::Curves => n.with("master", "0,0.05 0.3,0.25 0.7,0.8 1,0.95").with("red", "0,0 0.5,0.55 1,1").with("blue", "0,0.1 1,0.9"),
        ToolKind::Hsl => {
            n.with("hue", 15.0).with("saturation", 20.0).with("lightness", -10.0).with("redSat", 40.0).with("blueHue", -50.0).with("greenLum", 30.0)
        }
        ToolKind::ChannelMixer => n.with("rg", 20.0).with("bb", 80.0).with("gr", -10.0),
        ToolKind::HueRemap => n.with("shift", 40.0).with("saturation", 30.0).with("luminance", -20.0).with("range", 60.0),
        ToolKind::FilmGrain => n.with("colored", true).with("amount", 40.0).with("type", 1.0),
        ToolKind::SelectiveColor => {
            n.with("hueWidth", 60.0).with("hueShift", 20.0).with("exposure", 0.5).with("tintAmount", 30.0).with("tint", [1.0, 0.8, 0.6, 1.0])
        }
        ToolKind::GradientExposure | ToolKind::RadialExposure | ToolKind::SpotExposure => n.with("tintAmount", 40.0).with("tint", [1.0, 0.7, 0.5, 1.0]),
        ToolKind::LensDistortion => n.with("amount", 40.0).with("zoom", 110.0),
        ToolKind::Lut => n.with("data", lut_text()).with("input", 3.0),
        _ => n,
    }
}

fn lut_text() -> String {
    let n = 9;
    let k = (n - 1) as f32;
    let mut s = format!("LUT_1D_SIZE 4\nLUT_3D_SIZE {n}\n0 0 0\n0.4 0.38 0.36\n0.75 0.72 0.7\n1 1 1\n");
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let (r, g, b) = (r as f32 / k, g as f32 / k, b as f32 / k);
                s.push_str(&format!("{} {} {}\n", r * 0.9 + 0.1 * g * b, g * 0.95 + 0.05 * r, b.powf(1.2)));
            }
        }
    }
    s
}

#[test]
fn every_tool_matches_the_cpu() {
    for t in effectcraft_cinelooks::TOOLS {
        let look = Look::new("t", vec![tool_case(Node::new(t.kind))]);
        direct(t.name, &look, 100.0, EffectEnv::default());
    }
}

#[test]
fn chained_look_strengths_and_working_spaces() {
    let nodes: Vec<Node> = effectcraft_cinelooks::TOOLS
        .iter()
        .enumerate()
        .filter(|(_, t)| t.kind != ToolKind::SelectiveColor)
        .map(|(i, t)| tool_case(Node::new(t.kind)).strength(if i % 3 == 0 { 55.0 } else { 100.0 }))
        .collect();
    let look = Look::new("chain", nodes);
    direct("chain", &look, 100.0, EffectEnv::default());
    direct("chain at 60 %", &look, 60.0, EffectEnv::default());
    direct("chain linearised", &look, 100.0, EffectEnv { working_space: Some(ColorSpace::Srgb), working_linear: true, ..Default::default() });
    direct("chain Rec. 709", &look, 100.0, EffectEnv { working_space: Some(ColorSpace::Rec709), ..Default::default() });
    direct("chain ACEScg", &look, 100.0, EffectEnv { working_space: Some(ColorSpace::AcesCg), ..Default::default() });
}

#[test]
fn composited_frames_match() {
    let look = Look::new(
        "comp",
        vec![
            tool_case(Node::new(ToolKind::Exposure)),
            tool_case(Node::new(ToolKind::Wheels)),
            Node::new(ToolKind::FilmResponse),
            Node::new(ToolKind::Halation),
            tool_case(Node::new(ToolKind::FilmGrain)),
            Node::new(ToolKind::Vignette),
        ],
    );
    for depth in [BitDepth::Bpc8, BitDepth::Bpc32] {
        let mut s = Scene::new(depth);
        let bg = s.solid([0.15, 0.1, 0.2], 97, 61);
        s.push(bg);
        let mut l = s.footage(70, 44);
        let size = effectcraft_render::effect_bounds(&s.p, &s.comp, &l).0;
        let mut next = s.p.next_id;
        let g = instance(&look, size, &mut next);
        s.p.next_id = next;
        l.props.sub_mut("effects").unwrap().children.push(g.into());
        set(&mut l, "transform/rotation", Value::Scalar(8.0));
        s.push(l);
        let t = Tick::from_seconds_f64(0.5);
        check(&format!("CineLooks comp {depth:?}"), compare_at(&s, opts(), t), 0.0);
        check(&format!("CineLooks comp {depth:?} half"), compare_at(&s, RenderOpts { scale: 0.5, ..opts() }, t), 0.0);
    }
}

/// The kernels parse and validate against the shared prelude without a device.
#[test]
fn shader_validates_and_declares_every_kernel() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    let src = [read("common.wgsl"), read("kernels.wgsl"), read("fx_lut.wgsl"), read("fx_cinelooks.wgsl")].concat();
    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
    let entries: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
    for k in crate::fx_cinelooks::KERNELS {
        assert!(entries.contains(k), "kernel {k} is not declared");
    }
}
