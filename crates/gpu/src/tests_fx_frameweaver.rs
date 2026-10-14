//! GPU FrameWeaver vs the CPU effect (the oracle). Both render the same plan
//! (`effectcraft_effects::frameweaver_plan`: frames, cuts, flow, guides, time fields); the GPU
//! then samples it. A precomp whose content moves, at a time that falls between two source
//! frames, composited at 8 and 32 bpc, at full and half resolution. The direct cases have no
//! host and pass the layer through.

use effectcraft_color::Label;
use effectcraft_keyframe::{Keyframe, Value};
use effectcraft_project::{BitDepth, Comp, ItemKind, LayerSource, build};
use effectcraft_render::RenderOpts;
use effectcraft_time::{FrameRate, Tick};

use crate::tests::{Scene, check, compare_at, effect_case, n, opts, set, v3};

const ID: &str = "ec.time.frameweaver";

fn e(v: u32) -> Value {
    Value::Enum(v)
}

fn on() -> Value {
    Value::Bool(true)
}

/// FrameWeaver with `vals` on a moving precomp; `maps` add a hidden footage layer each and set
/// that layer parameter to it.
fn moving(vals: &[(&str, Value)], maps: &[&str]) {
    for depth in [BitDepth::Bpc8, BitDepth::Bpc32] {
        let mut s = Scene::new(depth);
        let bg = s.solid([0.15, 0.1, 0.2], 97, 61);
        s.push(bg);
        let mut vals = vals.to_vec();
        for param in maps {
            let mut m = s.footage(50, 36);
            m.switches.video = false;
            let mid = s.push(m);
            vals.push((param, Value::Layer(Some(mid.0))));
        }
        let inner = Comp::new(60, 40, FrameRate::FPS_30, Tick::from_seconds_f64(2.0));
        let inner_id = s.p.add_item("Inner", Label::Sandstone, None, ItemKind::Comp(inner.into()));
        let mut a = s.footage(30, 24);
        let pos = a.props.prop_mut("transform/position").unwrap();
        pos.keys = vec![Keyframe::new(Tick::ZERO, v3(10.0, 12.0)), Keyframe::new(Tick::from_seconds_f64(1.0), v3(50.0, 30.0))];
        let rot = a.props.prop_mut("transform/rotation").unwrap();
        rot.keys = vec![Keyframe::new(Tick::ZERO, Value::Scalar(0.0)), Keyframe::new(Tick::from_seconds_f64(1.0), Value::Scalar(90.0))];
        s.p.comp_mut(inner_id).unwrap().layers = vec![a];
        let mut l = build::layer(&mut s.p, &s.comp, "Inner", LayerSource::Comp { item: inner_id }, (60, 40), None);
        s.effect(&mut l, ID, &vals);
        set(&mut l, "transform/rotation", Value::Scalar(8.0));
        s.push(l);
        // Speed 50 % at 30 fps: 0.43 s is source frame 6.45, between frames 6 and 7.
        let t = Tick::from_seconds_f64(0.43);
        let label = format!("{ID} {vals:?} {depth:?}");
        check(&label, compare_at(&s, opts(), t), 0.0);
        check(&format!("{label} half"), compare_at(&s, RenderOpts { scale: 0.5, ..opts() }, t), 0.0);
    }
}

#[test]
fn without_a_host() {
    effect_case(ID, &[]);
}

#[test]
fn interpolation_modes_and_warping() {
    for mode in 0..3 {
        moving(&[("interp/mode", e(mode))], &[]);
    }
    for warping in 0..2 {
        moving(&[("interp/warping", e(warping)), ("motion/quality", e(0))], &[]);
    }
    moving(&[("interp/smartBlend", Value::Bool(false))], &[]);
    moving(&[("motion/vectors", e(0))], &[]);
    moving(&[("motion/imagePrep", e(1)), ("motion/sensitivity", n(30.0))], &[]);
}

#[test]
fn quality_presets_and_time_remapping() {
    for q in 0..4 {
        moving(&[("motion/quality", e(q)), ("timeRemap/speed", n(37.0))], &[]);
    }
    moving(&[("timeRemap/mode", e(1)), ("timeRemap/frameNum", n(7.4))], &[]);
    moving(&[("timeRemap/speed", n(-60.0)), ("timeRemap/startFrame", n(20.0))], &[]);
}

#[test]
fn motion_blur() {
    moving(&[("motionBlur/enable", on()), ("motionBlur/samples", n(5.0)), ("timeRemap/speed", n(160.0))], &[]);
    moving(&[("motionBlur/enable", on()), ("motionBlur/shutterAngle", n(360.0)), ("interp/warping", e(1))], &[]);
}

#[test]
fn separation_motion_source_vectors_and_guides() {
    for show in [0, 5, 6, 7] {
        moving(&[("diagnostics/display", e(show))], &["guidance/separation/matteLayer"]);
    }
    moving(&[("guidance/separation/bgFill", e(1))], &["guidance/separation/matteLayer"]);
    moving(&[("guidance/separation/matteChannel", e(1)), ("motionBlur/enable", on())], &["guidance/separation/matteLayer"]);
    moving(&[], &["motion/sourceLayer"]);
    moving(&[("motion/vectors", e(2)), ("guidance/externalVectors/scale", n(8.0))], &["guidance/externalVectors/layer"]);
    moving(&[("guidance/guides/guide1/enable", on()), ("guidance/guides/guide5/enable", on()), ("guidance/guides/radius", n(25.0))], &[]);
}

#[test]
fn diagnostic_displays() {
    for d in 1..9 {
        moving(&[("diagnostics/display", e(d)), ("guidance/guides/guide1/enable", on())], &[]);
    }
}

/// The FrameWeaver kernels parse and validate against the shared prelude without a device (the
/// parity tests above skip on machines with no GPU adapter, which would otherwise leave a
/// typo in the shader unnoticed).
#[test]
fn shader_validates_and_declares_every_kernel() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    let src = [read("common.wgsl"), read("kernels.wgsl"), read("fx_frameweaver.wgsl")].concat();
    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
    let entries: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
    for k in crate::fx_frameweaver::KERNELS {
        assert!(entries.contains(k), "kernel {k} is not declared");
    }
}
