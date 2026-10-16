//! CineLooks commands: applying looks and presets, editing with undo, keyframes kept, the user
//! library, import / export, and a saved project reopening with the same grading.

use std::sync::Arc;

use effectcraft_cinelooks::look::{Look, Node};
use effectcraft_cinelooks::tools::ToolKind;
use effectcraft_project::LayerId;
use serde_json::json;

use super::*;
use crate::config::MemoryConfig;

fn session() -> (Session, u64) {
    let mut s = Session { config: Some(Arc::new(MemoryConfig::default())), ..Default::default() };
    s.execute("comp.new", json!({"name": "Main", "width": 320, "height": 180, "frameRate": 24, "duration": 4})).unwrap();
    let l = s.execute("layer.newSolid", json!({"color": "#806040", "width": 320, "height": 180})).unwrap()["layer"].as_u64().unwrap();
    (s, l)
}

fn look_of(s: &mut Session, l: u64) -> Look {
    let v = s.execute("cinelooks.getLook", json!({"layer": l})).unwrap();
    serde_json::from_value(v["look"].clone()).unwrap()
}

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("cinelooks-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).to_string_lossy().to_string()
}

#[test]
fn apply_a_preset_then_undo_and_redo() {
    let (mut s, l) = session();
    let r = s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Warm Cinema"})).unwrap();
    let uid = r["effects"][0].as_u64().unwrap();
    let look = look_of(&mut s, l);
    let preset = presets::find("builtin.warm-cinema").unwrap();
    assert_eq!(look.name, "Warm Cinema");
    assert_eq!(look.nodes.len(), preset.look.nodes.len());
    assert!(look.nodes.iter().all(|n| n.uid != 0));
    // The effect is found by its display name too (scripts address effects by name).
    assert_eq!(effectcraft_effects::lookup("CineLooks").unwrap().id, ID);
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.execute("cinelooks.getLook", json!({"layer": l})).is_err());
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(look_of(&mut s, l).nodes.len(), preset.look.nodes.len());
    // Applying another look reuses the instance.
    let r = s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Noir"})).unwrap();
    assert_eq!(r["effects"][0].as_u64(), Some(uid));
    assert_eq!(look_of(&mut s, l).name, "Noir");
}

#[test]
fn set_look_keeps_keyframes_and_is_one_undo_step() {
    let (mut s, l) = session();
    s.execute("cinelooks.apply", json!({"layers": [l], "look": Look::new("A", vec![Node::new(ToolKind::Exposure).with("exposure", 0.5)])})).unwrap();
    let look = look_of(&mut s, l);
    let cid = s.active_comp_id().unwrap();
    let fx = target(&s, &json!({"layer": l}), "t").unwrap().2;
    let node = node_uids(&s, cid, LayerId(l), fx)[0];
    // Animate exposure: 0.5 at 0 s, 1.5 at 2 s.
    let prop = s.active_comp().unwrap().layer(LayerId(l)).unwrap().props.find_group(node).unwrap().get("exposure").unwrap().uid;
    for (t, v) in [(0.0, 0.5), (2.0, 1.5)] {
        s.execute("prop.addKey", json!({"layer": l, "prop": prop, "time": t, "timeBase": "comp", "value": v})).unwrap();
    }
    // Change exposure at 1 s and add a tool, in one step.
    let mut edited = look.clone();
    edited.nodes[0] = edited.nodes[0].clone().with("exposure", 3.0);
    edited.nodes.push(Node::new(ToolKind::FilmGrain));
    let before = s.project.clone();
    s.execute("cinelooks.setLook", json!({"layer": l, "look": edited, "time": 1.0})).unwrap();
    let pr = s.active_comp().unwrap().layer(LayerId(l)).unwrap().props.find(prop).unwrap().clone();
    assert_eq!(pr.keys.len(), 3, "a key at 1 s joins the two");
    let at = |s: &mut Session, t: f64| {
        let v = s.execute("cinelooks.getLook", json!({"layer": l, "time": t})).unwrap();
        v["look"]["nodes"][0]["params"]["exposure"].as_f64().unwrap()
    };
    assert_eq!(at(&mut s, 1.0), 3.0);
    assert_eq!(at(&mut s, 0.0), 0.5);
    assert_eq!(look_of(&mut s, l).nodes.len(), 2);
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(*s.project, *before);
}

#[test]
fn add_tool_inserts_where_asked() {
    let (mut s, l) = session();
    s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Cool Shadows"})).unwrap();
    let n = look_of(&mut s, l).nodes.len();
    let r = s.execute("cinelooks.addTool", json!({"layer": l, "tool": "Halation", "index": 1, "params": {"amount": 70.0}})).unwrap();
    assert_eq!(r["index"], 1);
    let look = look_of(&mut s, l);
    assert_eq!(look.nodes.len(), n + 1);
    assert_eq!(look.nodes[0].tool, ToolKind::Halation);
    assert_eq!(look.nodes[0].num("amount"), 70.0);
    assert!(s.execute("cinelooks.addTool", json!({"layer": l, "tool": "teleporter"})).is_err());
}

#[test]
fn user_library_favorites_import_and_export() {
    let (mut s, l) = session();
    s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Golden Hour"})).unwrap();
    let id = s.execute("cinelooks.savePreset", json!({"name": "My Sunset", "category": "Mine", "layer": l})).unwrap()["id"].as_str().unwrap().to_string();
    assert_eq!(id, "user.my-sunset");
    let list = s.execute("cinelooks.presets", json!({"category": "user"})).unwrap();
    assert_eq!(list["presets"].as_array().unwrap().len(), 1);
    s.execute("cinelooks.setFavorite", json!({"id": id})).unwrap();
    s.execute("cinelooks.setFavorite", json!({"id": "builtin.noir", "value": true})).unwrap();
    let favs = s.execute("cinelooks.presets", json!({"category": "favorites"})).unwrap();
    assert_eq!(favs["presets"].as_array().unwrap().len(), 2);
    // Export / import a .cinelook.
    let path = tmp("sunset.cinelook");
    s.execute("cinelooks.exportPreset", json!({"path": path, "id": id})).unwrap();
    s.execute("cinelooks.deletePreset", json!({"id": id})).unwrap();
    assert!(s.execute("cinelooks.deletePreset", json!({"id": "builtin.noir"})).is_err());
    let r = s.execute("cinelooks.importPreset", json!({"path": path})).unwrap();
    assert_eq!(r["name"], "My Sunset");
    let imported = s.execute("cinelooks.apply", json!({"layers": [l], "preset": r["id"]})).unwrap();
    assert!(imported["effects"].as_array().is_some());
    // A .cube becomes a LUT look that keeps a copy of the table.
    let cube = tmp("teal.cube");
    std::fs::write(&cube, "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n").unwrap();
    let r = s.execute("cinelooks.importPreset", json!({"path": cube})).unwrap();
    let luts = s.execute("cinelooks.presets", json!({"category": "LUTs"})).unwrap();
    assert_eq!(luts["presets"][0]["name"], "teal");
    assert!(r["id"].as_str().unwrap().starts_with("user."));
    let bad = tmp("bad.cube");
    std::fs::write(&bad, "LUT_3D_SIZE 2\n0 0 0\n").unwrap();
    let e = s.execute("cinelooks.importPreset", json!({"path": bad})).unwrap_err().to_string();
    assert!(e.contains("promises"), "{e}");
    // The colour tools as a .cube; the spatial ones are reported.
    s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Golden Hour"})).unwrap();
    let out = tmp("golden.cube");
    let r = s.execute("cinelooks.exportCube", json!({"path": out, "layer": l, "size": 17})).unwrap();
    let skipped: Vec<String> = serde_json::from_value(r["skipped"].clone()).unwrap();
    assert!(skipped.contains(&"Optical Diffusion".to_string()) && skipped.contains(&"Graduated Exposure".to_string()));
    let text = std::fs::read_to_string(&out).unwrap();
    assert_eq!(effectcraft_cinelooks::lut::parse_cube(&text).unwrap().n3, 17);
}

#[test]
fn a_saved_project_reopens_with_the_same_grading() {
    let (mut s, l) = session();
    s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Teal Dusk", "strength": 80.0})).unwrap();
    let before = s.execute("cinelooks.getLook", json!({"layer": l})).unwrap();
    let frame = |s: &Session| effectcraft_render::render_frame(&s.project, s.active_comp_id().unwrap(), effectcraft_time::Tick::ZERO, 0.5);
    let img = frame(&s);
    let path = tmp("graded.ecproj");
    s.execute("file.saveAs", json!({"path": path})).unwrap();
    let mut t = Session::default();
    t.execute("file.open", json!({"path": path})).unwrap();
    let after = t.execute("cinelooks.getLook", json!({"layer": l})).unwrap();
    assert_eq!(before["look"], after["look"]);
    assert_eq!(after["strength"], 80.0);
    assert_eq!(frame(&t).data, img.data, "the reopened project renders the same frame");
}

#[test]
fn hostile_command_input_is_an_error_not_a_crash() {
    let (mut s, l) = session();
    assert!(s.execute("cinelooks.setLook", json!({"layer": l, "look": {}})).is_err(), "no CineLooks yet");
    s.execute("cinelooks.apply", json!({"layers": [l]})).unwrap();
    for bad in [json!("{"), json!({"nodes": [{"tool": "nope"}]}), json!(42)] {
        assert!(s.execute("cinelooks.setLook", json!({"layer": l, "look": bad})).is_err());
    }
    // Absurd values are clamped, not rejected.
    s.execute("cinelooks.setLook", json!({"layer": l, "look": {"nodes": [{"tool": "exposure", "params": {"exposure": 1e300}, "strength": -5}]}})).unwrap();
    let look = look_of(&mut s, l);
    assert_eq!(look.nodes[0].num("exposure"), 8.0);
    assert_eq!(look.nodes[0].strength, 0.0);
    assert!(s.execute("cinelooks.importPreset", json!({"path": "/no/such/file.cinelook"})).is_err());
    assert!(s.execute("cinelooks.apply", json!({"layers": [l], "preset": "no such look"})).is_err());
}

#[test]
fn the_rendered_frame_matches_the_editor_preview() {
    // A full-frame layer in a 32 bpc project: the composition frame (what export renders) equals
    // the CineLooks engine run on the effect's input (what the editor previews), grain included.
    let (mut s, l) = session();
    Arc::make_mut(&mut s.project).settings.bit_depth = effectcraft_project::BitDepth::Bpc32;
    s.execute("effect.apply", json!({"layers": [l], "effect": "Gradient Ramp"})).unwrap();
    let fx = s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Warm Cinema"})).unwrap()["effects"][0].as_u64().unwrap();
    let cid = s.active_comp_id().unwrap();
    let t = effectcraft_time::Tick::from_seconds_f64(1.25);
    let comp = s.project.comp(cid).unwrap();
    let layer = comp.layer(LayerId(l)).unwrap();
    let idx = layer.effects().unwrap().groups().position(|g| g.uid == fx).unwrap();
    let opts = effectcraft_render::RenderOpts { backend: effectcraft_render::Backend::Cpu, ..Default::default() };
    let r = effectcraft_render::Renderer::new(&s.project, s.footage.as_ref(), opts);
    let ectx = effectcraft_render::EvalCtx { project: &s.project, comp_id: cid, comp, time: t, expr: None, footage: None };
    let input = r.layer_input(&ectx, layer, idx).unwrap();
    let (look, strength, size) = read_look(&s, cid, LayerId(l), fx, t).unwrap();
    let env = Env {
        layer_size: size,
        offset: input.offset,
        scale: input.scale,
        time: layer.layer_time(t).seconds(),
        fps: comp.frame_rate.as_f64(),
        seed: fx as u32,
        working: working(&s),
    };
    let preview = effectcraft_cinelooks::render(&effectcraft_cinelooks::compile(&look, &env, (strength / 100.0) as f32), &input.img);
    let frame = r.comp_frame_cpu(cid, t);
    assert_eq!((frame.width, frame.height), (preview.width, preview.height));
    let worst = frame.data.iter().zip(&preview.data).map(|(a, b)| (0..4).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "export and preview differ by {worst}");
}
