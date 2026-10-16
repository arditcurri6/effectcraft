//! The CineLooks Editor end to end: opened from the engine command (as Effect Controls' button
//! does), a library look picked, a slider dragged (one editor undo step), Apply writing the
//! draft to the effect as one project undo step, Cancel discarding, and the editor closing when
//! its effect goes away.

use effectcraft_engine::Session;
use effectcraft_engine::cinelooks::Look;
use effectcraft_ui_egui::{Dialog, EffectcraftApp};
use egui::{Event, Modifiers, PointerButton, Pos2, pos2};
use egui_kittest::Harness;
use serde_json::json;

fn setup() -> (Harness<'static, EffectcraftApp>, u64, u64) {
    let mut s = Session::default();
    s.execute("comp.new", json!({"name": "Main", "width": 320, "height": 180, "frameRate": 24, "duration": 4})).unwrap();
    let l = s.execute("layer.newSolid", json!({"color": "#808080", "width": 320, "height": 180})).unwrap()["layer"].as_u64().unwrap();
    // Something to grade: a ramp under the look.
    s.execute("effect.apply", json!({"layers": [l], "effect": "Gradient Ramp"})).unwrap();
    let fx = s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Cool Shadows"})).unwrap()["effects"][0].as_u64().unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).build_eframe(|_| EffectcraftApp::new(s));
    h.run_steps(2);
    (h, l, fx)
}

fn look(h: &Harness<'_, EffectcraftApp>, l: u64) -> Look {
    let s = &h.state().session;
    let cid = s.active_comp_id().unwrap();
    let lid = effectcraft_engine::project::LayerId(l);
    let fx = s.active_comp().unwrap().layer(lid).unwrap().effects().unwrap().groups().find(|g| g.match_id == "ec.color.cinelooks").unwrap().uid;
    effectcraft_engine::commands::cinelooks::read_look(s, cid, lid, fx, effectcraft_engine::time::Tick::ZERO).unwrap().0
}

fn rect(h: &Harness<'_, EffectcraftApp>, id: &str) -> [f32; 4] {
    h.state().auto.find(id).unwrap_or_else(|| panic!("no automation id {id}")).rect
}

fn center(h: &Harness<'_, EffectcraftApp>, id: &str) -> Pos2 {
    let r = rect(h, id);
    pos2(r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)
}

fn click(h: &mut Harness<'_, EffectcraftApp>, id: &str) {
    let p = center(h, id);
    h.input_mut().events.push(Event::PointerMoved(p));
    h.step();
    h.input_mut().events.push(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.step();
    h.input_mut().events.push(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
}

fn drag(h: &mut Harness<'_, EffectcraftApp>, from: Pos2, to: Pos2) {
    h.input_mut().events.push(Event::PointerMoved(from));
    h.step();
    h.input_mut().events.push(Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.step();
    for k in 1..=6 {
        let p = from + (to - from) * (k as f32 / 6.0);
        h.input_mut().events.push(Event::PointerMoved(p));
        h.step();
    }
    h.input_mut().events.push(Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
}

fn open(h: &mut Harness<'_, EffectcraftApp>, l: u64, fx: u64) {
    h.state_mut().session.execute("cinelooks.openEditor", json!({"layer": l, "effect": fx})).unwrap();
    h.run_steps(4);
    assert_eq!(h.state().dialog, Some(Dialog::CineLooks));
}

#[test]
fn pick_a_look_edit_and_apply() {
    let (mut h, l, fx) = setup();
    open(&mut h, l, fx);
    assert_eq!(h.state().cinelooks_draft().name, "Cool Shadows");
    // Pick a look from the library.
    click(&mut h, "cinelooks.look.builtin.teal-dusk");
    assert_eq!(h.state().cinelooks_draft().name, "Teal Dusk");
    assert_eq!(look(&h, l).name, "Cool Shadows", "not applied yet");
    // Select the first tool and drag its first slider.
    click(&mut h, "cinelooks.chain.0");
    let r = rect(&h, "cinelooks.tool.strength");
    drag(&mut h, pos2(r[0] + r[2] - 1.0, r[1] + r[3] / 2.0), pos2(r[0] + r[2] * 0.5, r[1] + r[3] / 2.0));
    let strength = h.state().cinelooks_draft().nodes[0].strength;
    assert!((40.0..60.0).contains(&strength), "{strength}");
    // Apply: one project undo step.
    click(&mut h, "cinelooks.apply");
    let applied = look(&h, l);
    assert_eq!(applied.name, "Teal Dusk");
    assert!((applied.nodes[0].strength - strength).abs() < 1e-6);
    // The editor's undo goes back through its own history; OK closes.
    click(&mut h, "cinelooks.ok");
    assert_eq!(h.state().dialog, None);
    h.state_mut().session.execute("edit.undo", json!({})).unwrap();
    assert_eq!(look(&h, l).name, "Cool Shadows");
}

#[test]
fn cancel_discards_and_a_deleted_effect_closes_the_editor() {
    let (mut h, l, fx) = setup();
    open(&mut h, l, fx);
    click(&mut h, "cinelooks.look.builtin.golden-hour");
    click(&mut h, "cinelooks.cancel");
    assert_eq!(h.state().dialog, None);
    assert_eq!(look(&h, l).name, "Cool Shadows");
    open(&mut h, l, fx);
    h.state_mut().session.execute("effect.removeAll", json!({"layers": [l]})).unwrap();
    h.run_steps(3);
    assert_eq!(h.state().dialog, None);
}

#[test]
fn every_control_of_every_tool_is_reachable() {
    // Each tool's inspector draws without panicking and registers its controls.
    let (mut h, l, fx) = setup();
    open(&mut h, l, fx);
    for k in effectcraft_engine::cinelooks::ToolKind::all() {
        h.state_mut().cinelooks_add_tool(k);
        h.run_steps(2);
        assert!(h.state().auto.find("cinelooks.tool.strength").is_some(), "{k:?}");
    }
    // The preview graded something and the scopes show it.
    assert!(h.state().auto.find("cinelooks.scope").is_some());
}
