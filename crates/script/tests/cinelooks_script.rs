//! Scripts find CineLooks by its display name (and its id), and its tools' parameters are
//! ordinary properties scripts can read and set.

use effectcraft_engine::Session;
use serde_json::json;

#[test]
fn a_script_adds_cinelooks_by_name_and_sets_a_tool_parameter() {
    let mut s = Session::default();
    effectcraft_script::install(&mut s);
    let out = effectcraft_script::run_code(
        &mut s,
        r#"
        var comp = app.project.items.addComp("Main", 320, 180, 1, 2, 24);
        var bg = comp.layers.addSolid([0.5, 0.4, 0.3], "BG", 320, 180, 1);
        var fx = bg.Effects.addProperty("CineLooks");
        var byId = bg.Effects.addProperty("ec.color.cinelooks");
        [fx.name, byId.name, bg.Effects.numProperties]
        "#,
        "cinelooks.jsx",
    );
    assert!(out.error.is_none(), "{:?}", out.error);
    assert_eq!(out.result, json!(["CineLooks", "CineLooks 2", 2]));
    // Give the first instance a look, then read and set a tool parameter from a script.
    let l = s.active_comp().unwrap().layers[0].id.0;
    s.execute("cinelooks.apply", json!({"layers": [l], "preset": "Noir"})).unwrap();
    let out = effectcraft_script::run_code(
        &mut s,
        r#"
        var fx = app.project.activeItem.layer(1).Effects.property("CineLooks");
        var contrast = fx.property("contrast");
        contrast.property("contrast").setValue(10);
        [fx.numProperties, contrast.property("contrast").value, contrast.property("strength").value]
        "#,
        "cinelooks2.jsx",
    );
    assert!(out.error.is_none(), "{:?}", out.error);
    let r = out.result.as_array().unwrap();
    assert_eq!(r[1].as_f64(), Some(10.0));
    assert_eq!(r[2].as_f64(), Some(100.0));
}
