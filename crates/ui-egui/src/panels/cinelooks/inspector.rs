//! The right sidebar: the selected tool's controls. Colour Wheels get wheels, Curves a curve
//! graph per channel, HSL its eight colour ranges; every other parameter a row of its kind
//! (slider, angle, colour, choice, switch, position, LUT file). Double-click a label to reset
//! that parameter; Reset Tool resets them all.

use effectcraft_engine::cinelooks::PV;
use effectcraft_engine::cinelooks::look::Node;
use effectcraft_engine::cinelooks::tools::{Def, HSL_RANGES, Kind, ParamDef, ToolKind};
use egui::{Align2, Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};

use super::widgets::{Changed, curve, slider, wheel};
use super::{aid, button};
use crate::EffectcraftApp;
use crate::theme::Tokens;

const ROW: f32 = 26.0;

/// Apply a control's change to node `i`'s parameter `id`.
fn commit(app: &mut EffectcraftApp, i: usize, id: &str, c: Changed<PV>) {
    let st = &mut app.dialog_state.cinelooks;
    let f = |l: &mut effectcraft_engine::cinelooks::Look, _: &mut f64| {
        if let Some(n) = l.nodes.get_mut(i) {
            n.params.insert(id.to_string(), c.value.clone());
        }
    };
    if c.dragging { st.gesture_edit(f) } else { st.edit(f) }
}

fn def_num(d: &ParamDef) -> f64 {
    match d.def {
        Def::Num(v) => v,
        Def::Choice(v) => v as f64,
        _ => 0.0,
    }
}

pub fn show(app: &mut EffectcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    let p = ui.painter().with_clip_rect(rect);
    p.rect_filled(rect, 6.0, t.panel_bg);
    let st = &app.dialog_state.cinelooks;
    let (Some(i), Some(node)) = (st.selected, st.node().cloned()) else {
        p.text(rect.center(), Align2::CENTER_CENTER, "Select a tool in the chain, or add one.", Tokens::ui(12.0), t.text_dim);
        return;
    };
    let def = node.tool.def();
    // Header.
    let hx = rect.min.x + 12.0;
    p.text(pos2(hx, rect.min.y + 18.0), Align2::LEFT_CENTER, node.name(), Tokens::semibold(15.0), t.tab_text_active);
    let on = super::super::panel_kit::checkbox(app, ui, pos2(rect.max.x - 150.0, rect.min.y + 10.0), &aid("tool.enabled"), "On", node.enabled);
    if on != node.enabled {
        app.dialog_state.cinelooks.edit(|l, _| {
            if let Some(n) = l.nodes.get_mut(i) {
                n.enabled = on;
            }
        });
    }
    let rr = Rect::from_min_size(pos2(rect.max.x - 92.0, rect.min.y + 6.0), vec2(80.0, 24.0));
    if button(app, ui, rr, "tool.reset", "Reset Tool", false) {
        app.dialog_state.cinelooks.edit(|l, _| {
            if let Some(n) = l.nodes.get_mut(i) {
                n.params.clear();
                n.strength = 100.0;
            }
        });
    }
    let about_r = Rect::from_min_max(pos2(hx, rect.min.y + 32.0), pos2(rect.max.x - 12.0, rect.min.y + 66.0));
    let galley = ui.painter().layout(def.about.to_string(), Tokens::ui(11.5), t.text_dim, about_r.width());
    let about_h = galley.size().y.min(34.0);
    p.galley(about_r.min, galley, t.text_dim);
    let sr = Rect::from_min_size(pos2(hx, about_r.min.y + about_h + 6.0), vec2(rect.width() - 24.0, ROW));
    if let Some(c) = slider(app, ui, sr, "tool.strength", "Strength", node.strength, (0.0, 100.0), (0.0, 100.0), 1, 100.0) {
        let st = &mut app.dialog_state.cinelooks;
        let f = |l: &mut effectcraft_engine::cinelooks::Look, _: &mut f64| {
            if let Some(n) = l.nodes.get_mut(i) {
                n.strength = c.value;
            }
        };
        if c.dragging { st.gesture_edit(f) } else { st.edit(f) }
    }
    ui.painter().line_segment([pos2(rect.min.x + 8.0, sr.max.y + 6.0), pos2(rect.max.x - 8.0, sr.max.y + 6.0)], Stroke::new(1.0, t.separator));
    let body = Rect::from_min_max(pos2(rect.min.x, sr.max.y + 10.0), rect.max);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).layout(egui::Layout::top_down(egui::Align::Min)));
    child.set_clip_rect(body.intersect(ui.clip_rect()));
    egui::ScrollArea::vertical().id_salt(("cl-insp", i, node.uid)).auto_shrink([false, false]).show(&mut child, |ui| {
        ui.add_space(4.0);
        let w = ui.available_width() - 16.0;
        match node.tool {
            ToolKind::Wheels => wheels(app, ui, i, &node, w),
            ToolKind::Curves => curves(app, ui, i, &node, w),
            ToolKind::Hsl => hsl(app, ui, i, &node, w),
            _ => {
                for d in def.params {
                    row(app, ui, i, &node, d, w);
                }
            }
        }
        ui.add_space(12.0);
    });
}

/// Allocate a row of height `h` (inset) and return its rect.
fn alloc(ui: &mut egui::Ui, w: f32, h: f32) -> Rect {
    let (r, _) = ui.allocate_exact_size(vec2(w + 16.0, h), Sense::hover());
    Rect::from_min_max(pos2(r.min.x + 12.0, r.min.y), pos2(r.max.x - 4.0, r.max.y))
}

fn row(app: &mut EffectcraftApp, ui: &mut egui::Ui, i: usize, node: &Node, d: &ParamDef, w: f32) {
    let t = app.tokens;
    let key = format!("param.{}", d.id);
    match d.kind {
        Kind::Num { min, max, smin, smax, dec } => {
            let r = alloc(ui, w, ROW);
            if let Some(c) = slider(app, ui, r, &key, d.name, node.num(d.id), (min, max), (smin, smax), dec as usize, def_num(d)) {
                commit(app, i, d.id, Changed { value: PV::Num(c.value), dragging: c.dragging });
            }
        }
        Kind::Angle => {
            let r = alloc(ui, w, ROW);
            let v = node.num(d.id).rem_euclid(360.0);
            if let Some(c) = slider(app, ui, r, &key, d.name, v, (0.0, 360.0), (0.0, 360.0), 1, def_num(d)) {
                commit(app, i, d.id, Changed { value: PV::Num(c.value), dragging: c.dragging });
            }
            // A hue strip under hue angles.
            let strip = Rect::from_min_max(pos2(r.min.x + (r.width() * 0.42).clamp(90.0, 150.0) + 4.0, r.max.y - 3.0), pos2(r.max.x - 70.0, r.max.y - 1.0));
            for k in 0..36 {
                let a = Rect::from_min_max(
                    pos2(strip.min.x + strip.width() * k as f32 / 36.0, strip.min.y),
                    pos2(strip.min.x + strip.width() * (k + 1) as f32 / 36.0, strip.max.y),
                );
                let (rr, gg, bb) = effectcraft_engine::color::hsv_to_rgb(k as f32 / 36.0, 0.8, 0.9);
                ui.painter().rect_filled(a, 0.0, Color32::from_rgb((rr * 255.0) as u8, (gg * 255.0) as u8, (bb * 255.0) as u8));
            }
        }
        Kind::Color => {
            let r = alloc(ui, w, ROW);
            let p = ui.painter().clone();
            p.text(pos2(r.min.x, r.center().y), Align2::LEFT_CENTER, d.name, Tokens::ui(12.0), t.text_dim);
            let c = node.color64(d.id);
            let mut rgb = [c[0] as f32, c[1] as f32, c[2] as f32];
            let sw = Rect::from_min_size(pos2(r.min.x + (r.width() * 0.42).clamp(90.0, 150.0) + 4.0, r.center().y - 8.0), vec2(44.0, 16.0));
            let pop = egui::Id::new(("cl-color", i, d.id));
            if crate::widgets::swatch(ui, sw, [rgb[0], rgb[1], rgb[2], 1.0], egui::Id::new(("cl-sw", i, d.id)), &t).clicked() {
                crate::widgets::open_popup(ui, pop);
            }
            app.auto.add(&aid(&key), sw, d.name);
            if crate::widgets::color_popup(ui, pop, sw.left_bottom(), &mut rgb) {
                commit(app, i, d.id, Changed { value: PV::List(vec![rgb[0] as f64, rgb[1] as f64, rgb[2] as f64, 1.0]), dragging: false });
            }
            let rr = Rect::from_min_size(pos2(sw.max.x + 8.0, r.center().y - 9.0), vec2(48.0, 18.0));
            if button(app, ui, rr, &format!("{key}.reset"), "Reset", false) {
                let mut n = node.clone();
                n.params.remove(d.id);
                commit(app, i, d.id, Changed { value: PV::from_def(d.def), dragging: false });
            }
        }
        Kind::Choice(opts) => {
            let r = alloc(ui, w, ROW);
            ui.painter().text(pos2(r.min.x, r.center().y), Align2::LEFT_CENTER, d.name, Tokens::ui(12.0), t.text_dim);
            let x0 = r.min.x + (r.width() * 0.42).clamp(90.0, 150.0) + 4.0;
            let dr = Rect::from_min_max(pos2(x0, r.center().y - 11.0), pos2(r.max.x - 4.0, r.center().y + 11.0));
            if let Some(k) = super::super::panel_kit::dropdown(app, ui, dr, &aid(&key), opts, node.choice(d.id) as usize) {
                commit(app, i, d.id, Changed { value: PV::Num(k as f64), dragging: false });
            }
        }
        Kind::Toggle => {
            let r = alloc(ui, w, ROW);
            let on = super::super::panel_kit::checkbox(app, ui, pos2(r.min.x, r.center().y - 8.0), &aid(&key), d.name, node.flag(d.id));
            if on != node.flag(d.id) {
                commit(app, i, d.id, Changed { value: PV::Bool(on), dragging: false });
            }
        }
        Kind::Point => {
            let r = alloc(ui, w, ROW);
            ui.painter().text(pos2(r.min.x, r.center().y), Align2::LEFT_CENTER, d.name, Tokens::ui(12.0), t.text_dim);
            let f = node.point(d.id);
            let x0 = r.min.x + (r.width() * 0.42).clamp(90.0, 150.0) + 4.0;
            let mut nf = f;
            for (k, axis) in ["X", "Y"].iter().enumerate() {
                let at = pos2(x0 + k as f32 * 74.0, r.center().y - 9.0);
                ui.painter().text(at + vec2(0.0, 9.0), Align2::LEFT_CENTER, *axis, Tokens::ui(11.0), t.text_faint);
                let (hr, nv, _) = crate::widgets::hot_number_at(
                    ui,
                    at + vec2(12.0, 0.0),
                    egui::Id::new(("cl-pt", i, d.id, k)),
                    f[k] * 100.0,
                    0.2,
                    (-200.0, 300.0),
                    1,
                    "%",
                    &t,
                );
                app.auto.add(&aid(&format!("{key}.{}", axis.to_lowercase())), hr, d.name);
                if let Some(v) = nv {
                    nf[k] = v / 100.0;
                }
            }
            if nf != f {
                commit(app, i, d.id, Changed { value: PV::List(nf.to_vec()), dragging: ui.input(|i| i.pointer.any_down()) });
            }
            let r2 = alloc(ui, w, 16.0);
            ui.painter().text(pos2(r2.min.x, r2.center().y), Align2::LEFT_CENTER, "Drag the handle on the preview", Tokens::ui(10.5), t.text_faint);
        }
        Kind::File(_) => lut_file(app, ui, i, node, w),
        Kind::Curve | Kind::Data | Kind::Wheel => {}
    }
}

fn lut_file(app: &mut EffectcraftApp, ui: &mut egui::Ui, i: usize, node: &Node, w: f32) {
    let t = app.tokens;
    let path = node.text("file");
    let data = node.text("data");
    let r = alloc(ui, w, ROW);
    let name = std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let shown = if name.is_empty() { if data.is_empty() { "No LUT".to_string() } else { "Embedded LUT".to_string() } } else { name };
    crate::widgets::text_fit(ui.painter(), pos2(r.min.x, r.center().y), Align2::LEFT_CENTER, &shown, Tokens::medium(12.0), r.width() - 100.0, t.text);
    let br = Rect::from_min_size(pos2(r.max.x - 92.0, r.center().y - 11.0), vec2(88.0, 22.0));
    if button(app, ui, br, "lut.choose", "Choose…", true) {
        let ctx = ui.ctx().clone();
        if let Err(e) = crate::menus::invoke(app, &ctx, "cinelooks.chooseLut", serde_json::json!({"node": i})) {
            app.dialog_state.cinelooks.message = e;
        }
    }
    let r2 = alloc(ui, w, 18.0);
    let status = match effectcraft_engine::cinelooks::lut::load(&path, &data) {
        Ok(l) => {
            let kind = match (l.n1 >= 2, l.n3 >= 2) {
                (true, true) => format!("1D {} + 3D {}³", l.n1, l.n3),
                (true, false) => format!("1D {}", l.n1),
                _ => format!("3D {}³", l.n3),
            };
            let missing = !path.is_empty() && !std::path::Path::new(&path).exists();
            (
                format!(
                    "{kind}{}{}",
                    if l.title.is_empty() { String::new() } else { format!(" · {}", l.title) },
                    if missing { " · file moved: using the embedded copy" } else { "" }
                ),
                t.text_dim,
            )
        }
        Err(e) => (e, t.warning),
    };
    crate::widgets::text_fit(ui.painter(), pos2(r2.min.x, r2.center().y), Align2::LEFT_CENTER, &status.0, Tokens::ui(11.0), r2.width(), status.1);
}

/// `cinelooks.chooseLut {node, path}`: the LUT picked for node `node` (validated; the table is
/// embedded so the project keeps it).
pub fn lut_chosen(app: &mut EffectcraftApp, p: &serde_json::Value) -> Result<serde_json::Value, String> {
    let path = p.get("path").and_then(serde_json::Value::as_str).ok_or("missing `path`")?.to_string();
    let i = p.get("node").and_then(serde_json::Value::as_u64).map(|v| v as usize).or(app.dialog_state.cinelooks.selected).ok_or("no LUT tool selected")?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("can't read {path}: {e}"))?;
    effectcraft_engine::cinelooks::lut::parse_cube(&text).map_err(|e| format!("{path}: {e}"))?;
    let st = &mut app.dialog_state.cinelooks;
    if st.draft.nodes.get(i).is_none_or(|n| n.tool != ToolKind::Lut) {
        return Err("the selected tool is not a LUT".into());
    }
    st.edit(|l, _| {
        if let Some(n) = l.nodes.get_mut(i) {
            n.params.insert("file".into(), PV::Text(path.clone()));
            n.params.insert("data".into(), PV::Text(text.clone()));
        }
    });
    Ok(serde_json::json!({"node": i, "path": path}))
}

fn wheels(app: &mut EffectcraftApp, ui: &mut egui::Ui, i: usize, node: &Node, w: f32) {
    let t = app.tokens;
    let rows: [&[(&str, &str)]; 3] = [
        &[("lift", "Lift"), ("gamma", "Gamma"), ("gain", "Gain")],
        &[("offset", "Offset")],
        &[("shadows", "Shadows"), ("midtones", "Midtones"), ("highlights", "Highlights")],
    ];
    for (k, row) in rows.iter().enumerate() {
        if k == 2 {
            let r = alloc(ui, w, 18.0);
            ui.painter().text(pos2(r.min.x, r.center().y), Align2::LEFT_CENTER, "Tonal Balance", Tokens::medium(11.5), t.text_dim);
        }
        let cell_w = (w / 3.0).min(120.0);
        let r = alloc(ui, w, cell_w + 46.0);
        let x0 = r.center().x - cell_w * row.len() as f32 * 0.5;
        for (j, (id, name)) in row.iter().enumerate() {
            let cr = Rect::from_min_size(pos2(x0 + j as f32 * cell_w, r.min.y), vec2(cell_w - 6.0, r.height()));
            if let Some(c) = wheel(app, ui, cr, &format!("param.{id}"), name, node.wheel64(id)) {
                commit(app, i, id, Changed { value: PV::List(c.value.to_vec()), dragging: c.dragging });
            }
        }
    }
}

const CHANNELS: [(&str, &str, [u8; 3]); 4] =
    [("master", "RGB", [230, 230, 230]), ("red", "Red", [235, 80, 80]), ("green", "Green", [90, 210, 110]), ("blue", "Blue", [90, 140, 255])];

fn curves(app: &mut EffectcraftApp, ui: &mut egui::Ui, i: usize, node: &Node, w: f32) {
    let t = app.tokens;
    let r = alloc(ui, w, 26.0);
    let cur = app.dialog_state.cinelooks.curve_channel.min(3);
    let mut x = r.min.x;
    for (k, (id, name, c)) in CHANNELS.iter().enumerate() {
        let br = Rect::from_min_size(pos2(x, r.min.y), vec2(56.0, 22.0));
        if button(app, ui, br, &format!("curves.{id}"), name, k == cur) {
            app.dialog_state.cinelooks.curve_channel = k;
        }
        let modified = node.params.get(*id).is_some_and(|v| *v != PV::Text("0,0 1,1".into()));
        if modified {
            ui.painter().circle_filled(br.right_top() + vec2(-5.0, 5.0), 3.0, Color32::from_rgb(c[0], c[1], c[2]));
        }
        x += 60.0;
    }
    let (id, _, c) = CHANNELS[cur];
    let side = w.min(300.0);
    let gr = alloc(ui, w, side);
    let gr = Rect::from_min_size(pos2(gr.center().x - side * 0.5 + 4.0, gr.min.y), vec2(side - 8.0, side - 8.0));
    if let Some(ch) = curve(app, ui, gr, &format!("curves.{id}.graph"), &node.text(id), Color32::from_rgb(c[0], c[1], c[2])) {
        commit(app, i, id, Changed { value: PV::Text(ch.value), dragging: ch.dragging });
    }
    let r = alloc(ui, w, 30.0);
    let br = Rect::from_min_size(pos2(r.min.x, r.min.y + 4.0), vec2(96.0, 22.0));
    if button(app, ui, br, "curves.reset", "Reset Curve", false) {
        commit(app, i, id, Changed { value: PV::Text("0,0 1,1".into()), dragging: false });
    }
    ui.painter().text(pos2(br.max.x + 8.0, br.center().y), Align2::LEFT_CENTER, "Click to add · double-click to remove", Tokens::ui(10.5), t.text_faint);
}

fn hsl(app: &mut EffectcraftApp, ui: &mut egui::Ui, i: usize, node: &Node, w: f32) {
    let t = app.tokens;
    let def = node.tool.def();
    for id in ["hue", "saturation", "lightness"] {
        if let Some(d) = def.param(id) {
            row(app, ui, i, node, d, w);
        }
    }
    let r = alloc(ui, w, 22.0);
    ui.painter().text(pos2(r.min.x, r.center().y), Align2::LEFT_CENTER, "Color Ranges", Tokens::medium(11.5), t.text_dim);
    let sel_id = egui::Id::new(("cl-hsl-range", i));
    let mut sel: usize = ui.data(|d| d.get_temp(sel_id)).unwrap_or(0);
    let r = alloc(ui, w, 26.0);
    let cw = (r.width() / 8.0).min(36.0);
    for (k, (rid, name, deg)) in HSL_RANGES.iter().enumerate() {
        let sr = Rect::from_min_size(pos2(r.min.x + k as f32 * cw, r.min.y + 2.0), vec2(cw - 4.0, 20.0));
        let (rr, gg, bb) = effectcraft_engine::color::hsv_to_rgb(deg / 360.0, 0.75, 0.9);
        ui.painter().rect_filled(sr, 3.0, Color32::from_rgb((rr * 255.0) as u8, (gg * 255.0) as u8, (bb * 255.0) as u8));
        let used = ["Hue", "Sat", "Lum"].iter().any(|s| node.num(&format!("{rid}{s}")) != 0.0);
        if k == sel {
            ui.painter().rect_stroke(sr.expand(2.0), 4.0, Stroke::new(2.0, Color32::WHITE), StrokeKind::Outside);
        }
        if used {
            ui.painter().circle_filled(sr.center_bottom() + vec2(0.0, -4.0), 2.5, Color32::BLACK);
        }
        let resp = ui.interact(sr, egui::Id::new(("cl-hsl-sw", i, k)), Sense::click()).on_hover_text(*name);
        app.auto.add(&aid(&format!("hsl.range.{rid}")), sr, name);
        if resp.clicked() {
            sel = k;
        }
    }
    ui.data_mut(|d| d.insert_temp(sel_id, sel));
    if let Some((rid, _, _)) = HSL_RANGES.get(sel) {
        for s in ["Hue", "Sat", "Lum"] {
            let id = format!("{rid}{s}");
            if let Some(d) = def.param(&id) {
                row(app, ui, i, node, d, w);
            }
        }
    }
}
