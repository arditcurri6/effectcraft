//! The editor's own controls: a labelled slider with a typed value and reset, a colour wheel
//! with a master slider, and a curve graph drawn and edited with the engine's own spline (so
//! what you see is what renders).

use effectcraft_engine::cinelooks::curve::Points;
use egui::{Align2, Color32, Pos2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};

use super::aid;
use crate::EffectcraftApp;
use crate::theme::Tokens;

/// What a control did this frame.
pub struct Changed<T> {
    pub value: T,
    /// Part of a continuous drag (one undo step for the gesture).
    pub dragging: bool,
}

/// `label  [track]  value`: drag the track or scrub / type the value; double-click the label to
/// reset to `default`. `range` is the hard range, `soft` the track's.
#[allow(clippy::too_many_arguments)]
pub fn slider(
    app: &mut EffectcraftApp,
    ui: &mut egui::Ui,
    r: Rect,
    id: &str,
    label: &str,
    value: f64,
    range: (f64, f64),
    soft: (f64, f64),
    dec: usize,
    default: f64,
) -> Option<Changed<f64>> {
    let t = app.tokens;
    let p = ui.painter().clone();
    // Narrow sliders (a wheel's master) drop the label area.
    let label_w = if label.is_empty() { 0.0 } else { (r.width() * 0.42).clamp(90.0, 150.0) };
    let lr = Rect::from_min_size(r.min, vec2(label_w, r.height()));
    let lresp = ui.interact(lr, egui::Id::new(("cl-sl-label", id)), Sense::click());
    let modified = (value - default).abs() > 1e-9;
    if !label.is_empty() {
        crate::widgets::text_fit(
            &p,
            pos2(lr.min.x, lr.center().y),
            Align2::LEFT_CENTER,
            label,
            Tokens::ui(12.0),
            label_w - 6.0,
            if modified { t.text } else { t.text_dim },
        );
    }
    let lresp = lresp.on_hover_text("Double-click to reset");
    app.auto.add(&aid(&format!("{id}.label")), lr, label);
    let mut out = None;
    if lresp.double_clicked() && modified {
        out = Some(Changed { value: default, dragging: false });
    }
    let num_w = if r.width() < 160.0 { 40.0 } else { 62.0 };
    let track = Rect::from_min_max(pos2(lr.max.x + 4.0, r.center().y - 3.0), pos2(r.max.x - num_w - 8.0, r.center().y + 3.0));
    let (lo, hi) = (soft.0.max(range.0), soft.1.min(range.1));
    let span = (hi - lo).max(1e-12);
    let fx = |v: f64| track.min.x + track.width() * ((v.clamp(lo, hi) - lo) / span) as f32;
    p.rect_filled(track, 3.0, t.field_bg);
    let zero = if (lo..=hi).contains(&0.0) && lo < 0.0 { 0.0 } else { default.clamp(lo, hi) };
    let (a, b) = (fx(zero).min(fx(value)), fx(zero).max(fx(value)));
    p.rect_filled(Rect::from_min_max(pos2(a, track.min.y), pos2(b, track.max.y)), 3.0, t.accent.gamma_multiply(0.75));
    let tresp = ui.interact(track.expand2(vec2(4.0, 7.0)), egui::Id::new(("cl-sl-track", id)), Sense::click_and_drag());
    app.auto.add(&aid(id), track, label);
    let knob = pos2(fx(value), track.center().y);
    p.circle_filled(knob, if tresp.hovered() || tresp.dragged() { 6.5 } else { 5.5 }, Color32::from_gray(230));
    if (tresp.dragged() || tresp.clicked())
        && let Some(pp) = tresp.interact_pointer_pos()
    {
        let fine = ui.input(|i| i.modifiers.shift);
        let v = if fine && tresp.dragged() {
            value + tresp.drag_delta().x as f64 / track.width().max(1.0) as f64 * span * 0.1
        } else {
            lo + ((pp.x - track.min.x) / track.width().max(1.0)).clamp(0.0, 1.0) as f64 * span
        };
        let step = 10f64.powi(-(dec as i32));
        let v = ((v / step).round() * step).clamp(range.0, range.1);
        out = Some(Changed { value: v, dragging: tresp.dragged() });
    }
    let (nr, nv, _) =
        crate::widgets::hot_number_at(ui, pos2(r.max.x - num_w, r.center().y - 9.0), egui::Id::new(("cl-sl-num", id)), value, span / 300.0, range, dec, "", &t);
    app.auto.add(&aid(&format!("{id}.value")), nr, label);
    if let Some(v) = nv {
        out = Some(Changed { value: v, dragging: ui.input(|i| i.pointer.any_down()) });
    }
    out
}

fn hue_color(deg: f32, s: f32, v: f32) -> Color32 {
    let (r, g, b) = effectcraft_engine::color::hsv_to_rgb(deg.rem_euclid(360.0) / 360.0, s, v);
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// A colour wheel for `[x, y, master]` in `r` (square-ish area including the master slider and
/// the label). Drag the puck (relative), click to jump, double-click to reset the colour.
pub fn wheel(app: &mut EffectcraftApp, ui: &mut egui::Ui, r: Rect, id: &str, label: &str, v: [f64; 3]) -> Option<Changed<[f64; 3]>> {
    let t = app.tokens;
    let p = ui.painter().clone();
    p.text(pos2(r.center().x, r.min.y + 8.0), Align2::CENTER_CENTER, label, Tokens::medium(12.0), t.text);
    let disc_h = r.height() - 44.0;
    let rad = (disc_h.min(r.width()) * 0.5 - 4.0).max(10.0);
    let c = pos2(r.center().x, r.min.y + 18.0 + rad);
    // Disc: a fan from grey to the hue at the rim; a thin saturated ring outside.
    let n: u32 = 72;
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(c, Color32::from_gray(70));
    for k in 0..=n {
        let a = k as f32 / n as f32 * std::f32::consts::TAU;
        let deg = a.to_degrees();
        mesh.colored_vertex(c + vec2(a.cos(), -a.sin()) * rad, hue_color(deg, 0.55, 0.55));
    }
    for k in 0..n {
        mesh.add_triangle(0, k + 1, k + 2);
    }
    let base = mesh.vertices.len() as u32;
    for k in 0..=n {
        let a = k as f32 / n as f32 * std::f32::consts::TAU;
        let col = hue_color(a.to_degrees(), 1.0, 1.0);
        mesh.colored_vertex(c + vec2(a.cos(), -a.sin()) * rad, col);
        mesh.colored_vertex(c + vec2(a.cos(), -a.sin()) * (rad + 4.0), col);
    }
    for k in 0..n {
        let i = base + 2 * k;
        mesh.add_triangle(i, i + 1, i + 2);
        mesh.add_triangle(i + 1, i + 3, i + 2);
    }
    p.add(egui::Shape::mesh(mesh));
    p.line_segment([c - vec2(rad, 0.0), c + vec2(rad, 0.0)], Stroke::new(1.0, Color32::from_white_alpha(25)));
    p.line_segment([c - vec2(0.0, rad), c + vec2(0.0, rad)], Stroke::new(1.0, Color32::from_white_alpha(25)));
    let puck = c + vec2(v[0] as f32, -(v[1] as f32)) * rad;
    let disc = Rect::from_center_size(c, vec2(rad * 2.0 + 8.0, rad * 2.0 + 8.0));
    let resp = ui.interact(disc, egui::Id::new(("cl-wheel", id)), Sense::click_and_drag());
    app.auto.add(&aid(id), disc, label);
    let hot = resp.hovered() || resp.dragged();
    p.circle_filled(puck, 5.0, Color32::from_black_alpha(120));
    p.circle_stroke(puck, 5.0, Stroke::new(2.0, if hot { t.accent_hover } else { Color32::WHITE }));
    let mut out = None;
    if resp.double_clicked() {
        out = Some(Changed { value: [0.0, 0.0, v[2]], dragging: false });
    } else if resp.dragged() {
        // Relative and finer than the pointer (Shift: finer still).
        let k = if ui.input(|i| i.modifiers.shift) { 0.15 } else { 0.6 };
        let d = resp.drag_delta() * k / rad;
        let (mut x, mut y) = (v[0] + d.x as f64, v[1] - d.y as f64);
        let m = (x * x + y * y).sqrt();
        if m > 1.0 {
            x /= m;
            y /= m;
        }
        out = Some(Changed { value: [x, y, v[2]], dragging: true });
    } else if resp.clicked()
        && let Some(pp) = resp.interact_pointer_pos()
    {
        let d = (pp - c) / rad;
        let (mut x, mut y) = (d.x as f64, -d.y as f64);
        let m = (x * x + y * y).sqrt();
        if m > 1.0 {
            x /= m;
            y /= m;
        }
        out = Some(Changed { value: [x, y, v[2]], dragging: false });
    }
    // Master below.
    let mr = Rect::from_min_max(pos2(r.min.x + 2.0, r.max.y - 20.0), pos2(r.max.x - 2.0, r.max.y - 2.0));
    if let Some(m) = slider(app, ui, mr, &format!("{id}.master"), "", v[2], (-1.0, 1.0), (-1.0, 1.0), 2, 0.0) {
        let cur = out.as_ref().map_or(v, |o| o.value);
        out = Some(Changed { value: [cur[0], cur[1], m.value], dragging: m.dragging });
    }
    out
}

/// A curve graph for `text` (`"x,y …"`), drawn in `col`. Click adds a point, drag moves one,
/// double-click or dragging it far off the graph removes it.
pub fn curve(app: &mut EffectcraftApp, ui: &mut egui::Ui, r: Rect, id: &str, text: &str, col: Color32) -> Option<Changed<String>> {
    let t = app.tokens;
    let p = ui.painter().with_clip_rect(r.expand(8.0));
    p.rect_filled(r, 3.0, Color32::from_gray(26));
    for k in 1..4 {
        let f = k as f32 / 4.0;
        p.line_segment([pos2(r.min.x + r.width() * f, r.min.y), pos2(r.min.x + r.width() * f, r.max.y)], Stroke::new(1.0, Color32::from_white_alpha(16)));
        p.line_segment([pos2(r.min.x, r.min.y + r.height() * f), pos2(r.max.x, r.min.y + r.height() * f)], Stroke::new(1.0, Color32::from_white_alpha(16)));
    }
    p.line_segment([r.left_bottom(), r.right_top()], Stroke::new(1.0, Color32::from_white_alpha(30)));
    p.rect_stroke(r, 3.0, Stroke::new(1.0, t.separator), StrokeKind::Inside);
    let mut pts = Points::parse(text);
    let to_screen = |q: [f32; 2]| pos2(r.min.x + q[0] * r.width(), r.max.y - q[1] * r.height());
    let to_graph = |s: Pos2| [((s.x - r.min.x) / r.width().max(1.0)), ((r.max.y - s.y) / r.height().max(1.0))];
    let line: Vec<Pos2> = (0..=96).map(|k| k as f32 / 96.0).map(|x| to_screen([x, pts.eval(x)])).collect();
    p.add(egui::Shape::line(line, Stroke::new(2.0, col)));
    let drag_id = egui::Id::new(("cl-curve-drag", id));
    let mut dragging: Option<usize> = ui.data(|d| d.get_temp(drag_id));
    let resp = ui.interact(r.expand(6.0), egui::Id::new(("cl-curve", id)), Sense::click_and_drag());
    app.auto.add(&aid(id), r, "Curve");
    for (i, q) in pts.0.iter().enumerate() {
        let s = to_screen(*q);
        let pr = Rect::from_center_size(s, vec2(10.0, 10.0));
        app.auto.add(&aid(&format!("{id}.point.{i}")), pr, &format!("Point {}", i + 1));
        let hot = dragging == Some(i) || resp.hover_pos().is_some_and(|h| h.distance(s) < 7.0);
        p.circle_filled(s, if hot { 5.5 } else { 4.5 }, if hot { Color32::WHITE } else { col });
        p.circle_stroke(s, 4.5, Stroke::new(1.0, Color32::from_black_alpha(160)));
    }
    let hit = |h: Pos2, pts: &Points| pts.0.iter().position(|q| to_screen(*q).distance(h) < 8.0);
    let mut changed = false;
    if resp.double_clicked()
        && let Some(h) = resp.interact_pointer_pos()
        && let Some(i) = hit(h, &pts)
    {
        changed = pts.remove(i);
        dragging = None;
    } else if resp.drag_started()
        && let Some(h) = resp.interact_pointer_pos()
    {
        dragging = Some(match hit(h, &pts) {
            Some(i) => i,
            None => {
                let g = to_graph(h);
                changed = true;
                pts.insert(g[0], g[1])
            }
        });
    } else if resp.dragged()
        && let (Some(i), Some(h)) = (dragging, resp.interact_pointer_pos())
    {
        let far = !r.expand(40.0).contains(h);
        let is_end = i == 0 || i + 1 == pts.0.len();
        if far && !is_end {
            changed = pts.remove(i);
            dragging = None;
        } else {
            let g = to_graph(h);
            pts.move_point(i, g[0].clamp(0.0, 1.0), g[1].clamp(0.0, 1.0));
            changed = true;
        }
    } else if resp.clicked()
        && let Some(h) = resp.interact_pointer_pos()
        && hit(h, &pts).is_none()
    {
        let g = to_graph(h);
        pts.insert(g[0], g[1]);
        changed = true;
    }
    if resp.drag_stopped() {
        dragging = None;
    }
    ui.data_mut(|d| match dragging {
        Some(i) => {
            d.insert_temp(drag_id, i);
        }
        None => {
            d.remove::<usize>(drag_id);
        }
    });
    changed.then(|| Changed { value: pts.format(), dragging: resp.dragged() })
}
