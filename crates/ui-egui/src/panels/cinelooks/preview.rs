//! The preview: the layer's input to the CineLooks effect (the layer with the effects above it,
//! rendered by the compositor at a preview scale), graded by the CineLooks engine exactly as the
//! effect renders it (same program, seed, time and working space), then converted for display
//! as the Composition panel converts its frames. Compare modes, zoom and pan, a clipping
//! overlay, and on-canvas handles for the selected tool's positions.

use std::collections::HashMap;
use std::sync::Arc;

use effectcraft_engine::cinelooks::look::Look;
use effectcraft_engine::cinelooks::tools::{Kind, ToolKind};
use effectcraft_engine::cinelooks::{Env, PV};
use effectcraft_engine::effects::Buf;
use effectcraft_engine::project::{ItemId, LayerId};
use effectcraft_engine::raster::Image;
use effectcraft_engine::render::{EvalCtx, RenderOpts, Renderer};
use effectcraft_engine::time::Tick;
use egui::{Align2, Color32, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, pos2, vec2};

use super::{Compare, EditorState, Target, aid, button};
use crate::EffectcraftApp;
use crate::theme::Tokens;

/// What the preview has computed (textures and images).
#[derive(Default)]
pub struct Cache {
    input_key: Option<(u64, u64, u64, i64)>,
    pub input: Option<Arc<Buf>>,
    /// Bumped when the input changes.
    pub input_gen: u64,
    pub layer_size: [f64; 2],
    pub layer_time: f64,
    pub fps: f64,
    original: Option<TextureHandle>,
    graded_key: Option<(u64, u64, u64, bool)>,
    /// The graded preview as displayed (output-encoded), for the scopes.
    pub display: Option<Arc<Image>>,
    pub display_gen: u64,
    graded: Option<TextureHandle>,
    clip_tex: Option<TextureHandle>,
    /// Fractions of pixels clipped in the highlights and the shadows.
    pub clip: [f32; 2],
    pub warnings: Vec<String>,
    /// Milliseconds the last grade took.
    pub ms: f32,
    pub scope_key: Option<(u64, usize)>,
    pub scope_tex: Option<TextureHandle>,
    /// Histogram bins (R, G, B) when the histogram is shown.
    pub scope_bars: Option<[Vec<u32>; 3]>,
    pub thumb_src: Option<(u64, Arc<Buf>)>,
    pub thumbs: HashMap<String, (u64, TextureHandle)>,
    pub error: Option<String>,
}

impl Cache {
    pub fn reset(&mut self) {
        *self = Cache::default();
    }
}

/// Preview scale for a layer of `size` (the compositor's quality steps).
fn preview_scale(size: [f64; 2]) -> f64 {
    let m = size[0].max(size[1]);
    if m <= 1400.0 {
        1.0
    } else if m <= 2800.0 {
        0.5
    } else {
        0.25
    }
}

/// Render the effect's input at composition time `time` and `scale`.
pub(crate) fn render_input(app: &EffectcraftApp, t: &Target, time: f64, scale: f64) -> Option<(Arc<Buf>, [f64; 2], f64, f64)> {
    let cid = ItemId(t.comp);
    let project = &app.session.project;
    let comp = project.comp(cid)?;
    let layer = comp.layer(LayerId(t.layer))?;
    let idx = layer.effects()?.groups().position(|g| g.uid == t.effect)?;
    let size = effectcraft_engine::render::effect_bounds(project, comp, layer).0;
    let ct = Tick::from_seconds_f64(time);
    let ectx = EvalCtx { project, comp_id: cid, comp, time: ct, expr: app.session.expr.as_deref(), footage: None };
    let mut r = Renderer::new(project, &*app.session.footage, RenderOpts { scale, ..Default::default() });
    r.expr = app.session.expr.as_deref();
    r.cache = Some(&app.session.layer_cache);
    let buf = r.layer_input(&ectx, layer, idx)?;
    Some((buf, size, layer.layer_time(ct).seconds(), comp.frame_rate.as_f64()))
}

/// The render environment for a buffer of the target layer.
pub(crate) fn env_for(app: &EffectcraftApp, t: &Target, buf: &Buf, size: [f64; 2], layer_time: f64, fps: f64) -> Env {
    Env { layer_size: size, offset: buf.offset, scale: buf.scale, time: layer_time, fps, seed: t.effect as u32, working: super::working(app) }
}

/// Working-space pixels → what the Composition panel shows (the project's output transform).
pub(crate) fn to_display(app: &EffectcraftApp, img: &Image) -> Image {
    let pipe = effectcraft_engine::render::color::Pipe::of(&app.session.project.settings);
    let mut out = img.clone();
    if let Some(c) = pipe.output() {
        effectcraft_engine::render::color::convert(&mut out, &c);
    } else if let Some((lin, mode, enc)) = pipe.output_hdr() {
        effectcraft_engine::render::color::output_hdr(&mut out, lin, mode, enc);
    }
    out
}

fn texture(ctx: &egui::Context, name: &str, img: &Image) -> TextureHandle {
    ctx.load_texture(name, crate::frames::to_color_image(img), egui::TextureOptions::LINEAR)
}

/// The clipping overlay (red: a channel at or above white, blue: black) and its fractions.
fn clipping(img: &Image) -> (egui::ColorImage, [f32; 2]) {
    let (mut hi, mut lo) = (0usize, 0usize);
    let px: Vec<Color32> = img
        .data
        .iter()
        .map(|p| {
            if p[3] < 0.5 {
                return Color32::TRANSPARENT;
            }
            let a = p[3];
            let c = [p[0] / a, p[1] / a, p[2] / a];
            if c.iter().any(|v| *v >= 0.999) {
                hi += 1;
                Color32::from_rgba_unmultiplied(255, 40, 40, 200)
            } else if c.iter().all(|v| *v <= 0.002) {
                lo += 1;
                Color32::from_rgba_unmultiplied(40, 110, 255, 200)
            } else {
                Color32::TRANSPARENT
            }
        })
        .collect();
    let n = img.data.len().max(1) as f32;
    (egui::ColorImage::new([img.width as usize, img.height as usize], px), [hi as f32 / n, lo as f32 / n])
}

/// Bring the input and the graded preview up to date.
pub(crate) fn update(app: &mut EffectcraftApp, ctx: &egui::Context) {
    let st = &app.dialog_state.cinelooks;
    let Some(t) = st.target.clone() else { return };
    let (time, rev, strength, draft) = (st.preview_time, st.rev, st.strength, st.draft.clone());
    let arc = st.caches.0.clone();
    let mut c = arc.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let key = (app.session.revision, t.layer, t.effect, Tick::from_seconds_f64(time).0);
    if c.input_key != Some(key) {
        let size = app
            .session
            .project
            .comp(ItemId(t.comp))
            .and_then(|comp| comp.layer(LayerId(t.layer)).map(|l| effectcraft_engine::render::effect_bounds(&app.session.project, comp, l).0))
            .unwrap_or([1920.0, 1080.0]);
        match render_input(app, &t, time, preview_scale(size)) {
            Some((buf, size, lt, fps)) => {
                c.original = Some(texture(ctx, "cinelooks-original", &to_display(app, &buf.img)));
                c.input = Some(buf);
                c.layer_size = size;
                c.layer_time = lt;
                c.fps = fps;
                c.error = None;
            }
            None => {
                c.input = None;
                c.original = None;
                c.error = Some("Nothing to preview at this time (the layer is outside its in/out points or empty).".into());
            }
        }
        c.input_key = Some(key);
        c.input_gen += 1;
        c.graded_key = None;
    }
    let gkey = (c.input_gen, rev, strength.to_bits(), app.dialog_state.cinelooks.clipping);
    if c.graded_key != Some(gkey)
        && let Some(buf) = c.input.clone()
    {
        let started = web_time::Instant::now();
        let env = env_for(app, &t, &buf, c.layer_size, c.layer_time, c.fps);
        let prog = effectcraft_engine::cinelooks::compile(&draft, &env, (strength / 100.0) as f32);
        let graded = effectcraft_engine::cinelooks::render(&prog, &buf.img);
        let disp = to_display(app, &graded);
        c.graded = Some(texture(ctx, "cinelooks-graded", &disp));
        let (overlay, frac) = clipping(&disp);
        c.clip = frac;
        c.clip_tex = app.dialog_state.cinelooks.clipping.then(|| ctx.load_texture("cinelooks-clip", overlay, egui::TextureOptions::NEAREST));
        c.display = Some(Arc::new(disp));
        c.display_gen += 1;
        c.warnings = prog.warnings;
        c.ms = started.elapsed().as_secs_f32() * 1000.0;
        c.graded_key = Some(gkey);
    }
}

fn checker(p: &egui::Painter, r: Rect) {
    let cell = 10.0;
    p.rect_filled(r, 0.0, Color32::from_gray(58));
    let mut y = r.min.y;
    let mut row = 0;
    while y < r.max.y {
        let mut x = r.min.x + if row % 2 == 0 { 0.0 } else { cell };
        while x < r.max.x {
            p.rect_filled(Rect::from_min_max(pos2(x, y), pos2((x + cell).min(r.max.x), (y + cell).min(r.max.y))), 0.0, Color32::from_gray(74));
            x += 2.0 * cell;
        }
        y += cell;
        row += 1;
    }
}

const COMPARE: [(Compare, &str, &str); 4] = [
    (Compare::Graded, "graded", "Graded"),
    (Compare::Original, "original", "Original"),
    (Compare::Split, "split", "Split"),
    (Compare::SideBySide, "sideBySide", "Side by Side"),
];
const ZOOMS: [(Option<f32>, &str); 5] = [(None, "Fit"), (Some(0.25), "25%"), (Some(0.5), "50%"), (Some(1.0), "100%"), (Some(2.0), "200%")];

pub fn show(app: &mut EffectcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    update(app, ui.ctx());
    let p = ui.painter().with_clip_rect(rect);
    // Toolbar.
    let bar = Rect::from_min_size(rect.min + vec2(8.0, 6.0), vec2(rect.width() - 16.0, 26.0));
    let mut x = bar.min.x;
    for (mode, id, label) in COMPARE {
        let wdt = 14.0 + label.len() as f32 * 6.6;
        let r = Rect::from_min_size(pos2(x, bar.min.y), vec2(wdt, 24.0));
        if button(app, ui, r, &format!("compare.{id}"), label, app.dialog_state.cinelooks.compare == mode) {
            app.dialog_state.cinelooks.compare = mode;
        }
        x += wdt + 4.0;
    }
    x += 10.0;
    let cur_zoom = app.dialog_state.cinelooks.zoom;
    let zi = ZOOMS.iter().position(|(z, _)| *z == cur_zoom).unwrap_or(0);
    let labels: Vec<&str> = ZOOMS.iter().map(|(_, l)| *l).collect();
    if let Some(i) = super::super::panel_kit::dropdown(app, ui, Rect::from_min_size(pos2(x, bar.min.y), vec2(70.0, 24.0)), &aid("zoom"), &labels, zi) {
        app.dialog_state.cinelooks.zoom = ZOOMS.get(i).and_then(|z| z.0);
        app.dialog_state.cinelooks.pan = [0.0, 0.0];
    }
    x += 80.0;
    let clip = app.dialog_state.cinelooks.clipping;
    let nc = super::super::panel_kit::checkbox(app, ui, pos2(x, bar.min.y + 4.0), &aid("clipping"), "Clipping", clip);
    if nc != clip {
        app.dialog_state.cinelooks.clipping = nc;
    }
    x += 92.0;
    // Look Strength.
    p.text(pos2(x, bar.center().y), Align2::LEFT_CENTER, "Look Strength", Tokens::ui(12.0), t.text_dim);
    let s = app.dialog_state.cinelooks.strength;
    let (sr, nv, _) = crate::widgets::hot_number_at(ui, pos2(x + 88.0, bar.min.y + 3.0), egui::Id::new("cl-strength"), s, 0.5, (0.0, 100.0), 1, "%", t);
    app.auto.add(&aid("lookStrength"), sr, "Look Strength");
    if let Some(v) = nv {
        app.dialog_state.cinelooks.gesture_edit(|_, st| *st = v);
    }
    // Timing / status on the right.
    let (ms, clip_frac, warnings, err) = {
        let c = app.dialog_state.cinelooks.caches.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (c.ms, c.clip, c.warnings.clone(), c.error.clone())
    };
    let status = if clip {
        format!("highlights {:.1}% · shadows {:.1}% clipped · {:.0} ms", clip_frac[0] * 100.0, clip_frac[1] * 100.0, ms)
    } else {
        format!("{ms:.0} ms")
    };
    p.text(pos2(bar.max.x, bar.center().y), Align2::RIGHT_CENTER, status, Tokens::ui(11.0), t.text_faint);

    // Time scrubber along the bottom.
    let scrub = Rect::from_min_max(pos2(rect.min.x + 8.0, rect.max.y - 30.0), pos2(rect.max.x - 8.0, rect.max.y - 6.0));
    time_scrubber(app, ui, scrub, t);

    let area = Rect::from_min_max(pos2(rect.min.x + 8.0, bar.max.y + 6.0), pos2(rect.max.x - 8.0, scrub.min.y - 6.0));
    p.rect_filled(area, 4.0, t.pasteboard);
    if let Some(e) = err {
        p.text(area.center(), Align2::CENTER_CENTER, e, Tokens::ui(13.0), t.text_dim);
        return;
    }
    if !warnings.is_empty() {
        crate::widgets::text_fit(
            &p,
            pos2(area.min.x + 8.0, area.max.y - 12.0),
            Align2::LEFT_CENTER,
            &warnings.join("; "),
            Tokens::ui(11.5),
            area.width() - 16.0,
            t.warning,
        );
    }
    canvas(app, ui, area, t);
}

/// Image rect for an image of `px` buffer pixels (scale `s`) in `area`.
fn image_rect(area: Rect, px: [f32; 2], s: f32, zoom: Option<f32>, pan: [f32; 2]) -> Rect {
    let layer = vec2(px[0] / s.max(1e-6), px[1] / s.max(1e-6));
    let k = match zoom {
        None => ((area.width() - 16.0) / layer.x).min((area.height() - 16.0) / layer.y).max(0.01),
        Some(z) => z,
    };
    Rect::from_center_size(area.center() + vec2(pan[0], pan[1]), layer * k)
}

fn canvas(app: &mut EffectcraftApp, ui: &mut egui::Ui, area: Rect, t: &Tokens) {
    let (input, original, graded, clip_tex) = {
        let c = app.dialog_state.cinelooks.caches.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (c.input.clone(), c.original.clone(), c.graded.clone(), c.clip_tex.clone())
    };
    let Some(buf) = input else { return };
    let st = &app.dialog_state.cinelooks;
    let (compare, zoom, pan, split) = (st.compare, st.zoom, st.pan, st.split);
    let px = [buf.img.width as f32, buf.img.height as f32];
    let s = buf.scale as f32;
    let p = ui.painter().with_clip_rect(area);
    let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
    // Navigation: wheel zooms, middle- or Alt-drag pans.
    let nav = ui.interact(area, egui::Id::new("cl-canvas-nav"), Sense::click_and_drag());
    app.auto.add(&aid("preview"), area, "Preview");
    if nav.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.5 {
            let cur = zoom.unwrap_or_else(|| image_rect(area, px, s, None, pan).width() / (px[0] / s));
            app.dialog_state.cinelooks.zoom = Some((cur * (scroll / 400.0).exp()).clamp(0.05, 16.0));
        }
    }
    let panning = nav.dragged_by(egui::PointerButton::Middle) || (nav.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.alt));
    if panning {
        let d = nav.drag_delta();
        let st = &mut app.dialog_state.cinelooks;
        st.pan = [st.pan[0] + d.x, st.pan[1] + d.y];
    }
    let draw = |r: Rect, tex: &Option<TextureHandle>, label: Option<&str>| {
        checker(&p, r.intersect(area));
        if let Some(tx) = tex {
            p.image(tx.id(), r, uv, Color32::WHITE);
        }
        p.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::from_black_alpha(160)), StrokeKind::Outside);
        if let Some(l) = label {
            let lr = Rect::from_min_size(r.min + vec2(6.0, 6.0), vec2(l.len() as f32 * 7.0 + 12.0, 18.0));
            p.rect_filled(lr, 3.0, Color32::from_black_alpha(150));
            p.text(lr.center(), Align2::CENTER_CENTER, l, Tokens::ui(11.0), Color32::WHITE);
        }
    };
    let main = match compare {
        Compare::SideBySide => {
            let half_a = Rect::from_min_max(area.min, pos2(area.center().x - 3.0, area.max.y));
            let half_b = Rect::from_min_max(pos2(area.center().x + 3.0, area.min.y), area.max);
            let ra = image_rect(half_a, px, s, zoom, pan);
            let rb = image_rect(half_b, px, s, zoom, pan);
            draw(ra, &original, Some("Original"));
            draw(rb, &graded, Some("Graded"));
            rb
        }
        _ => {
            let r = image_rect(area, px, s, zoom, pan);
            match compare {
                Compare::Original => draw(r, &original, Some("Original")),
                Compare::Split => {
                    draw(r, &graded, None);
                    let sx = r.min.x + r.width() * split;
                    let left = Rect::from_min_max(r.min, pos2(sx, r.max.y));
                    if let Some(tx) = &original {
                        p.with_clip_rect(left.intersect(area)).image(tx.id(), r, uv, Color32::WHITE);
                    }
                    p.line_segment([pos2(sx, r.min.y), pos2(sx, r.max.y)], Stroke::new(2.0, Color32::WHITE));
                    let h = Rect::from_center_size(pos2(sx, r.center().y), vec2(16.0, 40.0));
                    p.rect_filled(h, 4.0, Color32::from_gray(235));
                    let resp = ui.interact(Rect::from_center_size(pos2(sx, r.center().y), vec2(14.0, r.height())), egui::Id::new("cl-split"), Sense::drag());
                    app.auto.add(&aid("split"), h, "Split");
                    if resp.hovered() || resp.dragged() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                    }
                    if resp.dragged()
                        && let Some(pp) = resp.interact_pointer_pos()
                    {
                        app.dialog_state.cinelooks.split = ((pp.x - r.min.x) / r.width().max(1.0)).clamp(0.0, 1.0);
                    }
                    for (txt, at, al) in
                        [("Original", r.min + vec2(8.0, 8.0), Align2::LEFT_TOP), ("Graded", pos2(r.max.x - 8.0, r.min.y + 8.0), Align2::RIGHT_TOP)]
                    {
                        p.text(at, al, txt, Tokens::semibold(11.0), Color32::WHITE);
                    }
                }
                _ => draw(r, &graded, None),
            }
            r
        }
    };
    if let Some(ct) = clip_tex.filter(|_| compare != Compare::Original) {
        p.image(ct.id(), main, uv, Color32::WHITE);
    }
    if compare != Compare::Original {
        handles(app, ui, &p, main, &buf, t);
    }
}

/// On-canvas handles for the selected tool's point parameters (and their shapes).
fn handles(app: &mut EffectcraftApp, ui: &mut egui::Ui, p: &egui::Painter, img: Rect, buf: &Buf, t: &Tokens) {
    let st = &app.dialog_state.cinelooks;
    let (Some(i), Some(node)) = (st.selected, st.node().cloned()) else { return };
    let size = app.dialog_state.cinelooks.caches.0.lock().map(|c| c.layer_size).unwrap_or([1920.0, 1080.0]);
    let (bw, bh) = (buf.img.width as f64, buf.img.height as f64);
    // Layer fraction → screen, and back.
    let to_screen = |f: [f64; 2]| {
        let bx = f[0] * size[0] * buf.scale + buf.offset[0];
        let by = f[1] * size[1] * buf.scale + buf.offset[1];
        pos2(img.min.x + (bx / bw) as f32 * img.width(), img.min.y + (by / bh) as f32 * img.height())
    };
    let to_frac = |q: Pos2| {
        let bx = ((q.x - img.min.x) / img.width().max(1.0)) as f64 * bw;
        let by = ((q.y - img.min.y) / img.height().max(1.0)) as f64 * bh;
        [((bx - buf.offset[0]) / buf.scale / size[0].max(1.0)), ((by - buf.offset[1]) / buf.scale / size[1].max(1.0))]
    };
    // Screen points per layer pixel (the image spans the buffer, which may be padded).
    let px_per_layer = (img.width() as f64 / bw.max(1.0) * buf.scale) as f32;
    let line = Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 255, 255, 200));
    // Shapes.
    match node.tool {
        ToolKind::GradientExposure => {
            let (a, b) = (to_screen(node.point("start")), to_screen(node.point("end")));
            p.line_segment([a, b], line);
            let d = b - a;
            let n = vec2(-d.y, d.x).normalized() * 2000.0;
            for q in [a, b] {
                p.line_segment([q - n, q + n], Stroke::new(1.0, Color32::from_white_alpha(90)));
            }
        }
        ToolKind::RadialExposure | ToolKind::SpotExposure | ToolKind::Vignette => {
            let c = to_screen(node.point("center"));
            let (rx, ry, rot) = match node.tool {
                ToolKind::RadialExposure => (node.num("width") / 200.0 * size[1], node.num("height") / 200.0 * size[1], node.num("rotation")),
                ToolKind::SpotExposure => (node.num("size") / 200.0 * size[1], node.num("size") / 200.0 * size[1], 0.0),
                _ => {
                    let r = node.num("radius") / 100.0;
                    let a = node.num("aspect").clamp(0.25, 4.0).sqrt();
                    (size[0] * 0.5 * r * a, size[1] * 0.5 * r / a, 0.0)
                }
            };
            let (sn, cs) = rot.to_radians().sin_cos();
            let pts: Vec<Pos2> = (0..=64)
                .map(|k| {
                    let a = k as f64 / 64.0 * std::f64::consts::TAU;
                    let (x, y) = (a.cos() * rx, a.sin() * ry);
                    let (x, y) = (x * cs - y * sn, x * sn + y * cs);
                    c + vec2(x as f32, y as f32) * px_per_layer
                })
                .collect();
            p.add(egui::Shape::line(pts, line));
        }
        _ => {}
    }
    // Point handles.
    for d in node.tool.def().params.iter().filter(|d| matches!(d.kind, Kind::Point)) {
        let q = to_screen(node.point(d.id));
        let r = Rect::from_center_size(q, vec2(16.0, 16.0));
        let resp = ui.interact(r, egui::Id::new(("cl-handle", i, d.id)), Sense::drag());
        app.auto.add(&aid(&format!("handle.{}", d.id)), r, d.name);
        let hot = resp.hovered() || resp.dragged();
        p.circle_filled(q, if hot { 7.0 } else { 6.0 }, Color32::from_black_alpha(140));
        p.circle_stroke(q, 6.0, Stroke::new(2.0, if hot { t.accent_hover } else { Color32::WHITE }));
        p.line_segment([q - vec2(3.0, 0.0), q + vec2(3.0, 0.0)], Stroke::new(1.0, Color32::WHITE));
        p.line_segment([q - vec2(0.0, 3.0), q + vec2(0.0, 3.0)], Stroke::new(1.0, Color32::WHITE));
        if hot {
            p.text(q + vec2(10.0, -10.0), Align2::LEFT_BOTTOM, d.name, Tokens::ui(11.0), Color32::WHITE);
        }
        if resp.dragged()
            && let Some(pp) = resp.interact_pointer_pos()
        {
            let f = to_frac(pp);
            app.dialog_state.cinelooks.gesture_edit(|l, _| {
                if let Some(n) = l.nodes.get_mut(i) {
                    n.params.insert(d.id.to_string(), PV::List(vec![f[0].clamp(-2.0, 3.0), f[1].clamp(-2.0, 3.0)]));
                }
            });
        }
    }
}

fn time_scrubber(app: &mut EffectcraftApp, ui: &mut egui::Ui, r: Rect, t: &Tokens) {
    let Some(tg) = app.dialog_state.cinelooks.target.clone() else { return };
    let Some(comp) = app.session.project.comp(ItemId(tg.comp)) else { return };
    let Some(layer) = comp.layer(LayerId(tg.layer)) else { return };
    let (a, b) = (layer.in_point.seconds().max(0.0), layer.out_point.seconds().min(comp.duration.seconds()));
    let (a, b) = if b > a { (a, b) } else { (0.0, comp.duration.seconds().max(0.04)) };
    let fr = comp.frame_rate;
    let p = ui.painter();
    let label = super::super::timecode(&app.session, comp, Tick::from_seconds_f64(app.dialog_state.cinelooks.preview_time));
    p.text(pos2(r.min.x, r.center().y), Align2::LEFT_CENTER, label, Tokens::mono(12.0), t.timecode);
    let track = Rect::from_min_max(pos2(r.min.x + 110.0, r.center().y - 3.0), pos2(r.max.x - 130.0, r.center().y + 3.0));
    p.rect_filled(track, 3.0, t.field_bg);
    let cur = app.dialog_state.cinelooks.preview_time;
    let f = ((cur - a) / (b - a)).clamp(0.0, 1.0) as f32;
    let kx = track.min.x + track.width() * f;
    p.rect_filled(Rect::from_min_max(track.min, pos2(kx, track.max.y)), 3.0, t.accent.gamma_multiply(0.6));
    p.circle_filled(pos2(kx, track.center().y), 6.0, t.cti);
    let resp = ui.interact(track.expand2(vec2(6.0, 8.0)), egui::Id::new("cl-scrub"), Sense::click_and_drag());
    app.auto.add(&aid("frame"), track, "Preview frame");
    if (resp.dragged() || resp.clicked())
        && let Some(pp) = resp.interact_pointer_pos()
    {
        let ft = ((pp.x - track.min.x) / track.width().max(1.0)).clamp(0.0, 1.0) as f64;
        let ct = fr.snap_nearest(Tick::from_seconds_f64(a + (b - a) * ft));
        app.dialog_state.cinelooks.preview_time = ct.seconds();
    }
    let opened = tg.time;
    let back = Rect::from_min_size(pos2(r.max.x - 120.0, r.min.y), vec2(120.0, 22.0));
    if button(app, ui, back, "frame.current", "Current Time", (cur - opened).abs() < 1e-9) {
        app.dialog_state.cinelooks.preview_time = opened;
    }
}

impl EditorState {
    /// [`EditorState::edit`] for continuous gestures: the whole drag is one undo step.
    pub fn gesture_edit(&mut self, f: impl FnOnce(&mut Look, &mut f64)) {
        self.edit(f);
        self.gesture = true;
    }
}
