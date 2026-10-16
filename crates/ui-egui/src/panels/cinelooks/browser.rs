//! The left sidebar: the look library. Search, a category (All, Favorites, User, the built-in
//! categories and the user's own), and thumbnails of the layer's current frame through each look
//! (rendered by the engine a few per frame, cached until the frame changes). Click a look to
//! load it into the draft; the star marks a favourite; right-click for more. Save Look, Import
//! and Export at the bottom.

use std::sync::Arc;

use effectcraft_engine::effects::Buf;
use egui::{Align2, Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use serde_json::json;

use super::{Entry, aid, button, preview};
use crate::EffectcraftApp;
use crate::theme::Tokens;

const THUMB_W: f32 = 116.0;
const THUMB_H: f32 = 66.0;
/// Time spent on thumbnails per frame (ms).
const BUDGET_MS: f32 = 14.0;

fn categories(app: &EffectcraftApp) -> Vec<String> {
    let mut v: Vec<String> = ["All", "Favorites", "User"].iter().map(|s| s.to_string()).collect();
    v.extend(effectcraft_engine::cinelooks::presets::CATEGORIES.iter().map(|s| s.to_string()));
    for e in &app.dialog_state.cinelooks.library {
        if e.user && !v.contains(&e.category) {
            v.push(e.category.clone());
        }
    }
    v
}

fn visible(app: &EffectcraftApp) -> Vec<Entry> {
    let st = &app.dialog_state.cinelooks;
    let q = st.search.to_lowercase();
    st.library
        .iter()
        .filter(|e| match st.category.as_str() {
            "All" | "" => true,
            "Favorites" => st.favorites.contains(&e.id),
            "User" => e.user,
            c => e.category == c,
        })
        .filter(|e| q.is_empty() || e.name.to_lowercase().contains(&q) || e.category.to_lowercase().contains(&q) || e.description.to_lowercase().contains(&q))
        .cloned()
        .collect()
}

/// The small source frame thumbnails are graded from (rebuilt with the preview's input).
fn thumb_source(app: &EffectcraftApp) -> Option<(u64, Arc<Buf>)> {
    let arc = super::cache_arc(app);
    let mut c = super::lock(&arc);
    if let Some(ts) = &c.thumb_src
        && ts.0 == c.input_gen
    {
        return Some(ts.clone());
    }
    let input = c.input.clone()?;
    let k = (THUMB_W as f64 * 2.0 / input.img.width.max(1) as f64).min(1.0);
    let (w, h) = (((input.img.width as f64 * k).round() as u32).max(1), ((input.img.height as f64 * k).round() as u32).max(1));
    let img = effectcraft_engine::raster::resample(&input.img, w, h);
    let kx = w as f64 / input.img.width.max(1) as f64;
    let buf = Arc::new(Buf { img, offset: [input.offset[0] * kx, input.offset[1] * kx], scale: input.scale * kx });
    c.thumb_src = Some((c.input_gen, buf.clone()));
    c.thumbs.clear();
    Some((c.input_gen, buf))
}

fn ensure_thumbs(app: &mut EffectcraftApp, ctx: &egui::Context, entries: &[Entry]) {
    let Some(target) = app.dialog_state.cinelooks.target.clone() else { return };
    let Some((generation, src)) = thumb_source(app) else { return };
    let arc = super::cache_arc(app);
    let (size, lt, fps) = {
        let c = super::lock(&arc);
        (c.layer_size, c.layer_time, c.fps)
    };
    let started = web_time::Instant::now();
    let mut pending = false;
    for e in entries {
        let have = super::lock(&arc).thumbs.get(&e.id).is_some_and(|(g, _)| *g == generation);
        if have {
            continue;
        }
        if started.elapsed().as_secs_f32() * 1000.0 > BUDGET_MS {
            pending = true;
            break;
        }
        let env = preview::env_for(app, &target, &src, size, lt, fps);
        let prog = effectcraft_engine::cinelooks::compile(&e.look, &env, 1.0);
        let out = effectcraft_engine::cinelooks::render(&prog, &src.img);
        let disp = preview::to_display(app, &out);
        let tex = ctx.load_texture(format!("cinelooks-thumb-{}", e.id), crate::frames::to_color_image(&disp), egui::TextureOptions::LINEAR);
        super::lock(&arc).thumbs.insert(e.id.clone(), (generation, tex));
    }
    if pending {
        ctx.request_repaint();
    }
}

pub fn show(app: &mut EffectcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    // The user library may have changed (an import finished, another window saved a look).
    let frame = ui.ctx().cumulative_pass_nr();
    if frame.is_multiple_of(45) {
        super::reload_library(app);
    }
    let p = ui.painter().with_clip_rect(rect);
    p.text(pos2(rect.min.x + 12.0, rect.min.y + 18.0), Align2::LEFT_CENTER, "Looks", Tokens::semibold(14.0), t.tab_text_active);
    // Search.
    let sr = Rect::from_min_size(pos2(rect.min.x + 10.0, rect.min.y + 34.0), vec2(rect.width() - 20.0, 24.0));
    let mut q = app.dialog_state.cinelooks.search.clone();
    if crate::widgets::search_field(ui, sr, &mut q, "Search looks", t).changed() {
        app.dialog_state.cinelooks.search = q;
    }
    app.auto.add(&aid("search"), sr, "Search looks");
    // Category.
    let cats = categories(app);
    let cur = cats.iter().position(|c| *c == app.dialog_state.cinelooks.category).unwrap_or(0);
    let labels: Vec<&str> = cats.iter().map(String::as_str).collect();
    let cr = Rect::from_min_size(pos2(rect.min.x + 10.0, sr.max.y + 8.0), vec2(rect.width() - 20.0, 24.0));
    if let Some(i) = super::super::panel_kit::dropdown(app, ui, cr, &aid("category"), &labels, cur)
        && let Some(c) = cats.get(i)
    {
        app.dialog_state.cinelooks.category = c.clone();
    }
    // Bottom actions.
    let foot_h = if app.dialog_state.cinelooks.save_name.is_some() { 122.0 } else { 64.0 };
    let foot = Rect::from_min_max(pos2(rect.min.x + 10.0, rect.max.y - foot_h), pos2(rect.max.x - 10.0, rect.max.y - 8.0));
    footer(app, ui, foot, t);
    // Grid.
    let grid = Rect::from_min_max(pos2(rect.min.x + 6.0, cr.max.y + 10.0), pos2(rect.max.x - 6.0, foot.min.y - 8.0));
    let entries = visible(app);
    ensure_thumbs(app, ui.ctx(), &entries);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(grid).layout(egui::Layout::top_down(egui::Align::Min)));
    child.set_clip_rect(grid.intersect(ui.clip_rect()));
    let arc = super::cache_arc(app);
    let mut picked: Option<Entry> = None;
    let mut fav_toggle: Option<String> = None;
    let mut delete: Option<String> = None;
    egui::ScrollArea::vertical().id_salt("cl-browser").auto_shrink([false, false]).show(&mut child, |ui| {
        if entries.is_empty() {
            ui.add_space(20.0);
            ui.label(egui::RichText::new("No looks here.").color(t.text_dim));
        }
        let cols = 2;
        for row in entries.chunks(cols) {
            let (r, _) = ui.allocate_exact_size(vec2(grid.width() - 4.0, THUMB_H + 26.0), Sense::hover());
            for (k, e) in row.iter().enumerate() {
                let tile = Rect::from_min_size(pos2(r.min.x + 4.0 + k as f32 * (THUMB_W + 8.0), r.min.y + 2.0), vec2(THUMB_W, THUMB_H));
                let painter = ui.painter();
                painter.rect_filled(tile, 4.0, Color32::from_gray(40));
                if let Some((_, tex)) = super::lock(&arc).thumbs.get(&e.id) {
                    painter.image(tex.id(), tile, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                let resp = ui.interact(tile, egui::Id::new(("cl-thumb", &e.id)), Sense::click());
                let hot = resp.hovered();
                let current = app.dialog_state.cinelooks.draft.name == e.name;
                painter.rect_stroke(
                    tile,
                    4.0,
                    Stroke::new(
                        if current { 2.0 } else { 1.0 },
                        if current {
                            t.accent
                        } else if hot {
                            t.text_dim
                        } else {
                            t.field_border
                        },
                    ),
                    StrokeKind::Outside,
                );
                crate::widgets::text_fit(
                    painter,
                    pos2(tile.min.x + 2.0, tile.max.y + 11.0),
                    Align2::LEFT_CENTER,
                    &e.name,
                    Tokens::ui(11.5),
                    THUMB_W - 4.0,
                    if current { t.text } else { t.text_dim },
                );
                // Favourite star.
                let fav = app.dialog_state.cinelooks.favorites.contains(&e.id);
                let star = Rect::from_center_size(tile.right_top() + vec2(-11.0, 11.0), vec2(18.0, 18.0));
                if fav || hot {
                    painter.circle_filled(star.center(), 8.0, Color32::from_black_alpha(140));
                    painter.text(
                        star.center(),
                        Align2::CENTER_CENTER,
                        if fav { "★" } else { "☆" },
                        Tokens::ui(12.0),
                        if fav { Color32::from_rgb(255, 200, 60) } else { Color32::WHITE },
                    );
                }
                let sresp = ui.interact(star, egui::Id::new(("cl-star", &e.id)), Sense::click());
                if sresp.clicked() {
                    fav_toggle = Some(e.id.clone());
                } else if resp.clicked() {
                    picked = Some(e.clone());
                }
                let resp = resp.on_hover_text(format!("{} · {}\n{}", e.name, e.category, e.description));
                resp.context_menu(|ui| {
                    if ui.button(if fav { "Remove from Favorites" } else { "Add to Favorites" }).clicked() {
                        fav_toggle = Some(e.id.clone());
                        ui.close();
                    }
                    if e.user && ui.button("Delete Look").clicked() {
                        delete = Some(e.id.clone());
                        ui.close();
                    }
                });
                app.auto.add(&aid(&format!("look.{}", e.id)), tile, &e.name);
                app.auto.add(&aid(&format!("look.{}.favorite", e.id)), star, "Favorite");
            }
        }
    });
    if let Some(e) = picked {
        let st = &mut app.dialog_state.cinelooks;
        let mut look = e.look.clone();
        look.name = e.name.clone();
        st.edit(|l, s| {
            *l = look;
            *s = 100.0;
        });
        st.selected = (!st.draft.nodes.is_empty()).then_some(0);
        st.message = format!("Loaded “{}”. Apply to use it on the layer.", e.name);
    }
    if let Some(id) = fav_toggle {
        match app.session.execute("cinelooks.setFavorite", json!({"id": id})) {
            Ok(_) => super::reload_library(app),
            Err(e) => app.dialog_state.cinelooks.message = e.to_string(),
        }
    }
    if let Some(id) = delete {
        match app.session.execute("cinelooks.deletePreset", json!({"id": id})) {
            Ok(_) => super::reload_library(app),
            Err(e) => app.dialog_state.cinelooks.message = e.to_string(),
        }
    }
}

fn footer(app: &mut EffectcraftApp, ui: &mut egui::Ui, r: Rect, t: &Tokens) {
    let ctx = ui.ctx().clone();
    let mut y = r.min.y;
    if let Some(mut name) = app.dialog_state.cinelooks.save_name.clone() {
        let fr = Rect::from_min_size(pos2(r.min.x, y), vec2(r.width(), 24.0));
        let resp = crate::widgets::text_field(ui, fr, &mut name, "Look name", t);
        app.auto.add(&aid("save.name"), fr, "Look name");
        if resp.changed() {
            app.dialog_state.cinelooks.save_name = Some(name.clone());
        }
        y += 28.0;
        let half = (r.width() - 6.0) * 0.5;
        let ok = button(app, ui, Rect::from_min_size(pos2(r.min.x, y), vec2(half, 24.0)), "save.ok", "Save", true);
        let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if ok || enter {
            let st = &app.dialog_state.cinelooks;
            let p = json!({"name": name, "category": "User", "look": st.draft});
            match app.session.execute("cinelooks.savePreset", p) {
                Ok(_) => {
                    app.dialog_state.cinelooks.save_name = None;
                    app.dialog_state.cinelooks.message = format!("Saved “{name}” to your looks.");
                    app.dialog_state.cinelooks.category = "User".into();
                    super::reload_library(app);
                }
                Err(e) => app.dialog_state.cinelooks.message = e.to_string(),
            }
        }
        if button(app, ui, Rect::from_min_size(pos2(r.min.x + half + 6.0, y), vec2(half, 24.0)), "save.cancel", "Cancel", false) {
            app.dialog_state.cinelooks.save_name = None;
        }
        y += 30.0;
    }
    let third = (r.width() - 12.0) / 3.0;
    let b = |k: f32, yy: f32| Rect::from_min_size(pos2(r.min.x + k * (third + 6.0), yy), vec2(third, 24.0));
    if button(app, ui, b(0.0, y), "save", "Save Look…", false) {
        let n = app.dialog_state.cinelooks.draft.name.clone();
        app.dialog_state.cinelooks.save_name = Some(if n.is_empty() { "My Look".into() } else { n });
    }
    if button(app, ui, b(1.0, y), "import", "Import…", false) {
        if let Err(e) = crate::menus::invoke(app, &ctx, "cinelooks.importPreset", json!({})) {
            app.dialog_state.cinelooks.message = e;
        }
        super::reload_library(app);
    }
    if button(app, ui, b(2.0, y), "export", "Export…", false) {
        let st = &app.dialog_state.cinelooks;
        let p = json!({"look": st.draft, "name": st.draft.name});
        if let Err(e) = crate::menus::invoke(app, &ctx, "cinelooks.exportPreset", p) {
            app.dialog_state.cinelooks.message = e;
        }
    }
    let y2 = y + 30.0;
    if y2 + 24.0 <= r.max.y + 1.0 {
        let full = Rect::from_min_size(pos2(r.min.x, y2), vec2(r.width(), 24.0));
        if button(app, ui, full, "exportCube", "Export Color Tools as .cube LUT…", false) {
            let st = &app.dialog_state.cinelooks;
            if let Some(tg) = &st.target {
                let p = json!({"comp": tg.comp, "layer": tg.layer, "effect": tg.effect, "look": st.draft, "strength": st.strength});
                if let Err(e) = crate::menus::invoke(app, &ctx, "cinelooks.exportCube", p) {
                    app.dialog_state.cinelooks.message = e;
                }
            }
        }
    }
}
