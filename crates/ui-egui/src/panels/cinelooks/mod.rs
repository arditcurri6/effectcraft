//! The CineLooks Editor (`cinelooks.openEditor`, Effect Controls' "Open CineLooks Editor…"): a
//! full-window grading workspace for one CineLooks instance.
//!
//! - left: the look library (search, categories, Favorites, User) with thumbnails of the
//!   layer's own frame through each look;
//! - centre: the preview (the layer's input to the effect, graded by the real engine), zoom,
//!   Original / split wipe / side by side, clipping overlay and on-canvas handles;
//! - bottom: the tool chain (select, enable, drag to reorder, add, duplicate, delete);
//! - right: the selected tool's controls (sliders, colour wheels, curves, colours, LUT) and
//!   scopes of the graded preview.
//!
//! The editor works on a draft with its own undo / redo (one step per gesture). Apply writes it
//! to the effect with `cinelooks.setLook` (one project undo step; animated parameters get a key
//! at the time the editor opened); OK applies and closes; Cancel and Escape discard what wasn't
//! applied. Every control registers an automation id under `cinelooks.`.

mod browser;
mod chain;
mod inspector;
mod preview;
mod scopes;
mod widgets;

pub use inspector::lut_chosen;

use std::sync::{Arc, Mutex, PoisonError};

use effectcraft_engine::cinelooks::look::{Look, Node};
use effectcraft_engine::cinelooks::tools::ToolKind;
use egui::{Align2, Color32, Rect, Stroke, StrokeKind, pos2, vec2};
use serde_json::{Value, json};

use crate::theme::Tokens;
use crate::{Dialog, EffectcraftApp, widgets as w};

/// The instance being edited.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub comp: u64,
    pub layer: u64,
    pub effect: u64,
    /// Composition seconds the editor opened at: reads and writes happen there.
    pub time: f64,
}

/// How the preview compares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Compare {
    #[default]
    Graded,
    Original,
    Split,
    SideBySide,
}

/// One undoable state of the draft.
#[derive(Clone, Debug, PartialEq)]
struct Snapshot {
    look: Look,
    strength: f64,
    selected: Option<usize>,
}

/// A library entry (built-in or user).
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
    pub user: bool,
    pub look: Look,
}

/// Caches that hold textures and images (shared between clones of the state).
#[derive(Clone, Default)]
pub struct Caches(pub Arc<Mutex<preview::Cache>>);

impl std::fmt::Debug for Caches {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Caches")
    }
}

#[derive(Clone, Debug, Default)]
pub struct EditorState {
    pub target: Option<Target>,
    pub draft: Look,
    /// Look Strength, 0..100.
    pub strength: f64,
    /// What the effect holds (the draft as last applied or opened).
    baseline: Option<Snapshot>,
    pub selected: Option<usize>,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    /// A drag is changing the draft (one undo step for the whole gesture).
    gesture: bool,
    /// Draft revision (preview cache key).
    pub rev: u64,
    pub compare: Compare,
    /// Split position (0..1 of the image width).
    pub split: f32,
    /// Preview zoom (`None` = fit) and pan (screen points).
    pub zoom: Option<f32>,
    pub pan: [f32; 2],
    pub clipping: bool,
    pub category: String,
    pub search: String,
    pub favorites: Vec<String>,
    pub library: Vec<Entry>,
    pub scope: usize,
    /// Preview frame (composition seconds).
    pub preview_time: f64,
    pub curve_channel: usize,
    /// The Save Look prompt (name being typed).
    pub save_name: Option<String>,
    /// Status line (warnings, what Apply did).
    pub message: String,
    pub caches: Caches,
}

impl EditorState {
    fn snapshot(&self) -> Snapshot {
        Snapshot { look: self.draft.clone(), strength: self.strength, selected: self.selected }
    }

    fn restore(&mut self, s: Snapshot) {
        self.draft = s.look;
        self.strength = s.strength;
        self.selected = s.selected.filter(|i| *i < self.draft.nodes.len());
        self.rev += 1;
    }

    /// Change the draft: one undo step per gesture (a drag) or per click.
    pub fn edit(&mut self, f: impl FnOnce(&mut Look, &mut f64)) {
        let before = self.snapshot();
        let mut look = self.draft.clone();
        let mut strength = self.strength;
        f(&mut look, &mut strength);
        if look == self.draft && strength == self.strength {
            return;
        }
        if !self.gesture {
            self.undo.push(before);
            if self.undo.len() > 200 {
                self.undo.remove(0);
            }
        }
        self.redo.clear();
        self.draft = look;
        self.strength = strength;
        self.rev += 1;
    }

    /// Change one parameter of node `i`.
    pub fn set_param(&mut self, i: usize, id: &str, v: impl Into<effectcraft_engine::cinelooks::PV>) {
        let v = v.into();
        self.edit(|l, _| {
            if let Some(n) = l.nodes.get_mut(i) {
                n.params.insert(id.to_string(), v);
            }
        });
    }

    pub fn undo(&mut self) {
        if let Some(s) = self.undo.pop() {
            self.redo.push(self.snapshot());
            self.restore(s);
        }
    }

    pub fn redo(&mut self) {
        if let Some(s) = self.redo.pop() {
            self.undo.push(self.snapshot());
            self.restore(s);
        }
    }

    /// The draft differs from what the effect holds.
    pub fn modified(&self) -> bool {
        self.baseline.as_ref().is_none_or(|b| b.look != self.draft || b.strength != self.strength)
    }

    pub fn node(&self) -> Option<&Node> {
        self.selected.and_then(|i| self.draft.nodes.get(i))
    }
}

/// The automation id prefix.
pub(crate) fn aid(s: &str) -> String {
    format!("cinelooks.{s}")
}

fn read_look(app: &mut EffectcraftApp, t: &Target) -> Result<(Look, f64), String> {
    let v = app
        .session
        .execute("cinelooks.getLook", json!({"comp": t.comp, "layer": t.layer, "effect": t.effect, "time": t.time, "timeBase": "comp"}))
        .map_err(|e| e.to_string())?;
    let look: Look = serde_json::from_value(v["look"].clone()).map_err(|e| e.to_string())?;
    Ok((look, v["strength"].as_f64().unwrap_or(100.0)))
}

/// Reload the user library and favourites.
fn reload_library(app: &mut EffectcraftApp) {
    let lib = effectcraft_engine::commands::cinelooks::Library::load(&app.session);
    let st = &mut app.dialog_state.cinelooks;
    st.favorites = lib.favorites.clone();
    st.library = effectcraft_engine::cinelooks::presets::builtin()
        .iter()
        .map(|p| Entry {
            id: p.id.clone(),
            name: p.name.clone(),
            category: p.category.clone(),
            description: p.description.clone(),
            user: false,
            look: p.look.clone(),
        })
        .chain(lib.presets.into_iter().map(|u| Entry { id: u.id, name: u.name, category: u.category, description: u.description, user: true, look: u.look }))
        .collect();
}

/// `cinelooks.openEditor {comp, layer, effect}` (resolved by the engine command).
pub fn open(app: &mut EffectcraftApp, p: &Value) -> Result<Value, String> {
    let comp = p.get("comp").and_then(Value::as_u64).or_else(|| app.session.active_comp_id().map(|c| c.0)).ok_or("no composition is open")?;
    let layer = p.get("layer").and_then(Value::as_u64).ok_or("missing `layer`")?;
    let effect = p.get("effect").and_then(Value::as_u64).ok_or("missing `effect`")?;
    let cid = effectcraft_engine::project::ItemId(comp);
    let fr = app.session.project.comp(cid).map(|c| c.frame_rate).ok_or("no such composition")?;
    let time = fr.snap_nearest(app.session.time_of(cid)).seconds();
    let target = Target { comp, layer, effect, time };
    let (look, strength) = read_look(app, &target)?;
    let caches = app.dialog_state.cinelooks.caches.clone();
    let keep = std::mem::take(&mut app.dialog_state.cinelooks);
    app.dialog_state.cinelooks = EditorState {
        target: Some(target),
        selected: (!look.nodes.is_empty()).then_some(0),
        baseline: Some(Snapshot { look: look.clone(), strength, selected: None }),
        draft: look,
        strength,
        split: 0.5,
        category: if keep.category.is_empty() { "All".into() } else { keep.category },
        compare: keep.compare,
        scope: keep.scope,
        clipping: keep.clipping,
        preview_time: time,
        caches,
        ..Default::default()
    };
    if let Ok(mut c) = app.dialog_state.cinelooks.caches.0.lock() {
        c.reset();
    }
    reload_library(app);
    app.dialog = Some(Dialog::CineLooks);
    Ok(json!({"dialog": "cinelooks"}))
}

/// Write the draft to the effect (one undo step). Returns false on failure (the message says why).
fn apply(app: &mut EffectcraftApp) -> bool {
    let st = &app.dialog_state.cinelooks;
    let Some(t) = st.target.clone() else { return false };
    if !st.modified() {
        return true;
    }
    let label = if st.draft.name.is_empty() { "CineLooks Editor".to_string() } else { format!("CineLooks: {}", st.draft.name) };
    let p = json!({
        "comp": t.comp, "layer": t.layer, "effect": t.effect, "time": t.time, "timeBase": "comp",
        "look": st.draft, "strength": st.strength, "label": label,
    });
    match app.session.execute("cinelooks.setLook", p) {
        Ok(v) => {
            let st = &mut app.dialog_state.cinelooks;
            if let Ok(look) = serde_json::from_value::<Look>(v["look"].clone()) {
                // Keep the draft's values; take the new nodes' uids.
                for (d, n) in st.draft.nodes.iter_mut().zip(&look.nodes) {
                    d.uid = n.uid;
                }
            }
            st.baseline = Some(st.snapshot());
            st.message = "Applied to the effect (one undo step).".into();
            true
        }
        Err(e) => {
            app.dialog_state.cinelooks.message = format!("Couldn't apply: {e}");
            false
        }
    }
}

fn close(app: &mut EffectcraftApp) {
    app.dialog = None;
    let caches = app.dialog_state.cinelooks.caches.clone();
    if let Ok(mut c) = caches.0.lock() {
        c.reset();
    }
    app.dialog_state.cinelooks.target = None;
}

/// A button with an automation id.
pub(crate) fn button(app: &mut EffectcraftApp, ui: &mut egui::Ui, r: Rect, id: &str, text: &str, primary: bool) -> bool {
    let t = app.tokens;
    let clicked = w::text_button(ui, r, text, primary, &t, egui::Id::new(("cl-btn", id))).clicked();
    app.auto.add(&aid(id), r, text);
    clicked
}

pub fn show(app: &mut EffectcraftApp, ctx: &egui::Context, t: &Tokens) {
    let Some(target) = app.dialog_state.cinelooks.target.clone() else {
        app.dialog = None;
        return;
    };
    // The effect may be gone (deleted, undone from another panel): close.
    let exists = app
        .session
        .project
        .comp(effectcraft_engine::project::ItemId(target.comp))
        .and_then(|c| c.layer(effectcraft_engine::project::LayerId(target.layer)))
        .and_then(|l| l.props.find_group(target.effect))
        .is_some();
    if !exists {
        app.ui.status = "CineLooks: the effect is gone; the editor closed.".into();
        close(app);
        return;
    }
    // A gesture lasts while a pointer button is down.
    if !ctx.input(|i| i.pointer.any_down()) {
        app.dialog_state.cinelooks.gesture = false;
    }
    // Editor shortcuts (the app's are off while a dialog is open).
    let (undo, redo) = ctx.input_mut(|i| {
        let redo = i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z);
        let undo = i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z);
        (undo, redo)
    });
    let typing = ctx.memory(|m| m.focused().is_some());
    if undo && !typing {
        app.dialog_state.cinelooks.undo();
    }
    if redo && !typing {
        app.dialog_state.cinelooks.redo();
    }
    let screen = ctx.content_rect();
    let size = (screen.size() - vec2(24.0, 24.0)).max(vec2(900.0, 560.0));
    let id = egui::Id::new("cinelooks-editor");
    let mut action: Option<&str> = None;
    egui::Modal::new(id)
        .area(egui::Modal::default_area(id).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)))
        .backdrop_color(Color32::from_black_alpha(150))
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            let p = ui.painter().clone();
            p.rect_filled(rect, 8.0, t.app_bg);
            p.rect_stroke(rect, 8.0, Stroke::new(1.0, t.separator), StrokeKind::Inside);
            action = layout(app, ui, rect, t);
        });
    match action {
        Some("ok") => {
            if apply(app) {
                close(app);
            }
        }
        Some("apply") => {
            apply(app);
        }
        Some("cancel") => close(app),
        _ => {}
    }
}

const TITLE_H: f32 = 44.0;
const LEFT_W: f32 = 268.0;
const RIGHT_W: f32 = 340.0;
const CHAIN_H: f32 = 118.0;

/// The workspace; returns "ok" / "apply" / "cancel" when one was pressed.
fn layout(app: &mut EffectcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) -> Option<&'static str> {
    let title = Rect::from_min_size(rect.min, vec2(rect.width(), TITLE_H));
    let action = title_bar(app, ui, title, t);
    let body = Rect::from_min_max(pos2(rect.min.x + 8.0, title.max.y), pos2(rect.max.x - 8.0, rect.max.y - 8.0));
    let left = Rect::from_min_size(body.min, vec2(LEFT_W, body.height()));
    let right = Rect::from_min_max(pos2(body.max.x - RIGHT_W, body.min.y), body.max);
    let centre = Rect::from_min_max(pos2(left.max.x + 8.0, body.min.y), pos2(right.min.x - 8.0, body.max.y));
    let chain = Rect::from_min_max(pos2(centre.min.x, centre.max.y - CHAIN_H), centre.max);
    let view = Rect::from_min_max(centre.min, pos2(centre.max.x, chain.min.y - 8.0));
    for r in [left, right, chain, view] {
        ui.painter().rect_filled(r, 6.0, t.panel_bg);
    }
    browser::show(app, ui, left, t);
    preview::show(app, ui, view, t);
    chain::show(app, ui, chain, t);
    let scope_h = (right.height() * 0.36).clamp(190.0, 300.0);
    let insp = Rect::from_min_max(right.min, pos2(right.max.x, right.max.y - scope_h - 8.0));
    let sc = Rect::from_min_max(pos2(right.min.x, insp.max.y + 8.0), right.max);
    ui.painter().rect_filled(sc, 6.0, t.panel_bg);
    inspector::show(app, ui, insp, t);
    scopes::show(app, ui, sc, t);
    action
}

fn title_bar(app: &mut EffectcraftApp, ui: &mut egui::Ui, r: Rect, t: &Tokens) -> Option<&'static str> {
    let p = ui.painter().clone();
    let cy = r.center().y;
    // A small original mark: three overlapping lens circles in warm, neutral and cool.
    let m = pos2(r.min.x + 24.0, cy);
    for (dx, c) in [(-5.0, Color32::from_rgb(232, 140, 70)), (0.0, Color32::from_rgb(220, 220, 220)), (5.0, Color32::from_rgb(70, 160, 220))] {
        p.circle_stroke(m + vec2(dx, 0.0), 7.0, Stroke::new(1.6, c));
    }
    p.text(pos2(r.min.x + 42.0, cy), Align2::LEFT_CENTER, "CineLooks", Tokens::semibold(16.0), t.tab_text_active);
    // The look's name (editable).
    let nr = Rect::from_min_size(pos2(r.min.x + 150.0, cy - 12.0), vec2(220.0, 24.0));
    let mut name = app.dialog_state.cinelooks.draft.name.clone();
    let resp = w::text_field(ui, nr, &mut name, "Untitled look", t);
    app.auto.add(&aid("lookName"), nr, "Look name");
    if resp.changed() {
        app.dialog_state.cinelooks.edit(|l, _| l.name = name.clone());
    }
    // Where it applies.
    let st = &app.dialog_state.cinelooks;
    let layer_name = st
        .target
        .as_ref()
        .and_then(|tg| {
            app.session
                .project
                .comp(effectcraft_engine::project::ItemId(tg.comp))?
                .layer(effectcraft_engine::project::LayerId(tg.layer))
                .map(|l| l.name.clone())
        })
        .unwrap_or_default();
    let info = format!("on {layer_name}{}", if st.modified() { "  •  not applied" } else { "" });
    p.text(pos2(nr.max.x + 12.0, cy), Align2::LEFT_CENTER, info, Tokens::ui(12.0), t.text_dim);
    // Right: Undo, Redo · Cancel, Apply, OK.
    let at = |right: f32, wdt: f32| Rect::from_min_size(pos2(right - wdt, cy - 13.0), vec2(wdt, 26.0));
    let ok_r = at(r.max.x - 12.0, 72.0);
    let apply_r = at(ok_r.min.x - 8.0, 72.0);
    let cancel_r = at(apply_r.min.x - 8.0, 72.0);
    let redo_r = at(cancel_r.min.x - 20.0, 56.0);
    let undo_r = at(redo_r.min.x - 8.0, 56.0);
    let x = undo_r.min.x - 8.0;
    let mut out = None;
    if button(app, ui, ok_r, "ok", "OK", true) {
        out = Some("ok");
    }
    if button(app, ui, apply_r, "apply", "Apply", false) {
        out = Some("apply");
    }
    if button(app, ui, cancel_r, "cancel", "Cancel", false) {
        out = Some("cancel");
    }
    let can_redo = !app.dialog_state.cinelooks.redo.is_empty();
    let can_undo = !app.dialog_state.cinelooks.undo.is_empty();
    if button(app, ui, redo_r, "redo", "Redo", false) && can_redo {
        app.dialog_state.cinelooks.redo();
    }
    if button(app, ui, undo_r, "undo", "Undo", false) && can_undo {
        app.dialog_state.cinelooks.undo();
    }
    let msg = app.dialog_state.cinelooks.message.clone();
    if !msg.is_empty() {
        w::text_fit(&p, pos2(x - 8.0, cy), Align2::RIGHT_CENTER, &msg, Tokens::ui(11.5), (x - nr.max.x - 220.0).max(60.0), t.text_dim);
    }
    out
}

/// Add a node of `kind` after the selection (or at the end) and select it.
pub(crate) fn add_node(app: &mut EffectcraftApp, kind: ToolKind) {
    let st = &mut app.dialog_state.cinelooks;
    let at = st.selected.map_or(st.draft.nodes.len(), |i| (i + 1).min(st.draft.nodes.len()));
    st.edit(|l, _| l.nodes.insert(at, Node::new(kind)));
    st.selected = Some(at);
}

/// Effect Controls, under a CineLooks header: the look's summary and the button that opens the
/// editor.
#[allow(clippy::too_many_arguments)]
pub fn effect_controls_header(
    app: &mut EffectcraftApp,
    ui: &mut egui::Ui,
    p: &egui::Painter,
    layer: &effectcraft_engine::project::Layer,
    g: &effectcraft_engine::project::PropGroup,
    r: Rect,
    actions: &mut Vec<(String, Value)>,
) {
    let t = app.tokens;
    let b = Rect::from_min_size(pos2(r.min.x + 24.0, r.min.y + 5.0), vec2(178.0, 22.0));
    let resp = w::text_button(ui, b, "Open CineLooks Editor…", true, &t, egui::Id::new(("cl-open", g.uid)));
    app.auto.add(&format!("effectControls.effect.{}.cinelooksEditor", g.uid), b, "Open CineLooks Editor");
    if resp.clicked() {
        actions.push(("cinelooks.openEditor".into(), json!({"layer": layer.id.0, "effect": g.uid})));
    }
    let tools = effectcraft_engine::effects::cinelooks::node_groups(g).count();
    let name = match g.get("lookName").map(|p| &p.value) {
        Some(effectcraft_engine::keyframe::Value::Str(s)) if !s.is_empty() => s.clone(),
        _ => "Custom look".into(),
    };
    w::text_fit(
        p,
        pos2(b.max.x + 10.0, b.center().y),
        Align2::LEFT_CENTER,
        &format!("{name} · {tools} tool{}", if tools == 1 { "" } else { "s" }),
        Tokens::ui(11.5),
        (r.max.x - b.max.x - 16.0).max(20.0),
        t.text_dim,
    );
}

impl EffectcraftApp {
    /// The CineLooks Editor's draft look (tests, agents reading the UI state).
    pub fn cinelooks_draft(&self) -> &Look {
        &self.dialog_state.cinelooks.draft
    }

    /// Add a tool to the CineLooks Editor's draft after the selection and select it.
    pub fn cinelooks_add_tool(&mut self, kind: ToolKind) {
        add_node(self, kind);
    }
}

/// The working-space environment of the project.
pub(crate) fn working(app: &EffectcraftApp) -> effectcraft_engine::cinelooks::Working {
    effectcraft_engine::commands::cinelooks::working(&app.session)
}

/// The caches (lock with [`lock`]).
pub(crate) fn cache_arc(app: &EffectcraftApp) -> Arc<Mutex<preview::Cache>> {
    app.dialog_state.cinelooks.caches.0.clone()
}

/// Lock the caches (a poisoned lock still hands out its data).
pub(crate) fn lock(m: &Mutex<preview::Cache>) -> std::sync::MutexGuard<'_, preview::Cache> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
