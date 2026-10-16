//! The bottom strip: the look's tools in processing order. Click a card to select it, its dot
//! to switch it on or off, drag it to reorder; Add Tool (grouped menu), Duplicate and Delete act
//! on the selection. The chain reads left to right, the order the image goes through it.

use effectcraft_engine::cinelooks::tools::{TOOLS, ToolGroup};
use egui::{Align2, Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};

use super::{aid, button};
use crate::EffectcraftApp;
use crate::theme::Tokens;

const CARD_W: f32 = 116.0;
const CARD_H: f32 = 66.0;

fn group_color(g: ToolGroup) -> Color32 {
    match g {
        ToolGroup::Primary => Color32::from_rgb(86, 156, 214),
        ToolGroup::Film => Color32::from_rgb(214, 150, 70),
        ToolGroup::Lens => Color32::from_rgb(160, 120, 220),
        ToolGroup::Selective => Color32::from_rgb(90, 190, 140),
        ToolGroup::Utility => Color32::from_rgb(150, 150, 150),
    }
}

pub fn show(app: &mut EffectcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    let p = ui.painter().with_clip_rect(rect);
    let head = Rect::from_min_size(rect.min + vec2(10.0, 6.0), vec2(rect.width() - 20.0, 24.0));
    p.text(pos2(head.min.x, head.center().y), Align2::LEFT_CENTER, "Tool Chain", Tokens::semibold(13.0), t.tab_text_active);
    let n = app.dialog_state.cinelooks.draft.nodes.len();
    p.text(
        pos2(head.min.x + 84.0, head.center().y),
        Align2::LEFT_CENTER,
        format!("{n} tool{}  ·  processed left to right", if n == 1 { "" } else { "s" }),
        Tokens::ui(11.0),
        t.text_faint,
    );
    // Buttons on the right.
    let add_r = Rect::from_min_size(pos2(head.max.x - 104.0, head.min.y), vec2(104.0, 24.0));
    let dup_r = Rect::from_min_size(pos2(add_r.min.x - 88.0, head.min.y), vec2(82.0, 24.0));
    let del_r = Rect::from_min_size(pos2(dup_r.min.x - 70.0, head.min.y), vec2(64.0, 24.0));
    let sel = app.dialog_state.cinelooks.selected;
    let pop = egui::Id::new("cl-add-tool");
    if button(app, ui, add_r, "chain.add", "+ Add Tool", true) {
        crate::widgets::open_popup(ui, pop);
    }
    if crate::widgets::popup_is_open(ui, pop) {
        let rows = TOOLS.len() + ToolGroup::ALL.len();
        let chosen = crate::widgets::popup_list(ui, pop, add_r.left_bottom(), add_r, rows, ToolGroup::ALL.len(), |ui| {
            ui.set_min_width(220.0);
            let mut chosen = None;
            for g in ToolGroup::ALL {
                ui.label(egui::RichText::new(g.label()).small().color(group_color(g)));
                for tool in TOOLS.iter().filter(|x| x.group == g) {
                    if ui.selectable_label(false, tool.name).on_hover_text(tool.about).clicked() {
                        chosen = Some(tool.kind);
                    }
                }
                ui.separator();
            }
            chosen
        });
        if let Some(k) = chosen {
            super::add_node(app, k);
        }
    }
    if button(app, ui, dup_r, "chain.duplicate", "Duplicate", false)
        && let Some(i) = sel
    {
        let st = &mut app.dialog_state.cinelooks;
        st.edit(|l, _| {
            if let Some(n) = l.nodes.get(i).cloned() {
                let mut n = n;
                n.uid = 0;
                l.nodes.insert(i + 1, n);
            }
        });
        st.selected = Some((i + 1).min(st.draft.nodes.len().saturating_sub(1)));
    }
    if button(app, ui, del_r, "chain.delete", "Delete", false)
        && let Some(i) = sel
    {
        let st = &mut app.dialog_state.cinelooks;
        st.edit(|l, _| {
            if i < l.nodes.len() {
                l.nodes.remove(i);
            }
        });
        let n = st.draft.nodes.len();
        st.selected = if n == 0 { None } else { Some(i.min(n - 1)) };
    }
    // Delete / Backspace with a selection (not while typing).
    let typing = ui.ctx().memory(|m| m.focused().is_some());
    if !typing
        && sel.is_some()
        && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Delete) || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace))
    {
        let i = sel.unwrap_or(0);
        let st = &mut app.dialog_state.cinelooks;
        st.edit(|l, _| {
            if i < l.nodes.len() {
                l.nodes.remove(i);
            }
        });
        let n = st.draft.nodes.len();
        st.selected = if n == 0 { None } else { Some(i.min(n - 1)) };
    }

    // Cards.
    let strip = Rect::from_min_max(pos2(rect.min.x + 10.0, head.max.y + 8.0), pos2(rect.max.x - 10.0, rect.max.y - 8.0));
    let nodes = app.dialog_state.cinelooks.draft.nodes.clone();
    if nodes.is_empty() {
        p.text(strip.center(), Align2::CENTER_CENTER, "An empty look. Pick one from the library on the left, or add tools.", Tokens::ui(12.0), t.text_dim);
        return;
    }
    let scroll_id = egui::Id::new("cl-chain-scroll");
    let total = nodes.len() as f32 * (CARD_W + 10.0);
    let mut scroll: f32 = ui.data(|d| d.get_temp(scroll_id)).unwrap_or(0.0);
    let max_scroll = (total - strip.width()).max(0.0);
    if ui.rect_contains_pointer(strip) {
        let d = ui.input(|i| i.smooth_scroll_delta);
        scroll = (scroll - d.x - d.y).clamp(0.0, max_scroll);
    }
    scroll = scroll.clamp(0.0, max_scroll);
    ui.data_mut(|d| d.insert_temp(scroll_id, scroll));
    let pc = p.with_clip_rect(strip);
    let drag_id = egui::Id::new("cl-chain-drag");
    let dragging: Option<usize> = ui.data(|d| d.get_temp(drag_id));
    let mut drop_at: Option<usize> = None;
    for (i, node) in nodes.iter().enumerate() {
        let x = strip.min.x + i as f32 * (CARD_W + 10.0) - scroll;
        let card = Rect::from_min_size(pos2(x, strip.center().y - CARD_H * 0.5), vec2(CARD_W, CARD_H));
        let selected = app.dialog_state.cinelooks.selected == Some(i);
        let g = node.tool.def().group;
        let dim = !node.enabled || node.mix() <= 0.0;
        let bg = if selected { t.row_selected } else { t.field_bg };
        pc.rect_filled(card, 6.0, bg);
        pc.rect_filled(Rect::from_min_size(card.min, vec2(CARD_W, 4.0)), 2.0, if dim { group_color(g).gamma_multiply(0.35) } else { group_color(g) });
        pc.rect_stroke(card, 6.0, Stroke::new(if selected { 2.0 } else { 1.0 }, if selected { t.accent } else { t.field_border }), StrokeKind::Inside);
        // Order number and name.
        pc.text(card.min + vec2(8.0, 14.0), Align2::LEFT_CENTER, format!("{}", i + 1), Tokens::mono(10.5), t.text_faint);
        crate::widgets::text_fit(
            &pc,
            card.min + vec2(8.0, 32.0),
            Align2::LEFT_CENTER,
            &node.name(),
            Tokens::medium(12.0),
            CARD_W - 14.0,
            if dim { t.text_faint } else { t.text },
        );
        // Strength bar.
        let sb = Rect::from_min_size(card.min + vec2(8.0, CARD_H - 14.0), vec2(CARD_W - 16.0, 4.0));
        pc.rect_filled(sb, 2.0, Color32::from_black_alpha(90));
        pc.rect_filled(Rect::from_min_size(sb.min, vec2(sb.width() * node.mix(), sb.height())), 2.0, if dim { t.text_faint } else { t.accent });
        // On/off dot.
        let dot = Rect::from_center_size(pos2(card.max.x - 14.0, card.min.y + 14.0), vec2(14.0, 14.0));
        pc.circle_filled(dot.center(), 5.0, if node.enabled { Color32::from_rgb(110, 220, 120) } else { Color32::from_gray(80) });
        let dresp = ui.interact(dot, egui::Id::new(("cl-card-on", i)), Sense::click()).on_hover_text(if node.enabled { "Switch off" } else { "Switch on" });
        app.auto.add(&aid(&format!("chain.{i}.enabled")), dot, &node.name());
        if dresp.clicked() {
            let on = !node.enabled;
            app.dialog_state.cinelooks.edit(|l, _| {
                if let Some(n) = l.nodes.get_mut(i) {
                    n.enabled = on;
                }
            });
            continue;
        }
        let resp = ui.interact(card, egui::Id::new(("cl-card", i)), Sense::click_and_drag());
        app.auto.add(&aid(&format!("chain.{i}")), card, &node.name());
        if resp.clicked() || resp.drag_started() {
            app.dialog_state.cinelooks.selected = Some(i);
        }
        if resp.drag_started() {
            ui.data_mut(|d| d.insert_temp(drag_id, i));
        }
        if resp.hovered() {
            resp.on_hover_text(node.tool.def().about);
        }
    }
    // Reordering.
    if let Some(from) = dragging {
        if let Some(pp) = ui.input(|i| i.pointer.interact_pos()) {
            let k = ((pp.x - strip.min.x + scroll + (CARD_W + 10.0) * 0.5) / (CARD_W + 10.0)).floor().clamp(0.0, nodes.len() as f32) as usize;
            drop_at = Some(k);
            let mx = strip.min.x + k as f32 * (CARD_W + 10.0) - scroll - 5.0;
            pc.line_segment([pos2(mx, strip.min.y + 4.0), pos2(mx, strip.max.y - 4.0)], Stroke::new(3.0, t.accent));
        }
        if !ui.input(|i| i.pointer.any_down()) {
            ui.data_mut(|d| d.remove::<usize>(drag_id));
            if let Some(k) = drop_at {
                let to = if k > from { k - 1 } else { k };
                if to != from && from < nodes.len() {
                    let st = &mut app.dialog_state.cinelooks;
                    st.edit(|l, _| {
                        if from < l.nodes.len() {
                            let n = l.nodes.remove(from);
                            l.nodes.insert(to.min(l.nodes.len()), n);
                        }
                    });
                    st.selected = Some(to.min(st.draft.nodes.len().saturating_sub(1)));
                }
            }
        }
    }
}
