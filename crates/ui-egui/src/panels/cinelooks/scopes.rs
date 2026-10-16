//! Scopes of the graded preview as displayed (`effectcraft_raster::scopes`, the maths of the
//! Lumetri Scopes panel): histogram, luma waveform, RGB parade and vectorscope, with the share
//! of pixels clipped at white and at black. They follow every change to the draft.

use effectcraft_engine::raster::scopes::{self, ColorStandard, ScopeKind, ScopeOpts};
use egui::{Align2, Color32, Rect, Stroke, pos2, vec2};

use super::{aid, button};
use crate::EffectcraftApp;
use crate::theme::Tokens;

const KINDS: [(ScopeKind, &str, &str); 4] = [
    (ScopeKind::Histogram, "histogram", "Histogram"),
    (ScopeKind::WaveformLuma, "waveform", "Waveform"),
    (ScopeKind::ParadeRgb, "parade", "Parade"),
    (ScopeKind::VectorscopeYuv, "vectorscope", "Vector"),
];

/// Clipping of a display image: fractions of pixels at white and at black (opaque pixels).
pub fn clip_fractions(img: &effectcraft_engine::raster::Image) -> [f32; 2] {
    let (mut hi, mut lo, mut n) = (0usize, 0usize, 0usize);
    for p in &img.data {
        if p[3] < 0.5 {
            continue;
        }
        n += 1;
        let c = [p[0] / p[3], p[1] / p[3], p[2] / p[3]];
        if c.iter().any(|v| *v >= 0.999) {
            hi += 1;
        } else if c.iter().all(|v| *v <= 0.002) {
            lo += 1;
        }
    }
    let n = n.max(1) as f32;
    [hi as f32 / n, lo as f32 / n]
}

pub fn show(app: &mut EffectcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    let mut x = rect.min.x + 8.0;
    let cur = app.dialog_state.cinelooks.scope.min(KINDS.len() - 1);
    for (k, (_, id, label)) in KINDS.iter().enumerate() {
        let r = Rect::from_min_size(pos2(x, rect.min.y + 6.0), vec2(76.0, 22.0));
        if button(app, ui, r, &format!("scope.{id}"), label, k == cur) {
            app.dialog_state.cinelooks.scope = k;
        }
        x += 80.0;
    }
    let plot = Rect::from_min_max(pos2(rect.min.x + 8.0, rect.min.y + 34.0), pos2(rect.max.x - 8.0, rect.max.y - 26.0));
    let p = ui.painter().with_clip_rect(rect);
    p.rect_filled(plot, 3.0, Color32::from_gray(14));
    let (kind, _, label) = KINDS[cur];
    let arc = super::cache_arc(app);
    let (display, generation) = {
        let c = super::lock(&arc);
        (c.display.clone(), c.display_gen)
    };
    let Some(img) = display else {
        p.text(plot.center(), Align2::CENTER_CENTER, "No image", Tokens::ui(12.0), t.text_faint);
        return;
    };
    // Square for the vectorscope.
    let area = if kind.is_vectorscope() { Rect::from_center_size(plot.center(), vec2(plot.height(), plot.height())) } else { plot };
    let (w, h) = ((area.width() as u32).clamp(64, 512), (area.height() as u32).clamp(64, 512));
    let (tex, bars) = {
        let mut c = super::lock(&arc);
        let stale = c.scope_key != Some((generation, cur))
            || (kind == ScopeKind::Histogram && c.scope_bars.as_ref().is_none_or(|b| b[0].len() != w as usize))
            || (kind != ScopeKind::Histogram && c.scope_tex.as_ref().is_none_or(|t| t.size() != [w as usize, h as usize]));
        if stale {
            let o = ScopeOpts { standard: ColorStandard::Rec709, float: false, clamp: true, width: w, height: h };
            let sc = scopes::compute(&img, kind, &o);
            if kind == ScopeKind::Histogram {
                c.scope_bars = Some(scopes::histogram_bars(&sc));
                c.scope_tex = None;
            } else {
                let rgba = sc.to_rgba8(matches!(kind, ScopeKind::WaveformLuma));
                let ci = egui::ColorImage::from_rgba_unmultiplied([sc.width as usize, sc.height as usize], &rgba);
                c.scope_tex = Some(ui.ctx().load_texture("cinelooks-scope", ci, egui::TextureOptions::LINEAR));
                c.scope_bars = None;
            }
            c.scope_key = Some((generation, cur));
        }
        (c.scope_tex.clone(), c.scope_bars.clone())
    };
    if let Some(tx) = tex {
        p.image(tx.id(), area, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    }
    if let Some(b) = bars {
        // R, G, B bins as overlapping bars (square-root scale so small counts stay visible).
        let max = b.iter().flat_map(|v| v.iter()).copied().max().unwrap_or(1).max(1) as f32;
        let cols = [
            Color32::from_rgba_unmultiplied(255, 70, 70, 120),
            Color32::from_rgba_unmultiplied(70, 230, 90, 120),
            Color32::from_rgba_unmultiplied(80, 140, 255, 120),
        ];
        let n = b[0].len().max(1) as f32;
        for (ch, bins) in b.iter().enumerate() {
            for (x, cnt) in bins.iter().enumerate() {
                if *cnt == 0 {
                    continue;
                }
                let hh = (*cnt as f32 / max).sqrt() * (area.height() - 2.0);
                let x0 = area.min.x + area.width() * x as f32 / n;
                let x1 = area.min.x + area.width() * (x + 1) as f32 / n;
                p.rect_filled(Rect::from_min_max(pos2(x0, area.max.y - hh), pos2(x1.max(x0 + 1.0), area.max.y)), 0.0, cols[ch]);
            }
        }
    }
    app.auto.add(&aid("scope"), area, label);
    // Graticule.
    let grid = Stroke::new(1.0, Color32::from_white_alpha(22));
    if kind.is_vectorscope() {
        p.circle_stroke(area.center(), area.width() * 0.5 - 1.0, grid);
        p.circle_stroke(area.center(), area.width() * 0.25, grid);
        for (name, xy) in scopes::vector_targets(ColorStandard::Rec709, false) {
            // Targets are (u, v) in −0.5…0.5 of the plot.
            let q = pos2(area.center().x + xy[0] * area.width(), area.center().y - xy[1] * area.height());
            p.rect_stroke(Rect::from_center_size(q, vec2(7.0, 7.0)), 0.0, Stroke::new(1.0, Color32::from_white_alpha(70)), egui::StrokeKind::Middle);
            p.text(q + vec2(6.0, -6.0), Align2::LEFT_BOTTOM, name, Tokens::ui(9.0), Color32::from_white_alpha(90));
        }
    } else {
        for k in 0..=4 {
            let y = area.max.y - area.height() * k as f32 / 4.0;
            p.line_segment([pos2(area.min.x, y), pos2(area.max.x, y)], grid);
        }
        if kind != ScopeKind::Histogram {
            for (k, txt) in ["0", "25", "50", "75", "100"].iter().enumerate() {
                let y = area.max.y - area.height() * k as f32 / 4.0;
                let (at, al) =
                    if k == 4 { (pos2(area.min.x + 2.0, y + 1.0), Align2::LEFT_TOP) } else { (pos2(area.min.x + 2.0, y - 1.0), Align2::LEFT_BOTTOM) };
                p.text(at, al, *txt, Tokens::ui(9.0), Color32::from_white_alpha(80));
            }
        }
    }
    // Clipping readout.
    let [hi, lo] = clip_fractions(&img);
    let y = rect.max.y - 13.0;
    let col = |v: f32, warn: Color32| if v > 0.001 { warn } else { t.text_faint };
    p.circle_filled(pos2(rect.min.x + 14.0, y), 4.0, col(lo, Color32::from_rgb(70, 130, 255)));
    p.text(pos2(rect.min.x + 22.0, y), Align2::LEFT_CENTER, format!("Shadows clipped {:.2}%", lo * 100.0), Tokens::ui(11.0), t.text_dim);
    p.circle_filled(pos2(rect.center().x + 8.0, y), 4.0, col(hi, Color32::from_rgb(255, 70, 70)));
    p.text(pos2(rect.center().x + 16.0, y), Align2::LEFT_CENTER, format!("Highlights clipped {:.2}%", hi * 100.0), Tokens::ui(11.0), t.text_dim);
}
