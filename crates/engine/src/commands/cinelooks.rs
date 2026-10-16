//! CineLooks commands: apply a look or a library preset, read and write an instance's look
//! (one undo step each), add tools, the user look library (save, delete, favourites, import,
//! export) and `.cube` export. The CineLooks Editor and agents use the same commands.
//!
//! An instance's look is its nested tool groups (see `effectcraft_effects::cinelooks`); the
//! look JSON (`effectcraft_cinelooks::Look`) is the exchange format. User looks and favourites
//! live in the settings store (`cinelooks/library.json`).

use effectcraft_cinelooks::look::{Look, LookFile, Node};
use effectcraft_cinelooks::presets;
use effectcraft_cinelooks::tools::{Def, Kind, TOOLS, ToolKind};
use effectcraft_cinelooks::{Env, Working};
use effectcraft_effects::cinelooks::{ID, NAME, look_from_params, merge_look, node_groups};
use effectcraft_project::build::Ids;
use effectcraft_project::{GroupKind, ItemId, LayerId, Uid};
use effectcraft_time::Tick;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{CommandSpec, bad, f_p, has_layers, layer_mut, layer_p, layers_p, str_p};
use crate::{EngineError, Result, Session, cmd, query};

const LIBRARY: &str = "cinelooks/library.json";

/// The CineLooks instance a command targets: `effect` (uid, 1-based index or name), else the
/// selected effect when it is CineLooks, else the layer's first CineLooks.
pub fn target(s: &Session, p: &Value, c: &str) -> Result<(ItemId, LayerId, Uid)> {
    let (cid, lid) = layer_p(s, p, c)?;
    let layer = s.project.comp(cid).and_then(|comp| comp.layer(lid)).ok_or(EngineError::NoComp)?;
    let fx = layer.effects().ok_or_else(|| bad(c, "this layer has no effects"))?;
    let is_cl = |g: &&effectcraft_project::PropGroup| matches!(&g.kind, GroupKind::Effect { effect } if effect == ID);
    let g = match p.get("effect") {
        Some(Value::Number(n)) => {
            let n = n.as_u64().unwrap_or(0);
            fx.groups().find(|g| g.uid == n).or_else(|| usize::try_from(n.saturating_sub(1)).ok().and_then(|i| fx.groups().nth(i)))
        }
        Some(Value::String(name)) => fx.groups().find(|g| &g.name == name),
        _ => s
            .state
            .selected_props
            .iter()
            .find_map(|(l, u)| (*l == lid).then(|| fx.groups().filter(is_cl).find(|g| g.uid == *u)).flatten())
            .or_else(|| fx.groups().find(is_cl)),
    }
    .ok_or_else(|| bad(c, "no CineLooks effect on this layer (apply one with cinelooks.apply)"))?;
    if !is_cl(&g) {
        return Err(bad(c, format!("`{}` is not a CineLooks effect", g.name)));
    }
    Ok((cid, lid, g.uid))
}

/// The project's colour environment for CineLooks (what the renderer hands the effect).
pub fn working(s: &Session) -> Working {
    let st = &s.project.settings;
    Working::new(st.working_space, st.working_space.is_some_and(|w| st.linearize || w.is_linear()))
}

/// The instance's look and look strength (0..100) evaluated at comp time `t`, its layer size.
pub fn read_look(s: &Session, cid: ItemId, lid: LayerId, uid: Uid, t: Tick) -> Result<(Look, f64, [f64; 2])> {
    let comp = s.project.comp(cid).ok_or(EngineError::NoComp)?;
    let layer = comp.layer(lid).ok_or(EngineError::NoComp)?;
    let g = layer.props.find_group(uid).ok_or_else(|| bad("cinelooks", "the effect is gone"))?;
    let size = effectcraft_render::effect_bounds(&s.project, comp, layer).0;
    let ctx = effectcraft_render::EvalCtx { project: &s.project, comp_id: cid, comp, time: t, expr: s.expr.as_deref(), footage: None };
    let params = effectcraft_effects::flatten_params(g, &mut |pr| ctx.value(layer, pr));
    Ok((look_from_params(&params, size), params.f("lookStrength"), size))
}

/// A look from a command parameter: a JSON object or a JSON string.
fn look_p(p: &Value, c: &str) -> Result<Look> {
    match p.get("look") {
        Some(Value::String(t)) => Look::parse(t).map_err(|e| bad(c, e)),
        Some(v @ Value::Object(_)) => Look::parse(&v.to_string()).map_err(|e| bad(c, e)),
        _ => Err(bad(c, "missing `look`")),
    }
}

/// A look by preset id (built-in or `user.…`) or name.
fn preset_look(s: &Session, id: &str) -> Option<Look> {
    if let Some(p) = presets::find(id) {
        return Some(p.look.clone());
    }
    let lib = Library::load(s);
    lib.presets.into_iter().find(|p| p.id == id || p.name.eq_ignore_ascii_case(id)).map(|p| p.look)
}

/// Merge `look` (and optionally the look strength) into instance `uid` in one undo step.
fn write_look(s: &mut Session, label: &str, cid: ItemId, lid: LayerId, uid: Uid, look: &Look, strength: Option<f64>, t: Tick) -> Result<Look> {
    let (size, lt) = {
        let comp = s.project.comp(cid).ok_or(EngineError::NoComp)?;
        let layer = comp.layer(lid).ok_or(EngineError::NoComp)?;
        (effectcraft_render::effect_bounds(&s.project, comp, layer).0, layer.layer_time(t))
    };
    s.edit(label, None, |proj, _| {
        let mut next = proj.next_id;
        let g = layer_mut(proj, cid, lid)?.props.find_group_mut(uid).ok_or_else(|| bad("cinelooks", "the effect is gone"))?;
        let out = merge_look(g, look, &mut Ids(&mut next), size, lt);
        if let Some(v) = strength.filter(|v| v.is_finite())
            && let Some(pr) = g.get_mut("lookStrength")
        {
            pr.set_value_at(lt, effectcraft_keyframe::Value::Scalar(v.clamp(0.0, 100.0)));
        }
        proj.next_id = next;
        Ok(out)
    })
}

fn look_json(look: &Look) -> Value {
    serde_json::to_value(look).unwrap_or(Value::Null)
}

/// `cinelooks.apply {layers?, preset? | look?, strength?}`: apply CineLooks to the layers (or
/// reuse the instance they have) with a library look or the given look. One undo step.
fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.apply";
    let look = match (str_p(p, "preset"), p.get("look")) {
        (Some(id), _) => preset_look(s, id).ok_or_else(|| bad(c, format!("no look `{id}`")))?,
        (None, Some(_)) => look_p(p, c)?,
        (None, None) => Look::default(),
    };
    let (cid, ids) = layers_p(s, p)?;
    if ids.is_empty() {
        return Err(bad(c, "no layer"));
    }
    let t = super::time_p(s, p, s.project.comp(cid));
    let strength = f_p(p, "strength");
    let comp = s.project.comp(cid).ok_or(EngineError::NoComp)?;
    let mut plan = vec![];
    for lid in &ids {
        let Some(l) = comp.layer(*lid) else { continue };
        let existing = l.effects().and_then(|fx| fx.groups().find(|g| matches!(&g.kind, GroupKind::Effect { effect } if effect == ID)).map(|g| g.uid));
        plan.push((*lid, existing, effectcraft_render::effect_bounds(&s.project, comp, l).0, l.layer_time(t)));
    }
    let spec = effectcraft_effects::find(ID).ok_or_else(|| bad(c, "CineLooks is not registered"))?;
    let label = if look.name.is_empty() { "Apply CineLooks".to_string() } else { format!("Apply Look {}", look.name) };
    let uids = s.edit(&label, None, |proj, st| {
        let mut out = vec![];
        for (lid, existing, size, lt) in &plan {
            let mut next = proj.next_id;
            let l = layer_mut(proj, cid, *lid)?;
            let fx = l.props.sub_mut("effects").ok_or_else(|| bad(c, "this layer type has no effects"))?;
            let uid = match existing {
                Some(u) => *u,
                None => {
                    let g = effectcraft_effects::instantiate(spec, &mut Ids(&mut next), NAME, *size);
                    let u = g.uid;
                    fx.children.push(g.into());
                    u
                }
            };
            let g = fx.find_group_mut(uid).ok_or_else(|| bad(c, "the effect is gone"))?;
            merge_look(g, &look, &mut Ids(&mut next), *size, *lt);
            if let Some(v) = strength.filter(|v| v.is_finite())
                && let Some(pr) = g.get_mut("lookStrength")
            {
                pr.set_value_at(*lt, effectcraft_keyframe::Value::Scalar(v.clamp(0.0, 100.0)));
            }
            proj.next_id = next;
            st.selected_props = vec![(*lid, uid)];
            out.push(uid);
        }
        st.last_effect = Some(ID.to_string());
        Ok(out)
    })?;
    Ok(json!({"effects": uids, "look": look.name}))
}

/// `cinelooks.getLook {layer?, effect?, time?}`.
fn get_look(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.getLook";
    let (cid, lid, uid) = target(s, p, c)?;
    let t = super::time_p(s, p, s.project.comp(cid));
    let (look, strength, size) = read_look(s, cid, lid, uid, t)?;
    let env = Env { layer_size: size, working: working(s), ..Env::default() };
    let warnings = effectcraft_cinelooks::compile(&look, &env, 1.0).warnings;
    Ok(json!({"layer": lid.0, "effect": uid, "look": look_json(&look), "strength": strength, "warnings": warnings}))
}

/// `cinelooks.setLook {layer?, effect?, look, strength?, time?}`: one undo step.
fn set_look(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.setLook";
    let (cid, lid, uid) = target(s, p, c)?;
    let look = look_p(p, c)?;
    let t = super::time_p(s, p, s.project.comp(cid));
    let label = str_p(p, "label").unwrap_or("Edit Look").to_string();
    let out = write_look(s, &label, cid, lid, uid, &look, f_p(p, "strength"), t)?;
    Ok(json!({"effect": uid, "look": look_json(&out)}))
}

/// `cinelooks.addTool {layer?, effect?, tool, index?, params?}`.
fn add_tool(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.addTool";
    let (cid, lid, uid) = target(s, p, c)?;
    let tool = str_p(p, "tool").ok_or_else(|| bad(c, "missing `tool`"))?;
    let kind = ToolKind::from_id(tool)
        .or_else(|| ToolKind::all().find(|k| k.name().eq_ignore_ascii_case(tool)))
        .ok_or_else(|| bad(c, format!("unknown tool `{tool}` (see cinelooks.tools)")))?;
    let t = super::time_p(s, p, s.project.comp(cid));
    let (mut look, _, _) = read_look(s, cid, lid, uid, t)?;
    let mut node = Node::new(kind);
    if let Some(Value::Object(m)) = p.get("params") {
        for (k, v) in m {
            if let Ok(pv) = serde_json::from_value(v.clone()) {
                node.params.insert(k.clone(), pv);
            }
        }
    }
    let at = p.get("index").and_then(Value::as_u64).map_or(look.nodes.len(), |i| (i as usize).saturating_sub(1).min(look.nodes.len()));
    look.nodes.insert(at, node);
    let out = write_look(s, &format!("Add {}", kind.name()), cid, lid, uid, &look, None, t)?;
    let node_uid = out.nodes.get(at).map(|n| n.uid);
    Ok(json!({"effect": uid, "node": node_uid, "index": at + 1}))
}

/// `cinelooks.tools`: the tool catalogue (ids, names, groups, parameters with ranges).
fn tools(_: &mut Session, _: &Value) -> Result<Value> {
    let list: Vec<Value> = TOOLS
        .iter()
        .map(|t| {
            let params: Vec<Value> = t
                .params
                .iter()
                .map(|d| {
                    let (kind, range) = match d.kind {
                        Kind::Num { min, max, .. } => ("number", json!([min, max])),
                        Kind::Angle => ("angle", Value::Null),
                        Kind::Color => ("color", Value::Null),
                        Kind::Wheel => ("wheel", Value::Null),
                        Kind::Point => ("point", Value::Null),
                        Kind::Choice(o) => ("choice", json!(o)),
                        Kind::Toggle => ("toggle", Value::Null),
                        Kind::Curve => ("curve", Value::Null),
                        Kind::File(_) => ("file", Value::Null),
                        Kind::Data => ("data", Value::Null),
                    };
                    let default = match d.def {
                        Def::Num(v) => json!(v),
                        Def::Vec3(v) => json!(v),
                        Def::Point(v) => json!(v),
                        Def::Color(v) => json!(v),
                        Def::Bool(v) => json!(v),
                        Def::Choice(v) => json!(v),
                        Def::Text(v) => json!(v),
                    };
                    json!({"id": d.id, "name": d.name, "kind": kind, "range": range, "default": default})
                })
                .collect();
            json!({"id": t.kind.id(), "name": t.name, "group": t.group.label(), "about": t.about, "spatial": !t.kind.is_color_only(), "params": params})
        })
        .collect();
    Ok(json!({"tools": list}))
}

// ---------------------------------------------------------------------------- user library

/// A user look as stored.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UserPreset {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub description: String,
    pub look: Look,
}

/// The user's look library and favourites.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Library {
    #[serde(default)]
    pub presets: Vec<UserPreset>,
    #[serde(default)]
    pub favorites: Vec<String>,
}

impl Library {
    /// The stored library (empty when there is none or it can't be read).
    pub fn load(s: &Session) -> Library {
        let Some(cfg) = &s.config else { return Library::default() };
        match cfg.read_bounded(LIBRARY, 64 << 20) {
            Ok(Some(t)) => serde_json::from_str::<Library>(&t)
                .map(|mut l| {
                    for p in &mut l.presets {
                        p.look.sanitize();
                    }
                    l
                })
                .unwrap_or_default(),
            _ => Library::default(),
        }
    }

    fn save(&self, s: &Session, c: &str) -> Result<()> {
        let cfg = s.config.as_ref().ok_or_else(|| bad(c, "no settings store: the user look library is unavailable"))?;
        let text = serde_json::to_string_pretty(self).map_err(|e| EngineError::Other(e.to_string()))?;
        cfg.write(LIBRARY, &text).map_err(|e| EngineError::Other(format!("cannot save the look library: {e}")))
    }
}

fn slug(name: &str) -> String {
    let s: String = name.to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect();
    let s = s.split('-').filter(|x| !x.is_empty()).collect::<Vec<_>>().join("-");
    if s.is_empty() { "look".into() } else { s.chars().take(60).collect() }
}

/// Add (or replace, by name) a user look; its id.
fn store(s: &Session, c: &str, name: &str, category: &str, description: &str, look: Look) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(bad(c, "the look needs a name"));
    }
    let mut lib = Library::load(s);
    let mut look = look.detached();
    look.name = name.to_string();
    look.sanitize();
    let id = match lib.presets.iter().find(|p| p.name.eq_ignore_ascii_case(name)) {
        Some(p) => p.id.clone(),
        None => {
            let base = format!("user.{}", slug(name));
            let mut id = base.clone();
            let mut n = 2;
            while lib.presets.iter().any(|p| p.id == id) {
                id = format!("{base}-{n}");
                n += 1;
            }
            id
        }
    };
    let entry = UserPreset {
        id: id.clone(),
        name: name.into(),
        category: if category.trim().is_empty() { "User".into() } else { category.trim().into() },
        description: description.into(),
        look,
    };
    match lib.presets.iter_mut().find(|p| p.id == id) {
        Some(p) => *p = entry,
        None => lib.presets.push(entry),
    }
    lib.save(s, c)?;
    Ok(id)
}

/// `cinelooks.presets {category?, search?}`: the built-in library, then the user's looks.
fn list_presets(s: &mut Session, p: &Value) -> Result<Value> {
    let lib = Library::load(s);
    let cat = str_p(p, "category").map(str::to_lowercase);
    let q = str_p(p, "search").map(str::to_lowercase).unwrap_or_default();
    let mut out = vec![];
    let built = presets::builtin().iter().map(|b| (b.id.as_str(), b.name.as_str(), b.category.as_str(), b.description.as_str(), false, b.look.nodes.len()));
    let user = lib.presets.iter().map(|u| (u.id.as_str(), u.name.as_str(), u.category.as_str(), u.description.as_str(), true, u.look.nodes.len()));
    for (id, name, category, description, is_user, nodes) in built.chain(user) {
        if cat
            .as_ref()
            .is_some_and(|c| *c != category.to_lowercase() && !(c == "favorites" && lib.favorites.iter().any(|f| f == id)) && !(c == "user" && is_user))
        {
            continue;
        }
        if !q.is_empty() && !name.to_lowercase().contains(&q) && !category.to_lowercase().contains(&q) && !description.to_lowercase().contains(&q) {
            continue;
        }
        out.push(json!({"id": id, "name": name, "category": category, "description": description, "user": is_user, "favorite": lib.favorites.iter().any(|f| f == id), "tools": nodes}));
    }
    Ok(json!({"categories": presets::CATEGORIES, "presets": out}))
}

/// `cinelooks.savePreset {name, category?, description?, look? | layer?/effect?}`.
fn save_preset(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.savePreset";
    let name = str_p(p, "name").ok_or_else(|| bad(c, "missing `name`"))?;
    let look = if p.get("look").is_some() {
        look_p(p, c)?
    } else {
        let (cid, lid, uid) = target(s, p, c)?;
        let t = super::time_p(s, p, s.project.comp(cid));
        read_look(s, cid, lid, uid, t)?.0
    };
    let id = store(s, c, name, str_p(p, "category").unwrap_or(""), str_p(p, "description").unwrap_or(""), look)?;
    s.toast(format!("Saved look “{name}”"));
    Ok(json!({"id": id}))
}

/// `cinelooks.deletePreset {id}` (user looks only).
fn delete_preset(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.deletePreset";
    let id = str_p(p, "id").ok_or_else(|| bad(c, "missing `id`"))?;
    let mut lib = Library::load(s);
    let before = lib.presets.len();
    lib.presets.retain(|x| x.id != id);
    if lib.presets.len() == before {
        return Err(bad(c, format!("no user look `{id}` (built-in looks can't be deleted)")));
    }
    lib.favorites.retain(|f| f != id);
    lib.save(s, c)?;
    Ok(json!({"deleted": id}))
}

/// `cinelooks.setFavorite {id, value?}` (toggles without `value`).
fn set_favorite(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.setFavorite";
    let id = str_p(p, "id").ok_or_else(|| bad(c, "missing `id`"))?;
    if preset_look(s, id).is_none() {
        return Err(bad(c, format!("no look `{id}`")));
    }
    let mut lib = Library::load(s);
    let on = p.get("value").and_then(Value::as_bool).unwrap_or(!lib.favorites.iter().any(|f| f == id));
    lib.favorites.retain(|f| f != id);
    if on {
        lib.favorites.push(id.to_string());
    }
    lib.save(s, c)?;
    Ok(json!({"id": id, "favorite": on}))
}

/// `cinelooks.importPreset {path, category?}`: a `.cinelook` file, or a `.cube` LUT (a look
/// with one LUT tool that keeps a copy of the table).
fn import_preset(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.importPreset";
    let path = str_p(p, "path").ok_or_else(|| bad(c, "missing `path`"))?;
    let bytes = s.services.read_file(path).map_err(|e| bad(c, format!("cannot read {path}: {e}")))?;
    if bytes.len() > 256 << 20 {
        return Err(bad(c, "the file is larger than 256 MB"));
    }
    let text = String::from_utf8_lossy(&bytes).to_string();
    let stem = std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "Imported Look".into());
    let (name, category, description, look) = if path.to_ascii_lowercase().ends_with(".cube") {
        effectcraft_cinelooks::lut::parse_cube(&text).map_err(|e| bad(c, format!("{stem}.cube: {e}")))?;
        let look = Look::new(&stem, vec![Node::new(ToolKind::Lut).with("file", path).with("data", text)]);
        (stem, "LUTs".to_string(), "Imported .cube LUT".to_string(), look)
    } else {
        let v: Value = serde_json::from_str(&text).map_err(|e| bad(c, format!("not a CineLooks look file: {e}")))?;
        let look = Look::parse(&text).map_err(|e| bad(c, e))?;
        let name = v
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| if look.name.is_empty() { stem.clone() } else { look.name.clone() });
        let category = v.get("category").and_then(Value::as_str).unwrap_or("Imported").to_string();
        let description = v.get("description").and_then(Value::as_str).unwrap_or("").to_string();
        (name, category, description, look)
    };
    let category = str_p(p, "category").map(str::to_string).unwrap_or(category);
    let id = store(s, c, &name, &category, &description, look)?;
    Ok(json!({"id": id, "name": name}))
}

/// `cinelooks.exportPreset {path, id? | layer?/effect?}`: write a `.cinelook` file.
fn export_preset(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.exportPreset";
    let path = str_p(p, "path").ok_or_else(|| bad(c, "missing `path`"))?;
    let (look, category, description) = match (str_p(p, "id"), p.get("look")) {
        (_, Some(_)) => (look_p(p, c)?, String::new(), String::new()),
        (id, None) => match id {
            Some(id) => {
                if let Some(b) = presets::find(id) {
                    (b.look.clone(), b.category.clone(), b.description.clone())
                } else {
                    let lib = Library::load(s);
                    let u = lib.presets.into_iter().find(|u| u.id == id).ok_or_else(|| bad(c, format!("no look `{id}`")))?;
                    (u.look, u.category, u.description)
                }
            }
            None => {
                let (cid, lid, uid) = target(s, p, c)?;
                let t = super::time_p(s, p, s.project.comp(cid));
                (read_look(s, cid, lid, uid, t)?.0, String::new(), String::new())
            }
        },
    };
    let mut look = look;
    if let Some(n) = str_p(p, "name") {
        look.name = n.to_string();
    }
    let file: LookFile = look.to_file(&category, &description);
    let text = serde_json::to_string_pretty(&file).map_err(|e| EngineError::Other(e.to_string()))?;
    s.services.write_file(path, text.as_bytes()).map_err(|e| bad(c, format!("cannot write {path}: {e}")))?;
    Ok(json!({"path": path, "tools": look.nodes.len()}))
}

/// `cinelooks.exportCube {path, size?, layer?, effect?, time?}`: the colour tools of the
/// instance's look as a `.cube` (spatial / positional tools are reported, not baked).
fn export_cube(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.exportCube";
    let path = str_p(p, "path").ok_or_else(|| bad(c, "missing `path`"))?;
    let (cid, lid, uid) = target(s, p, c)?;
    let t = super::time_p(s, p, s.project.comp(cid));
    let (look, strength, size) = read_look(s, cid, lid, uid, t)?;
    // An editor's draft instead of the instance's look.
    let (look, strength) = if p.get("look").is_some() { (look_p(p, c)?, f_p(p, "strength").unwrap_or(strength)) } else { (look, strength) };
    let env = Env { layer_size: size, working: working(s), ..Env::default() };
    let n = p.get("size").and_then(Value::as_u64).unwrap_or(33) as usize;
    let title = if look.name.is_empty() { "CineLooks" } else { look.name.as_str() };
    let out = effectcraft_cinelooks::export::bake_cube(&look, &env, (strength / 100.0) as f32, n, title).map_err(|e| bad(c, e))?;
    s.services.write_file(path, out.text.as_bytes()).map_err(|e| bad(c, format!("cannot write {path}: {e}")))?;
    if !out.skipped.is_empty() {
        s.toast(format!("Exported the colour tools; not in the LUT: {}", out.skipped.join(", ")));
    }
    Ok(json!({"path": path, "size": n.clamp(2, effectcraft_cinelooks::lut::MAX_3D), "skipped": out.skipped}))
}

/// `cinelooks.openEditor {layer?, effect?}`: check the target, then the UI opens the editor.
fn open_editor(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.openEditor";
    let (cid, lid, uid) = target(s, p, c)?;
    super::frontend(s, c, &json!({"comp": cid.0, "layer": lid.0, "effect": uid}))
}

/// `cinelooks.chooseLut {node?, path}`: hand a picked `.cube` to the editor's LUT tool
/// (checked here so agents get an error for a bad file).
fn choose_lut(s: &mut Session, p: &Value) -> Result<Value> {
    let c = "cinelooks.chooseLut";
    let path = str_p(p, "path").ok_or_else(|| bad(c, "missing `path`"))?;
    let bytes = s.services.read_file(path).map_err(|e| bad(c, format!("cannot read {path}: {e}")))?;
    effectcraft_cinelooks::lut::parse_cube(&String::from_utf8_lossy(&bytes)).map_err(|e| bad(c, format!("{path}: {e}")))?;
    super::frontend(s, c, p)
}

/// Node uids of an instance, in order (tests, agents).
pub fn node_uids(s: &Session, cid: ItemId, lid: LayerId, uid: Uid) -> Vec<Uid> {
    s.project.comp(cid).and_then(|c| c.layer(lid)).and_then(|l| l.props.find_group(uid)).map(|g| node_groups(g).map(|n| n.uid).collect()).unwrap_or_default()
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "cinelooks.apply",
            "Apply CineLooks Look",
            [],
            None,
            "{layers?, preset?: id|name, look?: look JSON, strength?: 0-100} — adds CineLooks (or reuses the layer's) with the look; one undo step",
            has_layers,
            apply
        ),
        cmd!(
            "cinelooks.openEditor",
            "CineLooks Editor...",
            [],
            None,
            "{layer?, effect?: uid|index|name} — opens the CineLooks Editor on the effect",
            has_layers,
            open_editor
        ),
        cmd!(
            "cinelooks.setLook",
            "Edit Look",
            [],
            None,
            "{layer?, effect?, look: look JSON, strength?, time?, label?} — applies the look to the instance (keys animated parameters at the time); one undo step",
            has_layers,
            set_look
        ),
        cmd!(
            "cinelooks.chooseLut",
            "Choose LUT...",
            [],
            None,
            "{path, node?} — the CineLooks Editor's LUT tool loads the .cube",
            crate::commands::always,
            choose_lut
        ),
        cmd!("cinelooks.addTool", "Add Look Tool", [], None, "{layer?, effect?, tool: id|name, index?: 1-based, params?: {id: value}}", has_layers, add_tool),
        cmd!(
            "cinelooks.savePreset",
            "Save Look",
            [],
            None,
            "{name, category?, description?, look? | layer?/effect?/time?} → {id}",
            crate::commands::always,
            save_preset
        ),
        cmd!("cinelooks.deletePreset", "Delete Look", [], None, "{id} (user looks)", crate::commands::always, delete_preset),
        cmd!("cinelooks.setFavorite", "Favorite Look", [], None, "{id, value?}", crate::commands::always, set_favorite),
        cmd!("cinelooks.importPreset", "Import Look...", [], None, "{path: .cinelook or .cube, category?} → {id}", crate::commands::always, import_preset),
        cmd!(
            "cinelooks.exportPreset",
            "Export Look...",
            [],
            None,
            "{path, id? | look? | layer?/effect?, name?} — writes a .cinelook file",
            crate::commands::always,
            export_preset
        ),
        cmd!(
            "cinelooks.exportCube",
            "Export Look as LUT...",
            [],
            None,
            "{path, size?: 2-129 (33), layer?, effect?, look?, strength?, time?} — colour tools only",
            has_layers,
            export_cube
        ),
        query!("cinelooks.getLook", "Get Look", "{layer?, effect?, time?} → {look, strength, warnings}", get_look),
        query!("cinelooks.presets", "List Looks", "{category?, search?} → built-in and user looks", list_presets),
        query!("cinelooks.tools", "List Look Tools", "{} → the tool catalogue with parameters", tools),
    ]
}

#[cfg(test)]
mod tests;
