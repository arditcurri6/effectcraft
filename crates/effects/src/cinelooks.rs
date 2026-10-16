//! CineLooks (`ec.color.cinelooks`): modular cinematic grading. The grading engine is
//! `effectcraft-cinelooks`; this adapter maps an effect instance to a [`Look`] and back.
//!
//! An instance has three top-level parameters (Look Strength, Show Original, the hidden Look
//! Name) and one nested `Indexed` group per tool node, in processing order: the group's match
//! id is the tool (`exposure`, `wheels`…), its uid the node's identity, `enabled` its switch,
//! and its properties the node's Strength and the tool's parameters. So every tool parameter
//! animates, shows in the Timeline, serialises and undoes like any other property, and the
//! generic `prop.moveGroup` / `setGroupEnabled` / `removeGroup` / `duplicateGroup` reorder,
//! switch, delete and duplicate nodes. Looks store points as fractions of the layer size;
//! instances store layer pixels.

use effectcraft_cinelooks::look::{Look, Node, PV};
use effectcraft_cinelooks::tools::{Kind, ParamDef, ToolKind};
use effectcraft_cinelooks::{Env, Program, Working};
use effectcraft_keyframe::Value;
use effectcraft_project::build::Ids;
use effectcraft_project::{GroupKind, ParamUi, PropGroup, Property};
use effectcraft_time::Tick;

use crate::{Buf, EffectCtx, EffectSpec, Params, num, p, slider};

pub const ID: &str = "ec.color.cinelooks";
pub const NAME: &str = "CineLooks";

pub fn specs() -> Vec<EffectSpec> {
    vec![EffectSpec {
        id: ID,
        name: NAME,
        category: "Color Correction",
        params: vec![
            p("lookStrength", "Look Strength", num(100.0), slider(0.0, 100.0, 0.0, 100.0, 1)),
            p("showOriginal", "Show Original", Value::Bool(false), ParamUi::Checkbox),
            p("lookName", "Look Name", Value::Str(String::new()), ParamUi::Hidden),
        ],
        render,
        gpu: false,
        float: true,
    }]
}

/// The render environment of an effect call on a buffer with this `offset` and `scale`.
pub fn env(ctx: &EffectCtx, offset: [f64; 2], scale: f64) -> Env {
    Env {
        layer_size: ctx.layer_size,
        offset,
        scale,
        time: ctx.time,
        fps: ctx.fps(),
        seed: ctx.seed,
        working: Working::new(ctx.env.working_space, ctx.env.working_linear),
    }
}

/// The compiled program for this call on a buffer with this `offset` and `scale` (`None`: pass
/// the input through).
pub fn program(ctx: &EffectCtx, offset: [f64; 2], scale: f64) -> Option<Program> {
    if ctx.params.b("showOriginal") {
        return None;
    }
    let look = look_from_params(ctx.params, ctx.layer_size);
    let prog = effectcraft_cinelooks::compile(&look, &env(ctx, offset, scale), (ctx.params.f("lookStrength") / 100.0) as f32);
    (!prog.is_identity()).then_some(prog)
}

fn render(ctx: &EffectCtx, mut b: Buf) -> Buf {
    if let Some(prog) = program(ctx, b.offset, b.scale) {
        b.img = effectcraft_cinelooks::render(&prog, &b.img);
    }
    b
}

/// What Effect Controls shows under the effect when the look can't render as asked.
pub fn warning(params: &Params, layer_size: [f64; 2]) -> Option<String> {
    let look = look_from_params(params, layer_size);
    let w = effectcraft_cinelooks::compile(&look, &Env { layer_size, ..Env::default() }, 1.0).warnings;
    (!w.is_empty()).then(|| w.join("; "))
}

fn frac(v: f64, size: f64) -> f64 {
    if size.is_finite() && size > 0.0 { v / size } else { 0.5 }
}

/// An instance value as a look value.
pub fn value_to_pv(d: &ParamDef, v: &Value, layer_size: [f64; 2]) -> Option<PV> {
    Some(match (d.kind, v) {
        (Kind::Num { .. } | Kind::Angle, v) => PV::Num(v.as_f64()),
        (Kind::Choice(_), v) => PV::Num(v.as_enum() as f64),
        (Kind::Toggle, v) => PV::Bool(v.as_bool()),
        (Kind::Color, Value::Color(c)) => PV::List(c.to_vec()),
        (Kind::Wheel, Value::Vec3(c)) => PV::List(c.to_vec()),
        (Kind::Point, Value::Vec2(c)) => PV::List(vec![frac(c[0], layer_size[0]), frac(c[1], layer_size[1])]),
        (Kind::Curve | Kind::File(_) | Kind::Data, Value::Str(s)) => PV::Text(s.clone()),
        _ => return None,
    })
}

/// A look value as an instance value (the parameter's default when the look's is unusable).
pub fn pv_to_value(d: &ParamDef, n: &Node, layer_size: [f64; 2]) -> Value {
    match d.kind {
        Kind::Num { .. } | Kind::Angle => Value::Scalar(n.num(d.id)),
        Kind::Choice(_) => Value::Enum(n.choice(d.id)),
        Kind::Toggle => Value::Bool(n.flag(d.id)),
        Kind::Color => {
            let c = n.color64(d.id);
            Value::Color([c[0], c[1], c[2], 1.0])
        }
        Kind::Wheel => Value::Vec3(n.wheel64(d.id)),
        Kind::Point => {
            let f = n.point(d.id);
            Value::Vec2([f[0] * layer_size[0], f[1] * layer_size[1]])
        }
        Kind::Curve | Kind::File(_) | Kind::Data => Value::Str(n.text(d.id)),
    }
}

fn ui_of(d: &ParamDef) -> ParamUi {
    match d.kind {
        Kind::Num { min, max, smin, smax, dec } => ParamUi::Slider { min, max, slider_min: smin, slider_max: smax, decimals: dec },
        Kind::Angle => ParamUi::Angle,
        Kind::Color => ParamUi::Color,
        Kind::Wheel => ParamUi::Number,
        Kind::Point => ParamUi::Point,
        Kind::Choice(o) => ParamUi::Popup { options: o.iter().map(|s| s.to_string()).collect() },
        Kind::Toggle => ParamUi::Checkbox,
        Kind::File(e) => ParamUi::File { filters: e.iter().map(|s| s.to_string()).collect() },
        Kind::Curve | Kind::Data => ParamUi::Hidden,
    }
}

fn strength_prop(ids: &mut Ids, v: f64) -> Property {
    Property::new(ids.alloc(), "strength", "Strength", Value::Scalar(v)).with_ui(slider(0.0, 100.0, 0.0, 100.0, 1))
}

/// A property for parameter `d` holding `v`.
fn param_prop(ids: &mut Ids, d: &ParamDef, v: Value) -> Property {
    let mut pr = Property::new(ids.alloc(), d.id, d.name, v).with_ui(ui_of(d));
    match d.kind {
        Kind::Point => pr.spatial = true,
        Kind::Choice(_) | Kind::Toggle => pr.hold_only = true,
        Kind::Data | Kind::File(_) => pr.static_only = true,
        _ => {}
    }
    pr
}

/// The property group of node `n` (a new uid; `n.uid` is ignored).
pub fn node_group(ids: &mut Ids, n: &Node, layer_size: [f64; 2]) -> PropGroup {
    let mut g = ids.group(&n.tool.id(), &n.name());
    g.kind = GroupKind::Indexed;
    g.enabled = n.enabled;
    g.children.push(strength_prop(ids, n.strength.clamp(0.0, 100.0)).into());
    for d in n.tool.def().params {
        g.children.push(param_prop(ids, d, pv_to_value(d, n, layer_size)).into());
    }
    g
}

/// The tool node groups of an instance, in order.
pub fn node_groups(fx: &PropGroup) -> impl Iterator<Item = &PropGroup> {
    fx.groups().filter(|g| ToolKind::from_id(&g.match_id).is_some())
}

/// The look of an instance from its evaluated parameters ([`crate::flatten_params`]).
pub fn look_from_params(p: &Params, layer_size: [f64; 2]) -> Look {
    let mut nodes = vec![];
    for pre in p.groups("") {
        let Some(kind) = ToolKind::from_id(p.s(&format!("{pre}@match"))) else { continue };
        let mut n = Node::new(kind);
        n.uid = p.f(&format!("{pre}@uid")).max(0.0) as u64;
        n.enabled = p.b(&format!("{pre}@enabled"));
        let name = p.s(&format!("{pre}@name"));
        if !name.is_empty() && name != kind.name() {
            n.label = Some(name.to_string());
        }
        n.strength = p.get(&format!("{pre}strength")).map_or(100.0, Value::as_f64);
        for d in kind.def().params {
            if let Some(v) = p.get(&format!("{pre}{}", d.id))
                && let Some(pv) = value_to_pv(d, v, layer_size)
            {
                n.params.insert(d.id.to_string(), pv);
            }
        }
        nodes.push(n);
        if nodes.len() >= effectcraft_cinelooks::look::MAX_NODES {
            break;
        }
    }
    Look { name: p.s("lookName").to_string(), nodes }
}

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => x == y,
        _ => {
            let (x, y) = (a.components(), b.components());
            x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| (p - q).abs() <= 1e-9 * (1.0 + p.abs().max(q.abs())))
        }
    }
}

fn set_at(pr: &mut Property, t: Tick, v: Value) {
    if !same(&pr.value_at(t), &v) {
        pr.set_value_at(t, v);
    }
}

/// Make instance `fx` render `look` at layer time `t`: nodes are matched to existing groups by
/// uid (and tool); their static parameters are set and animated ones get a key at `t` (only
/// where the value changes); new nodes get groups, missing ones are removed and the order
/// follows the look. Top-level parameters stay first. Returns the look with the nodes' uids.
pub fn merge_look(fx: &mut PropGroup, look: &Look, ids: &mut Ids, layer_size: [f64; 2], t: Tick) -> Look {
    let mut look = look.clone();
    look.sanitize();
    let mut old: Vec<PropGroup> = vec![];
    let mut kept = vec![];
    for c in std::mem::take(&mut fx.children) {
        match c {
            effectcraft_project::Node::Group(g) if ToolKind::from_id(&g.match_id).is_some() => old.push(g),
            other => kept.push(other),
        }
    }
    fx.children = kept;
    for n in &mut look.nodes {
        let existing = old.iter().position(|g| n.uid != 0 && g.uid == n.uid && g.match_id == n.tool.id()).map(|i| old.remove(i));
        let g = match existing {
            Some(mut g) => {
                g.enabled = n.enabled;
                g.name = n.name();
                match g.get_mut("strength") {
                    Some(pr) => set_at(pr, t, Value::Scalar(n.strength)),
                    None => g.children.insert(0, strength_prop(ids, n.strength).into()),
                }
                for d in n.tool.def().params {
                    let v = pv_to_value(d, n, layer_size);
                    match g.get_mut(d.id) {
                        Some(pr) => set_at(pr, t, v),
                        None => g.children.push(param_prop(ids, d, v).into()),
                    }
                }
                g
            }
            None => node_group(ids, n, layer_size),
        };
        n.uid = g.uid;
        fx.children.push(g.into());
    }
    if let Some(pr) = fx.get_mut("lookName") {
        pr.value = Value::Str(look.name.clone());
    }
    look
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EffectEnv, find, flatten_params, instantiate};
    use effectcraft_raster::Image;

    fn instance(look: &Look, size: [f64; 2]) -> (PropGroup, u64) {
        let mut next = 1;
        let spec = find(ID).unwrap();
        let mut g = instantiate(spec, &mut Ids(&mut next), NAME, size);
        merge_look(&mut g, look, &mut Ids(&mut next), size, Tick::ZERO);
        (g, next)
    }

    fn params(g: &PropGroup) -> Params {
        flatten_params(g, &mut |p| p.value.clone())
    }

    fn sample() -> Look {
        Look::new(
            "Sample",
            vec![
                Node::new(ToolKind::Exposure).with("exposure", 0.5),
                Node::new(ToolKind::Vignette).with("center", [0.25, 0.75]),
                Node::new(ToolKind::Curves).with("master", "0,0 0.5,0.6 1,1").strength(70.0),
                Node { enabled: false, ..Node::new(ToolKind::FilmGrain) },
            ],
        )
    }

    #[test]
    fn instance_round_trips_the_look() {
        let size = [200.0, 100.0];
        let (g, _) = instance(&sample(), size);
        let back = look_from_params(&params(&g), size);
        assert_eq!(back.name, "Sample");
        assert_eq!(back.nodes.len(), 4);
        assert_eq!(back.nodes[0].num("exposure"), 0.5);
        assert_eq!(back.nodes[1].point("center"), [0.25, 0.75]);
        assert_eq!(back.nodes[2].text("master"), "0,0 0.5,0.6 1,1");
        assert_eq!(back.nodes[2].strength, 70.0);
        assert!(!back.nodes[3].enabled);
        // Point parameters are layer pixels in the instance.
        let vg = node_groups(&g).nth(1).unwrap();
        assert_eq!(vg.get("center").unwrap().value, Value::Vec2([50.0, 75.0]));
    }

    #[test]
    fn merge_keeps_identity_and_keys() {
        let size = [100.0, 100.0];
        let (mut g, mut next) = instance(&sample(), size);
        let uids: Vec<u64> = node_groups(&g).map(|n| n.uid).collect();
        // Animate exposure.
        let ex = node_groups(&g).next().unwrap().get("exposure").unwrap().uid;
        let pr = g.find_mut(ex).unwrap();
        pr.set_animated(true, Tick::ZERO);
        let mut look = look_from_params(&params(&g), size);
        // Reorder (curves first), change exposure, drop the vignette, add a LUT node.
        look.nodes.swap(0, 2);
        look.nodes[2] = look.nodes[2].clone().with("exposure", 1.5);
        look.nodes.remove(1);
        look.nodes.push(Node::new(ToolKind::Lut));
        let t = Tick::from_seconds_f64(1.0);
        let out = merge_look(&mut g, &look, &mut Ids(&mut next), size, t);
        let now: Vec<&PropGroup> = node_groups(&g).collect();
        assert_eq!(now.len(), 4);
        assert_eq!(now[0].uid, uids[2]);
        assert_eq!(now[1].uid, uids[0]);
        assert_eq!(now[2].match_id, "filmGrain");
        assert_eq!(now[3].match_id, "lut");
        assert_eq!(out.nodes[3].uid, now[3].uid);
        // The animated exposure got a key at t; the static value at 0 is untouched.
        let pr = g.find(ex).unwrap();
        assert_eq!(pr.keys.len(), 2);
        assert_eq!(pr.value_at(t), Value::Scalar(1.5));
        assert_eq!(pr.value_at(Tick::ZERO), Value::Scalar(0.5));
        // Top-level parameters stay first.
        assert_eq!(g.children[0].match_id(), "lookStrength");
        // Merging the same look again changes nothing.
        let before = g.clone();
        merge_look(&mut g, &out, &mut Ids(&mut next), size, t);
        assert_eq!(g, before);
    }

    #[test]
    fn renders_and_follows_show_original_and_strength() {
        let size = [32.0, 16.0];
        let (mut g, _) = instance(&Look::new("", vec![Node::new(ToolKind::Exposure).with("exposure", 1.0)]), size);
        let img = Image::filled(32, 16, [0.3, 0.3, 0.3, 1.0]);
        let run = |g: &PropGroup| {
            let p = params(g);
            let ctx = EffectCtx { params: &p, time: 0.0, layer_size: size, seed: 1, adjustment: false, env: EffectEnv::default() };
            crate::apply(find(ID).unwrap(), &ctx, Buf { img: img.clone(), offset: [0.0; 2], scale: 1.0 }).img
        };
        let out = run(&g);
        assert!(out.data[0][0] > 0.35);
        g.get_mut("showOriginal").unwrap().value = Value::Bool(true);
        assert_eq!(run(&g).data, img.data);
        g.get_mut("showOriginal").unwrap().value = Value::Bool(false);
        g.get_mut("lookStrength").unwrap().value = Value::Scalar(0.0);
        assert_eq!(run(&g).data, img.data);
    }

    #[test]
    fn node_groups_are_well_formed() {
        let mut next = 1;
        for t in effectcraft_cinelooks::TOOLS {
            let g = node_group(&mut Ids(&mut next), &Node::new(t.kind), [100.0, 100.0]);
            assert_eq!(g.match_id, t.kind.id());
            assert_eq!(g.children.len(), t.params.len() + 1);
            for d in t.params {
                let pr = g.get(d.id).unwrap();
                // The default round-trips through the instance.
                let pv = value_to_pv(d, &pr.value, [100.0, 100.0]).unwrap();
                assert_eq!(pv, PV::from_def(d.def), "{}.{}", t.name, d.id);
                if let (Kind::Num { .. }, effectcraft_cinelooks::tools::Def::Num(_)) = (d.kind, d.def) {
                    assert!(matches!(pr.ui, ParamUi::Slider { .. }));
                }
            }
        }
    }
}
