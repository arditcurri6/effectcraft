//! A look: an ordered chain of tool nodes with their parameter values. This is the exchange
//! format of presets, `.cinelook` files, the editor's draft and the `cinelooks.*` commands.
//!
//! Parameter values are loosely typed on the way in (JSON from files and agents) and read
//! through the catalogue: a missing, mistyped, NaN or out-of-range value reads as the default
//! or the nearest valid value, never as an error.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::tools::{Def, Kind, ParamDef, ToolKind, tool};

/// Current `.cinelook` format version.
pub const FORMAT_VERSION: u32 = 1;
/// Most nodes a look may hold (a hostile file can't make a render loop for ever).
pub const MAX_NODES: usize = 64;

/// A parameter value as stored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PV {
    Num(f64),
    Bool(bool),
    Text(String),
    List(Vec<f64>),
}

impl PV {
    /// The value `def` describes (points as fractions).
    pub fn from_def(d: Def) -> PV {
        match d {
            Def::Num(v) => PV::Num(v),
            Def::Vec3(v) => PV::List(v.to_vec()),
            Def::Point(v) => PV::List(v.to_vec()),
            Def::Color(v) => PV::List(v.to_vec()),
            Def::Bool(b) => PV::Bool(b),
            Def::Choice(i) => PV::Num(i as f64),
            Def::Text(s) => PV::Text(s.to_string()),
        }
    }
}

/// One tool in a chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// The tool.
    pub tool: ToolKind,
    /// Stable identity: the node's property-group uid in a project (0 = not in a project yet).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub uid: u64,
    #[serde(default = "yes")]
    pub enabled: bool,
    /// 0..100 %: the mix between the node's input and its result.
    #[serde(default = "hundred")]
    pub strength: f64,
    /// Display name override (`None` = the tool's name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Parameter values by id; missing ones are the defaults.
    #[serde(default)]
    pub params: BTreeMap<String, PV>,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}
fn yes() -> bool {
    true
}
fn hundred() -> f64 {
    100.0
}

/// A look.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Look {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub nodes: Vec<Node>,
}

/// A `.cinelook` file: a named, categorised look.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookFile {
    pub format: String,
    pub version: u32,
    pub name: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub description: String,
    pub look: Look,
}

impl Node {
    /// A node of `tool` with every parameter at its default.
    pub fn new(tool: ToolKind) -> Node {
        Node { tool, uid: 0, enabled: true, strength: 100.0, label: None, params: BTreeMap::new() }
    }

    /// Builder: set parameter `id`.
    pub fn with(mut self, id: &str, v: impl Into<PV>) -> Node {
        self.params.insert(id.to_string(), v.into());
        self
    }

    /// Builder: node strength.
    pub fn strength(mut self, s: f64) -> Node {
        self.strength = s;
        self
    }

    pub fn name(&self) -> String {
        self.label.clone().filter(|l| !l.trim().is_empty()).unwrap_or_else(|| self.tool.name().to_string())
    }

    /// Strength as a 0..1 fraction (NaN and out-of-range clamp).
    pub fn mix(&self) -> f32 {
        if self.strength.is_finite() { (self.strength / 100.0).clamp(0.0, 1.0) as f32 } else { 1.0 }
    }

    fn def(&self, id: &str) -> Option<&'static ParamDef> {
        tool(self.tool).param(id)
    }

    /// Numeric parameter (numbers, angles, choices as numbers), clamped to its range.
    pub fn num(&self, id: &str) -> f64 {
        let Some(d) = self.def(id) else { return 0.0 };
        let dv = match d.def {
            Def::Num(v) => v,
            Def::Choice(i) => i as f64,
            _ => 0.0,
        };
        let v = match self.params.get(id) {
            Some(PV::Num(v)) if v.is_finite() => *v,
            Some(PV::Bool(b)) => *b as u8 as f64,
            _ => dv,
        };
        match d.kind {
            Kind::Num { min, max, .. } => v.clamp(min, max),
            Kind::Angle => v.clamp(-1e6, 1e6),
            Kind::Choice(o) => v.round().clamp(0.0, o.len().saturating_sub(1) as f64),
            _ => v,
        }
    }

    pub fn f(&self, id: &str) -> f32 {
        self.num(id) as f32
    }

    /// A percentage parameter as a fraction.
    pub fn frac(&self, id: &str) -> f32 {
        (self.num(id) / 100.0) as f32
    }

    pub fn choice(&self, id: &str) -> u32 {
        self.num(id) as u32
    }

    pub fn flag(&self, id: &str) -> bool {
        match self.params.get(id) {
            Some(PV::Bool(b)) => *b,
            Some(PV::Num(v)) => *v != 0.0 && v.is_finite(),
            _ => matches!(self.def(id).map(|d| d.def), Some(Def::Bool(true))),
        }
    }

    fn list<const N: usize>(&self, id: &str, fallback: [f64; N]) -> [f64; N] {
        let dv = match self.def(id).map(|d| d.def) {
            Some(Def::Vec3(v)) if N == 3 => std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0)),
            Some(Def::Point(v)) if N == 2 => std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0)),
            Some(Def::Color(v)) if N == 4 => std::array::from_fn(|i| v.get(i).copied().unwrap_or(1.0)),
            _ => fallback,
        };
        match self.params.get(id) {
            Some(PV::List(l)) if l.len() >= N.min(3) => std::array::from_fn(|i| l.get(i).copied().filter(|v| v.is_finite()).unwrap_or(dv[i])),
            _ => dv,
        }
    }

    /// Colour parameter (RGB, straight, encoded like the working space), clamped to 0..4.
    pub fn color(&self, id: &str) -> [f32; 3] {
        self.color64(id).map(|v| v as f32)
    }

    /// [`Node::color`] at full precision.
    pub fn color64(&self, id: &str) -> [f64; 3] {
        let c = self.list(id, [1.0; 4]);
        [c[0], c[1], c[2]].map(|v| v.clamp(0.0, 4.0))
    }

    /// Wheel `[x, y, master]`: `(x, y)` pulled into the unit disc, master clamped to ±1.
    pub fn wheel(&self, id: &str) -> [f32; 3] {
        self.wheel64(id).map(|v| v as f32)
    }

    /// [`Node::wheel`] at full precision.
    pub fn wheel64(&self, id: &str) -> [f64; 3] {
        let [x, y, m] = self.list(id, [0.0; 3]);
        let r = (x * x + y * y).sqrt();
        let k = if r > 1.0 { 1.0 / r } else { 1.0 };
        [x * k, y * k, m.clamp(-1.0, 1.0)]
    }

    /// Point as fractions of the layer size (clamped to a generous ±10 layers).
    pub fn point(&self, id: &str) -> [f64; 2] {
        self.list(id, [0.5; 2]).map(|v| v.clamp(-10.0, 10.0))
    }

    /// Text parameter (curves, files, data).
    pub fn text(&self, id: &str) -> String {
        match self.params.get(id) {
            Some(PV::Text(s)) => s.clone(),
            _ => match self.def(id).map(|d| d.def) {
                Some(Def::Text(s)) => s.to_string(),
                _ => String::new(),
            },
        }
    }

    /// Whether every parameter is at its default (the tool's neutral or starting state).
    pub fn is_default(&self) -> bool {
        tool(self.tool).params.iter().all(|p| match self.params.get(p.id) {
            None => true,
            Some(v) => *v == PV::from_def(p.def),
        })
    }

    /// Drop parameters the tool doesn't have and normalise numbers into their ranges.
    pub fn sanitize(&mut self) {
        let t = tool(self.tool);
        self.params.retain(|k, _| t.param(k).is_some());
        let ids: Vec<&'static str> = t.params.iter().map(|p| p.id).collect();
        for id in ids {
            let Some(d) = t.param(id) else { continue };
            if let Some(v) = self.params.get(id).cloned() {
                let fixed = match (d.kind, &v) {
                    (Kind::Num { .. } | Kind::Angle | Kind::Choice(_), PV::Num(_) | PV::Bool(_)) => Some(PV::Num(self.num(id))),
                    (Kind::Toggle, _) => Some(PV::Bool(self.flag(id))),
                    (Kind::Color, PV::List(_)) => {
                        let c = self.color64(id);
                        Some(PV::List(vec![c[0], c[1], c[2], 1.0]))
                    }
                    (Kind::Wheel, PV::List(_)) => Some(PV::List(self.wheel64(id).to_vec())),
                    (Kind::Point, PV::List(_)) => Some(PV::List(self.point(id).to_vec())),
                    (Kind::Curve | Kind::File(_) | Kind::Data, PV::Text(_)) => Some(v.clone()),
                    _ => None,
                };
                match fixed {
                    Some(f) => {
                        self.params.insert(id.to_string(), f);
                    }
                    None => {
                        self.params.remove(id);
                    }
                }
            }
        }
        if !self.strength.is_finite() {
            self.strength = 100.0;
        }
        self.strength = self.strength.clamp(0.0, 100.0);
    }
}

impl From<f64> for PV {
    fn from(v: f64) -> PV {
        PV::Num(v)
    }
}
impl From<bool> for PV {
    fn from(v: bool) -> PV {
        PV::Bool(v)
    }
}
impl From<&str> for PV {
    fn from(v: &str) -> PV {
        PV::Text(v.to_string())
    }
}
impl From<String> for PV {
    fn from(v: String) -> PV {
        PV::Text(v)
    }
}
impl From<[f64; 2]> for PV {
    fn from(v: [f64; 2]) -> PV {
        PV::List(v.to_vec())
    }
}
impl From<[f64; 3]> for PV {
    fn from(v: [f64; 3]) -> PV {
        PV::List(v.to_vec())
    }
}
impl From<[f64; 4]> for PV {
    fn from(v: [f64; 4]) -> PV {
        PV::List(v.to_vec())
    }
}

impl Look {
    pub fn new(name: &str, nodes: Vec<Node>) -> Look {
        Look { name: name.to_string(), nodes }
    }

    /// Normalise every node (see [`Node::sanitize`]) and cap the chain length.
    pub fn sanitize(&mut self) {
        self.nodes.truncate(MAX_NODES);
        for n in &mut self.nodes {
            n.sanitize();
        }
    }

    /// Any enabled node with a non-zero strength.
    pub fn is_active(&self) -> bool {
        self.nodes.iter().any(|n| n.enabled && n.mix() > 0.0)
    }

    /// The look without project identities (uids), for presets and files.
    pub fn detached(&self) -> Look {
        let mut l = self.clone();
        for n in &mut l.nodes {
            n.uid = 0;
        }
        l
    }

    /// A `.cinelook` document.
    pub fn to_file(&self, category: &str, description: &str) -> LookFile {
        LookFile {
            format: "cinelook".into(),
            version: FORMAT_VERSION,
            name: self.name.clone(),
            category: category.into(),
            description: description.into(),
            look: self.detached(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Parse a look from JSON: a bare look or a `.cinelook` file. Errors name the problem.
    pub fn parse(text: &str) -> Result<Look, String> {
        if text.len() > 16 << 20 {
            return Err("the look file is larger than 16 MB".into());
        }
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("not a CineLooks look (JSON: {e})"))?;
        let mut look = if v.get("look").is_some() {
            let f: LookFile = serde_json::from_value(v).map_err(|e| format!("not a CineLooks look file: {e}"))?;
            if f.version > FORMAT_VERSION {
                return Err(format!("this look was saved by a newer CineLooks (format {}; this one reads {FORMAT_VERSION})", f.version));
            }
            let mut l = f.look;
            if l.name.is_empty() {
                l.name = f.name;
            }
            l
        } else {
            serde_json::from_value::<Look>(v).map_err(|e| format!("not a CineLooks look: {e}"))?
        };
        look.sanitize();
        Ok(look)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_ranges_and_hostile_values() {
        let n = Node::new(ToolKind::Exposure);
        assert_eq!(n.num("exposure"), 0.0);
        assert_eq!(n.num("nonexistent"), 0.0);
        let n = n.with("exposure", 99.0).with("compensation", f64::NAN);
        assert_eq!(n.num("exposure"), 8.0);
        assert_eq!(n.num("compensation"), 0.0);
        let w = Node::new(ToolKind::Wheels).with("lift", [3.0, 4.0, 7.0]);
        let l = w.wheel("lift");
        assert!((l[0] - 0.6).abs() < 1e-6 && (l[1] - 0.8).abs() < 1e-6 && l[2] == 1.0);
        let c = Node::new(ToolKind::Halation).with("color", PV::Text("red".into()));
        assert_eq!(c.color("color"), [1.0, 0.32, 0.12]);
        let g = Node::new(ToolKind::FilmGrain).with("type", 17.0);
        assert_eq!(g.choice("type"), 2);
    }

    #[test]
    fn json_round_trip_and_file_parse() {
        let look = Look::new("Test", vec![Node::new(ToolKind::Contrast).with("contrast", 20.0), Node::new(ToolKind::FilmGrain).strength(50.0)]);
        let back = Look::parse(&look.to_json()).unwrap();
        assert_eq!(back, look);
        let file = serde_json::to_string(&look.to_file("Film", "x")).unwrap();
        let back = Look::parse(&file).unwrap();
        assert_eq!(back.nodes, look.nodes);
        assert!(Look::parse("{").is_err());
        assert!(Look::parse(r#"{"format":"cinelook","version":99,"name":"x","look":{"nodes":[]}}"#).unwrap_err().contains("newer"));
        // Unknown tools are an error that names the problem, unknown params are dropped.
        assert!(Look::parse(r#"{"nodes":[{"tool":"teleport"}]}"#).is_err());
        let l = Look::parse(r#"{"nodes":[{"tool":"exposure","params":{"bogus":1,"exposure":"x"}}]}"#).unwrap();
        assert!(l.nodes[0].params.is_empty());
    }

    #[test]
    fn sanitize_caps_and_normalises() {
        let mut l = Look::new("x", vec![Node::new(ToolKind::Exposure).with("exposure", 1e9).strength(f64::INFINITY); 100]);
        l.sanitize();
        assert_eq!(l.nodes.len(), MAX_NODES);
        assert_eq!(l.nodes[0].params.get("exposure"), Some(&PV::Num(8.0)));
        assert_eq!(l.nodes[0].strength, 100.0);
    }
}
