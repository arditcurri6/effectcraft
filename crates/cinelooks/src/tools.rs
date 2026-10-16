//! The tool catalogue: every tool a look can contain, its parameters (ids, names, kinds,
//! defaults, ranges) and the colour domain it works in. This table is the single source of
//! truth for the effect's property groups, the editor's controls, preset validation and the
//! program compiler.

use serde::{Deserialize, Serialize};

/// Which representation of colour a tool works on (see the crate docs, "Domains").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Linear light in the working primaries.
    Linear,
    /// The grading encoding: the working space's transfer curve (sRGB for linear spaces).
    Encoded,
    /// Spatial tools that only move pixels: they work in whatever domain the chain is in.
    Any,
}

/// Groups of tools in the editor's Add Tool menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolGroup {
    Primary,
    Film,
    Lens,
    Selective,
    Utility,
}

impl ToolGroup {
    pub const ALL: [ToolGroup; 5] = [ToolGroup::Primary, ToolGroup::Film, ToolGroup::Lens, ToolGroup::Selective, ToolGroup::Utility];

    pub fn label(self) -> &'static str {
        match self {
            ToolGroup::Primary => "Primary Correction",
            ToolGroup::Film => "Film Emulation",
            ToolGroup::Lens => "Camera & Lens",
            ToolGroup::Selective => "Selective",
            ToolGroup::Utility => "LUT & Utility",
        }
    }
}

/// Every tool. The serde names are the property-group match ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub enum ToolKind {
    Exposure,
    Contrast,
    WhiteBalance,
    Wheels,
    Curves,
    Hsl,
    ChannelMixer,
    HueRemap,
    FilmResponse,
    FilmGrain,
    Halation,
    BleachBypass,
    FilmFade,
    FilmPrint,
    Diffusion,
    Vignette,
    ChromaticAberration,
    LensDistortion,
    AnamorphicGlow,
    EdgeSoftness,
    GradientExposure,
    RadialExposure,
    SpotExposure,
    ShadowHighlight,
    SelectiveColor,
    Lut,
}

/// How a parameter is edited and stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// A number: hard range `min..max`, slider range `smin..smax`, display decimals.
    Num {
        min: f64,
        max: f64,
        smin: f64,
        smax: f64,
        dec: u8,
    },
    /// Degrees (hue angles), wrapped where it matters.
    Angle,
    /// Straight RGBA, encoded like the working space (a picked colour).
    Color,
    /// A colour wheel: `[x, y, master]`, `(x, y)` in the unit disc.
    Wheel,
    /// A position on the layer; looks store fractions of the layer size.
    Point,
    /// One of the options (an index).
    Choice(&'static [&'static str]),
    Toggle,
    /// A curve: `"x,y x,y …"` control points in 0..1.
    Curve,
    /// A file path offering these extensions.
    File(&'static [&'static str]),
    /// Internal text data (an embedded `.cube`).
    Data,
}

/// Default value of a parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Def {
    Num(f64),
    /// Wheel `[x, y, master]`.
    Vec3([f64; 3]),
    /// Point as fractions of the layer size.
    Point([f64; 2]),
    Color([f64; 4]),
    Bool(bool),
    Choice(u32),
    Text(&'static str),
}

pub struct ParamDef {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: Kind,
    pub def: Def,
}

pub struct ToolDef {
    pub kind: ToolKind,
    pub name: &'static str,
    pub group: ToolGroup,
    pub domain: Domain,
    /// One line for the Add Tool menu and the inspector.
    pub about: &'static str,
    pub params: &'static [ParamDef],
}

const fn num(id: &'static str, name: &'static str, min: f64, max: f64, smin: f64, smax: f64, dec: u8, def: f64) -> ParamDef {
    ParamDef { id, name, kind: Kind::Num { min, max, smin, smax, dec }, def: Def::Num(def) }
}
/// A percentage `0..100`.
const fn pct(id: &'static str, name: &'static str, def: f64) -> ParamDef {
    num(id, name, 0.0, 100.0, 0.0, 100.0, 1, def)
}
/// A bipolar percentage `-100..100`.
const fn bip(id: &'static str, name: &'static str, def: f64) -> ParamDef {
    num(id, name, -100.0, 100.0, -100.0, 100.0, 1, def)
}
/// Exposure in stops.
const fn stops(id: &'static str, name: &'static str, lim: f64, def: f64) -> ParamDef {
    num(id, name, -lim, lim, -lim, lim, 2, def)
}
const fn color(id: &'static str, name: &'static str, r: f64, g: f64, b: f64) -> ParamDef {
    ParamDef { id, name, kind: Kind::Color, def: Def::Color([r, g, b, 1.0]) }
}
const fn wheel(id: &'static str, name: &'static str) -> ParamDef {
    ParamDef { id, name, kind: Kind::Wheel, def: Def::Vec3([0.0; 3]) }
}
const fn point(id: &'static str, name: &'static str, x: f64, y: f64) -> ParamDef {
    ParamDef { id, name, kind: Kind::Point, def: Def::Point([x, y]) }
}
const fn choice(id: &'static str, name: &'static str, opts: &'static [&'static str], def: u32) -> ParamDef {
    ParamDef { id, name, kind: Kind::Choice(opts), def: Def::Choice(def) }
}
const fn toggle(id: &'static str, name: &'static str, def: bool) -> ParamDef {
    ParamDef { id, name, kind: Kind::Toggle, def: Def::Bool(def) }
}
const fn angle(id: &'static str, name: &'static str, def: f64) -> ParamDef {
    ParamDef { id, name, kind: Kind::Angle, def: Def::Num(def) }
}
const fn curve(id: &'static str, name: &'static str) -> ParamDef {
    ParamDef { id, name, kind: Kind::Curve, def: Def::Text("0,0 1,1") }
}

/// Grain sizes (Film Grain's Type).
pub const GRAIN_TYPES: &[&str] = &["Fine", "Medium", "Coarse"];
/// Film Print's print styles.
pub const PRINT_STYLES: &[&str] = &["Neutral Print", "Warm Print", "Cool Print", "Dense Print", "Soft Print"];
/// What the LUT tool's table expects as input.
pub const LUT_INPUTS: &[&str] = &["Working (encoded)", "Working (linear)", "sRGB", "Rec. 709", "Rec. 2020", "Display P3", "ACEScg (linear)"];
pub const LUT_INTERP: &[&str] = &["Tetrahedral", "Trilinear"];
/// Lens Distortion's edges.
pub const EDGE_MODES: &[&str] = &["Transparent", "Extend Edges"];

/// The eight hue ranges of the HSL tool, with their centres in degrees.
pub const HSL_RANGES: [(&str, &str, f32); 8] = [
    ("red", "Red", 0.0),
    ("orange", "Orange", 30.0),
    ("yellow", "Yellow", 60.0),
    ("green", "Green", 120.0),
    ("aqua", "Aqua", 180.0),
    ("blue", "Blue", 240.0),
    ("purple", "Purple", 270.0),
    ("magenta", "Magenta", 300.0),
];

macro_rules! hsl_range {
    ($c:literal, $n:literal) => {
        [
            bip(concat!($c, "Hue"), concat!($n, " Hue"), 0.0),
            bip(concat!($c, "Sat"), concat!($n, " Saturation"), 0.0),
            bip(concat!($c, "Lum"), concat!($n, " Luminance"), 0.0),
        ]
    };
}

const HSL_PARAMS: [ParamDef; 27] = {
    let r = hsl_range!("red", "Red");
    let o = hsl_range!("orange", "Orange");
    let y = hsl_range!("yellow", "Yellow");
    let g = hsl_range!("green", "Green");
    let a = hsl_range!("aqua", "Aqua");
    let b = hsl_range!("blue", "Blue");
    let p = hsl_range!("purple", "Purple");
    let m = hsl_range!("magenta", "Magenta");
    let [r0, r1, r2] = r;
    let [o0, o1, o2] = o;
    let [y0, y1, y2] = y;
    let [g0, g1, g2] = g;
    let [a0, a1, a2] = a;
    let [b0, b1, b2] = b;
    let [p0, p1, p2] = p;
    let [m0, m1, m2] = m;
    [
        num("hue", "Hue", -180.0, 180.0, -180.0, 180.0, 1, 0.0),
        bip("saturation", "Saturation", 0.0),
        bip("lightness", "Lightness", 0.0),
        r0,
        r1,
        r2,
        o0,
        o1,
        o2,
        y0,
        y1,
        y2,
        g0,
        g1,
        g2,
        a0,
        a1,
        a2,
        b0,
        b1,
        b2,
        p0,
        p1,
        p2,
        m0,
        m1,
        m2,
    ]
};

/// The catalogue, in Add Tool menu order.
pub static TOOLS: &[ToolDef] = &[
    ToolDef {
        kind: ToolKind::Exposure,
        name: "Exposure",
        group: ToolGroup::Primary,
        domain: Domain::Linear,
        about: "Exposure in stops, midtone compensation and a highlight shoulder that keeps pushed highlights from clipping.",
        params: &[
            stops("exposure", "Exposure (stops)", 8.0, 0.0),
            stops("compensation", "Midtone Compensation (stops)", 2.0, 0.0),
            pct("highlightProtect", "Highlight Protection", 0.0),
        ],
    },
    ToolDef {
        kind: ToolKind::Contrast,
        name: "Contrast",
        group: ToolGroup::Primary,
        domain: Domain::Encoded,
        about: "Contrast around a pivot, from a straight slope to a filmic S-curve that never clips.",
        params: &[bip("contrast", "Contrast", 0.0), num("pivot", "Pivot", 0.05, 0.95, 0.05, 0.95, 3, 0.46), pct("filmic", "Filmic Response", 60.0)],
    },
    ToolDef {
        kind: ToolKind::WhiteBalance,
        name: "White Balance",
        group: ToolGroup::Primary,
        domain: Domain::Linear,
        about: "Temperature and tint as a chromatic adaptation along the daylight locus, and neutralising a picked colour.",
        params: &[
            bip("temperature", "Temperature", 0.0),
            bip("tint", "Tint", 0.0),
            color("neutral", "Neutral Color", 0.5, 0.5, 0.5),
            pct("neutralAmount", "Neutralize", 0.0),
        ],
    },
    ToolDef {
        kind: ToolKind::Wheels,
        name: "Color Wheels",
        group: ToolGroup::Primary,
        domain: Domain::Encoded,
        about: "Lift, gamma, gain and offset, then shadow / midtone / highlight balance.",
        params: &[
            wheel("lift", "Lift"),
            wheel("gamma", "Gamma"),
            wheel("gain", "Gain"),
            wheel("offset", "Offset"),
            wheel("shadows", "Shadows"),
            wheel("midtones", "Midtones"),
            wheel("highlights", "Highlights"),
        ],
    },
    ToolDef {
        kind: ToolKind::Curves,
        name: "Curves",
        group: ToolGroup::Primary,
        domain: Domain::Encoded,
        about: "Master RGB and per-channel curves through editable points (monotone cubic interpolation).",
        params: &[curve("master", "RGB"), curve("red", "Red"), curve("green", "Green"), curve("blue", "Blue")],
    },
    ToolDef {
        kind: ToolKind::Hsl,
        name: "HSL",
        group: ToolGroup::Primary,
        domain: Domain::Encoded,
        about: "Global hue, saturation and lightness, and hue / saturation / luminance for eight colour ranges.",
        params: &HSL_PARAMS,
    },
    ToolDef {
        kind: ToolKind::ChannelMixer,
        name: "Channel Mixer",
        group: ToolGroup::Primary,
        domain: Domain::Linear,
        about: "Each output channel as a mix of the input channels, or a monochrome mix.",
        params: &[
            num("rr", "Red from Red", -200.0, 200.0, -200.0, 200.0, 1, 100.0),
            num("rg", "Red from Green", -200.0, 200.0, -200.0, 200.0, 1, 0.0),
            num("rb", "Red from Blue", -200.0, 200.0, -200.0, 200.0, 1, 0.0),
            num("gr", "Green from Red", -200.0, 200.0, -200.0, 200.0, 1, 0.0),
            num("gg", "Green from Green", -200.0, 200.0, -200.0, 200.0, 1, 100.0),
            num("gb", "Green from Blue", -200.0, 200.0, -200.0, 200.0, 1, 0.0),
            num("br", "Blue from Red", -200.0, 200.0, -200.0, 200.0, 1, 0.0),
            num("bg", "Blue from Green", -200.0, 200.0, -200.0, 200.0, 1, 0.0),
            num("bb", "Blue from Blue", -200.0, 200.0, -200.0, 200.0, 1, 100.0),
            toggle("monochrome", "Monochrome", false),
            num("monoR", "Mono Red", -200.0, 200.0, -200.0, 200.0, 1, 30.0),
            num("monoG", "Mono Green", -200.0, 200.0, -200.0, 200.0, 1, 60.0),
            num("monoB", "Mono Blue", -200.0, 200.0, -200.0, 200.0, 1, 10.0),
        ],
    },
    ToolDef {
        kind: ToolKind::HueRemap,
        name: "Hue Remap",
        group: ToolGroup::Primary,
        domain: Domain::Encoded,
        about: "Select a hue range with a soft falloff and shift its hue, saturation and luminance.",
        params: &[
            angle("center", "Hue Center", 30.0),
            num("range", "Range (±°)", 0.0, 180.0, 0.0, 90.0, 1, 20.0),
            num("falloff", "Falloff (°)", 0.0, 180.0, 0.0, 90.0, 1, 25.0),
            num("shift", "Hue Shift (°)", -180.0, 180.0, -180.0, 180.0, 1, 0.0),
            bip("saturation", "Saturation", 0.0),
            bip("luminance", "Luminance", 0.0),
        ],
    },
    ToolDef {
        kind: ToolKind::FilmResponse,
        name: "Film Response",
        group: ToolGroup::Film,
        domain: Domain::Linear,
        about: "A filmic tone response in stops: contrast, toe (shadow compression), shoulder (highlight rolloff), latitude and colour response.",
        params: &[
            stops("exposure", "Exposure (stops)", 4.0, 0.0),
            bip("contrast", "Contrast", 0.0),
            pct("toe", "Shadow Compression", 35.0),
            pct("shoulder", "Highlight Rolloff", 55.0),
            num("latitude", "Latitude (stops)", 0.0, 4.0, 0.0, 4.0, 2, 1.0),
            bip("color", "Color Response", 0.0),
        ],
    },
    ToolDef {
        kind: ToolKind::FilmGrain,
        name: "Film Grain",
        group: ToolGroup::Film,
        domain: Domain::Encoded,
        about: "Procedural grain on the layer: size, softness, luminance response, mono or colour; stable per frame.",
        params: &[
            pct("amount", "Amount", 25.0),
            choice("type", "Type", GRAIN_TYPES, 0),
            num("size", "Size", 0.25, 4.0, 0.25, 4.0, 2, 1.0),
            pct("softness", "Softness", 40.0),
            pct("lumaResponse", "Luminance Response", 60.0),
            toggle("colored", "Colored", false),
            pct("colorAmount", "Color Variation", 50.0),
            num("animation", "Animation", 0.0, 1.0, 0.0, 1.0, 2, 1.0),
            num("seed", "Seed", 0.0, 10000.0, 0.0, 100.0, 0, 0.0),
        ],
    },
    ToolDef {
        kind: ToolKind::Halation,
        name: "Halation",
        group: ToolGroup::Film,
        domain: Domain::Linear,
        about: "Red-orange light bleeding around bright highlights, as on film without an anti-halation layer.",
        params: &[
            num("threshold", "Highlight Threshold", 0.0, 1.0, 0.0, 1.0, 3, 0.75),
            num("radius", "Radius (% of height)", 0.05, 25.0, 0.05, 10.0, 2, 1.2),
            pct("amount", "Amount", 45.0),
            pct("falloff", "Falloff", 50.0),
            color("color", "Color", 1.0, 0.32, 0.12),
        ],
    },
    ToolDef {
        kind: ToolKind::BleachBypass,
        name: "Bleach Bypass",
        group: ToolGroup::Film,
        domain: Domain::Encoded,
        about: "Silver retained in the image: the luminance overlaid on the colour, with reduced saturation.",
        params: &[pct("contrast", "Contrast", 60.0), pct("desaturation", "Desaturation", 55.0)],
    },
    ToolDef {
        kind: ToolKind::FilmFade,
        name: "Film Fade",
        group: ToolGroup::Film,
        domain: Domain::Encoded,
        about: "Lifted blacks, softened contrast and a faded split-tone colour response.",
        params: &[
            pct("blacks", "Lift Blacks", 25.0),
            pct("contrast", "Reduce Contrast", 20.0),
            num("saturation", "Saturation", -100.0, 0.0, -100.0, 0.0, 1, -15.0),
            color("shadowTint", "Shadow Tint", 0.35, 0.47, 0.55),
            color("highlightTint", "Highlight Tint", 1.0, 0.86, 0.68),
            pct("tintAmount", "Vintage Color", 40.0),
        ],
    },
    ToolDef {
        kind: ToolKind::FilmPrint,
        name: "Film Print",
        group: ToolGroup::Film,
        domain: Domain::Linear,
        about: "Print-inspired tone mapping: dye crosstalk and per-channel density, then a print curve with adjustable toe and shoulder.",
        params: &[
            choice("style", "Print Style", PRINT_STYLES, 0),
            stops("density", "Print Exposure (stops)", 3.0, 0.0),
            bip("highlights", "Highlight Response", 0.0),
            bip("shadows", "Shadow Response", 0.0),
            pct("saturation", "Saturation", 100.0),
        ],
    },
    ToolDef {
        kind: ToolKind::Diffusion,
        name: "Optical Diffusion",
        group: ToolGroup::Lens,
        domain: Domain::Linear,
        about: "A diffusion filter: highlight bloom plus an overall soft mist.",
        params: &[
            num("threshold", "Threshold", 0.0, 1.0, 0.0, 1.0, 3, 0.6),
            num("radius", "Radius (% of height)", 0.05, 30.0, 0.05, 10.0, 2, 2.5),
            pct("intensity", "Intensity", 35.0),
            pct("softness", "Soft Diffusion", 30.0),
            color("color", "Glow Color", 1.0, 0.97, 0.92),
        ],
    },
    ToolDef {
        kind: ToolKind::Vignette,
        name: "Lens Vignette",
        group: ToolGroup::Lens,
        domain: Domain::Linear,
        about: "Light falloff towards the edges (or a bright vignette), with highlight preservation.",
        params: &[
            bip("amount", "Amount", -35.0),
            num("radius", "Radius", 10.0, 200.0, 10.0, 160.0, 1, 85.0),
            pct("feather", "Feather", 60.0),
            point("center", "Center", 0.5, 0.5),
            num("aspect", "Aspect Ratio", 0.25, 4.0, 0.25, 4.0, 2, 1.0),
            pct("highlights", "Highlight Preservation", 50.0),
        ],
    },
    ToolDef {
        kind: ToolKind::ChromaticAberration,
        name: "Chromatic Aberration",
        group: ToolGroup::Lens,
        domain: Domain::Any,
        about: "Red/cyan and blue/yellow fringes, radial (lateral) or uniform.",
        params: &[
            num("redCyan", "Red/Cyan (px at 1080p)", -20.0, 20.0, -8.0, 8.0, 2, 1.5),
            num("blueYellow", "Blue/Yellow (px at 1080p)", -20.0, 20.0, -8.0, 8.0, 2, -0.75),
            pct("radial", "Radial", 100.0),
            pct("amount", "Amount", 100.0),
            point("center", "Center", 0.5, 0.5),
        ],
    },
    ToolDef {
        kind: ToolKind::LensDistortion,
        name: "Lens Distortion",
        group: ToolGroup::Lens,
        domain: Domain::Any,
        about: "Barrel (positive) or pincushion (negative) distortion, with zoom and edge handling.",
        params: &[
            bip("amount", "Distortion", 0.0),
            num("zoom", "Zoom", 50.0, 200.0, 75.0, 150.0, 1, 100.0),
            point("center", "Center", 0.5, 0.5),
            choice("edges", "Edges", EDGE_MODES, 1),
        ],
    },
    ToolDef {
        kind: ToolKind::AnamorphicGlow,
        name: "Anamorphic Glow",
        group: ToolGroup::Lens,
        domain: Domain::Linear,
        about: "Horizontal streaks from highlights.",
        params: &[
            num("threshold", "Threshold", 0.0, 1.0, 0.0, 1.0, 3, 0.8),
            num("length", "Length (% of width)", 0.5, 100.0, 0.5, 60.0, 1, 20.0),
            pct("intensity", "Intensity", 40.0),
            color("color", "Color", 0.45, 0.65, 1.0),
        ],
    },
    ToolDef {
        kind: ToolKind::EdgeSoftness,
        name: "Edge Softness",
        group: ToolGroup::Lens,
        domain: Domain::Any,
        about: "Blur that grows towards the frame edges.",
        params: &[num("radius", "Blur Radius (% of height)", 0.0, 10.0, 0.0, 5.0, 2, 0.8), pct("size", "Edge Size", 35.0), pct("feather", "Feather", 60.0)],
    },
    ToolDef {
        kind: ToolKind::GradientExposure,
        name: "Graduated Exposure",
        group: ToolGroup::Selective,
        domain: Domain::Linear,
        about: "A linear gradient from Start (full) to End (none): exposure, saturation and tint.",
        params: &[
            point("start", "Start", 0.5, 0.15),
            point("end", "End", 0.5, 0.55),
            stops("exposure", "Exposure (stops)", 4.0, -0.75),
            bip("saturation", "Saturation", 0.0),
            color("tint", "Tint", 1.0, 1.0, 1.0),
            pct("tintAmount", "Tint Amount", 0.0),
            pct("feather", "Feather", 100.0),
            toggle("invert", "Invert", false),
            pct("opacity", "Mask Opacity", 100.0),
        ],
    },
    ToolDef {
        kind: ToolKind::RadialExposure,
        name: "Radial Exposure",
        group: ToolGroup::Selective,
        domain: Domain::Linear,
        about: "An elliptical area (or everything outside it): exposure, saturation and tint.",
        params: &[
            point("center", "Center", 0.5, 0.5),
            num("width", "Width (% of height)", 1.0, 400.0, 5.0, 200.0, 1, 90.0),
            num("height", "Height (% of height)", 1.0, 400.0, 5.0, 200.0, 1, 70.0),
            angle("rotation", "Rotation", 0.0),
            pct("feather", "Feather", 60.0),
            stops("exposure", "Exposure (stops)", 4.0, 0.4),
            bip("saturation", "Saturation", 0.0),
            color("tint", "Tint", 1.0, 1.0, 1.0),
            pct("tintAmount", "Tint Amount", 0.0),
            toggle("invert", "Invert", false),
            pct("opacity", "Mask Opacity", 100.0),
        ],
    },
    ToolDef {
        kind: ToolKind::SpotExposure,
        name: "Spot Exposure",
        group: ToolGroup::Selective,
        domain: Domain::Linear,
        about: "A small round spot: brighten a face, darken a hot window.",
        params: &[
            point("center", "Center", 0.5, 0.4),
            num("size", "Size (% of height)", 1.0, 100.0, 1.0, 60.0, 1, 18.0),
            pct("feather", "Feather", 70.0),
            stops("exposure", "Exposure (stops)", 4.0, 0.5),
            color("tint", "Tint", 1.0, 1.0, 1.0),
            pct("tintAmount", "Tint Amount", 0.0),
            toggle("invert", "Invert", false),
            pct("opacity", "Mask Opacity", 100.0),
        ],
    },
    ToolDef {
        kind: ToolKind::ShadowHighlight,
        name: "Shadow & Highlight",
        group: ToolGroup::Selective,
        domain: Domain::Linear,
        about: "Local recovery: open up shadows and pull back highlights using the surrounding brightness.",
        params: &[
            pct("shadows", "Shadow Recovery", 30.0),
            pct("highlights", "Highlight Recovery", 30.0),
            num("radius", "Radius (% of height)", 0.0, 20.0, 0.0, 10.0, 2, 2.0),
            pct("width", "Tonal Width", 50.0),
        ],
    },
    ToolDef {
        kind: ToolKind::SelectiveColor,
        name: "Selective Color",
        group: ToolGroup::Selective,
        domain: Domain::Encoded,
        about: "An HSL qualifier (hue, saturation and luminance ranges with softness) and a correction applied through it.",
        params: &[
            angle("hue", "Hue Center", 25.0),
            num("hueWidth", "Hue Width (±°)", 0.0, 180.0, 0.0, 120.0, 1, 25.0),
            num("hueSoft", "Hue Softness (°)", 0.0, 180.0, 0.0, 90.0, 1, 20.0),
            num("satLow", "Saturation Low", 0.0, 100.0, 0.0, 100.0, 1, 10.0),
            num("satHigh", "Saturation High", 0.0, 100.0, 0.0, 100.0, 1, 100.0),
            num("lumLow", "Luminance Low", 0.0, 100.0, 0.0, 100.0, 1, 0.0),
            num("lumHigh", "Luminance High", 0.0, 100.0, 0.0, 100.0, 1, 100.0),
            pct("softness", "Range Softness", 20.0),
            toggle("invert", "Invert Selection", false),
            toggle("showMask", "Show Mask", false),
            num("hueShift", "Hue Shift (°)", -180.0, 180.0, -180.0, 180.0, 1, 0.0),
            bip("saturation", "Saturation", 0.0),
            stops("exposure", "Exposure (stops)", 4.0, 0.0),
            color("tint", "Tint", 1.0, 1.0, 1.0),
            pct("tintAmount", "Tint Amount", 0.0),
            pct("opacity", "Mask Opacity", 100.0),
        ],
    },
    ToolDef {
        kind: ToolKind::Lut,
        name: "LUT",
        group: ToolGroup::Utility,
        domain: Domain::Any,
        about: "A .cube LUT (1D, 3D or both), in the input space it was made for.",
        params: &[
            ParamDef { id: "file", name: "LUT File", kind: Kind::File(&["cube"]), def: Def::Text("") },
            choice("input", "LUT Input", LUT_INPUTS, 0),
            choice("interpolation", "Interpolation", LUT_INTERP, 0),
            ParamDef { id: "data", name: "Embedded LUT", kind: Kind::Data, def: Def::Text("") },
        ],
    },
];

/// The catalogue entry of `kind`.
pub fn tool(kind: ToolKind) -> &'static ToolDef {
    // Every kind is in the table (checked by a test); Exposure is the never-taken fallback.
    TOOLS.iter().find(|t| t.kind == kind).unwrap_or(&TOOLS[0])
}

impl ToolKind {
    /// Every kind, in catalogue order.
    pub fn all() -> impl Iterator<Item = ToolKind> {
        TOOLS.iter().map(|t| t.kind)
    }
    /// The property-group match id (`exposure`, `whiteBalance`…).
    pub fn id(self) -> String {
        serde_json::to_value(self).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
    }
    pub fn from_id(id: &str) -> Option<ToolKind> {
        serde_json::from_value(serde_json::Value::String(id.to_string())).ok()
    }
    pub fn name(self) -> &'static str {
        tool(self).name
    }
    pub fn def(self) -> &'static ToolDef {
        tool(self)
    }
    /// Tools that need neighbouring pixels (they cannot be baked into a LUT).
    pub fn is_spatial(self) -> bool {
        matches!(
            self,
            ToolKind::Halation
                | ToolKind::Diffusion
                | ToolKind::ChromaticAberration
                | ToolKind::LensDistortion
                | ToolKind::AnamorphicGlow
                | ToolKind::EdgeSoftness
                | ToolKind::ShadowHighlight
        )
    }
    /// Tools whose result depends on the pixel's position (grain, vignette, masks): also not
    /// representable as a colour LUT.
    pub fn is_positional(self) -> bool {
        matches!(self, ToolKind::FilmGrain | ToolKind::Vignette | ToolKind::GradientExposure | ToolKind::RadialExposure | ToolKind::SpotExposure)
    }
    /// A pure colour transform: can be baked into a `.cube`.
    pub fn is_color_only(self) -> bool {
        !self.is_spatial() && !self.is_positional()
    }
}

impl ToolDef {
    pub fn param(&self, id: &str) -> Option<&'static ParamDef> {
        // `self.params` is `'static`; find through it.
        self.params.iter().find(|p| p.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn catalogue_is_consistent() {
        let mut kinds = HashSet::new();
        for t in TOOLS {
            assert!(kinds.insert(t.kind), "duplicate {:?}", t.kind);
            assert_eq!(ToolKind::from_id(&t.kind.id()), Some(t.kind));
            let mut ids = HashSet::new();
            for p in t.params {
                assert!(ids.insert(p.id), "{}: duplicate param {}", t.name, p.id);
                assert_ne!(p.id, "strength", "{}: `strength` is every node's own", t.name);
                assert!(!p.name.is_empty());
                if let (Kind::Num { min, max, smin, smax, .. }, Def::Num(d)) = (p.kind, p.def) {
                    assert!(min <= smin && smin < smax && smax <= max, "{} {}", t.name, p.id);
                    assert!((min..=max).contains(&d), "{} {} default", t.name, p.id);
                }
                if let (Kind::Choice(o), Def::Choice(d)) = (p.kind, p.def) {
                    assert!((d as usize) < o.len());
                }
            }
        }
        // Every serde variant is in the table.
        for id in ["exposure", "lut", "selectiveColor", "filmGrain", "hsl"] {
            assert!(ToolKind::from_id(id).is_some(), "{id}");
        }
        assert_eq!(TOOLS.len(), 26);
    }
}
