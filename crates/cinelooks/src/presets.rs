//! The built-in look library: original looks designed for CineLooks (code-defined, MIT OR
//! Apache-2.0 like the rest of EffectCraft; listed in ATTRIBUTION.md). Each is a chain of tone,
//! colour, film and finishing tools with real parameters; none reproduces a commercial film
//! stock or another product's preset.

use std::sync::OnceLock;

use crate::look::{Look, Node};
use crate::tools::ToolKind as T;

/// Library categories, in browser order.
pub const CATEGORIES: &[&str] =
    &["Cinematic", "Film", "Vintage", "Modern", "Warm", "Cold", "Blockbuster", "Documentary", "Black & White", "Music Videos", "Portrait", "Night"];

/// A library look.
#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    /// Stable id (`builtin.warm-cinema`).
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
    pub look: Look,
}

/// A wheel position: hue angle in degrees (red 0°, yellow 60°, green 120°, cyan 180°, blue
/// 240°, magenta 300°), amount 0..1 and master.
fn wh(deg: f64, amount: f64, master: f64) -> [f64; 3] {
    let r = deg.to_radians();
    [r.cos() * amount, r.sin() * amount, master]
}

fn n(t: T) -> Node {
    Node::new(t)
}
fn grain(amount: f64, kind: f64, size: f64) -> Node {
    n(T::FilmGrain).with("amount", amount).with("type", kind).with("size", size)
}
fn vignette(amount: f64) -> Node {
    n(T::Vignette).with("amount", amount)
}
fn wb(temp: f64, tint: f64) -> Node {
    n(T::WhiteBalance).with("temperature", temp).with("tint", tint)
}
fn contrast(c: f64, filmic: f64) -> Node {
    n(T::Contrast).with("contrast", c).with("filmic", filmic)
}
fn film(contrast: f64, toe: f64, shoulder: f64, latitude: f64, color: f64) -> Node {
    n(T::FilmResponse).with("contrast", contrast).with("toe", toe).with("shoulder", shoulder).with("latitude", latitude).with("color", color)
}
fn curves(master: &str) -> Node {
    n(T::Curves).with("master", master)
}
fn halation(amount: f64, radius: f64) -> Node {
    n(T::Halation).with("amount", amount).with("radius", radius)
}
fn diffusion(threshold: f64, intensity: f64, softness: f64) -> Node {
    n(T::Diffusion).with("threshold", threshold).with("intensity", intensity).with("softness", softness)
}
fn mono(r: f64, g: f64, b: f64) -> Node {
    n(T::ChannelMixer).with("monochrome", true).with("monoR", r).with("monoG", g).with("monoB", b)
}
fn hsl(sat: f64) -> Node {
    n(T::Hsl).with("saturation", sat)
}
fn shadow_highlight(sh: f64, hi: f64) -> Node {
    n(T::ShadowHighlight).with("shadows", sh).with("highlights", hi)
}
fn fade(blacks: f64, contrast: f64, sat: f64, tint: f64) -> Node {
    n(T::FilmFade).with("blacks", blacks).with("contrast", contrast).with("saturation", sat).with("tintAmount", tint)
}

fn p(name: &str, category: &str, description: &str, nodes: Vec<Node>) -> Preset {
    let id = format!(
        "builtin.{}",
        name.to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    );
    Preset { id, name: name.into(), category: category.into(), description: description.into(), look: Look::new(name, nodes) }
}

fn library() -> Vec<Preset> {
    vec![
        // ---- Cinematic
        p(
            "Warm Cinema",
            "Cinematic",
            "A gentle filmic curve, warm highlights over cool shadows, soft halation and fine grain.",
            vec![
                film(10.0, 40.0, 60.0, 1.0, 5.0),
                wb(18.0, 4.0),
                n(T::Wheels).with("shadows", wh(195.0, 0.25, 0.0)).with("highlights", wh(35.0, 0.3, 0.0)),
                curves("0,0.02 0.25,0.22 0.75,0.8 1,0.98"),
                halation(25.0, 1.0),
                grain(15.0, 0.0, 1.0),
                vignette(-25.0),
            ],
        ),
        p(
            "Cinema Neutral",
            "Cinematic",
            "Neutral colour with a film shoulder and toe, a touch of contrast and texture.",
            vec![film(5.0, 35.0, 55.0, 1.5, 0.0), contrast(8.0, 75.0), hsl(-8.0), grain(10.0, 0.0, 1.0), vignette(-15.0)],
        ),
        p(
            "Teal Dusk",
            "Cinematic",
            "Teal shadows and amber skin tones with lifted mids and a little bloom.",
            vec![
                film(15.0, 35.0, 60.0, 1.0, 0.0),
                n(T::Wheels).with("shadows", wh(190.0, 0.35, 0.0)).with("highlights", wh(32.0, 0.3, 0.0)).with("gamma", wh(0.0, 0.0, 0.05)),
                n(T::Hsl).with("orangeSat", 10.0).with("blueHue", -10.0).with("aquaSat", -20.0),
                diffusion(0.65, 15.0, 20.0),
                vignette(-20.0),
            ],
        ),
        p(
            "Desaturated Drama",
            "Cinematic",
            "Muted colour with strong contrast, cool shadows and a dense toe.",
            vec![
                n(T::Hsl).with("saturation", -40.0).with("redSat", 10.0),
                contrast(30.0, 60.0),
                n(T::Wheels).with("shadows", wh(215.0, 0.25, -0.03)),
                film(0.0, 55.0, 50.0, 1.0, -10.0),
                vignette(-35.0),
                grain(18.0, 1.0, 1.0),
            ],
        ),
        // ---- Film
        p(
            "Natural Film",
            "Film",
            "Wide latitude, soft shoulder, slightly lifted blacks, halation and medium grain.",
            vec![film(0.0, 40.0, 65.0, 1.5, 10.0), curves("0,0.03 0.5,0.5 1,0.98"), halation(20.0, 1.2), grain(22.0, 1.0, 1.0)],
        ),
        p(
            "Warm Print",
            "Film",
            "A warm print curve with dense blacks, gentle halation and grain.",
            vec![n(T::FilmPrint).with("style", 1.0), halation(30.0, 1.5), grain(18.0, 0.0, 1.0), vignette(-15.0)],
        ),
        p(
            "Cool Print",
            "Film",
            "A cool, clean print response with crisp contrast.",
            vec![n(T::FilmPrint).with("style", 2.0).with("highlights", 10.0), n(T::Hsl).with("blueSat", 10.0).with("orangeSat", -5.0), grain(15.0, 0.0, 1.0)],
        ),
        p(
            "Silver Retention",
            "Film",
            "Bleach-bypass contrast and desaturation over a filmic curve, with coarse grain.",
            vec![
                n(T::BleachBypass).with("contrast", 70.0).with("desaturation", 55.0),
                film(10.0, 40.0, 50.0, 1.0, 0.0),
                grain(25.0, 1.0, 1.1),
                vignette(-30.0),
            ],
        ),
        // ---- Vintage
        p(
            "Vintage Summer",
            "Vintage",
            "Faded warm print: lifted blacks, warm cast, coarse grain and soft halation.",
            vec![
                fade(30.0, 25.0, -15.0, 45.0),
                wb(25.0, 5.0),
                n(T::Curves).with("red", "0,0.04 0.5,0.53 1,1"),
                grain(30.0, 2.0, 1.0),
                vignette(-35.0),
                halation(20.0, 1.5),
            ],
        ),
        p(
            "Faded Memories",
            "Vintage",
            "Low contrast, washed colour, yellowed greens and a soft glow.",
            vec![fade(40.0, 35.0, -30.0, 40.0), n(T::Hsl).with("greenHue", 20.0).with("blueSat", -30.0), grain(30.0, 1.0, 1.2), diffusion(0.5, 20.0, 40.0)],
        ),
        p(
            "Sunbleached",
            "Vintage",
            "Sun-faded print: warm and magenta-leaning, flattened blues, slight colour fringing.",
            vec![
                wb(30.0, 8.0),
                n(T::Curves).with("blue", "0,0.12 1,0.85"),
                fade(20.0, 15.0, -15.0, 30.0),
                grain(25.0, 1.0, 1.0),
                n(T::ChromaticAberration).with("redCyan", 1.0).with("blueYellow", -0.5),
            ],
        ),
        // ---- Modern
        p(
            "Clean Digital",
            "Modern",
            "Crisp, polished contrast with recovered shadows and highlights.",
            vec![contrast(15.0, 80.0), curves("0,0 0.25,0.23 0.75,0.78 1,1"), hsl(8.0), shadow_highlight(20.0, 30.0), vignette(-10.0)],
        ),
        p(
            "Urban Contrast",
            "Modern",
            "Hard contrast, muted colour with warm skin tones and teal shadows.",
            vec![
                contrast(35.0, 50.0),
                n(T::Hsl).with("saturation", -25.0).with("orangeSat", 15.0),
                n(T::Wheels).with("shadows", wh(195.0, 0.2, 0.0)),
                shadow_highlight(0.0, 40.0),
                grain(10.0, 0.0, 1.0),
                vignette(-30.0),
            ],
        ),
        p(
            "Crisp Pastel",
            "Modern",
            "Bright, airy and soft-coloured with pastel shadows.",
            vec![
                n(T::Exposure).with("exposure", 0.3).with("compensation", 0.3).with("highlightProtect", 40.0),
                fade(15.0, 15.0, 0.0, 25.0).with("shadowTint", [0.55, 0.62, 0.75, 1.0]).with("highlightTint", [1.0, 0.92, 0.9, 1.0]),
                n(T::Hsl).with("saturation", -10.0).with("lightness", 5.0),
            ],
        ),
        // ---- Warm
        p(
            "Golden Hour",
            "Warm",
            "Low sun: golden highlights, a warm graduated sky and glowing highlights.",
            vec![
                wb(35.0, 6.0),
                n(T::Wheels).with("highlights", wh(45.0, 0.35, 0.0)).with("midtones", wh(30.0, 0.15, 0.0)),
                n(T::GradientExposure).with("exposure", -0.4).with("tint", [1.0, 0.75, 0.45, 1.0]).with("tintAmount", 35.0),
                diffusion(0.55, 30.0, 20.0).with("color", [1.0, 0.85, 0.6, 1.0]),
                halation(20.0, 1.2),
            ],
        ),
        p(
            "Amber Interior",
            "Warm",
            "Tungsten warmth with rich oranges and a deep vignette.",
            vec![
                wb(28.0, 2.0),
                film(10.0, 45.0, 55.0, 1.0, 5.0),
                n(T::Hsl).with("orangeSat", 15.0).with("yellowHue", -10.0),
                vignette(-30.0),
                grain(12.0, 0.0, 1.0),
            ],
        ),
        p(
            "Desert Heat",
            "Warm",
            "Hot, dry and contrasty: pushed warmth, depressed blues, partial bleach.",
            vec![
                wb(40.0, 0.0),
                n(T::Curves).with("blue", "0,0 0.5,0.42 1,0.9"),
                contrast(20.0, 60.0),
                n(T::BleachBypass).with("contrast", 30.0).with("desaturation", 25.0).strength(60.0),
                grain(15.0, 1.0, 1.0),
            ],
        ),
        // ---- Cold
        p(
            "Midnight Blue",
            "Cold",
            "Cold, underexposed blue night with muted colour and a heavy vignette.",
            vec![
                wb(-45.0, 0.0),
                n(T::Exposure).with("exposure", -0.4),
                n(T::Wheels).with("shadows", wh(235.0, 0.35, 0.0)).with("highlights", wh(185.0, 0.15, 0.0)),
                hsl(-20.0),
                film(5.0, 50.0, 55.0, 1.0, 0.0),
                vignette(-35.0),
                grain(18.0, 1.0, 1.0),
            ],
        ),
        p(
            "Cool Shadows",
            "Cold",
            "Neutral highlights over teal-blue shadows with gentle contrast.",
            vec![
                n(T::Wheels).with("shadows", wh(200.0, 0.35, 0.0)),
                n(T::Hsl).with("blueSat", 10.0),
                curves("0,0.02 0.3,0.27 0.7,0.74 1,1"),
                contrast(15.0, 70.0),
            ],
        ),
        p(
            "Arctic Morning",
            "Cold",
            "Bright, cold and soft: protected highlights, a cool fade and mist.",
            vec![
                wb(-30.0, -5.0),
                n(T::Exposure).with("exposure", 0.3).with("highlightProtect", 50.0),
                fade(10.0, 15.0, -20.0, 30.0).with("shadowTint", [0.3, 0.45, 0.6, 1.0]).with("highlightTint", [0.92, 0.97, 1.0, 1.0]),
                diffusion(0.6, 20.0, 35.0).with("color", [0.9, 0.95, 1.0, 1.0]),
            ],
        ),
        // ---- Blockbuster
        p(
            "Blockbuster Teal & Orange",
            "Blockbuster",
            "Complementary teal and orange, punchy contrast, protected highlights and streaks.",
            vec![
                n(T::Wheels).with("shadows", wh(190.0, 0.45, 0.0)).with("highlights", wh(30.0, 0.4, 0.0)).with("gain", wh(0.0, 0.0, 0.05)),
                n(T::Hsl).with("orangeSat", 20.0).with("aquaHue", -20.0).with("greenSat", -30.0),
                contrast(30.0, 65.0),
                shadow_highlight(10.0, 35.0),
                n(T::AnamorphicGlow).with("intensity", 25.0),
                vignette(-25.0),
                grain(8.0, 0.0, 1.0),
            ],
        ),
        p(
            "Steel Action",
            "Blockbuster",
            "Hard, metallic and cold: bleach bypass, blue shadows and grit.",
            vec![
                n(T::BleachBypass).with("contrast", 60.0).with("desaturation", 45.0),
                wb(-15.0, 0.0),
                contrast(20.0, 55.0),
                n(T::Wheels).with("shadows", wh(225.0, 0.3, 0.0)),
                grain(20.0, 1.0, 1.0),
                vignette(-30.0),
            ],
        ),
        p(
            "Epic Sunset",
            "Blockbuster",
            "Warm print colour with lens bloom, orange streaks and halation.",
            vec![
                n(T::FilmPrint).with("style", 1.0),
                wb(20.0, 0.0),
                diffusion(0.6, 25.0, 15.0),
                n(T::AnamorphicGlow).with("intensity", 35.0).with("color", [1.0, 0.6, 0.3, 1.0]),
                halation(25.0, 1.5),
                vignette(-25.0),
            ],
        ),
        // ---- Documentary
        p(
            "Documentary Natural",
            "Documentary",
            "Honest colour, balanced exposure and opened-up shadows.",
            vec![contrast(8.0, 80.0), shadow_highlight(25.0, 25.0), hsl(-5.0), curves("0,0.01 0.5,0.51 1,0.99"), grain(8.0, 0.0, 1.0)],
        ),
        p(
            "Observational",
            "Documentary",
            "Wide latitude and restrained colour for long-form verité.",
            vec![film(-10.0, 30.0, 60.0, 2.0, -10.0), wb(5.0, 0.0), hsl(-12.0), grain(15.0, 1.0, 1.0)],
        ),
        p(
            "Field Report",
            "Documentary",
            "Clear, slightly lifted mids, held highlights and calm greens.",
            vec![
                n(T::Exposure).with("compensation", 0.2),
                contrast(12.0, 70.0),
                shadow_highlight(30.0, 20.0),
                n(T::Hsl).with("greenSat", -10.0),
                vignette(-10.0),
            ],
        ),
        // ---- Black & White
        p(
            "Classic Monochrome",
            "Black & White",
            "A balanced panchromatic mix with filmic contrast, grain and vignette.",
            vec![mono(40.0, 50.0, 10.0), film(15.0, 40.0, 55.0, 1.0, 0.0), grain(30.0, 1.0, 1.0), vignette(-30.0)],
        ),
        p(
            "High Key Mono",
            "Black & White",
            "Bright, soft monochrome with protected highlights and a glow.",
            vec![
                mono(30.0, 60.0, 10.0),
                n(T::Exposure).with("exposure", 0.5).with("highlightProtect", 60.0),
                fade(10.0, 10.0, 0.0, 0.0),
                diffusion(0.5, 25.0, 40.0),
            ],
        ),
        p(
            "Noir",
            "Black & White",
            "Red-filtered monochrome, hard contrast, a pool of light and heavy grain.",
            vec![
                mono(60.0, 40.0, 0.0),
                contrast(45.0, 50.0),
                n(T::RadialExposure).with("exposure", 0.3).with("width", 70.0).with("height", 60.0),
                vignette(-55.0),
                grain(35.0, 2.0, 1.0),
            ],
        ),
        // ---- Music Videos
        p(
            "Neon Streets",
            "Music Videos",
            "Magenta and cyan neon: saturated lights, blue-violet shadows, bloom and streaks.",
            vec![
                n(T::Hsl).with("magentaSat", 30.0).with("aquaSat", 30.0).with("blueHue", 15.0),
                n(T::Wheels).with("shadows", wh(270.0, 0.35, 0.0)).with("highlights", wh(185.0, 0.25, 0.0)),
                contrast(25.0, 60.0),
                diffusion(0.5, 35.0, 15.0).with("color", [1.0, 0.75, 0.95, 1.0]),
                n(T::AnamorphicGlow).with("intensity", 40.0).with("color", [0.7, 0.5, 1.0, 1.0]),
                n(T::ChromaticAberration).with("redCyan", 2.0),
                grain(10.0, 0.0, 1.0),
            ],
        ),
        p(
            "Pop Saturation",
            "Music Videos",
            "Bold, saturated colour with snappy contrast and a hint of fringing.",
            vec![
                hsl(30.0),
                contrast(30.0, 60.0),
                curves("0,0 0.25,0.2 0.75,0.82 1,1"),
                vignette(-20.0),
                n(T::ChromaticAberration).with("redCyan", 1.0).with("blueYellow", 0.0),
            ],
        ),
        p(
            "Analog Tape",
            "Music Videos",
            "Worn tape: colour fringes, soft edges, lifted blacks, coarse colour noise and a curved image.",
            vec![
                n(T::ChromaticAberration).with("redCyan", 3.0).with("blueYellow", -2.0).with("radial", 30.0),
                n(T::EdgeSoftness).with("radius", 1.0),
                fade(25.0, 15.0, -10.0, 20.0),
                n(T::Hsl).with("redHue", 5.0),
                grain(35.0, 2.0, 1.0).with("colored", true),
                n(T::LensDistortion).with("amount", 10.0).with("zoom", 105.0),
            ],
        ),
        // ---- Portrait
        p(
            "Soft Portrait",
            "Portrait",
            "Flattering skin, soft diffusion, lifted face light and a gentle vignette.",
            vec![
                n(T::Exposure).with("compensation", 0.25),
                diffusion(0.45, 30.0, 45.0),
                n(T::Hsl).with("orangeLum", 15.0).with("orangeSat", -8.0).with("redSat", -10.0),
                contrast(-8.0, 80.0),
                n(T::RadialExposure).with("exposure", 0.25),
                vignette(-20.0),
            ],
        ),
        p(
            "Golden Skin",
            "Portrait",
            "Warm, even skin tones (a narrow hue remap), with a soft glow.",
            vec![
                wb(12.0, 2.0),
                n(T::HueRemap)
                    .with("center", 25.0)
                    .with("range", 15.0)
                    .with("falloff", 20.0)
                    .with("shift", -4.0)
                    .with("saturation", -10.0)
                    .with("luminance", 10.0),
                curves("0,0.01 0.5,0.52 1,0.99"),
                diffusion(0.55, 15.0, 30.0),
            ],
        ),
        p(
            "Dreamy Highlights",
            "Portrait",
            "Glowing, airy highlights with a pastel fade.",
            vec![
                diffusion(0.5, 45.0, 50.0),
                fade(12.0, 10.0, -10.0, 20.0),
                n(T::Exposure).with("exposure", 0.2).with("highlightProtect", 50.0),
                halation(25.0, 1.5),
            ],
        ),
        // ---- Night
        p(
            "Night City",
            "Night",
            "Cool ambient light with warm practicals, glowing highlights and grain.",
            vec![
                wb(-20.0, 0.0),
                n(T::Exposure).with("exposure", -0.3),
                n(T::Wheels).with("shadows", wh(225.0, 0.3, 0.0)).with("highlights", wh(35.0, 0.3, 0.0)),
                contrast(20.0, 60.0),
                halation(40.0, 1.5),
                diffusion(0.5, 25.0, 15.0),
                grain(25.0, 1.0, 1.0),
                vignette(-30.0),
            ],
        ),
        p(
            "Moonlight",
            "Night",
            "Day-for-night: underexposed, cold and desaturated with compressed highlights.",
            vec![
                wb(-55.0, -5.0),
                n(T::Exposure).with("exposure", -1.3).with("compensation", -0.5),
                hsl(-45.0),
                curves("0,0 0.5,0.42 1,0.8"),
                vignette(-45.0),
                grain(20.0, 1.0, 1.0),
            ],
        ),
        p(
            "Sodium Glow",
            "Night",
            "Orange streetlight against deep blue shadows, with halation and streaks.",
            vec![
                n(T::Wheels).with("shadows", wh(230.0, 0.4, -0.03)).with("highlights", wh(28.0, 0.45, 0.0)),
                n(T::Hsl).with("orangeSat", 20.0).with("greenSat", -40.0),
                halation(45.0, 2.0),
                n(T::AnamorphicGlow).with("intensity", 20.0).with("color", [1.0, 0.65, 0.3, 1.0]),
                grain(22.0, 1.0, 1.0),
            ],
        ),
    ]
}

/// The built-in library.
pub fn builtin() -> &'static [Preset] {
    static L: OnceLock<Vec<Preset>> = OnceLock::new();
    L.get_or_init(library)
}

/// A built-in look by id or (case-insensitive) name.
pub fn find(id_or_name: &str) -> Option<&'static Preset> {
    builtin().iter().find(|p| p.id == id_or_name || p.name.eq_ignore_ascii_case(id_or_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::Env;
    use effectcraft_raster::Image;

    #[test]
    fn library_is_complete_and_original() {
        let lib = builtin();
        assert!(lib.len() >= 30, "{}", lib.len());
        let mut ids = std::collections::HashSet::new();
        for p in lib {
            assert!(ids.insert(p.id.clone()), "duplicate {}", p.id);
            assert!(CATEGORIES.contains(&p.category.as_str()), "{}", p.category);
            assert!(p.look.nodes.len() >= 3, "{} is too thin", p.name);
            assert!(!p.description.is_empty());
        }
        for c in CATEGORIES {
            assert!(lib.iter().filter(|p| p.category == *c).count() >= 3, "{c}");
        }
        assert_eq!(find("warm cinema").map(|p| p.id.as_str()), Some("builtin.warm-cinema"));
    }

    #[test]
    fn every_preset_renders_and_changes_the_image() {
        let (w, h) = (96u32, 54u32);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
                let (r, g, b) = effectcraft_color::hsv_to_rgb(v, 0.5, 0.1 + 0.85 * u);
                img.set(x, y, [r, g, b, 1.0]);
            }
        }
        let env = Env { layer_size: [w as f64, h as f64], time: 0.5, ..Env::default() };
        for p in builtin() {
            let mut look = p.look.clone();
            look.sanitize();
            assert_eq!(look, p.look, "{}: the preset has out-of-range values", p.name);
            let (out, warnings) = crate::grade(&p.look, &env, 1.0, &img);
            assert!(warnings.is_empty(), "{}: {warnings:?}", p.name);
            assert!(out.data.iter().all(|q| q.iter().all(|v| v.is_finite())));
            let mean: f32 =
                out.data.iter().zip(&img.data).map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).abs()).sum::<f32>()).sum::<f32>() / (3 * out.data.len()) as f32;
            assert!(mean > 0.01, "{} barely changes the image ({mean})", p.name);
        }
    }
}
