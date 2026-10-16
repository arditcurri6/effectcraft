//! Export a look's colour transform as a `.cube` 3D LUT.
//!
//! Only colour tools can live in a LUT: tools that read neighbours (halation, diffusion, glows,
//! lens tools, local recovery) or the pixel's position (grain, vignette, masks) are left out,
//! and the result says which ones, so nothing spatial is baked silently. The lattice samples
//! the project's pixel values in 0..1 (encoded with the working curve, or linear in a linear
//! project), exactly as the effect receives them; the look strength is included.

use crate::look::Look;
use crate::lut::Lut;
use crate::ops::run;
use crate::program::{Env, Stage, compile};

/// The baked LUT's `.cube` text and the names of the tools left out.
pub struct CubeExport {
    pub text: String,
    pub skipped: Vec<String>,
}

/// Bake `look` (with look strength `strength`, 0..1) into a `size`³ lattice (2..=129).
pub fn bake_cube(look: &Look, env: &Env, strength: f32, size: usize, title: &str) -> Result<CubeExport, String> {
    let size = size.clamp(2, crate::lut::MAX_3D);
    let mut color = look.clone();
    let mut skipped = vec![];
    color.nodes.retain(|n| {
        let keep = n.tool.is_color_only();
        if !keep && n.enabled && n.mix() > 0.0 {
            skipped.push(n.name());
        }
        keep
    });
    if !color.is_active() {
        return Err("the look has no colour tools to bake into a LUT".into());
    }
    let prog = compile(&color, env, strength);
    if !prog.warnings.is_empty() {
        return Err(prog.warnings.join("; "));
    }
    let ops: Vec<_> = prog
        .stages
        .iter()
        .flat_map(|s| match s {
            Stage::Pixel(ops) => ops.clone(),
            // Colour-only looks compile to pixel stages only.
            Stage::Spatial(_) => vec![],
        })
        .collect();
    let k = (size - 1) as f32;
    let mut d3 = Vec::with_capacity(size * size * size);
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let c = [r as f32 / k, g as f32 / k, b as f32 / k];
                let o = run(&ops, c, [0.0, 0.0], &prog.working);
                let s = prog.strength;
                d3.push([0, 1, 2].map(|i| c[i] + (o[i] - c[i]) * s));
            }
        }
    }
    let lut = Lut {
        title: title.chars().filter(|c| *c != '"' && !c.is_control()).take(120).collect(),
        n1: 0,
        d1: vec![],
        min1: [0.0; 3],
        max1: [1.0; 3],
        n3: size,
        d3,
        min3: [0.0; 3],
        max3: [1.0; 3],
    };
    let mut text = String::from("# Exported by EffectCraft CineLooks\n");
    if !skipped.is_empty() {
        text.push_str(&format!("# Not included (spatial or positional tools): {}\n", skipped.join(", ")));
    }
    text.push_str(&lut.to_cube());
    Ok(CubeExport { text, skipped })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::look::Node;
    use crate::tools::ToolKind;

    #[test]
    fn baked_lut_reproduces_the_colour_chain() {
        let look = Look::new(
            "x",
            vec![
                Node::new(ToolKind::WhiteBalance).with("temperature", 25.0),
                Node::new(ToolKind::Curves).with("master", "0,0.05 0.5,0.55 1,0.95"),
                Node::new(ToolKind::Hsl).with("saturation", -20.0),
                Node::new(ToolKind::FilmGrain),
                Node::new(ToolKind::Halation),
            ],
        );
        let env = Env::default();
        let out = bake_cube(&look, &env, 1.0, 33, "Test").unwrap();
        assert_eq!(out.skipped, vec!["Film Grain".to_string(), "Halation".to_string()]);
        let lut = crate::lut::parse_cube(&out.text).unwrap();
        // Compare with the engine on colours between lattice points.
        let colour_only = Look::new("", look.nodes[..3].to_vec());
        let mut worst = 0.0f32;
        for c in [[0.13f32, 0.52, 0.81], [0.7, 0.33, 0.21], [0.5, 0.5, 0.5], [0.92, 0.88, 0.1]] {
            let img = effectcraft_raster::Image::filled(1, 1, [c[0], c[1], c[2], 1.0]);
            let (want, _) = crate::grade(&colour_only, &env, 1.0, &img);
            let got = lut.apply(c, false);
            for i in 0..3 {
                worst = worst.max((want.data[0][i] - got[i]).abs());
            }
        }
        assert!(worst < 4e-3, "LUT error {worst}");
        assert!(bake_cube(&Look::new("", vec![Node::new(ToolKind::Vignette)]), &env, 1.0, 17, "").is_err());
    }
}
