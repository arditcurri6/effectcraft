//! CineLooks: the grading engine behind the `ec.color.cinelooks` effect.
//!
//! A [`Look`] is an ordered chain of tool [`Node`]s ([`tools`] lists every tool, its parameters
//! and the colour domain it works in). [`program::compile`] turns a look and a render
//! environment ([`Env`]: layer geometry, time, working space) into a [`Program`]: per-pixel
//! tools fuse into pixel stages ([`ops`]), tools that read neighbours become spatial stages
//! ([`spatial`]). [`program::render`] runs it on the CPU; `effectcraft-gpu` runs the same
//! program with the same arithmetic.
//!
//! Colour: the chain works on straight (un-premultiplied) colour, in linear light or in the
//! working space's encoding as each tool needs, converting only at domain changes and once
//! back at the end ([`cmath::Working`]). Over-range values pass through. Alpha is kept, except
//! where the lens tools resample it.
//!
//! Everything here is pure Rust and panic-free on hostile input (NaN parameters, empty or huge
//! images, malformed `.cube` and `.cinelook` files).

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod blur;
pub mod cmath;
pub mod curve;
pub mod export;
pub mod look;
pub mod lut;
pub mod ops;
pub mod presets;
pub mod program;
pub mod spatial;
pub mod tools;

pub use cmath::Working;
pub use look::{Look, LookFile, Node, PV};
pub use program::{Env, Program, Stage, compile, render};
pub use tools::{Domain, TOOLS, ToolGroup, ToolKind};

use effectcraft_raster::Image;

/// Grade a premultiplied image: compile `look` for `env` and run it on the CPU, mixed with the
/// input by `strength` (0..1). Returns the image and the program's warnings.
pub fn grade(look: &Look, env: &Env, strength: f32, img: &Image) -> (Image, Vec<String>) {
    let p = compile(look, env, strength);
    (render(&p, img), p.warnings)
}
