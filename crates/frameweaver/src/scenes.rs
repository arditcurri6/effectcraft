//! Procedural test scenes with analytic ground truth, for tests and benchmarks.
//!
//! A scene is a stack of layers, each a fractal value-noise texture (natural-image-like
//! 1/f statistics, so flow estimation faces realistic detail and ambiguity) seen through a
//! soft-edged mask and moved by a rigid / similarity transform of continuous time. Because
//! everything is a function of `t`, any frame — including the in-between frame an interpolator
//! is asked for — is rendered exactly, and the true flow and occlusion between two times come
//! from the transforms: no estimation anywhere in the reference.

use effectcraft_raster::Image;

/// The shape through which a layer is visible, in layer-local pixels.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// Everywhere (a background).
    Full,
    /// A rectangle with these half extents.
    Rect([f32; 2]),
    /// An ellipse with these radii.
    Ellipse([f32; 2]),
}

/// One layer. At time `t` its centre is `pos + vel·t`, rotated by `rot·t` radians and scaled by
/// `exp(zoom·t)`.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub seed: u32,
    /// Texture feature size (pixels of the coarsest octave).
    pub feature: f32,
    pub shape: Shape,
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub rot: f32,
    pub zoom: f32,
    /// Colour offset so layers differ in palette (0…1 each).
    pub tint: [f32; 3],
}

impl Layer {
    pub fn background(seed: u32, vel: [f32; 2]) -> Layer {
        Layer { seed, feature: 48.0, shape: Shape::Full, pos: [0.0, 0.0], vel, rot: 0.0, zoom: 0.0, tint: [0.0; 3] }
    }

    /// Screen position of the layer-local point `p` at time `t`.
    fn screen_of(&self, p: [f32; 2], t: f32) -> [f32; 2] {
        let (s, c) = ((self.rot * t).sin(), (self.rot * t).cos());
        let k = (self.zoom * t).exp();
        [self.pos[0] + self.vel[0] * t + k * (c * p[0] - s * p[1]), self.pos[1] + self.vel[1] * t + k * (s * p[0] + c * p[1])]
    }

    /// Layer-local position of the screen point `x` at time `t`.
    fn local_of(&self, x: [f32; 2], t: f32) -> [f32; 2] {
        let (s, c) = ((self.rot * t).sin(), (self.rot * t).cos());
        let k = (self.zoom * t).exp();
        let d = [(x[0] - self.pos[0] - self.vel[0] * t) / k, (x[1] - self.pos[1] - self.vel[1] * t) / k];
        [c * d[0] + s * d[1], -s * d[0] + c * d[1]]
    }

    /// Coverage 0…1 at the layer-local point (soft over about one screen pixel).
    fn coverage(&self, p: [f32; 2], t: f32) -> f32 {
        let k = (self.zoom * t).exp();
        let sdf = match self.shape {
            Shape::Full => return 1.0,
            Shape::Rect(h) => (h[0] - p[0].abs()).min(h[1] - p[1].abs()),
            Shape::Ellipse(r) => (1.0 - ((p[0] / r[0]).powi(2) + (p[1] / r[1]).powi(2)).sqrt()) * r[0].min(r[1]),
        };
        (0.5 + sdf * k).clamp(0.0, 1.0)
    }
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Smooth value noise in 0…1.
fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (tx, ty) = (x - xi, y - yi);
    let q = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (q(tx), q(ty));
    let (xi, yi) = (xi as i32, yi as i32);
    let a = hash(xi, yi, seed);
    let b = hash(xi + 1, yi, seed);
    let c = hash(xi, yi + 1, seed);
    let d = hash(xi + 1, yi + 1, seed);
    (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v
}

/// Fractal noise: six octaves, persistence 0.55 (a roughly 1/f spectrum), 0…1.
fn fractal(x: f32, y: f32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut f) = (0.0, 1.0, 0.0, 1.0);
    for o in 0..6u32 {
        sum += amp * value_noise(x * f, y * f, seed.wrapping_add(o * 7919));
        norm += amp;
        amp *= 0.55;
        f *= 2.0;
    }
    sum / norm
}

/// A scene: layers from back to front, a frame size and optional per-frame sensor noise.
#[derive(Clone, Debug)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<Layer>,
    /// Standard deviation of the temporal noise added to each channel (0 = none).
    pub noise: f32,
}

impl Scene {
    fn color(&self, l: &Layer, p: [f32; 2]) -> [f32; 3] {
        let (u, v) = (p[0] / l.feature, p[1] / l.feature);
        let lum = fractal(u, v, l.seed);
        let c = |k: u32| 0.5 + (0.65 * (lum - 0.5) + 0.35 * (fractal(u, v, l.seed.wrapping_add(k * 104729)) - 0.5)) * 1.9;
        [(c(1) + l.tint[0]).clamp(0.0, 1.0), (c(2) + l.tint[1]).clamp(0.0, 1.0), (c(3) + l.tint[2]).clamp(0.0, 1.0)]
    }

    /// Frame at time `t` (`frame_index` seeds the sensor noise).
    pub fn render(&self, t: f32, frame_index: u32) -> Image {
        let mut img = Image::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let sx = [x as f32 + 0.5, y as f32 + 0.5];
                let mut px = [0.0f32; 3];
                for l in &self.layers {
                    let p = l.local_of(sx, t);
                    let cov = l.coverage(p, t);
                    if cov <= 0.0 {
                        continue;
                    }
                    let c = self.color(l, p);
                    for k in 0..3 {
                        px[k] += (c[k] - px[k]) * cov;
                    }
                }
                if self.noise > 0.0 {
                    for (k, v) in px.iter_mut().enumerate() {
                        let n = hash(x as i32, y as i32, frame_index.wrapping_mul(31).wrapping_add(k as u32 + 0x5eed))
                            + hash(y as i32, x as i32, frame_index.wrapping_add(k as u32 * 977))
                            - 1.0;
                        *v = (*v + n * self.noise * 2.45).clamp(0.0, 1.0);
                    }
                }
                img.set(x, y, [px[0], px[1], px[2], 1.0]);
            }
        }
        img
    }

    /// The top-most layer index covering the screen point at time `t` (coverage > ½).
    fn top(&self, x: [f32; 2], t: f32) -> Option<usize> {
        (0..self.layers.len()).rev().find(|&i| self.layers[i].coverage(self.layers[i].local_of(x, t), t) > 0.5)
    }

    /// True flow from time `t0` to `t1` at every pixel (pixels) and whether the pixel is
    /// occluded: its target leaves the frame or is covered by another layer at `t1`.
    pub fn flow(&self, t0: f32, t1: f32) -> (Vec<[f32; 2]>, Vec<bool>) {
        let n = (self.width * self.height) as usize;
        let mut flow = vec![[0.0; 2]; n];
        let mut occ = vec![false; n];
        for y in 0..self.height {
            for x in 0..self.width {
                let sx = [x as f32 + 0.5, y as f32 + 0.5];
                let i = (y * self.width + x) as usize;
                let Some(li) = self.top(sx, t0) else { continue };
                let l = &self.layers[li];
                let target = l.screen_of(l.local_of(sx, t0), t1);
                flow[i] = [target[0] - sx[0], target[1] - sx[1]];
                let inside = target[0] >= 0.0 && target[1] >= 0.0 && target[0] < self.width as f32 && target[1] < self.height as f32;
                occ[i] = !inside || self.top(target, t1) != Some(li);
            }
        }
        (flow, occ)
    }
}

/// Mean endpoint error of `est` (A → B vectors at `scale` frame pixels per estimate pixel,
/// estimate size `ew × eh`) against the true flow of the non-occluded pixels at least `margin`
/// from the border.
pub fn endpoint_error(scene: &Scene, t0: f32, t1: f32, est: &crate::flow::FlowField, margin: u32) -> f32 {
    let (flow, occ) = scene.flow(t0, t1);
    let (sx, sy) = (est.w as f32 / scene.width as f32, est.h as f32 / scene.height as f32);
    let (mut s, mut n) = (0.0f64, 0u64);
    for y in margin..scene.height.saturating_sub(margin) {
        for x in margin..scene.width.saturating_sub(margin) {
            let i = (y * scene.width + x) as usize;
            if occ[i] {
                continue;
            }
            let v = est.sample((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5);
            let (ex, ey) = (v[0] / sx - flow[i][0], v[1] / sy - flow[i][1]);
            s += ex.hypot(ey) as f64;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { (s / n as f64) as f32 }
}

/// The named scene suite: `(name, scene, time step between source frames)`.
pub fn suite(width: u32, height: u32) -> Vec<(&'static str, Scene)> {
    let (w, h) = (width as f32, height as f32);
    let bg = |seed, vel| Layer::background(seed, vel);
    let obj = |seed: u32, shape, pos: [f32; 2], vel: [f32; 2], tint: [f32; 3]| Layer { seed, feature: 28.0, shape, pos, vel, rot: 0.0, zoom: 0.0, tint };
    let scene = |layers: Vec<Layer>, noise: f32| Scene { width, height, layers, noise };
    // Velocities are px per unit time; the suite's step between source frames is 1.
    vec![
        ("pan", scene(vec![bg(11, [4.3, -1.7])], 0.0)),
        ("fast pan", scene(vec![bg(12, [21.0, 6.0])], 0.0)),
        ("zoom+rotate", scene(vec![Layer { rot: 0.03, zoom: 0.04, pos: [w * 0.5, h * 0.5], ..bg(13, [0.0, 0.0]) }], 0.0)),
        ("parallax", scene(vec![bg(14, [3.0, 0.0]), obj(15, Shape::Rect([w * 0.18, h * 0.3]), [w * 0.5, h * 0.5], [-7.0, 1.0], [0.1, 0.0, 0.0])], 0.0)),
        (
            "crossing",
            scene(
                vec![
                    bg(16, [-1.5, 0.5]),
                    obj(17, Shape::Rect([w * 0.14, h * 0.22]), [w * 0.2, h * 0.35], [9.0, 0.0], [0.15, 0.0, 0.0]),
                    obj(18, Shape::Ellipse([w * 0.14, h * 0.2]), [w * 0.8, h * 0.62], [-9.0, 0.0], [0.0, 0.0, 0.15]),
                ],
                0.0,
            ),
        ),
        (
            "small fast object",
            scene(vec![bg(19, [0.0, 0.0]), obj(20, Shape::Ellipse([w * 0.05, w * 0.05]), [w * 0.2, h * 0.5], [w * 0.09, 0.0], [0.2, 0.1, 0.0])], 0.0),
        ),
        ("noisy pan", scene(vec![bg(21, [6.5, -2.2])], 0.02)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendering_is_deterministic_and_motion_is_rigid() {
        let s = &suite(96, 64)[0].1;
        assert_eq!(s.render(0.0, 0), s.render(0.0, 0));
        // The pan: frame at t = 1 is frame at t = 0 shifted by (4.3, −1.7).
        let (flow, occ) = s.flow(0.0, 1.0);
        let mid = 32 * 96 + 40;
        assert!((flow[mid][0] - 4.3).abs() < 1e-4 && (flow[mid][1] + 1.7).abs() < 1e-4);
        assert!(!occ[mid]);
        assert!(occ[32 * 96 + 95]); // the right edge leaves the frame
        assert!(occ[96 + 4]); // so does the top row (the pan moves up)
    }

    #[test]
    fn occluder_flow_and_occlusion_are_analytic() {
        let (w, h) = (128u32, 96u32);
        let s = &suite(w, h).into_iter().find(|(n, _)| *n == "parallax").unwrap().1;
        let (flow, occ) = s.flow(0.0, 1.0);
        let inside = ((h / 2) * w + w / 2) as usize; // inside the rectangle
        let outside = (6 * w + 6) as usize;
        assert!((flow[inside][0] + 7.0).abs() < 1e-4, "{:?}", flow[inside]);
        assert!((flow[outside][0] - 3.0).abs() < 1e-4);
        assert!(!occ[inside] && !occ[outside]);
        // Background just ahead of the moving rectangle (it moves left) is covered at t = 1.
        let covered = (((h / 2) * w) as f32 + w as f32 * 0.5 - w as f32 * 0.18 - 3.0) as usize;
        assert!(occ[covered], "{covered}");
    }

    #[test]
    fn texture_has_detail_at_every_scale() {
        let s = Scene { width: 128, height: 128, layers: vec![Layer::background(3, [0.0; 2])], noise: 0.0 };
        let img = s.render(0.0, 0);
        let mut grad = 0.0;
        for y in 0..127 {
            for x in 0..127 {
                grad += (img.get(x + 1, y)[0] - img.get(x, y)[0]).abs() + (img.get(x, y + 1)[0] - img.get(x, y)[0]).abs();
            }
        }
        assert!(grad / (127.0 * 127.0 * 2.0) > 0.004, "{}", grad / (127.0 * 127.0 * 2.0));
        let (mn, mx) = img.data.iter().fold((1.0f32, 0.0f32), |(a, b), p| (a.min(p[0]), b.max(p[0])));
        assert!(mx - mn > 0.3);
    }
}
