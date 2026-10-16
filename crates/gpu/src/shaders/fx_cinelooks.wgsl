// GPU effects, CineLooks: see src/fx_cinelooks.rs. The look is compiled on the CPU
// (effectcraft_cinelooks::compile); these kernels run its stages with the arithmetic of
// effectcraft_cinelooks::{ops, blur, spatial}, in the same order.
//
// Between stages the image is straight colour (in the chain's current domain) plus alpha.
// fcl_pixel interprets a pixel stage's op stream from `data` (see fx_cinelooks.rs `encode`):
// P.u[0] = (first: unpremultiply the input, last: premultiply and mix with `aux` (the
// original input) by the look strength, working curve code (1 sRGB, 2 gamma 2.4), 0);
// P.f[0] = (buffer offset x, y, scale, look strength); P.f[1] = (luma weights, 0).

// ---------------------------------------------------------------- shared maths (cmath.rs)

fn fcl_ss(e0: f32, e1: f32, x: f32) -> f32 {
    let d = e1 - e0;
    var t: f32;
    if (abs(d) < 1e-12) {
        t = select(1.0, 0.0, x < e0);
    } else {
        t = clamp((x - e0) / d, 0.0, 1.0);
    }
    return t * t * (3.0 - 2.0 * t);
}

fn fcl_dec(code: u32, v: f32) -> f32 {
    if (code == 1u) {
        return fxl_decode(1u, 0.0, v);
    }
    if (code == 2u) {
        return fxl_decode(2u, 2.4, v);
    }
    return v;
}

fn fcl_enc(code: u32, v: f32) -> f32 {
    if (code == 1u) {
        return fxl_encode(1u, 0.0, v);
    }
    if (code == 2u) {
        return fxl_encode(2u, 2.4, v);
    }
    return v;
}

fn fcl_dec3(code: u32, c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(fcl_dec(code, c.x), fcl_dec(code, c.y), fcl_dec(code, c.z));
}

fn fcl_enc3(code: u32, c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(fcl_enc(code, c.x), fcl_enc(code, c.y), fcl_enc(code, c.z));
}

fn fcl_spow(x: f32, e: f32) -> f32 {
    return fxl_copysign(powz(abs(x), e), x);
}

// hue_sat: HSV hue (degrees) and saturation.
fn fcl_hue_sat(c: vec3<f32>) -> vec2<f32> {
    let mx = max(max(c.x, c.y), c.z);
    let mn = min(min(c.x, c.y), c.z);
    let d = mx - mn;
    if (d <= 1e-7) {
        return vec2<f32>(0.0, 0.0);
    }
    var h: f32;
    if (mx == c.x) {
        h = 60.0 * ((c.y - c.z) / d);
    } else if (mx == c.y) {
        h = 60.0 * ((c.z - c.x) / d + 2.0);
    } else {
        h = 60.0 * ((c.x - c.y) / d + 4.0);
    }
    if (h < 0.0) {
        h = h + 360.0;
    }
    var s = 0.0;
    if (mx > 1e-7) {
        s = min(d / mx, 1.0);
    }
    return vec2<f32>(h, s);
}

// Rust's f32::rem_euclid for a positive modulus.
fn fcl_remeu(a: f32, n: f32) -> f32 {
    let r = a - n * trunc(a / n);
    return select(r, r + n, r < 0.0);
}

fn fcl_hue_dist(a: f32, b: f32) -> f32 {
    let d = fcl_remeu(a - b, 360.0);
    return select(d, 360.0 - d, d > 180.0);
}

fn fcl_rotate_hue(c: vec3<f32>, deg: f32, l: vec3<f32>) -> vec3<f32> {
    if (deg == 0.0) {
        return c;
    }
    let a = radians(deg);
    let s = sin(a);
    let co = cos(a);
    let k = 1.0 / sqrt(3.0);
    let kv = k * (c.x + c.y + c.z);
    let cr = vec3<f32>(k * (c.z - c.y), k * (c.x - c.z), k * (c.y - c.x));
    let r = c * co + cr * s + vec3<f32>(k * kv * (1.0 - co));
    let d = dot(l, c) - dot(l, r);
    return r + vec3<f32>(d);
}

fn fcl_saturate(c: vec3<f32>, l: vec3<f32>, f: f32) -> vec3<f32> {
    let y = dot(l, c);
    return vec3<f32>(y) + (c - vec3<f32>(y)) * f;
}

fn fcl_hash(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn fcl_lattice(ix: i32, iy: i32, seed: u32) -> f32 {
    let h = fcl_hash(bitcast<u32>(ix) ^ fcl_hash(bitcast<u32>(iy) ^ fcl_hash(seed)));
    return f32(h >> 8u) * (2.0 / 16777216.0) - 1.0;
}

fn fcl_sharp(t: f32, w: f32) -> f32 {
    let u = clamp((t - 0.5) / w + 0.5, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

fn fcl_noise(x: f32, y: f32, seed: u32, soft: f32) -> f32 {
    let fx = floor(x);
    let fy = floor(y);
    let ix = i32(fx);
    let iy = i32(fy);
    let w = 0.08 + 0.92 * clamp(soft, 0.0, 1.0);
    let tx = fcl_sharp(x - fx, w);
    let ty = fcl_sharp(y - fy, w);
    let a = fcl_lattice(ix, iy, seed);
    let b = fcl_lattice(ix + 1, iy, seed);
    let c = fcl_lattice(ix, iy + 1, seed);
    let d = fcl_lattice(ix + 1, iy + 1, seed);
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

fn fcl_grain_octaves(x: f32, y: f32, seed: u32, soft: f32) -> f32 {
    return 0.8 * fcl_noise(x, y, seed, soft) + 0.45 * fcl_noise(x * 2.0 + 17.31, y * 2.0 - 9.73, seed ^ 0x68bc21ebu, soft);
}

fn fcl_rsig(x: f32, p: f32) -> f32 {
    return x / powz(1.0 + powz(abs(x), p), 1.0 / p);
}

fn fcl_band(x: f32, lo: f32, hi: f32, soft: f32) -> f32 {
    var a = 1.0;
    if (lo > 0.0) {
        a = fcl_ss(lo - soft, lo + 1e-4, x);
    }
    var b = 1.0;
    if (hi < 100.0) {
        b = 1.0 - fcl_ss(hi - 1e-4, hi + soft, x);
    }
    return a * b;
}

fn fcl_v3(i: u32) -> vec3<f32> {
    return vec3<f32>(data[i], data[i + 1u], data[i + 2u]);
}

fn fcl_mat(i: u32, c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        data[i] * c.x + data[i + 1u] * c.y + data[i + 2u] * c.z,
        data[i + 3u] * c.x + data[i + 4u] * c.y + data[i + 5u] * c.z,
        data[i + 6u] * c.x + data[i + 7u] * c.y + data[i + 8u] * c.z,
    );
}

// curve::Table::eval for the table at `at`: [n, slope_lo, slope_hi, v0 …].
fn fcl_table(at: u32, x: f32) -> f32 {
    let n = u32(data[at]);
    let first = data[at + 3u];
    let last = data[at + 3u + n - 1u];
    if (x != x) {
        return first;
    }
    if (x <= 0.0) {
        return first + x * data[at + 1u];
    }
    if (x >= 1.0) {
        return last + (x - 1.0) * data[at + 2u];
    }
    let f = x * f32(n - 1u);
    let i = min(u32(f), n - 2u);
    let t = f - f32(i);
    let a = data[at + 3u + i];
    let b = data[at + 4u + i];
    return a + (b - a) * t;
}

// effectcraft_color::Conversion at `i`: [decode code, encode code, matrix flag, m (9)].
fn fcl_conv(i: u32, c0: vec3<f32>) -> vec3<f32> {
    var c = fcl_dec3(u32(data[i]), c0);
    if (data[i + 2u] > 0.5) {
        c = fcl_mat(i + 3u, c);
    }
    return fcl_enc3(u32(data[i + 1u]), c);
}

fn fcl_norm(c: f32, lo: f32, hi: f32) -> f32 {
    let t = (c - lo) / max(hi - lo, 1e-9);
    if (t != t) {
        return 0.0;
    }
    return clamp(t, 0.0, 1.0);
}

// LUT at `at`: [n1, n3, min1 (3), max1 (3), min3 (3), max3 (3), d1 (n1 × 3), d3 (n3³ × 3)].
fn fcl_lat(at: u32, n: u32, r: u32, g: u32, b: u32) -> vec3<f32> {
    let n1 = u32(data[at]);
    let i = at + 14u + n1 * 3u + (r + g * n + b * n * n) * 3u;
    return fcl_v3(i);
}

fn fcl_cell(x: f32, n: u32) -> vec2<f32> {
    let v = x * f32(n - 1u);
    let i = min(u32(v), n - 2u);
    return vec2<f32>(f32(i), v - f32(i));
}

fn fcl_lut(at: u32, c0: vec3<f32>, trilinear: bool) -> vec3<f32> {
    var c = c0;
    let n1 = u32(data[at]);
    let n3 = u32(data[at + 1u]);
    if (n1 >= 2u) {
        let s = f32(n1 - 1u);
        var o = vec3<f32>(0.0);
        for (var k = 0u; k < 3u; k = k + 1u) {
            let t = fcl_norm(c[k], data[at + 2u + k], data[at + 5u + k]);
            let x = t * s;
            let i0 = min(u32(x), n1 - 2u);
            let f = x - f32(i0);
            let a = data[at + 14u + i0 * 3u + k];
            let b = data[at + 14u + (i0 + 1u) * 3u + k];
            o[k] = a + (b - a) * f;
        }
        c = o;
    }
    if (n3 >= 2u) {
        let t = vec3<f32>(
            fcl_norm(c.x, data[at + 8u], data[at + 11u]),
            fcl_norm(c.y, data[at + 9u], data[at + 12u]),
            fcl_norm(c.z, data[at + 10u], data[at + 13u]),
        );
        let cr = fcl_cell(t.x, n3);
        let cg = fcl_cell(t.y, n3);
        let cb = fcl_cell(t.z, n3);
        let r = u32(cr.x);
        let g = u32(cg.x);
        let b = u32(cb.x);
        let fr = cr.y;
        let fg = cg.y;
        let fb = cb.y;
        if (trilinear) {
            let c00 = mix(fcl_lat(at, n3, r, g, b), fcl_lat(at, n3, r + 1u, g, b), fr);
            let c10 = mix(fcl_lat(at, n3, r, g + 1u, b), fcl_lat(at, n3, r + 1u, g + 1u, b), fr);
            let c01 = mix(fcl_lat(at, n3, r, g, b + 1u), fcl_lat(at, n3, r + 1u, g, b + 1u), fr);
            let c11 = mix(fcl_lat(at, n3, r, g + 1u, b + 1u), fcl_lat(at, n3, r + 1u, g + 1u, b + 1u), fr);
            c = mix(mix(c00, c10, fg), mix(c01, c11, fg), fb);
        } else {
            let c000 = fcl_lat(at, n3, r, g, b);
            let c111 = fcl_lat(at, n3, r + 1u, g + 1u, b + 1u);
            var w1: f32;
            var w2: f32;
            var w3: f32;
            var p1: vec3<f32>;
            var p2: vec3<f32>;
            if (fr > fg) {
                if (fg > fb) {
                    w1 = fr; w2 = fg; w3 = fb;
                    p1 = fcl_lat(at, n3, r + 1u, g, b); p2 = fcl_lat(at, n3, r + 1u, g + 1u, b);
                } else if (fr > fb) {
                    w1 = fr; w2 = fb; w3 = fg;
                    p1 = fcl_lat(at, n3, r + 1u, g, b); p2 = fcl_lat(at, n3, r + 1u, g, b + 1u);
                } else {
                    w1 = fb; w2 = fr; w3 = fg;
                    p1 = fcl_lat(at, n3, r, g, b + 1u); p2 = fcl_lat(at, n3, r + 1u, g, b + 1u);
                }
            } else if (fb > fg) {
                w1 = fb; w2 = fg; w3 = fr;
                p1 = fcl_lat(at, n3, r, g, b + 1u); p2 = fcl_lat(at, n3, r, g + 1u, b + 1u);
            } else if (fb > fr) {
                w1 = fg; w2 = fb; w3 = fr;
                p1 = fcl_lat(at, n3, r, g + 1u, b); p2 = fcl_lat(at, n3, r, g + 1u, b + 1u);
            } else {
                w1 = fg; w2 = fr; w3 = fb;
                p1 = fcl_lat(at, n3, r, g + 1u, b); p2 = fcl_lat(at, n3, r + 1u, g + 1u, b);
            }
            c = c000 + w1 * (p1 - c000) + w2 * (p2 - p1) + w3 * (c111 - p2);
        }
    }
    return c;
}

// ops::HSL_CENTERS.
fn fcl_hsl_center(i: u32) -> f32 {
    switch (i) {
        case 0u: { return 0.0; }
        case 1u: { return 30.0; }
        case 2u: { return 60.0; }
        case 3u: { return 120.0; }
        case 4u: { return 180.0; }
        case 5u: { return 240.0; }
        case 6u: { return 270.0; }
        default: { return 300.0; }
    }
}

// ---------------------------------------------------------------- the pixel stage (ops.rs)

fn fcl_run(c0: vec3<f32>, p: vec2<f32>) -> vec3<f32> {
    var c = c0;
    var saved = c0;
    let code = P.u[0].z;
    let l = P.f[1].xyz;
    var pc = 0u;
    for (var guard = 0u; guard < 4096u; guard = guard + 1u) {
        let op = u32(data[pc]);
        let a = pc + 1u;
        switch (op) {
            case 0u: {
                return c;
            }
            case 1u: {
                c = fcl_dec3(code, c);
                pc = a;
            }
            case 2u: {
                c = fcl_enc3(code, c);
                pc = a;
            }
            case 3u: {
                saved = c;
                pc = a;
            }
            case 4u: {
                c = mix(saved, c, data[a]);
                pc = a + 1u;
            }
            case 5u: {
                // Exposure: gain, gamma, protect, knee.
                let gamma = data[a + 1u];
                let protect = data[a + 2u];
                let knee = data[a + 3u];
                c = c * data[a];
                if (gamma != 1.0) {
                    for (var k = 0u; k < 3u; k = k + 1u) {
                        let v = c[k];
                        c[k] = select(1.0 + gamma * (v - 1.0), fcl_spow(v, gamma), v <= 1.0);
                    }
                }
                if (protect > 0.0) {
                    let m = max(max(c.x, c.y), c.z);
                    if (m > knee) {
                        let x = (m - knee) / max(1.0 - knee, 1e-4);
                        let t = knee + (1.0 - knee) * (x / (1.0 + x));
                        c = mix(c, c * (t / m), protect);
                    }
                }
                pc = a + 4u;
            }
            case 6u: {
                // Contrast: k, pivot, filmic.
                let k = data[a];
                let pv = data[a + 1u];
                let fm = data[a + 2u];
                for (var i = 0u; i < 3u; i = i + 1u) {
                    let v = c[i];
                    let lin = pv + (v - pv) * k;
                    var fil = v;
                    if (v > 0.0 && v < 1.0) {
                        if (v < pv) {
                            fil = pv * powz(v / pv, k);
                        } else {
                            fil = 1.0 - (1.0 - pv) * powz((1.0 - v) / (1.0 - pv), k);
                        }
                    }
                    c[i] = lin + (fil - lin) * fm;
                }
                pc = a + 3u;
            }
            case 7u: {
                c = fcl_mat(a, c);
                pc = a + 9u;
            }
            case 8u: {
                let g = dot(fcl_v3(a), c);
                c = vec3<f32>(g);
                pc = a + 3u;
            }
            case 9u: {
                // Wheels: offset, lift, gamma exponents, gain, shadows, mids, highs.
                let off = fcl_v3(a);
                let lift = fcl_v3(a + 3u);
                let gexp = fcl_v3(a + 6u);
                let gain = fcl_v3(a + 9u);
                for (var i = 0u; i < 3u; i = i + 1u) {
                    var v = c[i] + off[i];
                    v = v + lift[i] * (1.0 - v);
                    c[i] = fcl_spow(v, gexp[i]) * gain[i];
                }
                let y = dot(l, c);
                let ws = 1.0 - fcl_ss(0.0, 0.55, y);
                let wh = fcl_ss(0.45, 1.0, y);
                let wm = max(1.0 - ws - wh, 0.0);
                c = c + fcl_v3(a + 12u) * ws + fcl_v3(a + 15u) * wm + fcl_v3(a + 18u) * wh;
                pc = a + 21u;
            }
            case 10u: {
                // Curves: table offsets (master, red, green, blue; < 0 = identity).
                let tm = data[a];
                if (tm >= 0.0) {
                    let t = u32(tm);
                    c = vec3<f32>(fcl_table(t, c.x), fcl_table(t, c.y), fcl_table(t, c.z));
                }
                for (var i = 0u; i < 3u; i = i + 1u) {
                    let ti = data[a + 1u + i];
                    if (ti >= 0.0) {
                        c[i] = fcl_table(u32(ti), c[i]);
                    }
                }
                pc = a + 4u;
            }
            case 11u: {
                // HSL: rot, sat, light, 8 × (hue, sat, lum).
                let hs = fcl_hue_sat(c);
                var i = 7u;
                for (var k = 0u; k < 8u; k = k + 1u) {
                    if (hs.x >= fcl_hsl_center(k)) {
                        i = k;
                    }
                }
                let c0r = fcl_hsl_center(i);
                var c1r = 360.0;
                if (i < 7u) {
                    c1r = fcl_hsl_center(i + 1u);
                }
                let t = clamp((hs.x - c0r) / (c1r - c0r), 0.0, 1.0);
                let ra = fcl_v3(a + 3u + i * 3u);
                let rb = fcl_v3(a + 3u + ((i + 1u) % 8u) * 3u);
                let ws = fcl_ss(0.0, 0.2, hs.y);
                let adj = (ra + (rb - ra) * t) * ws;
                c = fcl_rotate_hue(c, data[a] + adj.x, l);
                c = fcl_saturate(c, l, data[a + 1u] * (1.0 + adj.y));
                c = c + vec3<f32>(adj.z * 0.2);
                let light = data[a + 2u];
                if (light >= 0.0) {
                    c = c + (vec3<f32>(1.0) - c) * light * 0.6;
                } else {
                    c = c * (1.0 + light * 0.6);
                }
                pc = a + 27u;
            }
            case 12u: {
                // Hue remap: center, range, falloff, shift, sat, lum.
                let hs = fcl_hue_sat(c);
                let d = fcl_hue_dist(hs.x, data[a]);
                let rng = data[a + 1u];
                let wgt = (1.0 - fcl_ss(rng, rng + data[a + 2u] + 1e-3, d)) * fcl_ss(0.03, 0.2, hs.y);
                if (wgt > 0.0) {
                    var c2 = fcl_rotate_hue(c, data[a + 3u] * wgt, l);
                    c2 = fcl_saturate(c2, l, 1.0 + data[a + 4u] * wgt);
                    c = c2 + vec3<f32>(data[a + 5u] * 0.25 * wgt);
                }
                pc = a + 6u;
            }
            case 13u: {
                // Film: pre_gain, has matrix, m (9), offsets (3), k, grey, p_lo, p_hi, sat.
                var s = vec3<f32>(0.0);
                for (var i = 0u; i < 3u; i = i + 1u) {
                    s[i] = log2(max(c[i] * data[a], 1e-6) / 0.18);
                }
                if (data[a + 1u] > 0.5) {
                    s = fcl_mat(a + 2u, s);
                }
                let k = data[a + 14u];
                let grey = data[a + 15u];
                var d = vec3<f32>(0.0);
                for (var i = 0u; i < 3u; i = i + 1u) {
                    let x = s[i] + data[a + 11u + i];
                    if (x >= 0.0) {
                        d[i] = grey + (1.0 - grey) * fcl_rsig(k * x / (1.0 - grey), data[a + 17u]);
                    } else {
                        d[i] = grey + grey * fcl_rsig(k * x / grey, data[a + 16u]);
                    }
                }
                d = fcl_saturate(d, l, data[a + 18u]);
                c = fcl_dec3(code, d);
                pc = a + 19u;
            }
            case 14u: {
                // Bleach bypass: contrast, desaturation.
                let yy = clamp(dot(l, c), 0.0, 1.0);
                var ov = vec3<f32>(0.0);
                for (var i = 0u; i < 3u; i = i + 1u) {
                    let v = c[i];
                    ov[i] = select(1.0 - 2.0 * (1.0 - v) * (1.0 - yy), 2.0 * v * yy, v < 0.5);
                }
                let r = mix(c, ov, data[a]);
                c = fcl_saturate(r, l, 1.0 - data[a + 1u]);
                pc = a + 2u;
            }
            case 15u: {
                // Fade: contrast, blacks, sat, shadow (3), high (3), amount.
                c = vec3<f32>(0.5) + (c - vec3<f32>(0.5)) * (1.0 - 0.5 * data[a]);
                let bl = data[a + 1u];
                c = vec3<f32>(bl * 0.3) + c * (1.0 - bl * 0.3);
                c = fcl_saturate(c, l, 1.0 + data[a + 2u]);
                let y = clamp(dot(l, c), 0.0, 1.0);
                let ws = (1.0 - y) * (1.0 - y);
                let wh = y * y;
                c = c + data[a + 9u] * 0.5 * (fcl_v3(a + 3u) * ws + fcl_v3(a + 6u) * wh);
                pc = a + 10u;
            }
            case 16u: {
                // Grain: amount, cell, soft, luma_resp, colored, color_amt, seed hi, seed lo.
                let cell = data[a + 1u];
                let x = (0.8 * p.x - 0.6 * p.y) / cell + 0.371;
                let y = (0.6 * p.x + 0.8 * p.y) / cell + 0.137;
                let soft = data[a + 2u];
                let seed = (u32(data[a + 6u]) << 16u) | u32(data[a + 7u]);
                let mono = fcl_grain_octaves(x, y, seed, soft);
                let yc = clamp(dot(l, c), 0.0, 1.0);
                let lr = data[a + 3u];
                let wgt = (1.0 - lr) + lr * clamp(4.0 * yc * (1.0 - yc), 0.0, 1.0);
                let am = data[a] * 0.12 * wgt;
                if (data[a + 4u] > 0.5) {
                    let ca = data[a + 5u];
                    for (var i = 0u; i < 3u; i = i + 1u) {
                        let ch = fcl_grain_octaves(x, y, seed ^ (0x02e5be93u * (i + 1u)), soft);
                        c[i] = c[i] + am * (mono + (ch - mono) * ca);
                    }
                } else {
                    c = c + vec3<f32>(am * mono);
                }
                pc = a + 8u;
            }
            case 17u: {
                // Vignette: cx, cy, 1/rx, 1/ry, feather, stops, keep.
                let dx = (p.x - data[a]) * data[a + 2u];
                let dy = (p.y - data[a + 1u]) * data[a + 3u];
                let d = sqrt(dx * dx + dy * dy);
                let m = fcl_ss(1.0 - data[a + 4u], 1.0, d);
                let stops = data[a + 5u];
                var g = exp2(stops * m);
                let keep = data[a + 6u];
                if (stops < 0.0 && keep > 0.0) {
                    g = g + (1.0 - g) * keep * fcl_ss(0.5, 2.0, dot(l, c));
                }
                c = c * g;
                pc = a + 7u;
            }
            case 18u: {
                // Masked: kind, a/c (2), b/r (2), rot (2), feather, invert, opacity, gain, sat, tint (3).
                var m: f32;
                let fe = data[a + 7u];
                if (data[a] < 0.5) {
                    let pa = vec2<f32>(data[a + 1u], data[a + 2u]);
                    let dd = vec2<f32>(data[a + 3u], data[a + 4u]) - pa;
                    let l2 = max(dot(dd, dd), 1e-6);
                    let t = dot(p - pa, dd) / l2;
                    let f = clamp(fe, 0.0, 1.0);
                    m = 1.0 - fcl_ss(0.5 - 0.5 * f, 0.5 + 0.5 * f + 1e-4, t);
                } else {
                    let dx = p.x - data[a + 1u];
                    let dy = p.y - data[a + 2u];
                    let rc = data[a + 5u];
                    let rs = data[a + 6u];
                    let lx = dx * rc + dy * rs;
                    let ly = -dx * rs + dy * rc;
                    let ex = lx / max(data[a + 3u], 1e-3);
                    let ey = ly / max(data[a + 4u], 1e-3);
                    let d = sqrt(ex * ex + ey * ey);
                    let f = clamp(fe, 0.001, 1.0);
                    m = 1.0 - fcl_ss(1.0 - f, 1.0, d);
                }
                if (data[a + 8u] > 0.5) {
                    m = 1.0 - m;
                }
                m = m * data[a + 9u];
                if (m > 0.0) {
                    var c2 = c * data[a + 10u] * fcl_v3(a + 12u);
                    c2 = fcl_saturate(c2, l, 1.0 + data[a + 11u]);
                    c = mix(c, c2, m);
                }
                pc = a + 15u;
            }
            case 19u: {
                // Qualifier: hue, hue width, hue soft, sat lo/hi, lum lo/hi, soft, invert, show,
                // opacity, shift, saturation, gain, tint (3).
                let hs = fcl_hue_sat(c);
                let y = dot(l, c);
                let hw = data[a + 1u];
                var wh = 1.0;
                if (hw < 180.0) {
                    wh = 1.0 - fcl_ss(hw, hw + data[a + 2u] + 1e-3, fcl_hue_dist(hs.x, data[a]));
                }
                let soft = data[a + 7u];
                var m = wh * fcl_band(hs.y * 100.0, data[a + 3u], data[a + 4u], soft) * fcl_band(y * 100.0, data[a + 5u], data[a + 6u], soft);
                if (data[a + 8u] > 0.5) {
                    m = 1.0 - m;
                }
                m = m * data[a + 10u];
                if (data[a + 9u] > 0.5) {
                    c = vec3<f32>(m);
                } else if (m > 0.0) {
                    var c2 = fcl_rotate_hue(c, data[a + 11u], l);
                    c2 = fcl_saturate(c2, l, 1.0 + data[a + 12u]);
                    let gain = data[a + 13u];
                    if (gain != 1.0) {
                        c2 = fcl_enc3(code, fcl_dec3(code, c2) * gain);
                    }
                    c2 = c2 * fcl_v3(a + 14u);
                    c = mix(c, c2, m);
                }
                pc = a + 17u;
            }
            case 20u: {
                // LUT: table offset, trilinear, pre? (12), post? (12).
                if (data[a + 2u] > 0.5) {
                    c = fcl_conv(a + 3u, c);
                }
                c = fcl_lut(u32(data[a]), c, data[a + 1u] > 0.5);
                if (data[a + 15u] > 0.5) {
                    c = fcl_conv(a + 16u, c);
                }
                pc = a + 28u;
            }
            default: {
                return c;
            }
        }
    }
    return c;
}

fn fcl_finite(v: f32) -> f32 {
    // NaN and ±inf become 0 (ops::run).
    return select(0.0, v, v == v && abs(v) <= 3.4e38);
}

@compute @workgroup_size(16, 16)
fn fcl_pixel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    var px = textureLoad(src, q, 0);
    if (P.u[0].x == 1u) {
        let a = px.w;
        if (a > 1e-6) {
            px = vec4<f32>(px.xyz / a, a);
        } else {
            px = vec4<f32>(0.0, 0.0, 0.0, max(a, 0.0));
        }
    }
    let s = max(P.f[0].z, 1e-6);
    let lp = vec2<f32>((f32(q.x) + 0.5 - P.f[0].x) / s, (f32(q.y) + 0.5 - P.f[0].y) / s);
    let r = fcl_run(px.xyz, lp);
    var o = vec4<f32>(fcl_finite(r.x), fcl_finite(r.y), fcl_finite(r.z), px.w);
    if (P.u[0].y == 1u) {
        let orig = textureLoad(aux, q, 0);
        let pm = vec4<f32>(o.xyz * o.w, o.w);
        o = orig + (pm - orig) * P.f[0].w;
    }
    textureStore(out, q, o);
}

// ---------------------------------------------------------------- spatial stages (spatial.rs)

// blur::sample: bilinear, clamped to the edge, of a level texture of a w×h image at full
// resolution position (x, y) scaled by (sx, sy).
fn fcl_level(t: texture_2d<f32>, x: f32, y: f32) -> vec4<f32> {
    let d = vec2<i32>(textureDimensions(t));
    let fx = clamp(x - 0.5, 0.0, f32(d.x - 1));
    let fy = clamp(y - 0.5, 0.0, f32(d.y - 1));
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let x1 = min(x0 + 1, d.x - 1);
    let y1 = min(y0 + 1, d.y - 1);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let a = textureLoad(t, vec2<i32>(x0, y0), 0);
    let b = textureLoad(t, vec2<i32>(x1, y0), 0);
    let c = textureLoad(t, vec2<i32>(x0, y1), 0);
    let e = textureLoad(t, vec2<i32>(x1, y1), 0);
    let top = a + (b - a) * tx;
    let bot = c + (e - c) * tx;
    return top + (bot - top) * ty;
}

// The blurred value for output pixel q (blur::blur's final upsample).
fn fcl_blurred(q: vec2<i32>) -> vec4<f32> {
    let od = out_dims();
    let ad = vec2<f32>(textureDimensions(aux));
    let sx = ad.x / f32(od.x);
    let sy = ad.y / f32(od.y);
    return fcl_level(aux, (f32(q.x) + 0.5) * sx, (f32(q.y) + 0.5) * sy);
}

// Glow sources. P.u[0].x mode: 0 excess luma × alpha (r), 1 excess RGB × alpha, 2 premultiplied
// colour, 3 luma (≥ 0); P.f[0].x = threshold, P.f[1].xyz = luma weights.
@compute @workgroup_size(16, 16)
fn fcl_extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let c = textureLoad(src, q, 0);
    let thr = P.f[0].x;
    let l = P.f[1].xyz;
    let a = clamp(c.w, 0.0, 1.0);
    var o = vec4<f32>(0.0);
    switch (P.u[0].x) {
        case 0u: {
            let e = max(c.xyz - vec3<f32>(thr), vec3<f32>(0.0));
            o = vec4<f32>(dot(l, e) * a, 0.0, 0.0, 0.0);
        }
        case 1u: {
            o = vec4<f32>(max(c.xyz - vec3<f32>(thr), vec3<f32>(0.0)) * a, 0.0);
        }
        case 2u: {
            o = vec4<f32>(c.xyz * a, a);
        }
        default: {
            o = vec4<f32>(max(dot(l, c.xyz), 0.0), 0.0, 0.0, 0.0);
        }
    }
    textureStore(out, q, o);
}

// blur::down: 2×2 box average, clamped.
@compute @workgroup_size(16, 16)
fn fcl_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let sd = vec2<i32>(textureDimensions(src));
    let x0 = min(2 * q.x, sd.x - 1);
    let x1 = min(2 * q.x + 1, sd.x - 1);
    let y0 = min(2 * q.y, sd.y - 1);
    let y1 = min(2 * q.y + 1, sd.y - 1);
    let s = textureLoad(src, vec2<i32>(x0, y0), 0) + textureLoad(src, vec2<i32>(x1, y0), 0) + textureLoad(src, vec2<i32>(x0, y1), 0)
        + textureLoad(src, vec2<i32>(x1, y1), 0);
    textureStore(out, q, s * 0.25);
}

// blur::pass: separable Gaussian, weights in `data` (centre first), P.u[0].x = 1 horizontal.
@compute @workgroup_size(16, 16)
fn fcl_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let r = i32(P.u[0].y);
    var acc = vec4<f32>(0.0);
    for (var i = -r; i <= r; i = i + 1) {
        let w = data[u32(abs(i))];
        var s = q;
        if (P.u[0].x == 1u) {
            s.x = clamp(q.x + i, 0, dims.x - 1);
        } else {
            s.y = clamp(q.y + i, 0, dims.y - 1);
        }
        acc = acc + textureLoad(src, s, 0) * w;
    }
    textureStore(out, q, acc);
}

// Add blurred light: out = src + P.f[0].xyz × (P.u[0].x == 1 ? blurred rgb : blurred r).
@compute @workgroup_size(16, 16)
fn fcl_addglow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let c = textureLoad(src, q, 0);
    let g = fcl_blurred(q);
    var add = P.f[0].xyz * g.x;
    if (P.u[0].x == 1u) {
        add = P.f[0].xyz * g.xyz;
    }
    textureStore(out, q, vec4<f32>(c.xyz + add, c.w));
}

// Diffusion's mist: mix towards the blurred premultiplied colour (un-premultiplied) by P.f[0].x.
@compute @workgroup_size(16, 16)
fn fcl_mist(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let c = textureLoad(src, q, 0);
    let s = fcl_blurred(q);
    let sc = s.xyz / max(s.w, 1e-4);
    textureStore(out, q, vec4<f32>(mix(c.xyz, sc, P.f[0].x), c.w));
}

// spatial::streak_rows, one thread per row (decay P.f[0].x): fcl_streak_fwd writes the forward
// sums of `src` (r); fcl_streak_bwd runs the backward sums and combines them with the forward
// ones in `aux`: (forward + backward − centre) × (1 − d) / (1 + d).
@compute @workgroup_size(1, 16)
fn fcl_streak_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let y = i32(gid.y);
    if (y >= dims.y || gid.x != 0u) {
        return;
    }
    let d = clamp(P.f[0].x, 0.0, 0.9999);
    var acc = 0.0;
    for (var x = 0; x < dims.x; x = x + 1) {
        acc = textureLoad(src, vec2<i32>(x, y), 0).x + d * acc;
        textureStore(out, vec2<i32>(x, y), vec4<f32>(acc, 0.0, 0.0, 0.0));
    }
}

@compute @workgroup_size(1, 16)
fn fcl_streak_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let y = i32(gid.y);
    if (y >= dims.y || gid.x != 0u) {
        return;
    }
    let d = clamp(P.f[0].x, 0.0, 0.9999);
    let norm = (1.0 - d) / (1.0 + d);
    var acc = 0.0;
    for (var x = dims.x - 1; x >= 0; x = x - 1) {
        let s = textureLoad(src, vec2<i32>(x, y), 0).x;
        acc = s + d * acc;
        let f = textureLoad(aux, vec2<i32>(x, y), 0).x;
        textureStore(out, vec2<i32>(x, y), vec4<f32>((f + acc - s) * norm, 0.0, 0.0, 0.0));
    }
}

// Straight-colour bilinear sample clamped to the edge (blur::sample at full resolution).
fn fcl_sample(t: texture_2d<f32>, x: f32, y: f32) -> vec4<f32> {
    return fcl_level(t, x, y);
}

// Chromatic aberration: P.f[0] = (rc, by, radial, inv_ref), P.f[1].xy = centre, P.f[1].z = mix.
@compute @workgroup_size(16, 16)
fn fcl_chroma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let rc = P.f[0].x;
    let by = P.f[0].y;
    let radial = P.f[0].z;
    let inv_ref = P.f[0].w;
    let qp = vec2<f32>(f32(q.x) + 0.5, f32(q.y) + 0.5);
    let v = (qp - P.f[1].xy) * inv_ref;
    let dr = vec2<f32>(rc * (1.0 - radial) + v.x * rc * radial, v.y * rc * radial);
    let db = vec2<f32>(by * (1.0 - radial) + v.x * by * radial, v.y * by * radial);
    let c = textureLoad(src, q, 0);
    let r = fcl_sample(src, qp.x - dr.x, qp.y - dr.y);
    let b = fcl_sample(src, qp.x - db.x, qp.y - db.y);
    let o = vec4<f32>(r.x, c.y, b.z, c.w);
    textureStore(out, q, c + (o - c) * P.f[1].z);
}

// spatial::sample_premul.
fn fcl_premul_at(x: i32, y: i32, extend: bool) -> vec4<f32> {
    let d = vec2<i32>(textureDimensions(src));
    var xx = x;
    var yy = y;
    if (extend) {
        xx = clamp(x, 0, d.x - 1);
        yy = clamp(y, 0, d.y - 1);
    } else if (x < 0 || y < 0 || x >= d.x || y >= d.y) {
        return vec4<f32>(0.0);
    }
    let c = textureLoad(src, vec2<i32>(xx, yy), 0);
    let a = clamp(c.w, 0.0, 1.0);
    return vec4<f32>(c.xyz * a, a);
}

fn fcl_sample_premul(x: f32, y: f32, extend: bool) -> vec4<f32> {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = floor(fx);
    let y0 = floor(fy);
    let tx = fx - x0;
    let ty = fy - y0;
    let xi = i32(x0);
    let yi = i32(y0);
    let a = fcl_premul_at(xi, yi, extend);
    let b = fcl_premul_at(xi + 1, yi, extend);
    let c = fcl_premul_at(xi, yi + 1, extend);
    let e = fcl_premul_at(xi + 1, yi + 1, extend);
    let top = a + (b - a) * tx;
    let bot = c + (e - c) * tx;
    let o = top + (bot - top) * ty;
    if (o.w > 1e-6) {
        return vec4<f32>(o.xyz / o.w, o.w);
    }
    return vec4<f32>(0.0);
}

// Lens distortion: P.f[0] = (k, 1/zoom, centre x, centre y), P.f[1] = (ref_r, extend, mix, 0).
@compute @workgroup_size(16, 16)
fn fcl_distort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let qp = vec2<f32>(f32(q.x) + 0.5, f32(q.y) + 0.5);
    let ctr = P.f[0].zw;
    let rr = P.f[1].x;
    let v = (qp - ctr) / rr;
    let f = (1.0 + P.f[0].x * dot(v, v)) * P.f[0].y;
    let s = ctr + v * rr * f;
    let c = textureLoad(src, q, 0);
    let o = fcl_sample_premul(s.x, s.y, P.f[1].y > 0.5);
    textureStore(out, q, c + (o - c) * P.f[1].z);
}

// Edge softness: aux = blurred premultiplied colour; P.f[0] = layer rect (pixels),
// P.f[1] = (size, feather, mix, 0).
@compute @workgroup_size(16, 16)
fn fcl_edgesoft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let rect = P.f[0];
    let qp = vec2<f32>(f32(q.x) + 0.5, f32(q.y) + 0.5);
    let hw = max((rect.z - rect.x) * 0.5, 1.0);
    let hh = max((rect.w - rect.y) * 0.5, 1.0);
    let ex = max(min(qp.x - rect.x, rect.z - qp.x) / hw, 0.0);
    let ey = max(min(qp.y - rect.y, rect.w - qp.y) / hh, 0.0);
    let e = min(ex, ey);
    let size = P.f[1].x;
    let m = 1.0 - fcl_ss(size * (1.0 - P.f[1].y), size + 1e-4, e);
    let c = textureLoad(src, q, 0);
    let s = fcl_blurred(q);
    let sc = s.xyz / max(s.w, 1e-4);
    let r = mix(c.xyz, sc, m);
    let o = vec4<f32>(r, c.w);
    textureStore(out, q, c + (o - c) * P.f[1].z);
}

// Shadow & highlight: aux = blurred luma; P.f[0] = (shadows, highlights, width, mix).
@compute @workgroup_size(16, 16)
fn fcl_local(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let q = vec2<i32>(gid.xy);
    if (q.x >= dims.x || q.y >= dims.y) {
        return;
    }
    let b = fcl_blurred(q).x;
    let lv = log2(max(b, 1e-5) / 0.18);
    let w = P.f[0].z;
    let ws = 1.0 - fcl_ss(-2.0 - 4.0 * w, 0.0, lv);
    let wh = fcl_ss(0.3, 1.0 + 2.5 * w, lv);
    let g = exp2(P.f[0].x * 2.0 * ws - P.f[0].y * 2.0 * wh);
    let c = textureLoad(src, q, 0);
    let o = vec4<f32>(c.xyz * g, c.w);
    textureStore(out, q, c + (o - c) * P.f[0].w);
}
