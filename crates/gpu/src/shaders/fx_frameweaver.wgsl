// GPU effects, FrameWeaver: see src/fx_frameweaver.rs. The flow, the cuts, the guides and the
// time fields are computed on the CPU (effectcraft_effects::frameweaver_plan); these kernels
// only sample, with the arithmetic and the order of effectcraft_frameweaver::synth.
//
// The time field arrives in `data`, seven floats per field pixel: offset to A (x, y) in frame
// pixels, offset to B (x, y), visibility of A, visibility of B, confidence.
// P.u[0] = (field width, field height, smart blend / sample count, 0);
// P.f[0] = (frame pixels per field pixel x, y, fraction w or shutter span, 0).

fn ffw_inside(p: vec2<i32>) -> bool {
    let d = out_dims();
    return p.x < d.x && p.y < d.y;
}

struct FfwSample {
    ta: vec2<f32>,
    tb: vec2<f32>,
    va: f32,
    vb: f32,
    conf: f32,
}

fn ffw_load(x: i32, y: i32) -> FfwSample {
    let fw = i32(P.u[0].x);
    let fh = i32(P.u[0].y);
    let i = u32(clamp(y, 0, fh - 1) * fw + clamp(x, 0, fw - 1)) * 7u;
    var s: FfwSample;
    s.ta = vec2<f32>(data[i], data[i + 1u]);
    s.tb = vec2<f32>(data[i + 2u], data[i + 3u]);
    s.va = data[i + 4u];
    s.vb = data[i + 5u];
    s.conf = data[i + 6u];
    return s;
}

// synth::TimeField::sample: bilinear over the field at frame position (px, py).
fn ffw_field(px: f32, py: f32) -> FfwSample {
    let fw = f32(P.u[0].x);
    let fh = f32(P.u[0].y);
    let fx = clamp(px / P.f[0].x - 0.5, -1.0, fw);
    let fy = clamp(py / P.f[0].y - 0.5, -1.0, fh);
    let x0 = floor(fx);
    let y0 = floor(fy);
    let tx = fx - x0;
    let ty = fy - y0;
    let xi = i32(x0);
    let yi = i32(y0);
    let v00 = ffw_load(xi, yi);
    let v10 = ffw_load(xi + 1, yi);
    let v01 = ffw_load(xi, yi + 1);
    let v11 = ffw_load(xi + 1, yi + 1);
    let w00 = (1.0 - tx) * (1.0 - ty);
    let w10 = tx * (1.0 - ty);
    let w01 = (1.0 - tx) * ty;
    let w11 = tx * ty;
    var s: FfwSample;
    s.ta = v00.ta * w00 + v10.ta * w10 + v01.ta * w01 + v11.ta * w11;
    s.tb = v00.tb * w00 + v10.tb * w10 + v01.tb * w01 + v11.tb * w11;
    s.va = v00.va * w00 + v10.va * w10 + v01.va * w01 + v11.va * w11;
    s.vb = v00.vb * w00 + v10.vb * w10 + v01.vb * w01 + v11.vb * w11;
    s.conf = v00.conf * w00 + v10.conf * w10 + v01.conf * w01 + v11.conf * w11;
    return s;
}

// Cross-fade (raster::flow::mix): src + (aux − src) × w.
@compute @workgroup_size(16, 16)
fn ffw_mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!ffw_inside(p)) {
        return;
    }
    let a = textureLoad(src, p, 0);
    textureStore(out, p, a + (textureLoad(aux, p, 0) - a) * P.f[1].x);
}

// synth::synthesize: src = frame A, aux = frame B. u[0].z = smart blend; f[1].x = w.
@compute @workgroup_size(16, 16)
fn ffw_synth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!ffw_inside(p)) {
        return;
    }
    let w = P.f[1].x;
    let d = vec2<f32>(out_dims());
    let c = vec2<f32>(f32(p.x) + 0.5, f32(p.y) + 0.5);
    let s = ffw_field(c.x, c.y);
    let pa = c + s.ta;
    let pb = c + s.tb;
    var va = s.va;
    var vb = s.vb;
    if (P.u[0].z != 0u) {
        if (pa.x < 0.0 || pa.y < 0.0 || pa.x > d.x || pa.y > d.y) {
            va = 0.0;
        }
        if (pb.x < 0.0 || pb.y < 0.0 || pb.x > d.x || pb.y > d.y) {
            vb = 0.0;
        }
    }
    let wa = (1.0 - w) * va;
    let wb = w * vb;
    var near = textureLoad(aux, p, 0);
    if (wa >= wb) {
        near = textureLoad(src, p, 0);
    }
    let den = wa + wb;
    var warped = near;
    var cf = 0.0;
    if (den > 1e-4) {
        let sa = sample_bilinear_clamped(src, pa.x, pa.y);
        let sb = sample_bilinear_clamped(aux, pb.x, pb.y);
        warped = (wa * sa + wb * sb) / den;
        cf = clamp(s.conf, 0.0, 1.0);
    }
    textureStore(out, p, near + (warped - near) * cf);
}

// synth::motion_blur: src = the synthesised frame; u[0].z = samples (≥ 2); f[1].x = span.
@compute @workgroup_size(16, 16)
fn ffw_mblur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!ffw_inside(p)) {
        return;
    }
    let k = P.u[0].z;
    let c = vec2<f32>(f32(p.x) + 0.5, f32(p.y) + 0.5);
    let s = ffw_field(c.x, c.y);
    let m = (s.tb - s.ta) * P.f[1].x;
    var acc = vec4<f32>(0.0);
    for (var i = 0u; i < k; i = i + 1u) {
        let t = f32(i) / f32(k - 1u) - 0.5;
        acc = acc + sample_bilinear_clamped(src, c.x + t * m.x, c.y + t * m.y);
    }
    textureStore(out, p, acc / f32(k));
}

// synth::over: src = foreground, aux = background (premultiplied).
@compute @workgroup_size(16, 16)
fn ffw_over(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!ffw_inside(p)) {
        return;
    }
    let f = textureLoad(src, p, 0);
    let b = textureLoad(aux, p, 0);
    textureStore(out, p, f + b * (1.0 - f.w));
}
