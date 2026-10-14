//! GPU effect, FrameWeaver (kernels in `shaders/fx_frameweaver.wgsl`, every entry point
//! prefixed `ffw_`).
//!
//! Like Timewarp, the data-dependent work stays on the CPU: `effectcraft_effects::frameweaver_plan`
//! fetches the frames, decides cuts, estimates (or reads) the optical flow, applies the guides
//! and builds the time fields. The GPU only *samples*: it uploads each frame and each field once
//! and runs the synthesis (`ffw_synth`), the motion blur (`ffw_mblur`) and the foreground over
//! background (`ffw_over`) with the CPU effect's arithmetic in its order, so the two agree to
//! float rounding. Diagnostic displays are finished images built on the CPU and uploaded.
//!
//! Any failure (an upload that does not fit, a kernel that failed to build) returns `None` and
//! the compositor's existing CPU fallback renders the effect.

use effectcraft_effects::{FwPlan, FwStep, LayerShow, TimeField};

use crate::context::{Enc, GpuImage, Params};
use crate::effects::GBuf;

/// Compute entry points in `fx_frameweaver.wgsl`.
pub(crate) const KERNELS: &[&str] = &["ffw_mix", "ffw_synth", "ffw_mblur", "ffw_over"];

/// Effect ids implemented here.
pub(crate) const IDS: &[&str] = &["ec.time.frameweaver"];

/// The largest time field (floats) uploaded as one storage buffer.
const MAX_FIELD_FLOATS: usize = 24 << 20;

/// Run FrameWeaver; `None` = render on the CPU.
pub(crate) fn apply(e: &mut Enc, _id: &str, ctx: &effectcraft_effects::EffectCtx, b: GBuf) -> Option<GBuf> {
    if ctx.env.host.is_none() {
        return Some(b);
    }
    let Some(plan) = effectcraft_effects::frameweaver_plan(ctx) else { return Some(b) };
    if plan.is_empty() {
        let (w, h) = (b.img.width, b.img.height);
        return Some(GBuf { img: e.image(w, h), ..b });
    }
    render_plan(e, &plan)
}

fn kernel(e: &mut Enc, entry: &str, p: &Params, src: &GpuImage, aux: Option<&GpuImage>, data: Option<&wgpu::Buffer>) -> GpuImage {
    let out = e.scratch(src.width, src.height);
    e.pixels(entry, p, src, aux, &out, data);
    out
}

/// A time field as the kernels read it: seven floats per pixel (offset to A, offset to B,
/// visibility of A, of B, confidence).
fn pack(f: &TimeField) -> Option<Vec<f32>> {
    if f.is_empty() || f.width * f.height * 7 > MAX_FIELD_FLOATS {
        return None;
    }
    let mut v = Vec::with_capacity(f.width * f.height * 7);
    for i in 0..f.width * f.height {
        let (a, b) = (f.to_a.get(i)?, f.to_b.get(i)?);
        v.extend_from_slice(&[a[0], a[1], b[0], b[1], *f.vis_a.get(i)?, *f.vis_b.get(i)?, *f.conf.get(i)?]);
    }
    Some(v)
}

/// Render a prepared FrameWeaver plan.
pub(crate) fn render_plan(e: &mut Enc, plan: &FwPlan) -> Option<GBuf> {
    let mut up: Vec<Option<GpuImage>> = vec![None; plan.frames.len()];
    let mut frame = |e: &mut Enc, i: usize| -> Option<GpuImage> {
        if up.get(i)?.is_none() {
            let img = plan.frames.get(i)?;
            *up.get_mut(i)? = Some(e.g.upload_image(img)?);
        }
        up.get(i)?.clone()
    };
    let mut buffers: Vec<Option<wgpu::Buffer>> = (0..plan.fields.len()).map(|_| None).collect();
    // synth::synthesize (+ motion_blur when `blur`) of frames a, b along field `field`.
    let mut synth = |e: &mut Enc, a: usize, b: usize, w: f32, field: usize, blur: bool| -> Option<GpuImage> {
        let tf = plan.fields.get(field)?;
        if buffers.get(field)?.is_none() {
            *buffers.get_mut(field)? = Some(e.data(&pack(tf)?));
        }
        let data = buffers.get(field)?.as_ref()?;
        let (ia, ib) = (frame(e, a)?, frame(e, b)?);
        let mut p = Params::default();
        p.u[0] = [tf.width as u32, tf.height as u32, plan.smart_blend as u32, 0];
        p.f[0] = [tf.scale[0], tf.scale[1], 0.0, 0.0];
        p.f[1][0] = w;
        let img = kernel(e, "ffw_synth", &p, &ia, Some(&ib), Some(data));
        match plan.mblur.filter(|_| blur) {
            Some(m) if m.samples > 1 && m.span.abs() >= 1e-6 => {
                let mut q = Params::default();
                q.u[0] = [tf.width as u32, tf.height as u32, m.samples.clamp(1, 64), 0];
                q.f[0] = [tf.scale[0], tf.scale[1], 0.0, 0.0];
                q.f[1][0] = m.span;
                Some(kernel(e, "ffw_mblur", &q, &img, None, Some(data)))
            }
            _ => Some(img),
        }
    };
    let img = match plan.step {
        FwStep::Empty => return None,
        FwStep::Hold(i) | FwStep::Image(i) => frame(e, i)?,
        FwStep::Mix { a, b, w } => {
            let (ia, ib) = (frame(e, a)?, frame(e, b)?);
            let mut p = Params::default();
            p.f[1][0] = w;
            kernel(e, "ffw_mix", &p, &ia, Some(&ib), None)
        }
        FwStep::Motion { a, b, w, field } => synth(e, a, b, w, field, true)?,
        FwStep::Layered { fg, bg, mt, w, show } => match show {
            LayerShow::FgOnly => synth(e, fg[0], fg[1], w, fg[2], true)?,
            LayerShow::BgOnly => synth(e, bg[0], bg[1], w, bg[2], true)?,
            LayerShow::Matte => synth(e, mt[0], mt[1], w, fg[2], false)?,
            LayerShow::Output => {
                let f = synth(e, fg[0], fg[1], w, fg[2], true)?;
                let g = synth(e, bg[0], bg[1], w, bg[2], true)?;
                kernel(e, "ffw_over", &Params::default(), &f, Some(&g), None)
            }
        },
    };
    Some(GBuf { img, offset: plan.grid.offset, scale: plan.grid.scale })
}
