//! Reference sequences generated with ffmpeg (an external test oracle only: it is never linked
//! or shipped), with known 60 fps ground truth. Every other frame is dropped, FrameWeaver
//! rebuilds it, and the result is scored against the real frame.

#![allow(dead_code)]

use std::io::Write;
use std::process::{Command, Stdio};

use effectcraft_raster::Image;

pub const W: u32 = 320;
pub const H: u32 = 180;

pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// RGB24 bytes → opaque images.
fn frames_from_rgb24(bytes: &[u8], w: u32, h: u32) -> Vec<Image> {
    let n = (w * h * 3) as usize;
    bytes
        .chunks_exact(n)
        .map(|f| {
            let mut img = Image::new(w, h);
            for (p, px) in img.data.iter_mut().zip(f.chunks_exact(3)) {
                *p = [px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0, 1.0];
            }
            img
        })
        .collect()
}

/// `graph` is a lavfi filtergraph ending in the output pad `[out]`; `n` frames of `w × h`.
pub fn generate(graph: &str, n: usize, w: u32, h: u32) -> Option<Vec<Image>> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-filter_complex", graph, "-map", "[out]", "-frames:v", &n.to_string(), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let frames = frames_from_rgb24(&out.stdout, w, h);
    (frames.len() == n).then_some(frames)
}

/// ffmpeg's motion-compensated interpolation (minterpolate, `mci` / `aobmc`, bidirectional
/// estimation) doubling the frame rate of `frames`: the `2i + 1`-th output is between inputs
/// `i` and `i + 1`.
pub fn ffmpeg_minterpolate(frames: &[Image]) -> Option<Vec<Image>> {
    let (w, h) = (frames.first()?.width, frames.first()?.height);
    let mut child = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{w}x{h}"),
            "-r",
            "30",
            "-i",
            "-",
            "-vf",
            "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let bytes: Vec<u8> = frames.iter().flat_map(|f| f.data.iter().flat_map(|p| (0..3).map(|c| (p[c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8))).collect();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&bytes);
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    out.status.success().then(|| frames_from_rgb24(&out.stdout, w, h))
}

/// (name, filtergraph, frames at 60 fps).
pub fn sequences() -> Vec<(&'static str, String, usize)> {
    let cross = |speed: u32| {
        format!(
            "gradients=s=320x180:r=60:speed=0.005[bg];testsrc2=s=80x80:r=60[a];mandelbrot=s=80x80:r=60:start_scale=0.5:end_scale=0.45[b];\
             [bg][a]overlay=x='40+t*{speed}':y=20[t1];[t1][b]overlay=x='230-t*{speed}':y=85[out]"
        )
    };
    vec![
        ("zoom", "mandelbrot=s=320x180:r=60:start_scale=3:end_scale=1.5:end_pts=48[out]".into(), 24),
        ("pattern", "testsrc2=s=320x180:r=60[out]".into(), 24),
        ("cross", cross(300), 24),
        ("fast", cross(700), 24),
        ("pan_noise", "testsrc2=s=640x360:r=60,crop=320:180:x='n*7':y=90,noise=alls=6:allf=t[out]".into(), 24),
    ]
}

pub struct Score {
    pub psnr: f64,
    pub ssim: f64,
}

/// Mean PSNR / SSIM of `made[i]` against `truth[i]` (border of 8 px ignored).
pub fn score(made: &[Image], truth: &[Image]) -> Score {
    let n = made.len().min(truth.len()).max(1) as f64;
    let (mut p, mut s) = (0.0, 0.0);
    for (m, t) in made.iter().zip(truth) {
        p += effectcraft_frameweaver::metrics::psnr(m, t, 8).min(60.0);
        s += effectcraft_frameweaver::metrics::ssim(m, t, 8);
    }
    Score { psnr: p / n, ssim: s / n }
}
