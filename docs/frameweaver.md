# FrameWeaver

Optical-flow retiming for EffectCraft: **Effect ▸ Time ▸ FrameWeaver** (`ec.time.frameweaver`).
It slows footage down, speeds it up, ramps it, freezes it and converts frame rates by building
the in-between frames along dense motion vectors instead of cross-fading neighbouring frames.
It is our own implementation (Dense Inverse Search flow, published algorithms only) and lives in
the `effectcraft-frameweaver` crate (L1) with an adapter in `effectcraft-effects` and GPU
synthesis in `effectcraft-gpu`. Design: [superpowers/specs/2026-10-14-frameweaver-design.md](superpowers/specs/2026-10-14-frameweaver-design.md).

## Quick start

* **Slow motion:** apply FrameWeaver, set *Speed %* to 50 (the default). Footage is read at the
  composition's frame rate unless *Input Frame Rate* says otherwise.
* **Speed ramp:** keyframe *Speed %*. The source time is the integral of the speed over layer
  time, so a ramp from 100 % to 25 % eases smoothly and never jumps.
* **Freeze / reverse / arbitrary curve:** set *Time Remap Mode* to *Frame Number* and keyframe
  *Frame Number* (source frame, fractional allowed). Or use a negative *Speed %* with *Start Frame*
  at the last frame.
* **Frame-rate conversion:** set *Input Frame Rate* to the footage's rate (24) in a 30 fps
  composition at 100 % speed.

Outside the footage the output is transparent; with only one of the two neighbouring frames
available (the last frame) that frame is held.

## Controls

| Group | Control | Purpose |
|---|---|---|
| Time Remapping | Input Frame Rate (0 = comp rate) | Source frame *k* is at layer time *k* / rate |
| | Time Remap Mode | Speed (integrated) or Frame Number (absolute) |
| | Speed % | −10000…10000; negative plays backwards |
| | Start Frame | Source frame shown at layer time 0 in Speed mode |
| | Frame Number | Source frame in Frame Number mode |
| Motion Estimation | Motion Vectors | Off (frame blend), Estimate, External Vector Layer |
| | Quality | Draft, Balanced, High, Best (below) |
| | Motion Sensitivity | High: finer, more independent vectors. Low: smoother, more rigid |
| | Image Prep | Contrast/Edge Enhance on the analysis image (low-contrast or exposure-flickering footage) |
| | Motion Source Layer | Estimate the motion on another layer (denoised, contrast-boosted) and apply it to this one |
| Interpolation | Frame Interpolation | Nearest Frame, Frame Blend, Motion Weighted Blend |
| | Warping | Inverse, or Forward (Depth-Tested, default) |
| | Smart Blend | A sample outside its frame gives its weight to the other frame instead of smearing the edge |
| Motion Blur | Enable, Shutter Angle, Samples | Synthetic blur along the motion at the output time |
| Segments | Auto-Detect Cuts, Cut Sensitivity, Segment Marker | No interpolation across a cut: the nearest frame is held |
| Advanced Guidance | FG Matte Layer, Matte Channel, BG Fill | Foreground and background move on separate flows |
| | Vector Layer, Encoding, Vector Scale, Direction, Flip Y | Imported motion vectors |
| | Guide Radius; Guide 1…8 Enable / Point | User-placed guide points |
| Diagnostics | Display | Output, Source, Motion Vectors, Occlusion, Confidence, FG Matte, FG Only, BG Only, Guides |

All controls are ordinary properties: keyframes, expressions, scripting (`addProperty("FrameWeaver")`),
`prop.set`, the control channel and MCP address them like any built-in effect.

## What the quality presets do

| | analysis size | patch | overlap | LK iterations | variational (outer × inner) | median | symmetrise | global fallback |
|---|---|---|---|---|---|---|---|---|
| Draft | 640 | 12 | 0.3 | 4 | – | – | – | – |
| Balanced | 960 | 10 | 0.4 | 6 | 2 × 6 | 1 | – | – |
| High | 1440 | 8 | 0.5 | 10 | 3 × 8 | 1 | – | yes |
| Best | 1920 | 8 | 0.6 | 12 | 5 × 10 | 1 | yes | yes |

Motion Sensitivity scales the patch size by 1.5 − s and the smoothness by 2 − 1.8 s. The ordering
is checked by the tests (flow error falls and PSNR rises from Draft to Best).

## How a frame is built

1. **Source time** (integrated speed, or Frame Number) gives a pair of source frames A, B and a
   fraction *w*. On a whole frame (*w* ≈ 0) the frame is returned untouched; no flow is estimated.
2. **Cut check:** colour-histogram + motion-compensated dissimilarity, or a changed Segment Marker.
3. **Flow** A→B and B→A: coarse-to-fine Dense Inverse Search (patch Lucas–Kanade on a pyramid,
   candidate initialisation from neighbours, bounded integer search for fast small objects,
   neighbour propagation, error-weighted densification) and, from Balanced up, a variational
   refinement with brightness and gradient constancy and a robust smoothness term.
4. **Consistency:** forward-backward check separates *occlusion* (target leaves the frame, or the
   flow jumps at a motion boundary: the pixel is visible in one frame only) from *unreliable
   motion* (inconsistent without a boundary: low confidence). Occluded vectors are replaced by the
   motion of the dominant surface (what stays behind a covering or uncovering object); unreliable
   ones by their neighbours'. Confidence is then measured again on the repaired pair.
5. **Guides** pull the flow near a point to the point's own motion (Gaussian falloff, radius = 2 σ).
6. **Time field** at *w*: *Forward* splats every pixel of A and B to its position at *w* with a depth
   test (an occluded pixel loses to a visible one), fills holes by push-pull; *Inverse* solves
   `x = u + w F(u)` per output pixel, and where the A and B points disagree (surfaces crossing)
   takes a single winner and samples both frames along its correspondence.
7. **Synthesis:** `O = [(1−w) v_a A(x+F_a) + w v_b B(x+F_b)] / [(1−w) v_a + w v_b]`, mixed with the
   nearest frame that can show the pixel by `1 − confidence`.
8. **Motion blur** (optional) and, with a matte, foreground over background.

## Workflows

* **Separation:** point *FG Matte Layer* at a matte of the moving subject. The foreground, and a
  background plate with the subject removed (*BG Fill*: neighbouring source frames where they show
  background, else membrane interpolation; *Inpaint*: membrane only), are estimated and built
  separately, then composited. Use when a subject crosses a busy background. Neighbour Frames
  assumes a locked-off camera; with a moving camera use *Inpaint*.
* **Motion Source:** a cleaned copy of the footage drives the estimate.
* **External Vectors:** a motion-vector pass (Signed RG with 0.5 = still, or Float RG), read at
  source frame A (*To Next Frame*) or B (*To Previous Frame*). A missing or empty layer falls back to
  estimating.
* **Guides:** enable a guide, place its *Point*, and keyframe (or track) it on the source footage:
  its positions at the two source frames define the motion it enforces.
* **Segments:** automatic cut detection stops interpolation across a cut; animate *Segment Marker*
  (any change of its value starts a new segment) for cuts that are not visible changes.

## Troubleshooting

* **Warping artefacts around a subject** → Separation, or a Guide on the subject.
* **Flicker between frames** → lower Motion Sensitivity, or Image Prep.
* **Smeared edges at the frame border** → leave Smart Blend on.
* **A small fast object is missed or tears** → raise Quality (Draft and Balanced can lose objects
  smaller than a few patches that move farther than a patch per frame); add a Guide.
* **Slow-motion looks blurry** → FrameWeaver cannot remove blur present in the source frames; it
  adds none when slowing down (the blur span scales with the source frames advanced per output frame).

## Caching

Flows are the expensive part and are cached by *content* (both analysis images, every estimator
setting, imported-vector data): at most 6 flow pairs and 256 MiB, least recently used evicted, so
scrubbing between the same two source frames, or rendering several output frames between them,
estimates once, and a stale entry cannot be served. Frames are cached by the renderer's frame
cache and the finished layer by its layer cache (which already keys time effects on layer time).
There is no cancellation hook in the effect API: a render that is abandoned finishes the estimate it
is in the middle of, and its result stays cached for the next request.

## Neural optical flow: status

Only the classical estimator exists, and every code path is classical. The seam for another
estimator is `FlowBackend` (it returns a `FlowPair`: flows both ways, occlusion, confidence); nothing
implements it, no control mentions it and no model is shipped. What a neural backend needs, from a
look at the candidates:

* **RAFT** (code BSD-3-Clause) and **GMFlow / UniMatch** (code MIT) have permissive code licences.
  Neither README states a licence for the pretrained weights, and the weights are trained on
  FlyingChairs, FlyingThings3D, Sintel and KITTI, whose own terms have to be checked for
  redistribution (Sintel and KITTI are commonly research-only). That check is outstanding, so no
  weights are bundled.
* A pure-Rust inference path (the `segment` crate already carries a TFLite interpreter and
  ndarray-based GEMM) or a wgpu one, with CPU fallback; a model download step with checksum
  verification and an explicit error when the model is missing.

## Known limitations

* Slow motion cannot recover detail that is not in the source frames; heavily blurred source gives
  blurred output.
* A small object that moves about its own width per frame is reconstructed with a distorted outline even at Best (visible in the montage; blending would show two ghosts instead). Draft and Balanced can miss such objects entirely (see above); Inverse warping can still lose thin
  structures at depth discontinuities that Forward handles.
* One foreground matte; no separate motion for several regions, and no spline/shape-driven guides
  (guides are points).
* Motion-vector *export* exists as a library function (`vectors::encode`), not as a control.
* No interlaced footage; no neural flow; the GPU path runs the synthesis only (the flow is CPU).

## Performance

Measured with `cargo run --release -p effectcraft-frameweaver --example bench 720 1080 2160` on the
reference machine: **Intel Xeon @ 2.8 GHz, 4 vCPU, 16 GB RAM, Linux, no GPU** (a cloud VM; a modern
8-core desktop is expected to be roughly twice as fast). Scene: the procedural "parallax" test scene
(fractal texture with a moving occluder), one in-between frame (w = 0.5), forward warping. Times are
for one pair of source frames; "cached re-render" is another output frame between the same pair
(flow found in the cache). Flow analysis is capped at the preset's analysis size (640 / 960 / 1440 /
1920 px long side), so 4K costs the same flow as 1080p and only synthesis grows.

| frame | quality | flow (both directions) | consistency + repair | time field | synthesis | **total** | cached re-render |
|---|---|---:|---:|---:|---:|---:|---:|
| 1280×720 | Draft | 70 ms | 58 ms | 45 ms | 24 ms | **208 ms** | 77 ms |
| 1280×720 | Balanced | 455 ms | 198 ms | 143 ms | 25 ms | **831 ms** | 136 ms |
| 1280×720 | High | 1081 ms | 276 ms | 183 ms | 24 ms | **1.57 s** | 234 ms |
| 1280×720 | Best | 1828 ms | 312 ms | 177 ms | 22 ms | **2.34 s** | 202 ms |
| 1920×1080 | Draft | 65 ms | 59 ms | 62 ms | 61 ms | **255 ms** | 94 ms |
| 1920×1080 | Balanced | 342 ms | 132 ms | 106 ms | 56 ms | **644 ms** | 161 ms |
| 1920×1080 | High | 1204 ms | 305 ms | 249 ms | 56 ms | **1.83 s** | 308 ms |
| 1920×1080 | Best | 3559 ms | 651 ms | 446 ms | 58 ms | **4.72 s** | 445 ms |
| 3840×2160 | Draft | 92 ms | 37 ms | 41 ms | 236 ms | **456 ms** | 261 ms |
| 3840×2160 | Balanced | 443 ms | 128 ms | 106 ms | 238 ms | **937 ms** | 328 ms |
| 3840×2160 | High | 1228 ms | 313 ms | 251 ms | 241 ms | **2.07 s** | 499 ms |
| 3840×2160 | Best | 3645 ms | 659 ms | 447 ms | 249 ms | **5.03 s** | 637 ms |

Peak resident memory of the benchmark process (all sizes and presets in one run): 0.4 GiB at 720p,
0.74 GiB through 1080p Best, ~1.0 GiB at 4K. The flow cache holds at most 256 MiB on top of that.
On a real timeline the flow is computed once per source-frame pair and reused for every output frame
between them, so slow motion by 4× pays the "cached re-render" column for three of every four frames.

What the profile showed and what was done (1080p Best, one direction, before → after):
the patch search 3.1 s → 0.9 s (the bilinear weights are the same for every pixel of a patch, so
the displaced patch is read as direct row slices instead of per-pixel clamped samples) and the
variational refinement 2.1 s → 1.7 s (shared tap indices, reused SOR buffers). Total flow per
direction 5.2 s → 2.6 s; output unchanged (the quality gates and the ablation table are identical).
Still on the table: the forward field builder is single-threaded (0.45 s at Best), the SOR sweeps are
the bulk of the remaining refinement time, and the flow could move to the GPU. The original target of
"Best in 600 ms at 1080p" is **not** met on this machine (Best 4.7 s, Balanced 644 ms); it was
aspirational and no quality was traded to approach it.

## Testing and measuring

```
cargo test -p effectcraft-frameweaver --release        # unit + ground-truth quality gates
cargo test -p effectcraft-effects --test frameweaver   # the effect through a mock host
cargo run --release -p effectcraft-frameweaver --example ablate [width]     # tuning table
cargo run --release -p effectcraft-frameweaver --example reference          # needs ffmpeg
cargo run --release -p effectcraft-frameweaver --example bench              # performance
cargo run --release -p effectcraft-frameweaver --example montage -- out.ppm parallax crossing   # look at it
```
