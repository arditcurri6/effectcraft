# FrameWeaver: optical-flow retiming effect — design

Date: 2026-10-14 · Branch: `frameweaver` · Status: awaiting review

## 1. Intent

**What was asked:** a plug-in for EffectCraft with the functionality of the commercial
optical-flow retiming plug-in for After Effects (Twixtor), under a different name.

**Decisions made with the requester:**

| Question | Decision |
|---|---|
| Delivery | Lives in the requester's **own fork** of EffectCraft, as a native Rust effect (no plug-in API change) |
| Scope | **Core + Pro** feature set in one design |
| Name | **FrameWeaver** (effect id `ec.time.frameweaver`, Effect ▸ Time ▸ FrameWeaver) |
| Architecture | Algorithms in a new `frameweaver` crate, a thin adapter in `effectcraft-effects`, **plus GPU synthesis** in `effectcraft-gpu` |

**Assumptions (not stated by the requester):**

- The fork should keep merging upstream EffectCraft cleanly, so edits to existing files stay
  one-liners where possible.
- The fork follows the repository's rules (`AGENTS.md`, `CLAUDE.md`): never crash, clean-room,
  pure Rust, layering, `cargo xtask ci` green on every commit.
- "Same functionality" means the same *capabilities* (speed ramps, motion-interpolated
  in-betweens, warping and blend choices, matte separation, guidance, cuts, external vectors),
  not identical parameter names or pixel-identical output. The behaviour reference is the
  public documentation only; no code, assets or names are copied.

**Success criteria:**

1. Applied to footage, FrameWeaver slows it down (Speed % or keyframed Frame Num) with
   in-between frames built along dense motion vectors, visibly sharper than Timewarp's Pixel
   Motion and Frame Mix on the same shot.
2. Every control in §4 works and is animatable, scriptable and agent-drivable like a built-in.
3. CPU and GPU produce matching output; the CPU path works everywhere, including wasm.
4. `cargo xtask ci` passes; no crash path is introduced.

## 2. Why not the existing extension points

- **Plug-in API v1** (`docs/plugins.md`) hands an effect only the current frame, processed in
  place. Retiming needs the neighbouring source frames, which only built-in effects can read
  (`EffectHost::self_at`). Extending the API (v2) is not needed in a fork, and dense flow in
  the wasmi interpreter would be 10–50× slower than native.
- **Timewarp** (`crates/effects/src/time_fx.rs`) is After Effects' built-in equivalent and is
  kept for parity. Modifying it would change a parity effect and conflict with upstream.
  FrameWeaver is a separate effect: it reuses the public frame fetcher
  (`effectcraft_effects::time_frames`) and follows Timewarp's GPU plan pattern, without changing
  Timewarp. Private Timewarp helpers it needs (fetching a layer parameter at source times) are
  copied into its adapter rather than made public upstream.

## 3. Architecture

### 3.1 Crates and files

```
crates/frameweaver/                 NEW crate `effectcraft-frameweaver`, layer L1
  Cargo.toml                        deps: effectcraft-raster, rayon (workspace)
  src/lib.rs                        public API, #![deny(unwrap/expect/panic/…)]
  src/flow.rs                       DIS dense optical flow, bidirectional, occlusion
  src/prep.rs                       Image Prep, matte planes, BG hole fill
  src/field.rs                      time fields (Inverse / Forward), visibility, guides
  src/synth.rs                      CPU synthesis, motion blur, FG/BG composite, Display
  src/vectors.rs                    external motion-vector decoding
  src/cache.rs                      bounded flow LRU
  src/tests/…                       unit tests (synthetic motion)

crates/effects/src/frameweaver.rs   NEW: spec, params, source-time maths, frame and layer
                                    fetching, cut decisions → FwPlan; FwPlan::render_cpu
crates/effects/tests/frameweaver.rs NEW: behaviour + golden tests with a mock EffectHost

crates/gpu/src/fx_frameweaver.rs    NEW: renders an FwPlan on the GPU
crates/gpu/src/shaders/fx_frameweaver.wgsl   NEW: kernels `ffw_synth`, `ffw_mblur`, `ffw_layer`
crates/gpu/src/tests_fx_frameweaver.rs       NEW: CPU-vs-GPU parity

docs/frameweaver.md                 NEW: user guide
```

### 3.2 Edits to existing files (kept minimal for upstream merges)

| File | Edit |
|---|---|
| `Cargo.toml` (workspace) | `effectcraft-frameweaver = { path = "crates/frameweaver" }` |
| `xtask/src/main.rs` | `("frameweaver", 1)` in `LAYERS`; `("frameweaver", "raster")` in `SAME_LAYER` |
| `docs/architecture.md` §1 | the crate in the layer table |
| `crates/effects/Cargo.toml` | depend on `effectcraft-frameweaver` |
| `crates/effects/src/lib.rs` | `mod frameweaver;` + `pub use` of the plan types; `v.extend(frameweaver::specs())` in `registry()`; `"ec.time.frameweaver"` in `TIME_DEPENDENT` and `GPU_EFFECTS`; new group display names (§4) in the group-name table |
| `crates/gpu/src/lib.rs` | `mod fx_frameweaver;` (+ test module) |
| `crates/gpu/src/effects.rs` | dispatch `ec.time.frameweaver` to `fx_frameweaver::apply` |
| `crates/gpu/src/context.rs` | chain `fx_frameweaver::KERNELS` |
| `docs/effects.md` | one entry under Time |

The id prefix `ec.time.` makes the layer cache treat the effect as reading other times
(`crates/render/src/cache.rs`, `reads_other_times`) without any change there. `ec.` is
reserved for built-ins, and in the fork FrameWeaver *is* a built-in.

`crates/script/src/matchnames.rs` maps EffectCraft ids to After Effects match names. FrameWeaver
has no After Effects counterpart, so it gets no entry; no test requires one. Scripts add it by
its display name (`addProperty("FrameWeaver")`), as they do plug-in effects.

### 3.3 Layering

`frameweaver` is **L1**, beside `raster`, which it depends on. That same-layer edge is declared
in `SAME_LAYER` as `("frameweaver", "raster")`, exactly as `("path", "raster")` is. `effects`
(L2) → `frameweaver` (L1) is then an ordinary downward edge, and `gpu` (L3) reaches it through
`effects`. It depends on nothing UI- or OS-specific and builds for `wasm32-unknown-unknown`.

### 3.4 Per-frame data flow

```
effect render (CPU fn or GPU apply)
  └─ frameweaver_plan(ctx) -> Option<FwPlan>          (None: no host → pass through)
       1. instants  = source times s (1, or N across the shutter) with weights
       2. pairs     = for each s: a = floor(s·fps_in), b = a + 1, w = frac
       3. frames    = self_at(a/fps_in), self_at(b/fps_in) on one grid
                      (+ matte, motion-source, vector layers via layer_at at the matching comp times)
       4. per pair  = cut? → Hold(a or b)
                      else flows A→B, B→A (cache → or estimate on the motion-source image,
                      after Image Prep; or decode external vectors), guides imposed
       5. fields    = time fields F_ta, F_tb + visibility v_a, v_b at w
                      (Inverse: fixed-point; Forward: splatting — both on the CPU)
  └─ FwPlan::render_cpu()  |  fx_frameweaver::render_plan() on the GPU
       synthesis per instant → motion-blur average → FG over BG → Display mode
```

### 3.5 Plan types (in `effectcraft-effects`, re-exported for the GPU crate)

```rust
pub struct FwPlan {
    pub grid: Buf,                 // output geometry (offset, scale, size)
    pub frames: Vec<Image>,        // fetched frames (and matte-split FG/BG frames)
    pub mattes: Vec<Vec<f32>>,     // matte planes, when separating
    pub fields: Vec<TimeField>,    // per-instant (per-layer) time fields
    pub steps: Vec<(FwStep, f32)>, // instants with their shutter weights
    pub interp: Interp,            // Nearest | Blend | MotionWeighted
    pub smart_blend: bool,
    pub mblur: Option<MotionBlur>, // taps, span
    pub display: Display,
}
pub enum FwStep {
    Hold(usize),                                   // a frame (Nearest, or across a cut)
    Mix { a: usize, b: usize, w: f32 },            // Blend
    Motion { a: usize, b: usize, w: f32, field: usize },
    Layered { fg: [usize; 3], bg: [usize; 3], matte: [usize; 2], w: f32 },
    Image(usize),                                  // a ready-made Display image
}
// in effectcraft-frameweaver:
pub struct TimeField {         // at analysis resolution, sampled bilinearly
    pub width: u32, pub height: u32, pub scale: f32,   // field px → frame px
    pub to_a: Vec<[f32; 2]>, pub to_b: Vec<[f32; 2]>,  // output pixel → sample position offsets
    pub vis_a: Vec<f32>,     pub vis_b: Vec<f32>,      // 0..1 visibility
}
```

The GPU only *samples*: everything data-dependent (flow, splatting, cuts, guides) is done on
the CPU and shipped in the plan, exactly as Timewarp's `TimewarpPlan` does.

## 4. Effect Controls

Every control is an ordinary property (keyframes, expressions, Effect Controls, `prop.set`,
scripting, MCP). Group paths follow the existing `group/param` convention.

| Control | Id | Type, range | Default |
|---|---|---|---|
| Input Frame Rate | `inputFrameRate` | slider 0–240 (UI 0–120), 3 decimals; **0 = use the composition's rate** | 0 |
| Time Remap Mode | `timeRemapMode` | popup Speed, Frame Num | Speed |
| Speed % | `speed` | slider −10000…10000 (UI 0–400), 1 decimal | 50 |
| Frame Num | `frameNum` | slider −100000…100000 (UI 0–300), 2 decimals | 0 |
| Frame Interp | `frameInterp` | popup Nearest, Blend, Motion Weighted Blend | Motion Weighted Blend |
| Motion Vectors | `motionVectors` | popup None, Sloppy, Best, External Vector Layer | Best |
| Warping | `warping` | popup Inverse, Forward | Inverse |
| Smart Blend | `smartBlend` | checkbox | on |
| Image Prep | `imagePrep` | popup None, Contrast/Edge Enhance | None |
| Motion Sensitivity | `motionSensitivity` | slider 0–100 | 70 |
| Display | `display` | popup Output, Source, Motion Vectors, Occlusion, FG Matte, FG Only, BG Only, Guides | Output |
| **Motion Blur** ▸ Enable Motion Blur | `motionBlur/enable` | checkbox | off |
| Motion Blur ▸ Shutter Angle | `motionBlur/shutterAngle` | slider 0–720 (UI 0–360) | 180 |
| Motion Blur ▸ Samples | `motionBlur/samples` | slider 1–32, 0 decimals | 8 |
| **Segments** ▸ Auto-Detect Cuts | `segments/autoDetect` | checkbox | on |
| Segments ▸ Cut Sensitivity | `segments/sensitivity` | slider 0–100 | 50 |
| Segments ▸ Segment Marker | `segments/marker` | slider −1000…1000, 0 decimals | 0 |
| **Separation** ▸ FG Matte Layer | `separation/matteLayer` | layer | none |
| Separation ▸ Matte Channel | `separation/matteChannel` | popup Alpha, Luminance, Inverted Alpha, Inverted Luminance | Alpha |
| Separation ▸ BG Fill | `separation/bgFill` | popup Neighbour Frames, Inpaint | Neighbour Frames |
| **Motion Source** ▸ Motion Source Layer | `motionSource/layer` | layer | none |
| **External Vectors** ▸ Vector Layer | `externalVectors/layer` | layer | none |
| External Vectors ▸ Encoding | `externalVectors/encoding` | popup Signed RG (0.5 = still), Float RG | Signed RG |
| External Vectors ▸ Vector Scale | `externalVectors/scale` | slider 0–10000 (UI 0–512) | 1 |
| External Vectors ▸ Direction | `externalVectors/direction` | popup To Next Frame, To Previous Frame | To Next Frame |
| External Vectors ▸ Flip Y | `externalVectors/flipY` | checkbox | off |
| **Guides** ▸ Guide Radius | `guides/radius` | slider 1–2000 px (UI 1–400) | 60 |
| Guides ▸ Guide *n* ▸ Enable (n = 1…8) | `guides/guideN/enable` | checkbox | off |
| Guides ▸ Guide *n* ▸ Point | `guides/guideN/point` | point | guides 1–4 at x = 0.2, 0.4, 0.6, 0.8 and y = 0.35; guides 5–8 at the same x and y = 0.65 (fractions of the layer) |

New group display names for the group-name table: Segments, Separation, Motion Source, External
Vectors, Guides, Guide 1 … Guide 8 (Motion Blur already exists).

### 4.1 Behaviour of the controls

- **Input Frame Rate** `fps_in`: source frame *k* is at layer time *k / fps_in*. 0 uses the
  composition's rate (`EffectCtx::fps`), which is right for footage interpreted at the comp
  rate; set it to the footage's real rate otherwise.
- **Source time** at output layer time *t*: Speed mode `s = t · speed / 100` (negative plays
  backwards); Frame Num mode `s = frameNum / fps_in` (keyframe it for ramps and freezes). Same
  conventions as Timewarp (`timewarp_source_time`).
- **Pair:** `a = floor(s · fps_in + 1e-6)`, `b = a + 1`, `w = s · fps_in − a`. When `w < 1e-4`
  the output is frame *a* exactly (no flow is estimated).
- **Frame Interp:** Nearest shows *a* if `w < 0.5`, else *b*. Blend cross-fades. Motion Weighted
  Blend warps along vectors (§5.3); with Motion Vectors = None it behaves as Blend.
- **Motion Vectors:** None skips estimation; Sloppy and Best are the two DIS presets (§5.1);
  External Vector Layer decodes the Vector Layer (§5.6), falling back to Best when no layer is
  set.
- **Motion Sensitivity** (0–100) maps to DIS patch size and the refinement's smoothness weight:
  high = finer, more independent vectors; low = smoother, more rigid ones.
- **Display:** Output; Source = nearest source frame; Motion Vectors = A→B flow as a colour
  wheel (hue = direction, saturation = magnitude normalised to the frame's 99th percentile);
  Occlusion = occluded pixels white; FG Matte / FG Only / BG Only = the separation's pieces;
  Guides = Output with each enabled guide drawn as a dot and its a→b vector.
- **Motion Blur:** span = `shutterAngle / 360` × source frames advanced per output frame
  (`|speed| / 100` in Speed mode, the Frame Num slope over one output frame in Frame Num mode).
  Taps are taken along the output-time vector field, box-weighted.
- **Segments:** a cut between *a* and *b* when Auto-Detect Cuts is on and the cut detector fires
  (§5.7), or when Segment Marker evaluated at *a* and at *b* (via `params_at`) differ. Across a
  cut the output holds *a* (`w < 0.5`) or *b*; no flow is estimated.
- **Separation:** the matte layer is read at the comp times matching the source frames (timed
  like the source). FG and BG get separate flows and are synthesised separately, then
  composited FG over BG. BG Fill chooses how the BG behind the FG is filled before its flow is
  estimated (§5.5).
- **Motion Source Layer:** flows are estimated on this layer's frames and applied to the
  effect's own frames (e.g. a denoised or contrast-boosted copy).
- **Guides:** evaluated at the source frame times *a* and *b* through `params_at` (keyframe or
  track them on the source footage; Layer ▸ Track Motion can target them). Each enabled guide
  forces the flow near it to its own motion `P(b) − P(a)`.

### 4.2 Out of scope

- Fields / interlaced footage.
- More than one FG matte; exporting the filled BG plate as its own output.
- After Effects SDK compatibility or a WebAssembly build of the effect (the fork links it in).
- Changing Timewarp or the plug-in API.

## 5. Algorithms

All methods are from published papers or textbooks; no third-party code is consulted or copied.

### 5.1 Dense optical flow — DIS (Kroeger, Timofte, Dai, Van Gool, ECCV 2016)

- **Input:** a single-channel analysis plane per frame: luma `0.2126R + 0.7152G + 0.0722B` of
  the un-premultiplied colour, times alpha (or the Motion Source layer's), after Image Prep.
  Downscaled so the long side ≤ 960 px (Sloppy) / 1920 px (Best), and further by the render's
  preview scale.
- **Pyramid:** ×2 levels down to a coarsest long side of ~32 px.
- **Per level, coarse to fine:** initialise from the coarser flow ×2 (zero at the top). Lay
  patches on a regular grid (stride = patch × (1 − overlap)). For each patch, run
  inverse-compositional Lucas–Kanade on the patch (precomputed template gradients and Hessian;
  stop at the iteration cap or when the update < 0.01 px), starting from the initial flow at the
  patch centre; reject an update that increases the patch error.
- **Densify:** each pixel's flow is the average of the vectors of the patches covering it,
  weighted by `1 / max(1, |I_b(x + u) − I_a(x)|)` per patch.
- **Variational refinement (Best only):** minimise brightness constancy + gradient constancy +
  smoothness (robust penalty) with a fixed number of SOR sweeps (5 outer × 10 inner) per level.
- **Presets:**

  | | patch | overlap | LK iterations | refinement |
  |---|---|---|---|---|
  | Sloppy | 12 px | 0.3 | 4 | no |
  | Best | 8 px | 0.6 | 12 | yes |

  Motion Sensitivity `m ∈ [0, 1]` scales patch size by `1.5 − m` (clamped to 6–16 px) and the
  refinement's smoothness weight α by `2 − 1.8m`.
- **Bidirectional:** compute `F_ab` and `F_ba`. **Occlusion** (Sundaram, Brox, Keutzer 2010):
  `x` is occluded in B when
  `|F_ab(x) + F_ba(x + F_ab(x))|² > 0.01 (|F_ab(x)|² + |F_ba(x + F_ab(x))|²) + 0.5`; the binary map
  is softened by a 3 × 3 box into visibility `v ∈ [0, 1]`.

### 5.2 Image Prep

Contrast/Edge Enhance: `I' = 0.5 + (I − mean₇ₓ₇(I)) / (4·std₇ₓ₇(I) + 0.02) · 0.25 + 0.5·|∇I|`,
clamped to [0, 1]. It normalises local contrast so flow survives exposure changes and flat
regions.

### 5.3 Time fields and synthesis

- **Inverse warping:** for each output pixel `x` at fraction `w`, find the source point `u` in
  A with `x = u + w·F_ab(u)` by fixed-point iteration `u ← x − w·F_ab(u)` (5 iterations, start
  `u = x`). `F_ta(x) = u − x`. Symmetrically `q ← x − (1 − w)·F_ba(q)` and `F_tb(x) = q − x`.
  **Visibility:** with `occ_ab(p)` = "A pixel *p* is hidden in B" and `occ_ba(q)` = "B pixel *q*
  is not present in A" (§5.1), `v_a(x) = 1 − occ_ba(q)` (content revealed between A and B comes
  only from B) and `v_b(x) = 1 − occ_ab(u)` (content being covered comes only from A); both
  sampled bilinearly from the softened maps.
- **Forward warping:** splat every A pixel `p` to `p + w·F_ab(p)` and every B pixel `q` to
  `q + (1 − w)·F_ba(q)` into the output grid with bilinear weights. Where splats collide, a
  depth test keeps the one whose pixel is not occluded (`occ_ab(p)` / `occ_ba(q)` lower; ties:
  smaller photometric error `|B(p + F_ab(p)) − A(p)|`). Visibility is then 1 where a splat
  landed from that frame and 0 where only the other frame's did. It records `F_ta = p − x`
  and `F_tb = q − x`. Holes are filled by push-pull (pyramid down with weights, up with
  fill-only). The result is the same `TimeField` format.
- **Guides** (applied to `F_ab` and `F_ba` before the time fields): for guides *i* with
  positions `P_i(a)`, `P_i(b)` in buffer pixels and `d_i = P_i(b) − P_i(a)`,
  `g_i(x) = exp(−|x − P_i(a)|² / (2σ²))`, `σ = radius / 2`;
  `F(x) ← F(x)(1 − G(x)) + Σ g_i d_i / max(Σ g_i, 1e-6) · G(x)` with `G = min(1, Σ g_i)`. With a
  matte, a guide applies to the FG flow when the matte at `P_i(a)` ≥ 0.5, otherwise to the BG
  flow.
- **Motion Weighted Blend** per pixel:
  `O = [(1 − w)·v_a·A(x + F_ta) + w·v_b·B(x + F_tb)] / [(1 − w)·v_a + w·v_b]`, bilinear sampling
  on premultiplied RGBA. When the denominator < 1e-4, fall back to the nearest frame at `x`.
- **Smart Blend:** a sample position outside its frame's bounds sets that frame's `v` to 0 for
  that pixel (so its weight moves to the other frame) instead of clamping to the edge.
  Without Smart Blend, out-of-frame samples clamp to the edge.

### 5.4 Motion blur

For each output pixel, `K = samples` taps at `x + (k / (K − 1) − 0.5)·span·F_t(x)` (where
`F_t = F_tb − F_ta`, the per-frame motion at the output time), sampled from the synthesised
frame and averaged with equal weights. Span as in §4.1. K = 1 or span = 0 is a no-op.

### 5.5 FG/BG separation

1. Matte planes `m_a`, `m_b` (channel per Matte Channel), resampled to the grid.
2. FG frames: `A·m_a`, `B·m_b`. FG flow is estimated with the analysis plane multiplied by the
   matte (BG pixels don't vote).
3. BG frames: FG area filled. **Neighbour Frames:** for each pixel with `m > 0.01`, take the
   same pixel from frames `a−1, b+1, a−2, b+2` (fetched only when needed) where their matte is
   < 0.01, else push-pull fill. **Inpaint:** `effectcraft_raster::inpaint`.
4. Synthesise FG (warping the matte too, giving FG alpha) and BG separately; output = FG over BG.

### 5.6 External vectors

Read the Vector Layer at the comp time matching source frame *a* (Direction = To Next Frame:
it holds A→B) or *b* (To Previous Frame: it holds B→A). Decode per pixel:
Signed RG `v = (c.rg − 0.5)·2·scale`; Float RG `v = c.rg·scale` (un-premultiplied colour);
Flip Y negates `v.y`. The reverse field is built by splatting the decoded field negated
(push-pull hole fill), so occlusion detection (§5.1) still applies.

### 5.7 Cut detection

`effectcraft_raster::cuts::signature` of A and B, `dissimilarity(sa, sb)`; a cut when it
exceeds `threshold = 0.6 − 0.5·sensitivity/100` (Cut Sensitivity 50 → 0.35). Signatures are
cheap (96 px thumbnails) and cached with the flows.

### 5.8 Flow cache

Process-wide LRU in `frameweaver::cache`: at most 6 entries and 256 MiB. The key is a 64-bit
hash of (A analysis plane, B analysis plane, preset, sensitivity, external-vector settings,
guide positions); the value is `F_ab`, `F_ba`, the visibility maps and the cut signatures.
Keyed by content, so it can never serve a stale result. The lock is a `Mutex`; a poisoned
lock is recovered (`into_inner`) and the cache cleared.

## 6. GPU synthesis

`crates/gpu/src/fx_frameweaver.rs`, following `fx_time.rs`:

- `IDS = ["ec.time.frameweaver"]`; `apply(e, id, ctx, b)` builds the plan with
  `effectcraft_effects::frameweaver_plan(ctx)` (`None` without a host → `Some(b)` unchanged,
  the same as the CPU) and renders it.
- Uploads the frames once each; uploads each `TimeField` as a storage buffer
  (`to_a`, `to_b`, `vis_a`, `vis_b`, plus dimensions and scale in `Params`).
- Kernels (prefix `ffw_`, file `shaders/fx_frameweaver.wgsl`):
  - `ffw_synth`: per output pixel, bilinear-sample the field, sample A and B, Smart Blend
    bounds test, visibility-weighted blend (§5.3). Nearest/Blend reuse `ftm_mix` semantics
    in the same kernel (a mode uniform).
  - `ffw_mblur`: K taps along the field (§5.4).
  - `ffw_layer`: FG over BG with the warped matte.
  - The shutter instants are averaged with the existing time-effect accumulate pattern
    (one weighted pass per instant, the same order as the CPU).
- Display images other than Output are built on the CPU and uploaded as `FwStep::Image`.
- Any failure (upload, buffer size) returns `None`, and the compositor's existing CPU fallback
  renders the effect.

The CPU `FwPlan::render_cpu` performs the identical arithmetic in the identical order, so the
parity tests can use a tight tolerance.

## 7. Error handling (never crash)

- `effectcraft-frameweaver` starts with
  `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]`
  and contains no `unsafe`. Input-derived indices use `get`; arithmetic on sizes uses checked or
  saturating operations; loops have fixed caps (pyramid levels ≤ 12, iterations as in §5.1).
- Sizes: grid ≤ 16384 × 16384 (as `time_fx::fetch`), analysis long side ≤ 1920 px, field
  buffers sized from the analysis plane. An allocation that would exceed these is refused and
  the effect passes its input through.
- Parameters: every value is checked with `is_finite()` and clamped to its range before use.
  `fps_in` ≤ 0 or non-finite → comp rate. Non-finite pixels are treated as 0 in the analysis
  plane.
- Fallbacks (each logged once per effect instance with `log::warn!`):

  | Situation | Behaviour |
  |---|---|
  | No `EffectHost` (thumbnails, tests) | pass the input through |
  | A source frame unavailable | use the nearest available frame |
  | Matte / motion-source / vector layer unset or unavailable | behave as if the feature were off |
  | External vectors unreadable | estimate with Best |
  | GPU upload or kernel failure | CPU path |
  | Cache lock poisoned | recover, clear, continue |

- Every crash found during implementation gets a regression test (repository rule).

## 8. Performance targets

Measured on a 1080p shot, release build, 8-core Apple Silicon; tracked with `cargo xtask bench`
and the CLI's per-effect timings, not asserted in tests:

| Operation | Target |
|---|---|
| Sloppy flow pair (both directions) | ≤ 150 ms |
| Best flow pair (both directions) | ≤ 600 ms |
| GPU synthesis of one output frame | ≤ 10 ms |
| CPU synthesis of one output frame | ≤ 60 ms |
| Re-render with cached flows (scrubbing) | ≤ 30 ms |

Preview at reduced resolution scales the analysis plane with the buffer, so half-resolution
previews cost roughly a quarter.

## 9. Testing

1. **`effectcraft-frameweaver` unit tests** (synthetic, deterministic textures):
   - Translation (integer and sub-pixel), rotation (3°), zoom (1.05×): average endpoint error
     ≤ 0.25 px (Best), ≤ 0.75 px (Sloppy), measured away from a 16 px border.
   - Square moving over a textured background: occlusion map marks ≥ 80 % of the revealed strip
     and ≤ 5 % elsewhere.
   - Interpolation at `w = 0.5` of a translating texture vs. the true half-shifted frame:
     PSNR ≥ 35 dB, and ≥ 8 dB above Blend.
   - Inverse vs. Forward on rigid motion: mean absolute difference ≤ 0.01.
   - Guides: the flow at a guide equals its `d` within 0.05 px; at 3σ the influence is < 2 %.
   - External vectors: encode/decode round trip for both encodings, Flip Y, Direction.
   - Cuts: fires on unrelated frames, not on a 20 px pan.
   - Cache: hit on identical content, miss on a one-pixel change, byte bound respected.
   - Never-crash: 0 × 0, 1 × 1, 1 × 4000, NaN / ∞ pixels, NaN params, zero-length guides.
2. **Effect tests** (`crates/effects/tests/frameweaver.rs`), mock `EffectHost` serving a
   texture that moves *k* px per source frame (as `timewarp_golden.rs`):
   - Speed 50 %, 200 %, −100 % and Frame Num pick the right frames and fractions.
   - Nearest and Blend are exact; Motion Weighted Blend matches the true in-between frame.
   - Separation: two objects moving in opposite directions, with a matte; both stay sharp
     (PSNR ≥ 30 dB vs. ground truth), better than without separation.
   - Segment Marker and Auto-Detect Cuts stop interpolation across a cut.
   - Motion Source layer and External Vectors drive the warp.
   - One check per Display mode.
   - Golden hashes of the CPU output for a matrix of settings (`SIM_GOLDEN_PRINT=1` re-pins).
   - The registry-wide tests (`registry_is_consistent`, `all_effects_run_with_defaults`,
     `time_dependence_is_declared`) cover it once it is registered.
3. **GPU parity** (`crates/gpu/src/tests_fx_frameweaver.rs`): the `compare_at` harness on a
   moving precomp (as `tests_fx_time.rs`) at 8 and 32 bpc, full and half resolution, for each
   Frame Interp, each Warping, Smart Blend on/off, Motion Blur on, and Separation; plus the
   no-host pass-through.
4. **Gate:** `cargo xtask ci` (fmt, clippy `-D warnings`, tests, layers, assets, wasm) on every
   commit.

## 10. Documentation

- `docs/frameweaver.md`: what it does; a quick start (slow motion with Speed %; a speed ramp with
  Frame Num); each control; workflows for Separation, Motion Source, External Vectors, Guides
  (with the tracker) and Segments; troubleshooting (warping artefacts → Separation or Guides;
  flicker → Motion Sensitivity; edges → Smart Blend).
- `docs/effects.md`: one entry under Time linking to it.
- `docs/architecture.md`: the crate in the layer table.

## 11. Implementation order (for the plan)

Each phase ends green (`cargo xtask ci`) and is one or more commits:

1. Crate scaffold, layering, `TimeField`/plan types, analysis plane + Image Prep.
2. DIS flow (Sloppy), densification, bidirectional + occlusion; unit tests.
3. Best: variational refinement; accuracy tests.
4. Time fields (Inverse), synthesis (Nearest, Blend, Motion Weighted Blend, Smart Blend).
5. Effect adapter: spec, params, source-time maths, frame fetching, registration; effect tests.
6. Forward warping (splatting + push-pull).
7. Motion blur.
8. Flow cache.
9. Pro: Segments (auto + marker).
10. Pro: Motion Source layer.
11. Pro: External Vectors.
12. Pro: Guides.
13. Pro: Separation (FG/BG, BG Fill).
14. Display modes.
15. GPU synthesis + parity tests.
16. Golden hashes, docs, benchmarks against §8.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Best flow misses the 600 ms target on CPU | Cap the analysis resolution lower for Best (1440 px); fewer SOR sweeps; profile before optimising further |
| GPU/CPU float differences beyond tolerance | Same operation order as the CPU (as `fx_time`); tolerance per test as `tests_fx_time.rs` uses |
| Upstream reorders the layer table or same-layer list | The two `xtask` lines are self-contained; re-apply on merge conflicts |
| Memory with Separation + motion blur (many frames on the grid) | Fetch frames once per unique time (dedupe, as `time_fx::fetch`); refuse plans over a byte budget and fall back to fewer shutter samples |
| Upstream changes to `time_fx` helpers FrameWeaver reuses | FrameWeaver calls only `pub` items (`time_frames`); private helpers it needs are copied into its adapter, not made public upstream |

## 13. Revision 2: corrections made during implementation

Status: implemented on branch `frameweaver`. Where this section and §1–12 disagree, this section
wins. Every change below is backed by a test or a measurement
(`cargo run --release -p effectcraft-frameweaver --example ablate`).

| # | Spec said | Problem | Now |
|---|---|---|---|
| 1 | Speed mode `s = t·speed/100` | Wrong for a keyframed speed (the source time jumps at each change) | `s(t) = start/fps_in + ∫₀ᵗ speed/100`, trapezoid on the comp frame grid through `params_at` (≤ 16384 steps); constant speed is exact. Start Frame added (reverse playback from the last frame) |
| 2 | Frames via `time_frames` | Cannot tell a missing frame from a blank one, so boundaries could blend with black | `time_frames_avail`; one available frame is held, none is transparent |
| 3 | Presets Sloppy / Best | Too coarse; "Best" regressed on some content | Draft / Balanced / High / Best differing in analysis size, patch, overlap, iterations, refinement, symmetrisation and global fallback (table in `docs/frameweaver.md`); ordering is a test |
| 4 | Visibility `v_a = 1 − occ_ba(q)`, `v_b = 1 − occ_ab(u)` | Assumes `u` and `q` are the same scene point; fails where surfaces cross, and a fixed point that oscillates is not a solution. Ignores unreliable (not occluded) motion | Candidates must solve their own equation; if both do and disagree, one winner (the one that survives in the other frame, else the better match) supplies both samples; a separate **confidence** (forward-backward consistency measured on the repaired pair) blends towards the nearest frame that can show the pixel |
| 5 | Occluded vectors: push-pull of neighbours | Averaged occluder and background motion, inventing a motion neither has | Weighted fill from vectors near the dominant (global affine) motion; low-confidence pixels fill from plain neighbours |
| 6 | Smoothness `α` 0.01·(2 − 1.8m) | Refinement over-fitted noise (flow error 0.14 → 0.52 px) | 0.1·(2 − 1.8m) (error 0.08 px on the noisy pan, 0.06 px average) |
| 7 | Default Warping Inverse | Measured worse at occlusions (parallax 29 vs 37 dB before the winner rule); forward is the better default | Default Forward (depth-tested); Inverse kept (comparable after the winner rule, faster, no splat) |
| 8 | Forward offsets `p − x` | Biased at partial coverage | `p − t` (landing position); visibility saturates at a quarter coverage; occluded splats ranked by occlusion alone |
| 9 | `ffw_acc` / shutter instants | Motion blur is defined by §5.4 (taps along the field), not by averaging instants | One instant per frame; `ffw_mblur` after `ffw_synth` |
| 10 | Controls flat (`speed`, `warping`, …) | Cannot be organised in Effect Controls | Groups: Time Remapping, Motion Estimation, Interpolation, Motion Blur, Segments, Advanced Guidance (separation, external vectors, guides), Diagnostics. Defaults otherwise as §4 (Quality Balanced, Vector Scale 16) |
| 11 | "Alpha-assisted" / splines / several regions | Not specified | Alpha is part of the analysis plane (luma × alpha); single FG matte; no spline guides, no multi-region |
| 12 | Cache keyed per §5.8 only | Output frames and decoded frames also cache | Flow pairs here (content-keyed, 6 / 256 MiB, poison-safe); decoded frames and output layers by the renderer. No cancellation hook exists in the effect API |
| 13 | Benchmarks in §8 assumed Apple Silicon | Not measurable here | Measured on the reference machine in `docs/frameweaver.md`; the targets stay aspirational |

**Verified rather than assumed:** the renderer already serves any layer time through
`EffectHost::self_at` (forward, reverse, freeze, non-monotonic, repeated requests) and keys time
effects on layer time (`ec.time.` prefix), so no engine change was needed for temporal access.

**Not done:** a neural flow backend (the `FlowBackend` seam exists; see the guide for what is
missing), GPU flow estimation, spline guides, several independent motion regions, a vector-export
control, cancellation.
