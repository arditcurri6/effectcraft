# CineLooks: design and implementation plan

Status: implemented (branch `cinelooks`, from `frameweaver-cloud-port`; commits CL1–CL7). User guide:
[`docs/cinelooks.md`](../../cinelooks.md). Effect id
`ec.color.cinelooks`, display name **CineLooks**, category Color Correction.

CineLooks is a modular cinematic colour-grading effect: a *look* is an ordered chain of tools
(exposure, wheels, curves, film response, grain, halation, LUTs, lens effects, selective
corrections…), edited in a dedicated workspace (the CineLooks Editor) with a preset library,
before/after comparison and scopes. It is an original implementation: no third-party code,
presets, UI, measurements or names. Film models are our own approximations and are never
described as reproductions of commercial stocks.

## 1. Where things live (follows FrameWeaver's split)

| Piece | Crate / file | Layer |
|---|---|---|
| Grading engine: tool catalogue, look model (serde), curve maths, `.cube` LUTs, per-pixel ops, spatial ops, program compiler, CPU executor, presets, cube export | `crates/cinelooks` (`effectcraft-cinelooks`) | L1 (beside raster) |
| Effect adapter: `EffectSpec`, node property groups, Params → `Look`, warnings | `crates/effects/src/cinelooks.rs` | L2 |
| GPU: fused per-pixel kernel + spatial kernels | `crates/gpu/src/fx_cinelooks.rs`, `shaders/fx_cinelooks.wgsl` | L3 |
| Commands: add tool, set/get look, presets, user presets, import/export, cube export | `crates/engine/src/commands/cinelooks.rs` | L4 |
| CineLooks Editor + Effect Controls button | `crates/ui-egui/src/panels/cinelooks/` | L5 |

Edits to upstream files stay one-liners (registry lists, GPU lists, dispatch arms, menu rows).

## 2. Data model: nodes are nested property groups

A CineLooks instance is an ordinary effect property group:

- top-level parameters: **Look Strength** (0–100 %, animatable), **Look Name** (hidden), **Show
  Original** (checkbox: bypass for comparison);
- one nested `GroupKind::Indexed` group per tool node, in processing order. The group's
  `match_id` is the tool kind (`exposure`, `wheels`, `lut`…), its `uid` is the node's stable
  identifier, `enabled` is the node's on/off switch, and its children are the tool's
  parameters plus **Strength** (0–100 %).

So every numeric tool parameter is a real `Property`: it animates with keyframes and
expressions, shows in the Timeline, serialises with the project, duplicates with the effect,
and undoes like any other edit. Reorder / enable / delete / duplicate use the generic
`prop.moveGroup` / `prop.setGroupEnabled` / `prop.removeGroup` / `prop.duplicateGroup`.
Curves are point-list strings (`"0,0 0.5,0.6 1,1"`, the format Lumetri uses); colour wheels are
`Vec3` `[x, y, master]`; LUTs are a file path plus optional embedded `.cube` text.

The engine-side `Look` struct (`Vec<Node>`, serde) is the exchange format: presets, user
presets, `.cinelook` files, the editor's draft and `cinelooks.getLook` / `setLook` all use
it. `setLook` *merges* into the instance (matching nodes by uid): static parameters are set,
animated ones get a key at the edit time, new nodes are built, missing ones removed, order
applied, all in one undo step.

## 3. Processing

- **Alpha:** input is premultiplied; the chain works on straight colour (unpremultiplied once)
  and premultiplies once at the end. Alpha is never changed except by the resampling tools
  (lens distortion, chromatic aberration, edge softness), which resample it with the colour.
- **Domains:** each tool declares the domain it works in: *linear light* (exposure, white
  balance, channel mixer, film response/print, halation, diffusion, vignette, anamorphic glow,
  selective exposure) or *encoded* (contrast, wheels, curves, HSL, hue remap, bleach bypass,
  fade, grain, LUT default). The chain starts in the project's domain (Linearize Working Space
  → linear, otherwise the working space's transfer curve; linear working spaces such as ACEScg
  use the sRGB curve as their grading encoding) and converts only where the next tool needs the
  other domain, then back once at the end. No primaries conversions happen; luma weights come
  from the working space's primaries.
- **Range:** float in, float out. Over-range (HDR) values pass through every tool (curves
  extrapolate with their end slopes, transfer curves mirror around zero); only LUTs clamp to
  their declared input domain.
- **Program:** a look compiles to stages. Consecutive per-pixel tools (everything without a
  neighbourhood: colour tools, grain, vignette, gradients, qualifiers, LUTs) fuse into one
  *pixel stage* evaluated in a single pass; blur-based and resampling tools are *spatial
  stages*. Node strength is a mix inside the stage; Look Strength mixes with the original
  input at the end.
- **Resolution independence:** spatial sizes are percentages of the layer height (offsets and
  centres are layer-space points), evaluated through the buffer's scale, so proxies, half
  resolution, 1080p and 4K render the same look.
- **Grain:** band-limited value noise on a lattice in layer space, hashed with an integer hash
  (identical on CPU and GPU), seeded by instance seed, node uid and the frame index
  `floor(time · fps · animation)`: the same frame always renders the same grain (preview =
  export); zero-mean with constant statistics, so no brightness flicker.
- **Blurs:** a deterministic pyramid (2×2 box downsample to a working level, separable
  Gaussian there, bilinear upsample) with clamp-to-edge borders; identical maths on the GPU.

## 4. GPU

`fx_cinelooks.rs` compiles the same program: pixel stages become one op stream in the storage
buffer, interpreted by `fcl_pixel` (curves, HSL tables and LUT lattices travel in the same
buffer, sampled with the CPU's interpolation); spatial stages get their own kernels
(`fcl_down`, `fcl_blur_h/v`, `fcl_glow`, `fcl_resample`, `fcl_streak`, `fcl_local`). Any
unsupported case returns `None` and the compositor's CPU fallback renders the effect. Parity
tests compare GPU output with the CPU effect within 1e-3 (absolute, float output).

## 5. Colour management (honest scope)

CineLooks follows the project's working space and linearisation and never applies a display
transform. The LUT tool takes an input space (working encoding, working linear, or a named
space from `effectcraft-color`) and converts in and back out around the table. Full OCIO
configs are not embedded in CineLooks: the existing OCIO effects (OCIO File/Color Space/Display
Transform) can be stacked before/after it, and the LUT tool's input-space hook is the
extension point. No ACES claims beyond the ACEScg/ACES2065-1 working spaces the project
already supports.

## 6. Editor (egui, EffectCraft theme)

Opened from the effect's header in Effect Controls ("Open CineLooks Editor…"), from
`cinelooks.openEditor`, or by double-clicking the effect. A full-window workspace:

- **left:** preset browser: search, categories, Favorites, User; thumbnails rendered from the
  layer's current frame through the real engine (cached by look + frame);
- **centre:** preview of the current frame (input = the layer before CineLooks, through the
  real engine), zoom, Original toggle, side-by-side before/after and a draggable split wipe;
  on-canvas handles for gradient / radial / spot / vignette centres;
- **bottom:** the tool chain: nodes in order with enable switches, selection, drag to reorder,
  add (categorised menu), duplicate, delete;
- **right:** inspector for the selected tool: sliders, colour wheels, curve editor, colour
  pickers, LUT chooser, per-parameter and per-tool reset; scopes (histogram, waveform, parade,
  vectorscope with clipping readouts) of the graded preview.

The editor works on a draft with its own undo/redo; **Apply** writes it with one `setLook`
(one project undo step), **Cancel** discards. Every widget registers an automation id.

## 7. Presets

At least 30 original looks in 12 categories, each a real multi-tool chain (tone, colour, film
response, finishing). Stored as Rust data (original work, listed in ATTRIBUTION.md). User
presets and favourites live in the preferences directory (`cinelooks/`), import/export as
`.cinelook` JSON; colour-only chains export as `.cube` (spatial tools are refused or reported,
never baked silently).

## 8. Phases

1. Engine crate: model, curves, primary tools, program + CPU executor, serde, unit tests.
2. LUTs, film tools, grain, halation, diffusion, lens and selective tools.
3. Effect adapter + registration; GPU kernels and parity tests; render integration.
4. Engine commands; CineLooks Editor.
5. Presets, thumbnails, user presets, import/export, cube export.
6. Scopes in the editor, colour-management checks.
7. Performance, end-to-end tests (project round trip, export = preview), packaging as
   `EffectCraft-Custom.app` (bundle id `ai.storyteller.effectcraft.custom`).
