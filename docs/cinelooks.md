# CineLooks

CineLooks (`ec.color.cinelooks`, **Effect ▸ Color Correction ▸ CineLooks**) is EffectCraft's
cinematic colour-grading effect: a *look* is a chain of grading tools (exposure, colour wheels,
curves, film response, grain, halation, LUTs, lens and selective tools…) that you build, preview
and fine-tune in the **CineLooks Editor**, a full-window grading workspace with a look library,
before/after comparison and scopes.

It is an original implementation: the engine, film models, look library and interface were
written for EffectCraft. Film tools are our own approximations of photochemical behaviour; no
look claims to reproduce a commercial film stock.

## Quick start

1. Select a layer and choose **Effect ▸ Color Correction ▸ CineLooks**.
2. In Effect Controls, click **Open CineLooks Editor…**.
3. Pick a look on the left (thumbnails show *your* frame through each look), then adjust its
   tools: click a tool in the chain at the bottom, edit it on the right.
4. **Apply** writes the look to the effect (one undo step) and keeps the editor open; **OK**
   applies and closes; **Cancel** (or Escape) discards what you haven't applied.

The editor has its own **Undo / Redo** (⌘Z / ⇧⌘Z while it is open); a whole slider drag is one
step.

## The editor

| Area | What it does |
|---|---|
| **Looks** (left) | Search, category (All, Favorites, User, the 12 library categories, your own), thumbnails, ★ favourites, right-click to favourite or delete your looks. **Save Look…**, **Import…** (`.cinelook` or `.cube`), **Export…** (`.cinelook`), **Export Color Tools as .cube LUT…** |
| **Preview** (centre) | The layer's input to CineLooks (with the effects above it) graded by the same engine that renders the effect. **Graded / Original / Split** (drag the divider) **/ Side by Side**, zoom (Fit, 25–200 %, wheel), pan (middle- or ⌥-drag), **Clipping** overlay (red: a channel at white, blue: black), **Look Strength**, a frame scrubber over the layer's duration. On-canvas handles move positions (gradient start/end, radial and spot centres, vignette centre) and show the shapes. |
| **Tool Chain** (bottom) | The tools in processing order, left to right. Click to select, the dot switches a tool on/off, the bar shows its strength, drag to reorder; **+ Add Tool**, **Duplicate**, **Delete** (or ⌫). |
| **Inspector** (right) | The selected tool: On, Strength, **Reset Tool**, and its controls (sliders, colour wheels, curve graph, colour pickers, choices, positions, LUT file). Double-click a label to reset that control. |
| **Scopes** (right) | Histogram, luma waveform, RGB parade and vectorscope of the graded preview, with the share of clipped shadows and highlights. |

## Tools

Every parameter is a real property of the effect: it can be keyframed and given expressions in
the Timeline or Effect Controls (each tool is a twirl-down group of the effect). Every tool has a
**Strength** (0–100 %) that mixes its result with its input.

**Primary Correction** — *Exposure* (stops, midtone compensation, highlight protection);
*Contrast* (amount, pivot, straight ↔ filmic S-curve); *White Balance* (temperature and tint as a
chromatic adaptation along the daylight locus, neutralise a picked colour); *Color Wheels* (lift,
gamma, gain, offset, then shadows / midtones / highlights balance, each with a master);
*Curves* (RGB, red, green, blue; monotone cubic through your points); *HSL* (global hue,
saturation, lightness, and hue / saturation / luminance for eight colour ranges);
*Channel Mixer* (3×3 mix or a monochrome mix); *Hue Remap* (a hue range with soft falloff shifted
in hue, saturation and luminance).

**Film Emulation** — *Film Response* (a filmic tone curve in stops: contrast, shadow compression,
highlight rolloff, latitude, colour response); *Film Grain* (procedural, fine / medium / coarse,
size, softness, luminance response, mono or colour, animation rate, seed); *Halation* (red-orange
glow around highlights: threshold, radius, amount, falloff, colour); *Bleach Bypass*; *Film Fade*
(lifted blacks, softened contrast, split-toned vintage colour); *Film Print* (five original print
styles: dye crosstalk and density in log space, then a print curve with adjustable toe and
shoulder).

**Camera & Lens** — *Optical Diffusion* (highlight bloom and soft mist); *Lens Vignette* (amount,
radius, feather, centre, aspect, highlight preservation); *Chromatic Aberration* (red/cyan and
blue/yellow, radial or uniform); *Lens Distortion* (barrel / pincushion, zoom, edges);
*Anamorphic Glow* (horizontal streaks); *Edge Softness*.

**Selective** — *Graduated Exposure* (linear gradient), *Radial Exposure* (ellipse, rotation),
*Spot Exposure*; each with exposure, saturation, tint, feather, invert and mask opacity.
*Shadow & Highlight* (local recovery from the surrounding brightness). *Selective Color* (an HSL
qualifier — hue, saturation and luminance ranges with softness, invert, **Show Mask** — and a
correction applied through it).

**LUT & Utility** — *LUT*: `.cube` files (1D, 3D, or a 1D shaper + 3D lattice; `DOMAIN_MIN/MAX` and
`LUT_*_INPUT_RANGE`), tetrahedral or trilinear interpolation, and the input space the LUT expects
(the working encoding, working linear, sRGB, Rec. 709, Rec. 2020, Display P3, ACEScg). The table is
embedded in the project when you choose it, so a moved or missing file still renders (the
inspector says so). A malformed file is reported with the line and the reason and renders
nothing. Use several LUT tools in one chain if you like.

## Looks

37 original looks in 12 categories (Cinematic, Film, Vintage, Modern, Warm, Cold, Blockbuster,
Documentary, Black & White, Music Videos, Portrait, Night). Each is a full chain you can edit.
Your own looks and favourites are kept in the settings folder (`cinelooks/library.json`);
`.cinelook` files are JSON (`{"format": "cinelook", "version": 1, "name", "category",
"description", "look": {"nodes": [...]}}`).

**Export Color Tools as .cube LUT…** bakes the colour tools of the current look (with its Look
Strength) into a 3D LUT. Tools that need neighbouring pixels or the pixel's position (grain,
glows, lens tools, vignette, masks, local recovery) cannot be expressed as a LUT: they are left
out and listed in the file's header and in a message, never baked silently.

## Colour management

- CineLooks follows **File ▸ Project Settings ▸ Color**: it receives the working space's pixels,
  works on them, and hands back the same encoding. It never applies a display transform.
- Each tool states the domain it needs — *linear light* (exposure, white balance, channel mixer,
  film response and print, glows, vignette, selective exposure) or the *grading encoding*
  (contrast, wheels, curves, HSL, grain, fades…). The chain converts only where the next tool
  needs the other domain, and once at the end. Linear working spaces (ACEScg, ACES2065-1, or
  Linearize Working Space) start linear; their grading encoding is the sRGB curve.
- Float in, float out: values above 1.0 (32 bpc) pass through; curves extend with their end
  slopes. Only LUTs clamp, to their declared input domain.
- Unmanaged projects are treated as sRGB (an unmanaged project set to Working Gamma 2.4 is still
  graded with the sRGB curve: small differences in the encoded tools, nothing applied twice).
- **OCIO / ACES**: CineLooks does not load OCIO configurations itself. For OCIO pipelines, place
  EffectCraft's OCIO effects (OCIO Color Space / File / Display Transform) before or after
  CineLooks; the LUT tool's input-space setting is the place a future OCIO input transform would
  plug in. Supporting the ACEScg / ACES2065-1 working spaces is all the ACES support claimed.

## Rendering

- The editor preview, thumbnails, Composition panel, RAM preview and export all run the same
  compiled program: the CPU and the GPU (Metal on Apple Silicon) agree within 1e-3 (tested for
  every tool, a 25-tool chain, every working space, 8 and 32 bpc, full and half resolution).
- Sizes (radii, grain, aberration) are relative to the layer, so proxies, half-resolution
  previews, 1080p and 4K give the same look.
- Grain is deterministic per frame: the same frame always renders the same grain (preview equals
  export); *Animation* 0 freezes it.
- Alpha is kept (transparent pixels stay transparent; glows are added within the alpha), except
  where Lens Distortion and Chromatic Aberration resample it.

## Scripting and agents

The effect is found by its display name or id (`CineLooks`, `ec.color.cinelooks`). Commands:

| Command | |
|---|---|
| `cinelooks.apply {layers?, preset? \| look?, strength?}` | add CineLooks (or reuse the layer's) with a look |
| `cinelooks.getLook {layer?, effect?, time?}` / `cinelooks.setLook {…, look, strength?, time?}` | read / write the look (one undo step; animated parameters get a key at `time`) |
| `cinelooks.addTool {layer?, effect?, tool, index?, params?}` | add a tool |
| `cinelooks.presets {category?, search?}`, `cinelooks.tools` | the library, the tool catalogue |
| `cinelooks.savePreset`, `deletePreset`, `setFavorite`, `importPreset`, `exportPreset`, `exportCube` | the user library and files |
| `cinelooks.openEditor {layer?, effect?}` | open the editor |

## Known limitations

- The editor edits the look at one time (the current time when it opened): animated parameters
  get a key there when you apply. Animate across time in the Timeline.
- Curves are hold-keyframed (a curve doesn't interpolate between keys).
- Hue-based tools (HSL ranges, Hue Remap, Selective Color) leave near-neutral pixels alone by
  design, so they cannot tint greys; use the wheels for that.
- The editor preview is graded on the CPU at the compositor's preview steps (full size up to
  1400 px, half for 1080p, quarter for 4K): typically 20–35 ms per change. Measured on an Apple M4
  (`cargo test -p effectcraft-gpu --lib cinelooks_bench -- --ignored --nocapture`), full frames:

  | Look (tools) | 1080p CPU | 1080p GPU* | 4K CPU | 4K GPU* |
  |---|---|---|---|---|
  | Cool Shadows (4) | 23 ms | 19 ms | 93 ms | 46 ms |
  | Warm Cinema (7) | 92 ms | 23 ms | 409 ms | 83 ms |
  | Blockbuster Teal & Orange (7) | 54 ms | 23 ms | 224 ms | 77 ms |
  | Neon Streets (7) | 58 ms | 29 ms | 225 ms | 92 ms |

  \*GPU times include uploading the frame and reading it back; inside the compositor the frame
  stays on the GPU.
