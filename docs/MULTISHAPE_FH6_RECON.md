# Multi-Shape Injection Recon Brief

> **Audience:** A Cursor agent (or any reverse-engineer) running FH6 locally
> with permission to attach a debugger / scan process memory.
>
> **Mission:** Decode FH6's in-memory binary format for **rectangle**,
> **triangle**, and **rotated_rectangle** layer types so the injector can
> write them, the same way it already writes ellipses.
>
> **Why this matters:** Our GPU shape-gen already produces these three shape
> types (eval notebooks shipped). The ONLY thing standing between us and
> shipping non-ellipse coverage is verified binary-format knowledge for these
> three layer types. Painter-fh6 and upstream geometrize/painter only do
> ellipses — solving this is our shape-vocabulary moat.

---

## 0. Status

| Shape type | GPU shape-gen | EXE injector | Verified in FH6? |
|---|---|---|---|
| `rotated_ellipse` | ✅ shipping | ✅ shipping | ✅ pinned (regression-tested) |
| `rectangle` (axis-aligned) | ✅ eval | ⚠️ writes `shape_id=101`, divisor `/127` — **byte ID confirmed, divisor unverified** | 🟡 partial |
| `rotated_rectangle` | ✅ eval notebook shipped | ⚠️ shares rect write path — **never tested with angle ≠ 0** | 🟡 partial |
| `triangle` (equilateral) | ✅ eval notebook shipped | ❌ injector writes `shape_id=101` (wrong) + `sx=sy=1.0` (no `hw`/`rx`/`r` field) | 🟡 byte ID = **103** confirmed; **vertex/size field unknown** |
| `triangle` (right) | ✅ eval | ❌ same issue | 🟡 byte ID = **104** confirmed; **vertex/size field unknown** |

We need to take this table from "🟡 / ❌" to "✅ pinned" with regression tests.

### 2026-05-27 — Cursor QUASAR recon update

| Finding | Evidence |
|---|---|
| **Equilateral triangle = `shape_id_byte` 103** at `layer + 0x7A` | 300-layer probe (`round3_verdict_300_triangles.json`) |
| **Right triangle = `shape_id_byte` 104** at `layer + 0x7A` | 300-layer probe (`round3_verdict_300_right_triangle.json`) |
| **Rectangle = `shape_id_byte` 101** confirmed (was assumption) | 300-layer probe (`round3_verdict_300_rectangles.json`) |
| **Triangles use the same `(x, -y)` position convention as ellipses** | Position-spread experiment confirmed visually in FH6 by user |
| **Byte-flip 104 → 103 turns right-triangle into equilateral in-game** | Visual confirmation: flipped all 300 layers, every shape changed |
| **Mixed 100×3 template repeats `103 → 101 → 102`** per logical triple | Heap dump of user's 3-shape texture |
| **Heap fingerprint for 103/104 templates scores 4, not 5** | Locator strict path skips them; must use `--reuse-prior --skip-heap` |

**What this resolves:**
- §3 **Q1** answered for rectangle (101), equilateral triangle (103), right triangle (104).
- §3 **Q5** Y-negation answered for triangles: same as ellipses.
- §3 **Q4** STILL UNKNOWN: writing only `shape_id_byte` + pos/scale/color/rotation works to set the TYPE, but the triangle's actual geometry (vertex coordinates or unit-template size) hasn't been located. The byte-flip experiment changes type while keeping whatever size/position FH6 originally allocated for the layer; it does not prove we can write a triangle of arbitrary size from scratch.

**What this implies for the injector (P0 gating, P1 inject):**

1. `inject/fh6_injector.py::_score_layer` currently rejects any template whose `shape_id_byte ∉ {101, 102}` — strict fingerprint score 4 fails. Triangle templates get skipped silently.
2. `GameProfile` (FH6) has `shape_id_other = 101` for ALL non-ellipses, including JSON `triangle`. Result: a `triangle` JSON gets written as a rectangle with `sx=sy=1.0` (no `hw`/`rx`/`r` field → default scale).
3. Locator's heap fingerprint demands score-5 match; triangle-only templates max out at score 4 and require `--reuse-prior --skip-heap` to inject into.

See §3 below for the corresponding QUASAR-validated answers + §8 for the roadmap that uses them.

**Artifacts (on SMB):**
- `\\QUASAR\ContentCreation\ForzaAbyssPainter_build\diagnostics\SHAPE_TYPE_MATRIX_UPSTREAM.md` — Cursor's full writeup
- `diagnostics/round3_verdict_300_{rectangles,triangles,right_triangle}.json` — raw layer dumps
- `diagnostics/layer_shape_experiment.py` — byte-flip + position-spread scripts
- `diagnostics/layer_shape_experiment_log.json` — flip evidence with before/after byte values
- `diagnostics/layer_experiment_snapshot.json` — 300-layer baseline (for restore)

---

## 1. What We Know (Ellipse Baseline — Do Not Re-Verify)

Pinned in `tests/test_inject_upstream_scale_convention.py` and
`tests/test_fh6_injector.py`. Trust these.

Layer struct (offsets relative to a layer pointer dereferenced from the
LiveryGroup table at `0x78`):

| Offset | Field | Type | Ellipse convention |
|---|---|---|---|
| `0x18` | Position | 2× f32 `(x, -y)` | Y is **negated** |
| `0x28` | Scale | 2× f32 `(sx, sy)` | `sx = rx / 63.0`, `sy = ry / 63.0` (radii — NOT halved again) |
| `0x50` | Rotation | f32 degrees | `(360 - angle) % 360` (negated) |
| `0x74` | Color | 4× u8 RGBA | alpha **forced to 255** |
| `0x78` | Mask | u8 | always 0 |
| `0x7A` | Shape ID | u8 | **102** = ellipse |

Total: 26 bytes written across 6 fields. Source of truth:
- Write path: `forza_abyss_painter/inject/fh6_injector.py:1315-1358`
- Field offsets: `forza_abyss_painter/inject/game_profiles.py:87-99`

**What's already coded for non-ellipses (but unverified):**
- `game_profiles.py:88-89` — `shape_id_other = 101` for everything that isn't an ellipse
- `fh6_injector.py:1327-1331` — non-ellipse scale uses divisor `/127` (vs ellipse `/63`)
- Both values are inherited from painter-fh6's `inject.cpp` and have not been
  exercised in-game from our side. They could be right. They could be
  partially right. They could be wrong.

---

## 2. What's Already Type-Agnostic (Don't Re-Investigate)

The locator pipeline doesn't care about shape type — it finds the
LiveryGroup by structural fingerprint, not semantics. So **once we know how
to write a rect/triangle, the existing locator code handles inject end-to-end
without changes.** Confirmed type-agnostic:

1. **Signature-chain locator** (`fh6_injector.py:671-719`) — scans 3× 32 MiB
   windows for the 8-byte sentinel + mirror gate at `+0x70`, walks the
   4-pointer chain `+0xB8 → +0xA58 → +0x8 → +0x20`. Shape-blind.
2. **Heap u16 layer_count scan** (`fh6_injector.py:721-826`) — finds
   LiveryGroup by `layer_count` value (e.g. `1000`, `3000`) + 5/5 sphere
   fingerprint over sampled table pointers. Shape-blind.
3. **RTTI vtable scan** (`fh6_injector.py:921-985`) — `CLiveryGroup` RTTI
   class-name match. Shape-blind.

**What IS ellipse-specific in the current codebase:**
- Random-param sampling in GPU shape-gen uses ellipse `w/8, h/8` semi-axis
  bounds (already replaced for the rect/triangle shape-gen — see eval
  notebooks)
- Joint-polish gradient optimizer is ellipse-only (out of scope here)
- Bbox-local scoring assumes ellipse aspect ratios (out of scope here)

---

## 3. Open Questions — Per Shape Type

For each question, do the minimum experiment to answer it, capture
memory-diff evidence, write a regression test that pins the answer in
`tests/test_inject_multishape.py`, and update this doc with the verified
value.

### Q1: Is `shape_id=101` really the right byte for rectangle AND triangle? **[ANSWERED 2026-05-27]**

**Answer:** No — only rectangle is 101. Triangles have their OWN IDs.

| Shape | `shape_id_byte` @ `layer + 0x7A` | Source |
|---|---|---|
| Ellipse / circle | **102** | Pre-existing (production-validated) |
| Rectangle / square | **101** | 300-layer probe `round3_verdict_300_rectangles.json` |
| Equilateral triangle | **103** | 300-layer probe `round3_verdict_300_triangles.json` |
| Right triangle | **104** | 300-layer probe `round3_verdict_300_right_triangle.json` |

Byte-flip experiment (writing `shape_id_byte` 103 → 104 → 103 across all
300 layers) visually confirmed in-game by user. Position writes at
`layer + 0x18` as `struct.pack('<2f', x, -y)` also confirmed visually.

**Still TODO:** pin in `tests/test_inject_multishape.py::test_shape_id_byte_per_type`
once `GameProfile` is updated to carry the byte map (see §8 roadmap).

### Q2: Is the `/127` scale divisor correct for rectangle?

**Method:**
1. Open painter-fh6, insert a rectangle with known half-extents (eyeball
   to roughly `64 × 32` game units — exact doesn't matter, just record
   it).
2. Locate layer base (Q1 method), dump bytes at `+0x28..+0x30` (8 bytes,
   2× f32).
3. Compute observed `sx, sy`. Confirm `sx ≈ hw / 127.0` and `sy ≈ hh / 127.0`.
4. If divisor is wrong, sweep candidate divisors (`63`, `100`, `127`,
   `128`, `255`) and report which matches.

**Acceptance:** Verified divisor with empirical evidence. Pin in
`tests/test_inject_multishape.py::test_rectangle_scale_divisor`.

### Q3: Is the scale field full-extent or half-extent for rectangle?

**Method:**
1. Same setup as Q2. Once divisor is known, derive whether the scale
   field encodes the half-width `hw` (so `sx*divisor = hw`) or the full
   width `w = 2*hw` (so `sx*divisor = w`).
2. Cross-check by inserting a second rect with double the width and
   confirming the byte ratio.

**Acceptance:** Verified extent convention. Update
`fh6_injector.py:1327-1331` if the current `(hw * 2) / 127.0` math is
wrong. Pin in `tests/test_inject_multishape.py::test_rectangle_extent_convention`.

### Q4: Where do triangle's 3 vertices live in memory? **[PARTIALLY ANSWERED — H1 LIKELY, NEEDS PROOF]**

**This is the hardest unknown.** The Layer struct is fixed-size and has
no 24-byte slot for 6 floats. Hypotheses, in decreasing order of
likelihood:

- **H1: Triangle reuses the scale field's 8 bytes** as `(scale_x, scale_y)`
  applied to a unit equilateral / right triangle template. The actual
  vertex offsets are baked into the shape mesh keyed by `shape_id`. This
  matches how axis-aligned-rectangle would work and is the simplest case.
- **H2: Triangle stores 3 vertices as offsets relative to position, packed
  into scale + rotation + an unused slot** (e.g., `scale_x = v2x - v1x`,
  `scale_y = v2y - v1y`, `rotation = atan2(v3-v1)`, third vertex implied).
  Hacky but possible.
- **H3: Triangle uses an entirely separate side-table** referenced by a
  pointer stored in one of the Layer struct's currently-unread fields
  (the offsets `0x00..0x18`, `0x30..0x50`, `0x54..0x74` are all unknown
  to us today).

**2026-05-27 evidence (Cursor QUASAR):** Byte-flipping `shape_id_byte`
between 103 and 104 on existing layers DOES change the rendered triangle
type, without rewriting any other Layer field. This is strong evidence
for **H1**: the triangle mesh is keyed by `shape_id`, and the existing
position/scale/rotation fields drive the placement. The flip experiment
keeps whatever size FH6 originally assigned to those layers (the layers
were created via the FH6 vinyl editor at known sizes); it does NOT prove
we can write a NEW triangle of arbitrary size from scratch.

**Open sub-question (Q4a):** Does the `(scale_x, scale_y)` at `+0x28`
control triangle size, the same way it controls rectangle size? If yes,
H1 is fully confirmed and we have a working inject path. If no, FH6 uses
a different slot for triangle scale that we haven't located.

**Method (for Q4a):**
1. Use `diagnostics/layer_shape_experiment.py` `spread` mode to write
   different `(scale_x, scale_y)` values to triangle layers at
   `+0x28`. Confirm visually whether triangle SIZE changes (vs just
   position).
2. If size changes: H1 confirmed. Divisor TBD by Q2-equivalent for
   triangles.
3. If size doesn't change: locate the actual size slot by scanning
   for the known size float across the Layer struct's unread offsets.

**Method (for original Q4 — locate-the-vertices, if H1 fails):**
1. Insert a triangle in painter-fh6 with all three vertices at known,
   distinct positions (e.g., `(100,100)`, `(200,100)`, `(150,200)`).
2. Locate the layer (use centroid `(150, 133)` for the float scan).
3. Dump full Layer struct: `forza_abyss_painter inject dump <addr> --size 256`.
4. Search the dump for any of the 6 known vertex coordinates (raw float
   bytes). Report which offsets contain them — that tells us the layout.
5. If no vertex coords appear in the Layer struct, follow pointer-shaped
   u64 fields and dump their targets to find the side-table.

**Acceptance:** Either Q4a confirms H1 + a divisor for triangle scale,
OR the original Q4 method locates the vertex storage. Either way, pin
in `tests/test_inject_multishape.py::test_triangle_geometry_layout`.

### Q5: Y-negation, rotation, color, mask — universal or ellipse-only? **[PARTIALLY ANSWERED 2026-05-27]**

| Field | Ellipse convention | Rectangle? | Triangle (103/104)? |
|---|---|---|---|
| Position Y sign | negated `(x, -y)` | ? | ✅ same — confirmed visually |
| Rotation | `(360 - angle) % 360` | ? | ? |
| Alpha byte | forced 255 | ? | ? |
| Mask byte | always 0 | ? | ? |

Triangle Y-negation was verified by Cursor's `layer_shape_experiment.py
spread` mode: writing position as `struct.pack('<2f', x, -y)` placed
triangles at the expected on-screen locations.

**Still TODO** for rectangle + triangle rotation/alpha/mask: same
memory-dump procedure as Q1, dump bytes at the relevant offset, confirm
the encoding matches the ellipse convention or document the difference.

**Acceptance:** Table above filled in with verified values. Pin in
`tests/test_inject_multishape.py::test_field_conventions_per_type`.

### Q6: Is rotated_rectangle a distinct shape ID, or just rectangle with non-zero rotation?

**Method:**
1. Inject one axis-aligned rect and one rotated rect (e.g., 45°) via
   painter-fh6.
2. Dump shape_id bytes for both. If identical, rotated_rect is just rect
   + rotation field set. If distinct, we have a third shape ID to track.

**Acceptance:** Single answer documented. Pin in
`tests/test_inject_multishape.py::test_rotated_rectangle_id_vs_rectangle`.

---

## 4. Discovery Toolkit

You have these CLI subcommands today (all in
`forza_abyss_painter/inject/cli.py`):

```
forza_abyss_painter inject find-pid             # locate forzahorizon6.exe
forza_abyss_painter inject scan-float <value>   # initial coord scan
forza_abyss_painter inject narrow <value>       # refine after moving shape
forza_abyss_painter inject dump <addr> [--size]  # hex + struct dump
```

The Python helpers backing them are in `inject/discovery.py`:
- `scan_float(pid, target, epsilon=0.01)` → list of candidate addresses
- `narrow(state, new_target)` → intersect with second value
- `dump_around(pid, addr, size=256, before=64)` → `DumpRow` objects with
  hex + parsed floats/u32 in context

Existing inject-time diagnostics (`fh6_injector.py:350-668`) are
persisted to `%LOCALAPPDATA%\ForzaAbyssPainter\logs\inject-*.log`. Mine
these for chain-hop addresses if you need to bootstrap a locator pass
manually.

**Suggested workflow per shape type:**

```
1. Open FH6 with a livery editor session active.
2. Use painter-fh6's UI to insert ONE rectangle at known (x, y), known size.
3. forza_abyss_painter inject find-pid
4. forza_abyss_painter inject scan-float <x>            → ~hundreds of hits
5. Move the rect to a new known (x', y') via painter-fh6 UI.
6. forza_abyss_painter inject narrow <x'>               → ~tens of hits
7. Move once more. forza_abyss_painter inject narrow <x'''> → ideally 1-10 hits.
8. For each surviving address, dump 256 bytes:
   forza_abyss_painter inject dump <addr> --size 256
9. The address whose dump matches the §1 Layer offset pattern (position
   at +0x18 reads as the known (x, -y), color at +0x74, etc.) is the
   real Layer base. Subtract 0x18 if the address is the position field
   itself; the dump tool offsets automatically.
10. Record byte values at +0x28, +0x50, +0x74, +0x7A, +0x78 — that
    answers Q1-Q3 + Q5 in one capture.
```

For triangle (Q4), repeat with three distinct vertex coordinates so
scan-float can find any of the 6 float values; if no vertex coords
appear in the Layer struct, follow u64-shaped fields as pointer
candidates.

---

## 5. Acceptance Criteria — When Is This Recon "Done"?

This brief is closed and the work shifts to implementation when ALL of
the following are true:

- [ ] Q1-Q6 each have a verified answer documented in this file
- [ ] `tests/test_inject_multishape.py` exists and contains a passing
      regression test per question above, with byte-level expected
      values (use the `_pack_*` helpers as the unit under test —
      analogous to `tests/test_inject_upstream_scale_convention.py`)
- [ ] `forza_abyss_painter/inject/game_profiles.py` is updated with any
      newly-discovered shape-type IDs and divisors (replace the
      currently-assumed `shape_id_other=101` / `scale_divisor_other=127`
      with verified per-type values, or confirm them as correct)
- [ ] `forza_abyss_painter/inject/fh6_injector.py` write path handles
      all three new types correctly (or has a stub + clear `NotImplementedError`
      for any type we discovered needs a separate side-table approach)
- [ ] End-to-end: inject a 100-shape JSON with mixed
      ellipse/rect/triangle/rotated_rect into FH6, all 100 render
      correctly in-game

When all four boxes are ticked, file a PR and close the parked
multi-shape branch on the ForzaDesigner6 side.

---

## 6. Out of Scope

Solving any of these would be welcome but is NOT what this brief asks
for:

- Multi-shape **joint polish** (the gradient optimizer is ellipse-only
  today; adding rect/triangle gradients is a separate, larger effort)
- New shape types beyond rect / triangle / rotated_rect (e.g., bezier,
  polygon — defer until the basic three are shipping)
- Performance work on the locator (signature-chain path is already
  painter-parity-fast)
- Replacing painter-fh6's UI dependency for the recon procedure itself
  — using painter-fh6 to GENERATE known shapes is the simplest way to
  bootstrap; we only need to OWN the inject path, not the editor

---

## 7. Background References

- Painter-fh6 ellipse inject reference (the only shape type their
  injector handles): `inject.cpp` lines as cited in our README's
  painter-fh6 credit section
- GPU shape-gen multi-shape renderers (already shipping in eval
  notebooks): `notebooks/fd6_gpu_colab_lineart_400.ipynb` (rect) and
  `notebooks/fd6_gpu_colab_headshots_700.ipynb` (triangle), and the
  upstream Python in the ForzaDesigner6 sister repo at
  `fd6/shapegen/gpu/shapes_gpu.py` lines 109-115 (rect) and 176-183
  (triangle)
- Game-profile constants in this repo:
  `forza_abyss_painter/inject/game_profiles.py`
- Ellipse-baseline regression test (template for the multishape test
  file): `tests/test_inject_upstream_scale_convention.py`

---

## 8. Roadmap — recon → injector → notebook compositions

The strategic goal beyond this recon brief is shipping multi-shape
shape-gen end-to-end: GPU generates a mix of ellipses, triangles, and
rectangles → injector writes them all correctly → FH6 renders them →
notebook eval pipeline enumerates compositions for quality A/B.

Today every stage downstream of GPU shape-gen is bottlenecked on this
recon. Sequence the work like this:

### Stage 1 — Honest gating (P0, ~1 hr)

**Goal:** stop the injector from silently writing wrong shapes.

- `inject/fh6_injector.py::_score_layer` — accept `shape_id_byte ∈
  {101, 102, 103, 104}` instead of hardcoding `{101, 102}`. Move the
  allow-list to `GameProfile.shape_id_allowed: set[int]`.
- `inject/game_profiles.py::FH6_PROFILE` — add `shape_id_allowed = {101,
  102, 103, 104}` and a `shape_id_byte_for_json_type` mapping.
- `inject/fh6_injector.py` write path — pick `shape_id_byte` from the
  JSON `type` via the mapping, not the hardcoded `shape_id_other = 101`.
- Triangle JSON write policy (pick one):
  - **A. Reject** — refuse to inject any `triangle` / `rotated_rectangle`
    JSON with a clear modal: "Triangle injection is not yet validated;
    convert to ellipse + rectangle first or wait for full recon."
  - **B. Experimental write** — write `shape_id_byte` 103 or 104 +
    pos/scale/color/rotation. Result: a triangle of FH6's default size
    appears at the right position with the right color. Geometry size
    may NOT match the JSON's intent until Q4a is verified.

  Recommend Option A until Q4a is verified — Option B's bogus geometry
  is exactly the failure mode users would have today, just with the
  right type instead of the wrong type.
- `fap-validate --inject-safe` flag (optional) — strict mode that
  ERRORs on any non-injector-safe type, for CI / pre-flight scripting.

**Acceptance:** Injecting a 100-shape JSON with mixed types either
writes them all correctly (Option B, post-Q4a) or fails fast with a
clear message (Option A).

### Stage 2 — Verify triangle scale (Q4a, ~1 session on QUASAR)

**Goal:** confirm whether triangle scale lives at `+0x28` (H1) or
elsewhere.

- Use `diagnostics/layer_shape_experiment.py spread` to vary
  `(scale_x, scale_y)` on triangle layers; observe size change.
- If size changes: confirm H1, determine divisor (analogous to Q2 for
  rect). Pin in regression test.
- If size doesn't change: scan unread Layer struct offsets for size
  floats; document layout.

**Acceptance:** §3 Q4 + Q4a fully answered, pinned in
`tests/test_inject_multishape.py::test_triangle_geometry_layout`.

### Stage 3 — Real triangle injection (P1, ~1-2 sessions)

**Goal:** GPU shape-gen → triangle JSON → injector writes correct-size,
correct-color, correct-position triangles in FH6.

- Update `inject/fh6_injector.py` write path: triangle scale field
  follows the convention discovered in Stage 2.
- Update `io/json_schema.py` triangle shape to carry the field set
  (size, vertex-count, etc.) the injector needs.
- Regression test: 10-shape golden inject fixture
  (`tests/golden_inject/fixture_triangles_10.json`) + FH6 screenshot
  comparison (manual on QUASAR initially; SSIM gate later per planning
  doc §6).

**Acceptance:** Inject 100 triangles of varying sizes/positions/colors,
all render correctly in-game. Validator's
`non_injector_safe_type_present` warning is removed for triangle.

### Stage 4 — Notebook compositions (the unlock, multi-session)

The Colab eval pipeline (`ForzaDesigner6` sister repo,
`notebooks/build_colab_notebook.py`) currently generates ellipse-only
output because the multi-shape eval is gated on injection. With Stage 3
shipped, the notebook side can:

- **Enumerate compositions:** mixed-shape presets where the GPU greedy
  loop picks the best shape TYPE per position, not just the best
  ellipse. Per planning doc §4: brute-force enumeration first, then UMAP
  / LSH for high-shape-count pools.
- **Algorithm-selection moat:** painter-fh6 is ellipse-only; geometrize/
  painter has no algorithm selection. Mixed-type compositions where the
  engine picks `triangle` for sharp edges, `ellipse` for soft gradients,
  `rectangle` for flat regions is the competitive feature.
- **Quality A/B vs ellipse-only:** the notebook eval already exists
  (mixed-shape rendering ships; what's broken is the inject pipeline
  proving the mixed output ends up correct in-game). Stage 3 closes that
  loop.

**Acceptance:** A user uploads a portrait → notebook generates a mixed
JSON with triangles for hair edges, ellipses for skin, rectangles for
flat backdrops → inject writes them all → in-game render is visibly
sharper than the ellipse-only baseline at the same shape budget.

### Stage 5 — Multi-shape polish + scoring (P2, blocked on #129)

The GPU polish optimizer is ellipse-only today (joint_polish:`shapegen/
gpu/joint_polish.py`). Polishing mixed-type compositions is a separate
optimizer with per-type gradients — out of scope for the recon brief
itself but is the next strategic bottleneck once Stage 4 ships.

Also blocked: `bbox_local` scoring (production EXE path) assumes
ellipse aspect ratios. For multi-shape, the engine must use
`full_canvas` scoring, which requires #129 chunked `rasterize_hard`
(per planning doc §4). #129 is the chunked-rasterize-hard work that
unblocks K=8192 + full_canvas on consumer GPUs.

### Dependency graph

```
recon (this doc)
    │
    ├── Stage 1 — Honest gating (P0, today)
    │
    └── Stage 2 — Triangle scale verification (Q4a)
            │
            └── Stage 3 — Real triangle injection (P1)
                    │
                    ├── Stage 4 — Notebook compositions (unlock!)
                    │
                    └── Stage 5 — Multi-shape polish + scoring (P2)
                                       │
                                       └── #129 chunked rasterize_hard
                                            (separate strategic task)
```

The shortest path to user-visible value is Stage 1 → Stage 2 → Stage 3.
Stage 4 + 5 are bigger sessions but together produce the moat.
