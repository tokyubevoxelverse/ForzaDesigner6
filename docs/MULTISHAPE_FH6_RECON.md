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
| `rotated_rectangle` | ✅ eval notebook shipped | ⚠️ writes `shape_id=101` and divisor `/127` — **unverified** | ❌ |
| `triangle` | ✅ eval notebook shipped | ❌ no path — 6 vertex floats don't fit standard 8-byte scale field | ❌ |
| `rectangle` (axis-aligned) | ✅ eval | ⚠️ same write path as rotated_rectangle, never tested with angle=0 | ❌ |

We need to take this table from "⚠️ / ❌" to "✅ pinned" with regression tests.

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

### Q1: Is `shape_id=101` really the right byte for rectangle AND triangle?

**Method:**
1. Open painter-fh6 (the C++ tool we forked from). Insert a single
   rectangle in the FH6 editor via painter's UI. Note its position.
2. Run `forza_abyss_painter inject scan-float <rect.x>` to locate the
   layer base address.
3. `forza_abyss_painter inject dump <layer_addr>` — read byte at
   `+0x7A`. Record value.
4. Repeat for triangle and rotated_rectangle.

**Acceptance:** Either confirm all three are `101`, or document the
distinct IDs we find. Pin in
`tests/test_inject_multishape.py::test_shape_id_byte_per_type`.

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

### Q4: Where do triangle's 3 vertices live in memory?

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

**Method:**
1. Insert a triangle in painter-fh6 with all three vertices at known,
   distinct positions (e.g., `(100,100)`, `(200,100)`, `(150,200)`).
2. Locate the layer (use centroid `(150, 133)` for the float scan).
3. Dump full Layer struct: `forza_abyss_painter inject dump <addr> --size 256`.
4. Search the dump for any of the 6 known vertex coordinates (raw float
   bytes). Report which offsets contain them — that tells us the layout.
5. If no vertex coords appear in the Layer struct, follow pointer-shaped
   u64 fields and dump their targets to find the side-table.

**Acceptance:** Verified vertex storage layout, documented as an
addendum to the offset table in §1. Pin in
`tests/test_inject_multishape.py::test_triangle_vertex_layout`. If H1 wins
this is straightforward; if H3 wins we may need a new inject code path.

### Q5: Y-negation, rotation, color, mask — universal or ellipse-only?

**Method:** Same memory-dump procedure as Q1-Q4 — for each non-ellipse
type, inject a known shape, dump bytes at the relevant offset, confirm
the encoding matches the ellipse convention or document the difference.

| Field | Ellipse convention | Rectangle? | Triangle? |
|---|---|---|---|
| Position Y sign | negated `(x, -y)` | ? | ? |
| Rotation | `(360 - angle) % 360` | ? | ? |
| Alpha byte | forced 255 | ? | ? |
| Mask byte | always 0 | ? | ? |

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
