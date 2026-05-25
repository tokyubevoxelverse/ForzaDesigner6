# Forza Abyss Painter — Working Rules

These rules govern how I (Claude) work in this repo and its sister
`/Users/kusanagi/Development/ForzaDesigner6` (the Colab/notebook side).
They are the user's standing instructions, not suggestions. Violating
any of them is a session-level failure even if the code "works" — the
process matters as much as the output.

## 0. Repos at a glance

Two repos in lockstep:

- **`forza-abyss-painter`** (this repo) — Windows EXE for **consumer**
  hardware (gaming GPUs that have to co-exist with FH6 in RAM/VRAM).
  PyInstaller bundle, Qt GUI, in-app injector + CPU shape-gen.
- **`ForzaDesigner6`** (sister repo) — Colab notebooks targeting
  **enterprise/server** GPUs (T4 / V100 / L4 / A100, FH6 not running).
  Big-budget overnight runs allowed.

Same shape-gen library + injector core, different deployment targets,
different constraints.

---

## 1. Shipping rules (the hardest line)

### 1a. No shipping a GUI/EXE feature without BOTH:
  (i) a **visible UI component** the user can reach — a menu item, a
      button, a settings toggle, a banner. Not "the function exists in
      the package" — actually reachable through the rendered GUI.
  (ii) a **local smoke test** that exercises the real code path
       end-to-end under offscreen Qt with real files (not mocks),
       observing the output flow through completion without exception.

**Unit tests passing ≠ smoke tested.** Tests mock things; smoke tests
don't. A green pytest run is necessary but not sufficient.

### 1b. Order of operations is FIXED, no exceptions:

```
write code
  → unit tests green
  → LOCAL smoke test against real code paths
  → SMB sync to /Volumes/ContentCreation/ForzaAbyssPainter_build/source/
  → Windows tester validates the rebuilt EXE
  → THEN consider PR / release / next feature
```

Skipping any step burns the tester's time and the rebuild cycle. They
are not the integration-test layer.

### 1c. No "Phase N stub" surface visible to users.
If buttons / menu items would print "not yet shipped" messages or do
nothing, **hide them behind a feature flag** in
`forza_abyss_painter/gui/feature_flags.py` defaulting to `False`. Flip
to `True` in the SAME commit that lands the real plumbing. Never
flip the flag independently of the plumbing landing.

### 1d. No incomplete features in shipped builds.
If a user can't actually use a feature in the EXE or notebook, **don't
ship it**. Either gate it behind an explicit experimental flag (see §5)
or keep it local-only/uncommitted until the path completes.

---

## 2. Code-path separation: Colab (enterprise) vs EXE (consumer)

These are **fundamentally different code paths** with different
constraints. You must verify them SEPARATELY.

| Dimension | Colab/enterprise | EXE/consumer |
|---|---|---|
| VRAM | 12–80 GiB exclusive | 6–24 GiB shared with FH6 |
| FH6 running | No | Yes (co-resident) |
| Session length | Hours, overnight ok | Minutes; OOM kills game |
| Presets | RANDOM_SAMPLES up to 24576, max_res 1500+ | Capped at 16384 + 1200 |
| Backend | Pure Python + cupy/torch | Qt threading + signals |
| OOM recovery | Retry next cell | Modal error + abort |

A smoke that proves the Colab path works does NOT prove the EXE path
works, and vice versa. When porting a fix between sides:

1. Write a port-target dossier first: which file:line on each side?
2. Make the change.
3. Smoke BOTH paths after.
4. Pin both with regression tests.

The three small ports this session (#59 / #60 / #61) are precedent —
each had a notebook-side reference + an EXE-side test + a smoke run.

---

## 3. JSON output spec compliance

**The output JSON MUST conform to the game's shape spec. You cannot
invent geometry the game's injector doesn't know how to read.**

Canonical schema: `fd6.shapes` v1 — defined in
`forza_abyss_painter/io/json_schema.py`. Every shape dict uses known
type strings (`"rotated_ellipse"`; future: `"rectangle"`, `"triangle"`)
with the field set the injector reads byte-by-byte.

### When adding a new shape type:
1. **First** verify the game's binary format for that shape via
   reverse-engineering on a live FH6 process (see
   `docs/MULTISHAPE_FH6_RECON.md` for the procedure).
2. **Then** update `io/json_schema.py` + `shapegen/shapes/<type>.py`
   with the verified contract.
3. **Pin** with a regression test that locks the byte-level
   expectations for the injector
   (`tests/test_inject_upstream_scale_convention.py` is the template).
4. **THEN** the shape-gen can emit JSON using the new type.

**Never the reverse order.** Generating shape JSON that the injector
doesn't understand ships data that won't render in-game — the worst
kind of regression because offline rendering looks fine.

### JSON validator hooks (in flight, task #100):
- Library `forza_abyss_painter/io/validator.py` + CLI `fap-validate`
- Hook into `load_json()` + `save_json()` (warn on load, block on save)
- Hook into `GenerationWorker` at `event.kind == "done"` BEFORE
  `save_json` writes — invalid output never lands on disk
- Tools menu surface for ad-hoc validation

---

## 4. Multi-shape inference strategy

For new shape types (rectangle, triangle, rotated_rectangle), the
strategic goal is **algorithmic superiority over both upstreams**:
painter-fh6 (ellipse-only) and geometrize/painter (no algorithm
selection at all). We win on math, not implementation polish.

### Approaches, in increasing order of complexity:

1. **Brute-force parameter enumeration** — enumerate
   (x, y, w, h, angle, color) for each candidate shape type at
   resolution-appropriate granularity. Pick best by score. Fast on
   GPU; baseline.
2. **Shape inference from existing ellipses** — given an ellipse-only
   solution, find slots where a rectangle/triangle fits better in the
   same geometric region. Replace if score improves. Cheap upgrade.
3. **Dimensionality reduction (UMAP, t-SNE)** — embed the full
   parameter tuple space into 2D, surface **shape collisions**: at
   certain rotations a thin triangle ≈ a thin rectangle ≈ a line.
   Use the embedding to prune redundant candidates + accelerate
   selection. Also a debugging surface for "why did the engine pick
   this shape over that one".
4. **LSH (locality-sensitive hashing)** — for huge candidate pools
   (ellipse + rect + triangle + rotated_rect together), bucket
   near-identical-rendering shapes so we don't score them
   redundantly.

### Pick the cheapest approach that solves the case.

Don't burn weeks on a UMAP pipeline if brute-force on the GPU finishes
in seconds. Don't ship brute-force when LSH saves 10× compute on user
hardware. The advanced techniques are tools, not destinations.

### What this is NOT:
- Not an excuse to introduce ML where deterministic algorithms work.
- Not a license to "infer" game-side shape types — the game's binary
  layout still has to be reverse-engineered (see §3).

---

## 5. Experimental gating

If something is incomplete (user-facing path doesn't fully work) or
unstable (needs in-game validation on a Windows PC):

- **Default: local-only, uncommitted** until validation completes.
- **Exception:** gate it behind an explicit experimental flag the user
  (or Cursor) can flip for in-game testing.

If you commit gated experimental code:
- Flag default reflects "not ready for users".
- Dialog/menu copy labels it explicitly:
  `"EXPERIMENTAL — for in-game validation only, not production"`.
- A test pins the flag's value with an explicit checklist of what
  MUST land before flipping (see `tests/test_gpu_phase3_flag.py` for
  the template).

---

## 6. Repo hygiene

### 6a. Branding & voice
- Project name: **Forza Abyss Painter** (this repo) /
  **Forza Designer 6** (sister Colab repo).
- License: MIT, clean rebrand of upstream `tokyubevoxelverse/ForzaDesigner6`.
- Painter-fh6 credit in README with specific source-file citations —
  preserve.
- Commit identity: `whyKusanagi <169282093+whykusanagi@users.noreply.github.com>`
  **always**. NEVER personal email or dev paths. ("THE FUCKING RULE
  FOR WHYKUSANAGI IS TO USE THAT ACCOUNT ALWAYS.")
- Notebook theme: chibi logo + corrupted-palette
  (magenta `#d94f90`, purple `#8b5cf6`, dark `#0a0a0a`).

### 6b. CI/CD discipline
- Release pipeline: `.github/workflows/release.yml` — triggers on
  `v*` tag push OR workflow_dispatch.
- Build target: github-hosted windows-latest, PyInstaller.
- Build provenance: `forza_abyss_painter/_build_info.py` rewritten by
  CI before PyInstaller runs.
- Release body documents SmartScreen warning + admin-mode requirement.

### 6c. Version bumps
- **Only when explicitly authorized.** No silent bumps for fixes.
- Releases are deliberate events tied to confirmed-working features,
  not "we wrote some code this session".
- `pyproject.toml` version stays put until the user says ship.

### 6d. Push discipline
- Never push to `main` of either repo without explicit authorization
  for that specific push.
- Default workflow: commit on feature branch → smoke → SMB → tester →
  user authorizes PR → merge.
- Sister repo (`ForzaDesigner6`) is someone else's fork — never push
  /PR to `origin` without instruction on which remote to use.

### 6e. SMB build workflow
- Sync target: `/Volumes/ContentCreation/ForzaAbyssPainter_build/source/`
- QUASAR (Windows box) robocopies source → local → builds with
  `build_exe.bat` → copies EXE to `dist/`.
- Sync to SMB only AFTER local smoke tests pass (§1b).
- See `README_BUILD.md` on SMB for the QUASAR-side procedure.

---

## 7. Things that violate these rules

If you find yourself doing any of these, STOP and re-plan:

- "Tests pass, syncing to SMB" without a real-path smoke run.
- "Adding a menu item now, will wire the dialog later."
- "The Colab notebook tests cover this, the EXE side should be fine."
- "I'll just emit `type: 'triangle'` in the JSON, the injector will
  figure it out" (it won't — see §3).
- "This is experimental but I'll commit it anyway."
- "Pushing straight to main since the change is small."
- "Bumping the version because we made some progress."
- Personal email or local path in a commit author or body.

When in doubt, ask the user before acting. Two minutes of clarification
beats hours of unwinding a wrong assumption from SMB and the tester's
machine.

---

## 8. Lessons from past failure patterns

Rules earned the hard way. Each one corresponds to a real session
where I got something wrong and the user had to course-correct.

### 8a. Quality knobs are user-judged baselines — don't silently move them.
RANDOM_SAMPLES, shape count per preset, edge density, posterize levels,
joint-polish iterations, lock_alpha defaults — the user A/B-judges
output against these. Never change a default "to be safe" or "to round
the number"; the next render will look different and the user will
have lost their baseline. If a knob needs changing, surface the
proposal first with the trade-off.

### 8b. Free-text answers to multi-choice questions override the option.
When the user picks "Option B" but ALSO types free-text that adds
nuance, the free-text is the actual answer. Don't take the option-pick
at face value. Re-read the free-text first; if it changes the meaning,
confirm before acting. Example from this session: user picked
"Push all three" but typed "use the SMB so I can test?" — the real
answer was "sync to SMB, hold the GitHub push".

### 8c. Don't combine `git commit && git push` in one bash invocation.
The permission classifier can block the compound for push-safety
reasons and leave the commit unmade, creating ambiguous state. Do
them separately so each gets evaluated on its own and the commit
lands even if the push needs auth.

### 8d. Audits must give brutally honest severity ratings.
When auditing something the user thinks is broken, return verdicts
with explicit BLOCKER / GAP / DRIFT / NIT tags. Don't soften.
"There's some room for improvement" is a useless answer; "the
Install button is a stub that always reports failure" is what the
user needs to plan around. If the audit comes back rosier than the
user expects, say so clearly too — "the JSON spec is actually
correct, the bug is 100% in the UI surface" saved a session of work
this time.

### 8e. Background-task output reads are easy to get wrong.
When you launch a long bash with `run_in_background: true`, the
output file accumulates over time. Reading it before the task
finishes returns partial output that looks like a hang or failure
when it isn't. If a smoke or test seems to hang, check `ps -ef` for
the actual process state before re-running. Don't `kill` a pytest
mid-run unless you've confirmed it's actually stuck.

### 8f. Test fixtures must round-trip through real load/save.
The clean-dialog test hung in this session because I built a JSON
fixture with `"format": "fd6.shapegen.v1"` — invented, not the real
`"fd6.shapes"` constant. The fixture passed isolated string-equality
asserts but exploded when fed through `FD6Document.from_dict()`. Any
fixture that touches the schema loader/saver must be verified by an
actual load/save call BEFORE asserting on its contents.

### 8g. Real MainWindow construction is the highest-fidelity smoke.
Source-grep tests catch "did the code get written" but miss
"does the menu actually render this item". When changing the GUI
surface, write or run at least one smoke that constructs the real
`MainWindow()` under offscreen Qt and walks the menu tree. The
test_gpu_bundle_gui.py grep tests are necessary but not sufficient.

### 8h. Avoid two `MainWindow()` constructions in a single pytest run.
Qt object lifetimes get tangled — `deleteLater()` queued by the first
window's cleanup fires during the second window's menu iteration and
raises "Internal C++ object already deleted". If a test needs two
constructions, either split it into two files or process the event
queue explicitly between them. Better: design the test to need only
one.

### 8i. The harness's `gitStatus` snapshot at session start is stale.
That snapshot reflects the moment the session began, not the current
state. If you're checking branch / unpushed-commits / file state,
run `git status` / `git log` fresh — don't trust the system-prompt
snapshot for decisions.

### 8j. The memory file's content is also point-in-time.
Memories named in `<system-reminder>` blocks are background context
that reflects what was true when written. File paths, branch names,
flag values cited in memories must be verified against the current
repo before being asserted as fact. The memory layer never auto-updates.

### 8k. When the user says "update X", default to comprehensive.
"Update CLAUDE.md" usually means more than the literal one thing
named — they want the doc to cover the full set of failure patterns
they've been seeing. After the obvious edit, scan the recent
conversation for related issues and add rules for those too. Asking
"anything else?" beats stopping short.

### 8l. Don't commit memory drift back to the repo.
Memory files live at
`/Users/kusanagi/.claude/projects/-Users-kusanagi-Development-ForzaDesigner6/memory/`
which is OUTSIDE both repos by design. They are session-spanning
context, not project artifacts. Never copy memory content into
repo-level docs unless the user explicitly says to.

### 8m. Sister repo (`ForzaDesigner6`) is someone else's fork.
The upstream is `tokyubevoxelverse/ForzaDesigner6`, NOT a personal
account. Never push or PR to that `origin` without explicit
instruction on which remote (this repo `forza-abyss-painter` lives
at `whykusanagi/forza-abyss-painter`). When working in
`ForzaDesigner6`, default to assuming pushes are blocked.
