"""Build-time feature flags for the GUI layer.

A feature flag lives here when the underlying plumbing is staged across
multiple sessions / commits and we don't want partial UI surfacing in
shipped EXEs in the meantime. Flag flips happen in the same commit that
lands the real plumbing — never independently.

Why a module instead of an env var: build-time flags are part of the
shipped binary's contract. An env var would let users (or accidental
PowerShell sessions) toggle stub UI on without the runtime actually
being ready. Hardcoding here means the EXE's UI always matches the
EXE's wiring.
"""
from __future__ import annotations


# In-EXE GPU shape-gen (task #62). Phase 2 GUI scaffolding (runtime
# install dialog + generate dialog) is in the tree but its buttons
# currently surface "Phase 2 scaffolding — Phase 3 not yet shipped"
# stub messages and do nothing. Flip to True in the same commit that
# wires the real HTTP downloader + embedded-Python bootstrap +
# subprocess shape-gen runner (tasks #93-#96).
GPU_PHASE_3_AVAILABLE: bool = True
