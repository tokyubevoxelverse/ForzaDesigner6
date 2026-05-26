"""Assetto Corsa Titles support for FD6 — separate code path from Forza.

Submodules:
  profiles        — AC title profiles (ACC, ACE, AC Rally, AC original)
  livery_paths    — resolve the user's per-title livery folder
  car_catalog     — discover available car models (per title)
  texture_pipeline— image preprocessing for livery export
  livery_writer   — write decals.png + decals.json into a livery folder
  slot_planner    — auto-assign source image to decal slots

Strict isolation rule: this package must NEVER import from fd6.inject,
fd6.shapegen, or fd6.io. AC liveries are raster-export only; no geometrize,
no live memory writes, no Forza JSON schema.
"""
