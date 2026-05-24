"""Command-line entry for Phase 2 discovery and injection.

Run with: `python -m forza_abyss_painter.inject <subcommand>`.

Subcommands:
  status              — show FH6 process info, region count, current patterns.json state
  find-pid            — print the PID of forzahorizon6.exe
  scan-float VALUE    — scan memory for a 4-byte float; saves results to .fd6_scan.json
  narrow VALUE        — keep only addresses whose current value matches; rewrites .fd6_scan.json
  dump ADDR [SIZE]    — hex/float dump around an address
  walk-struct ADDR N  — search backward from ADDR for a u32 == N (shape count); reports candidates
  find-refs ADDR      — find code locations that reference ADDR via RIP-relative addressing
  save-pattern NAME AOB [--offset N] [--no-relative]   — append a pattern to fh6_patterns.json
  test-injector       — try to attach + resolve patterns without injecting; reports each step

Workflow (typical):
  1. (in-game) Open FH6, load a 100-sphere template vinyl group at known position
  2. python -m forza_abyss_painter.inject status
  3. python -m forza_abyss_painter.inject scan-float 100.0          # known coord, e.g., a sphere's X
  4. (move sphere in-game to a different X)
  5. python -m forza_abyss_painter.inject narrow 200.0
  6. (repeat narrow until ~1 result)
  7. python -m forza_abyss_painter.inject dump <addr> 256           # identify struct fields
  8. python -m forza_abyss_painter.inject find-refs <struct_addr>   # get stable code-side AOBs
  9. python -m forza_abyss_painter.inject save-pattern shape_array_ref "<AOB>" --offset 3
  10. (edit fh6_patterns.json to fill in shape_struct.stride_bytes + fields manually based on dump)
  11. python -m forza_abyss_painter.inject test-injector
"""

from __future__ import annotations

import argparse
import datetime
import json
import struct
import sys
from pathlib import Path

from forza_abyss_painter.inject import discovery as disc
from forza_abyss_painter.inject.patterns_io import (
    DEFAULT_PATTERNS_PATH, PatternEntry, PatternsFile, load_patterns, save_patterns, has_usable_patterns
)
from forza_abyss_painter.inject.win_process import find_process_id


SCAN_STATE_PATH = Path.cwd() / ".fd6_scan.json"


def _resolve_pid(explicit_pid: int | None = None) -> int:
    if explicit_pid:
        return explicit_pid
    pid = disc.find_game_pid()
    if pid is None:
        print(f"[!] forzahorizon6.exe not found. Is FH6 running?", file=sys.stderr)
        sys.exit(2)
    return pid


def cmd_status(args) -> int:
    pid = disc.find_game_pid()
    if pid is None:
        print("forzahorizon6.exe: NOT RUNNING")
    else:
        info = disc.process_summary(pid)
        print(f"forzahorizon6.exe: PID {info.pid}")
        print(f"  committed regions: {info.region_count}")
        print(f"  private+writable bytes: {info.private_writable_bytes:,}")
        print(f"  image bytes:            {info.image_bytes:,}")
    pf = load_patterns()
    print(f"\npatterns file: {DEFAULT_PATTERNS_PATH}")
    print(f"  patterns:                {len(pf.patterns)}")
    print(f"  shape_struct.stride:     {pf.shape_struct.stride_bytes}")
    print(f"  shape_struct.fields:     {len(pf.shape_struct.fields)}")
    print(f"  vinyl_group meta keys:   {list(pf.vinyl_group.keys())}")
    print(f"  injector ready:          {has_usable_patterns(pf)}")
    if SCAN_STATE_PATH.exists():
        try:
            st = disc.ScanState.load(SCAN_STATE_PATH)
            print(f"\nlast scan: target={st.target} hits={len(st.hits)} pid={st.pid}")
        except Exception as e:
            print(f"\nlast scan: (unreadable: {e})")
    return 0


def cmd_find_pid(args) -> int:
    pid = disc.find_game_pid(args.exe or "forzahorizon6.exe")
    if pid is None:
        print("(not running)")
        return 1
    print(pid)
    return 0


def cmd_scan_float(args) -> int:
    pid = _resolve_pid(args.pid)
    print(f"Scanning PID {pid} for float ~= {args.value} (epsilon {args.epsilon})...")
    state = disc.scan_float(pid, args.value, args.epsilon, include_images=args.include_images)
    print(f"  {len(state.hits)} hits")
    if len(state.hits) <= 20:
        for h in state.hits:
            print(f"    0x{h.addr:016X}  {h.value}")
    elif len(state.hits) <= 200:
        for h in state.hits[:10]:
            print(f"    0x{h.addr:016X}  {h.value}")
        print(f"    ... ({len(state.hits) - 10} more)")
    state.save(SCAN_STATE_PATH)
    print(f"Saved to {SCAN_STATE_PATH}")
    return 0


def cmd_narrow(args) -> int:
    if not SCAN_STATE_PATH.exists():
        print(f"[!] No prior scan at {SCAN_STATE_PATH}. Run scan-float first.", file=sys.stderr)
        return 2
    state = disc.ScanState.load(SCAN_STATE_PATH)
    print(f"Narrowing {len(state.hits)} hits to those now ~= {args.value} (epsilon {args.epsilon})...")
    new_state = disc.narrow(state, args.value, args.epsilon)
    print(f"  {len(new_state.hits)} kept")
    for h in new_state.hits[:20]:
        print(f"    0x{h.addr:016X}  {h.value}")
    if len(new_state.hits) > 20:
        print(f"    ... ({len(new_state.hits) - 20} more)")
    new_state.save(SCAN_STATE_PATH)
    return 0


def cmd_dump(args) -> int:
    pid = _resolve_pid(args.pid)
    addr = int(args.addr, 0)
    rows = disc.dump_around(pid, addr, size=args.size, before=args.before)
    if not rows:
        print(f"[!] Could not read memory around 0x{addr:016X}", file=sys.stderr)
        return 1
    print(f"Dump around 0x{addr:016X} (before={args.before}, size={args.size}):")
    print(f"{'offset':<18} {'hex':<48} {'f32 (4)':<40} {'u32 (4)':<40} {'ascii':<16}")
    for r in rows:
        marker = " <-- " if r.offset == addr else ""
        f_str = " ".join(f"{x:>9.3g}" for x in r.as_f32)
        u_str = " ".join(f"{x:>9d}" for x in r.as_u32)
        print(f"0x{r.offset:016X} {r.raw_hex} {f_str} {u_str} {r.as_chars}{marker}")
    return 0


def cmd_walk_struct(args) -> int:
    pid = _resolve_pid(args.pid)
    addr = int(args.addr, 0)
    candidates = disc.find_struct_start(pid, addr, max_back=args.max_back, expected_shape_count=args.count)
    if not candidates:
        print(f"No u32 == {args.count} found within {args.max_back} bytes before 0x{addr:016X}.")
        return 1
    print(f"Candidates for struct-start u32 == {args.count}:")
    for cand_addr, dist in candidates:
        print(f"  0x{cand_addr:016X}   (-{dist} bytes from given addr)")
    return 0


def cmd_find_refs(args) -> int:
    pid = _resolve_pid(args.pid)
    addr = int(args.addr, 0)
    print(f"Scanning forzahorizon6.exe image for RIP-relative refs to 0x{addr:016X}...")
    refs = disc.find_code_references(pid, addr)
    if not refs:
        print("  (none found)")
        return 1
    for instr_addr, seed_bytes in refs[:20]:
        aob = " ".join(f"{b:02X}" for b in seed_bytes)
        # Wildcard the last 4 bytes (the RIP-relative displacement)
        bytes_list = aob.split()
        for i in range(max(0, len(bytes_list) - 4), len(bytes_list)):
            bytes_list[i] = "??"
        wildcarded = " ".join(bytes_list)
        print(f"  instr_at=0x{instr_addr:016X}  seed_aob={wildcarded}")
    if len(refs) > 20:
        print(f"  ... ({len(refs) - 20} more)")
    return 0


def cmd_save_pattern(args) -> int:
    pf = load_patterns()
    pf.patterns.append(PatternEntry(
        name=args.name,
        pattern=args.aob,
        offset_after_match=args.offset,
        relative_addressing=not args.no_relative,
        extra_offset=args.extra,
        note=args.note or "",
    ))
    pf.discovered_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = save_patterns(pf)
    print(f"Saved pattern '{args.name}' to {path}")
    return 0


def cmd_test_injector(args) -> int:
    from forza_abyss_painter.inject.fh6_injector import FH6Injector
    pid = _resolve_pid(args.pid)
    inj = FH6Injector(pid=pid)
    try:
        inj.attach()
        print(f"[OK] attached to PID {pid}")
    except Exception as e:
        print(f"[!] attach failed: {e}")
        return 1
    try:
        handle = inj.find_active_vinyl_group()
        print(f"[OK] resolved vinyl group:")
        print(f"    base_addr        0x{handle.base_addr:016X}")
        print(f"    layer_count      {handle.layer_count}")
        print(f"    shape_array_addr 0x{handle.shape_array_addr:016X}")
        print(f"    shape_stride     {handle.shape_stride}")
        if handle.meta:
            for k, v in handle.meta.items():
                print(f"    meta.{k:<14} {v}")
    except Exception as e:
        print(f"[!] find_active_vinyl_group failed: {e}")
        return 1
    finally:
        inj.detach()
    return 0


def cmd_inject(args) -> int:
    from forza_abyss_painter.inject.fh6_injector import FH6Injector
    from forza_abyss_painter.io.exporter import load_json
    pid = _resolve_pid(args.pid)
    try:
        doc = load_json(args.json_path)
    except Exception as e:
        print(f"[!] could not load {args.json_path}: {e}")
        return 1
    shapes = doc.materialize_shapes()
    print(f"Loaded {len(shapes)} shapes from {args.json_path} (image_size={doc.image_size}, profile={doc.profile})")

    inj = FH6Injector(pid=pid)
    try:
        inj.attach()
        print(f"[OK] attached to PID {pid}")
        handle = inj.find_active_vinyl_group()
        print(f"[OK] found {handle.layer_count} shape slots in active vinyl group")
        if len(shapes) > handle.layer_count:
            print(f"[!] {len(shapes)} shapes > {handle.layer_count} slots — load a larger template vinyl group "
                  f"(e.g., 3000 spheres for full quality).")
            return 2
        result = inj.inject(shapes, handle)
        if result.success:
            print(f"[OK] {result.message}")
            return 0
        else:
            print(f"[!] inject failed: {result.message}")
            return 1
    except Exception as e:
        print(f"[!] {type(e).__name__}: {e}")
        return 1
    finally:
        inj.detach()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m forza_abyss_painter.inject", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("status", help="show injector & process state")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("find-pid", help="print PID of forzahorizon6.exe")
    sp.add_argument("--exe", default="forzahorizon6.exe")
    sp.set_defaults(func=cmd_find_pid)

    sp = sub.add_parser("scan-float", help="scan memory for a float")
    sp.add_argument("value", type=float)
    sp.add_argument("--epsilon", type=float, default=0.01)
    sp.add_argument("--pid", type=int, default=None)
    sp.add_argument("--include-images", action="store_true", help="also scan module memory (usually unnecessary)")
    sp.set_defaults(func=cmd_scan_float)

    sp = sub.add_parser("narrow", help="narrow last scan to addresses now equal to value")
    sp.add_argument("value", type=float)
    sp.add_argument("--epsilon", type=float, default=0.01)
    sp.set_defaults(func=cmd_narrow)

    sp = sub.add_parser("dump", help="hex/float dump around address")
    sp.add_argument("addr")
    sp.add_argument("--size", type=int, default=256)
    sp.add_argument("--before", type=int, default=64)
    sp.add_argument("--pid", type=int, default=None)
    sp.set_defaults(func=cmd_dump)

    sp = sub.add_parser("walk-struct", help="search backward for a struct-start matching a known shape count")
    sp.add_argument("addr")
    sp.add_argument("count", type=int, help="expected shape count (e.g., 100 if template has 100 spheres)")
    sp.add_argument("--max-back", type=int, default=4096)
    sp.add_argument("--pid", type=int, default=None)
    sp.set_defaults(func=cmd_walk_struct)

    sp = sub.add_parser("find-refs", help="find code refs (RIP-relative) to an address")
    sp.add_argument("addr")
    sp.add_argument("--pid", type=int, default=None)
    sp.set_defaults(func=cmd_find_refs)

    sp = sub.add_parser("save-pattern", help="append a pattern to fh6_patterns.json")
    sp.add_argument("name")
    sp.add_argument("aob", help="AOB pattern string, e.g. '48 8B 05 ?? ?? ?? ??'")
    sp.add_argument("--offset", type=int, default=3, help="bytes from match start to disp32 (default 3)")
    sp.add_argument("--no-relative", action="store_true", help="treat disp as absolute, not RIP-relative")
    sp.add_argument("--extra", type=int, default=0, help="byte offset added to resolved address")
    sp.add_argument("--note", default="")
    sp.set_defaults(func=cmd_save_pattern)

    sp = sub.add_parser("test-injector", help="try to attach + resolve patterns; does not write")
    sp.add_argument("--pid", type=int, default=None)
    sp.set_defaults(func=cmd_test_injector)

    sp = sub.add_parser("inject", help="inject an FD6 shapes JSON into a running FH6 vinyl group")
    sp.add_argument("json_path", help="path to FD6 .json file")
    sp.add_argument("--pid", type=int, default=None)
    sp.set_defaults(func=cmd_inject)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
