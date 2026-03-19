#!/usr/bin/env python3
# postprocess/cli.py
from __future__ import annotations
import argparse, json, os, sys, copy
from typing import Any, List
from .pipeline import run_pipeline
import re

# Matches {token[:format]} where token is identifier with dots; avoids matching '{{' '}}'
_PLACEHOLDER = re.compile(r"""
    (?<!\{)            # not preceded by '{'  (so we don't hit '{{')
    \{
      ([A-Za-z_][A-Za-z0-9_\.]*)   # token (allow dots)
      (?:
        :([^}]+)        # optional format spec (no closing brace)
      )?
    \}
    (?!\})             # not followed by '}' (so we don't hit '}}')
""", re.VERBOSE)

import tomllib as toml 

# --- Small helpers
def deep_get(d: dict, path: str, default=None):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _assert_no_braces(path: str, label: str):
    if path and ("{" in path or "}" in path):
        raise SystemExit(f"{label} not fully expanded: {path}")

def deep_set(d: dict, path: str, value: Any):
    cur = d
    parts = path.split('.')
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value

def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, path))
        else:
            flat[path] = v
            flat[k] = v  # expose short key too
    return flat

def _format_once(obj, ctx: dict):
    if isinstance(obj, str):
        # only simple types in the format map
        fm = {k: v for k, v in ctx.items() if isinstance(v, (str, int, float))}
        try:
            return obj.format_map(fm)
        except KeyError:
            return obj  # leave for next pass
    if isinstance(obj, list):
        return [_format_once(x, ctx) for x in obj]
    if isinstance(obj, dict):
        return {k: _format_once(v, ctx) for k, v in obj.items()}
    return obj

def _build_context(cfg: dict) -> dict:
    ctx = {}
    # 1) High-priority sections
    vars_sec   = cfg.get("vars", {}) or {}
    global_sec = cfg.get("global", {}) or {}
    Global_sec = cfg.get("Global", {}) or {}
    ctx.update(vars_sec)
    ctx.update(global_sec)
    ctx.update(Global_sec)

    # 2) Allow dotted lookups but ONLY from vars/global/Global
    def _flatten_section(name: str, sec: dict) -> dict:
        # returns {"vars.nside": 512, "nside": 512, ...}
        flat = {}
        def rec(d, prefix):
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    rec(v, path)
                else:
                    flat[path] = v
                    # short alias for the last segment (only if not set yet)
                    short = path.split(".")[-1]
                    if short not in flat:
                        flat[short] = v
        rec(sec, name)
        return flat

    flat_allowed = {}
    flat_allowed.update(_flatten_section("vars", vars_sec))
    flat_allowed.update(_flatten_section("global", global_sec))
    flat_allowed.update(_flatten_section("Global", Global_sec))

    # Merge in dotted names and (first-win) short aliases
    for k, v in flat_allowed.items():
        if k not in ctx:
            ctx[k] = v

    # 3) DO NOT pull aliases from the whole cfg (matrix/constraints would clobber us)
    return ctx

def _format_value(val, fmt: str | None):
    if fmt is None:
        return f"{val}"
    try:
        return format(val, fmt)
    except Exception:
        # if bad format spec, just fall back to str
        return f"{val}"

def _interpolate_string(s: str, ctx: dict) -> str:
    def repl(m: re.Match):
        token = m.group(1)
        fmt   = m.group(2)
        # try exact token, then its last segment (e.g. vars.nside -> nside)
        if token in ctx:
            val = ctx[token]
        else:
            short = token.split(".")[-1]
            val = ctx.get(short, None)
        if val is None:
            # leave unresolved for the next pass
            return m.group(0)
        return _format_value(val, fmt)
    return _PLACEHOLDER.sub(repl, s)

def _interpolate_all(cfg: dict, max_passes: int = 5) -> dict:
    out = cfg
    for _ in range(max_passes):
        ctx = _build_context(out)
        changed = False
        def walk(x):
            nonlocal changed
            if isinstance(x, str):
                y = _interpolate_string(x, ctx)
                if y != x:
                    changed = True
                return y
            if isinstance(x, list):
                return [walk(v) for v in x]
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            return x
        new_out = walk(out)
        out = new_out
        if not changed:
            break
    return out

def _assert_no_real_placeholders(cfg: dict):
    # accept escaped braces '{{...}}'; only fail on real single-brace placeholders
    def walk(x):
        if isinstance(x, str):
            if _PLACEHOLDER.search(x):
                raise SystemExit(f"Unresolved template in config: {x}")
            return
        if isinstance(x, list):
            for v in x: walk(v)
            return
        if isinstance(x, dict):
            for v in x.values(): walk(v)
            return
    walk(cfg)

def resolve(cfg: dict, overrides: dict) -> dict:
    merged = deep_merge(cfg, overrides)
    resolved = _interpolate_all(merged, max_passes=6)
    _assert_no_real_placeholders(resolved)
    return resolved

def coerce_scalar(s: str) -> Any:
    # Turn "true"/"false"/numbers into proper types for --set
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s

def parse_overrides(pairs: List[str]) -> dict:
    out = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--set override must be key=val, got: {item}")
        k, v = item.split("=", 1)
        deep_set(out, k, coerce_scalar(v))
    return out

def apply_profile(cfg: dict, name: str | None) -> dict:
    if not name:
        return cfg
    prof = deep_get(cfg, f"profiles.{name}")
    if prof is None:
        raise SystemExit(f"Profile '{name}' not found under [profiles].")
    # Profiles may contain a 'modules' list and/or arbitrary key overrides
    base = copy.deepcopy(cfg)
    if isinstance(prof, dict):
        if 'modules' in prof:
            base['stages']['order'] = prof['modules']
        overrides = {k: v for k, v in prof.items() if k != 'modules'}
        if overrides:
            base = deep_merge(base, overrides)
    return base

def run_once(cfg: dict) -> int:
    _, _ = run_pipeline(cfg)
    return 0

def main():
    ap = argparse.ArgumentParser(
        prog="postprocess.cli",
        description="C-BASS post-processing orchestrator",
        epilog=(
            "Example: python -m postprocess.cli run --config defaults.toml "
            "--profile with_deconv --set vars.nside=512 vars.nside_out=256 vars.coords=C"
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="Path to TOML config")
    common.add_argument("--profile", help="Profile under [profiles] to apply")
    common.add_argument("--set", nargs="*", default=[], help="Override keys: key=val ...")

    sp_run = sub.add_parser("run", parents=[common], help="Run a single job")
    sp_run.add_argument("--dry-run", action="store_true", help="Load/resolve only, do not execute")

    args = ap.parse_args()

    # Load base config
    with open(args.config, "rb") as f:
        base_cfg = toml.load(f)

    # Apply profile (optional)
    cfg = apply_profile(base_cfg, args.profile)

    # Parse overrides
    overrides = parse_overrides(args.set)


    if args.cmd == "run":
        cfg = resolve(cfg, overrides)
        if args.dry_run:
            print(json.dumps(cfg, indent=2))
            return
        # Execute one job
        rc = run_once(cfg)
        sys.exit(rc)

if __name__ == "__main__":
    main()
