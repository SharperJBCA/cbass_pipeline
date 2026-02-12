#!/usr/bin/env python3
"""Run a matrix of NSIDE/configuration experiments.

This sweep always keeps MonoDipoleSub enabled while toggling
Deconvolution and SourceSubtraction.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib

from postprocess.pipeline import run_pipeline
from postprocess.run import resolve


COMBINATIONS = {
    "no_deconv_no_src": ["Masks", "MonoDipoleSub", "FinalMap"],
    "deconv_only": ["Masks", "MonoDipoleSub", "Deconvolution", "FinalMap"],
    "src_only": ["Masks", "MonoDipoleSub", "SourceSubtraction", "FinalMap"],
    "src_and_deconv": ["Masks", "MonoDipoleSub", "SourceSubtraction", "Deconvolution", "FinalMap"],
}


def parse_list_arg(value: str, cast=int) -> list:
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="C-BASS NSIDE/config sweep runner")
    parser.add_argument("--config", default="postprocess/configs/defaults.toml")
    parser.add_argument("--nsides", default="1024,512,256", help="Comma-separated input NSIDE values")
    parser.add_argument("--coords", default="G,C", help="Comma-separated coordinates (e.g. G,C)")
    parser.add_argument("--include-half", action="store_true", help="Also run nside_out=nside/2 when valid")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        base_cfg = tomllib.load(f)

    nsides = parse_list_arg(args.nsides, int)
    coords = parse_list_arg(args.coords, str)

    for nside in nsides:
        nside_out_values = [nside]
        if args.include_half and nside // 2 >= 1:
            nside_out_values.append(nside // 2)

        for nside_out in nside_out_values:
            if nside_out > nside:
                continue

            for coord in coords:
                for combo_name, stage_order in COMBINATIONS.items():
                    output_root = f"out_sweep/{combo_name}/ns{nside}_to_ns{nside_out}_{coord}"

                    cfg = copy.deepcopy(base_cfg)
                    cfg["stages"]["order"] = stage_order

                    overrides = {
                        "vars": {
                            "nside": nside,
                            "nside_out": nside_out,
                            "coords": coord,
                            "output_root": output_root,
                            "fig_root": f"{output_root}/figures",
                            "prefix": f"_{combo_name}",
                        }
                    }
                    resolved = resolve(cfg, overrides)

                    print(
                        f"[sweep] {combo_name}: nside={nside} nside_out={nside_out} "
                        f"coord={coord} output={resolved['FinalMap']['output']}"
                    )

                    if not args.dry_run:
                        pathlib.Path(output_root).mkdir(parents=True, exist_ok=True)
                        run_pipeline(resolved)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
