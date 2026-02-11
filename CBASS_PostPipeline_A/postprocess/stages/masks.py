# postprocess/stages/masks.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Callable, List
from dataclasses import dataclass
import os
import numpy as np
import healpy as hp

try:
    import tomllib as toml  # py311
except Exception:
    import tomli as toml

from . import Stage
from ..types import MapBundle, StageReport

# ---- import your existing functions (geometry, thresholds, etc.)
# Keep them in a single module (e.g., postprocess/masks/define_regions.py)
from ..masks.define_regions import (
    smooth_mask,
    planck_source_catalogue,
    query_arc_pixels,
    query_elliptical_pixels,
    threshold_mask,
    remove_pixels_with_masked_neighbour,
    remove_overlap_pixels,
    subtract_elliptical_pixels,
    latitude_cut,
    bad_data_mask,
    remove_bad_data_mask,
)

# ---- function registry (explicit; no globals() spelunking)
FN_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "smooth_mask": smooth_mask,
    "planck_source_catalogue": planck_source_catalogue,
    "query_arc_pixels": query_arc_pixels,
    "query_elliptical_pixels": query_elliptical_pixels,
    "threshold_mask": threshold_mask,
    "remove_pixels_with_masked_neighbour": remove_pixels_with_masked_neighbour,
    "remove_overlap_pixels": remove_overlap_pixels,
    "subtract_elliptical_pixels": subtract_elliptical_pixels,
    "latitude_cut": latitude_cut,
    "bad_data_mask": bad_data_mask,
    "remove_bad_data_mask": remove_bad_data_mask,
}

@dataclass
class RegionDef:
    name: str
    cmap: str
    processes: List[Dict[str, Any]]

class Masks(Stage):
    name = "Masks"

    def run(self, bundle: MapBundle, stage_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> Tuple[MapBundle, StageReport]:
        """
        Expected stage_cfg keys (after interpolation):
          - mask_definitions: path to TOML with regions/processes
          - nside: int (target nside for masks)
          - sky_map: optional path to input map; if omitted we use [input].map
          - smooth_sky_map: float degrees (optional)
          - output_dir: optional; defaults to "{vars.output_root}/masks"
          - write_regions_h5: bool (optional, default True)
          - write_figures: bool (optional, default False)
        """
        # ---- resolve basics
        mask_toml = stage_cfg.get("mask_definitions")
        if not mask_toml or not os.path.exists(mask_toml):
            raise RuntimeError(f"[Masks] mask_definitions not found: {mask_toml}")
        target_nside = int(stage_cfg.get("nside") or (bundle.nside or 128))

        out_root = stage_cfg.get("output_dir") or os.path.join(full_cfg.get("vars", {}).get("output_root", "out"), "masks")
        os.makedirs(out_root, exist_ok=True)

        # ---- load sky map once (for threshold/bad-data steps)
        sky_map_path = stage_cfg.get("sky_map") or full_cfg.get("input", {}).get("map")
        print(sky_map_path)
        if not sky_map_path or not os.path.exists(sky_map_path):
            raise RuntimeError(f"[Masks] sky_map not found: {sky_map_path}")

        sky_map = hp.read_map(sky_map_path, verbose=False)
        base_nside = hp.get_nside(sky_map)
        smooth_deg = float(stage_cfg.get("smooth_sky_map") or 0.0)
        if smooth_deg > 0:
            sky_map = hp.smoothing(sky_map, fwhm=np.deg2rad(smooth_deg))
        sky_map = hp.ud_grade(sky_map, target_nside) if base_nside != target_nside else sky_map

        # ---- load region definitions
        with open(mask_toml, "rb") as f:
            raw = toml.load(f)
        regions: List[RegionDef] = []
        for key, block in raw.items():
            regions.append(RegionDef(
                name=block["name"],
                cmap=block.get("cmap", "viridis"),
                processes=block.get("processes", []),
            ))

        # ---- runtime variables available to processes
        runtime = {
            "nside": target_nside,
            "sky_map": sky_map,            # already smoothed/downgraded
        }

        # ---- build masks
        written: List[str] = []
        metrics = {"regions": [], "nside": target_nside}

        for r in regions:
            # start with all pixels, then apply each process
            pixels = np.arange(12 * target_nside * target_nside, dtype=int)
            for step in r.processes:
                fn_name = step.get("function")
                if fn_name not in FN_REGISTRY:
                    raise RuntimeError(f"[Masks] Unknown function '{fn_name}' in region '{r.name}'")
                fn = FN_REGISTRY[fn_name]
                # Merge step params with runtime defaults
                params = {**{k: v for k, v in step.items() if k != "function"}, **runtime}
                # Some funcs expect 'map', honor that name
                if "map" in fn.__code__.co_varnames and "map" not in params:
                    params["map"] = sky_map
                pixels = fn(pixels, **params)

            # Write FITS
            mask_vec = np.zeros(12 * target_nside * target_nside, dtype=np.float32)
            mask_vec[pixels] = 1.0
            fout = os.path.join(out_root, f"{r.name.replace(' ', '')}_{target_nside:04d}.fits")
            hp.write_map(fout, mask_vec, overwrite=True)
            written.append(fout)
            metrics["regions"].append({"name": r.name, "pixels": int(pixels.size), "file": fout})

            # Optional: preview figure (off by default)
            if bool(stage_cfg.get("write_figures", False)):
                try:
                    import matplotlib.pyplot as plt
                    m = np.zeros_like(mask_vec)
                    m[pixels] = 1.0
                    hp.mollview(m, title=r.name)
                    fig = plt.gcf()
                    figpath = fout.replace(".fits", ".png")
                    fig.savefig(figpath, dpi=200)
                    plt.close(fig)
                except Exception:
                    pass

        # ---- HDF archive of regions (optional)
        if bool(stage_cfg.get("write_regions_h5", True)):
            try:
                import h5py
                h5path = os.path.join(out_root, f"regions_nside{target_nside:04d}.h5")
                with h5py.File(h5path, "w") as h:
                    for r in metrics["regions"]:
                        grp = h.create_group(r["name"])
                        # store pixel indices to reconstruct quickly
                        # reloaders can get them back via np.where(mask==1)
                        grp.create_dataset("file", data=np.string_(r["file"]))
                        grp.attrs["nside"] = target_nside
                metrics["regions_h5"] = h5path
            except Exception:
                pass

        # ---- update bundle minimally (don’t stuff the map in here)
        bundle.history.append(f"Masks: {len(written)} masks @ NSIDE={target_nside}")
        # You can stash a pointer to masks for later stages if helpful:

        rep = StageReport(
            name=self.name,
            summary=f"created {len(written)} masks",
            metrics=metrics,
            figures=[p.replace(".fits", ".png") for p in written] if stage_cfg.get("write_figures") else [],
        )
        return bundle, rep
