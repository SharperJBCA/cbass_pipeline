from __future__ import annotations
from typing import Dict, Any, Tuple
import os
import numpy as np
import healpy as hp

from . import Stage
from ..types import MapBundle, StageReport
from ..zerolevel.core import run_zero_level


class MonopoleDipole(Stage):
    name = "MonoDipoleSub"

    def _load_input_I(self, stage_cfg: Dict[str, Any], full_cfg: Dict[str, Any], bundle: MapBundle) -> Tuple[np.ndarray, str]:
        """
        Returns (I_map, coord) — loads from bundle if present,
        else from [input].map. Accepts 1-comp I or 3-comp IQU; we operate on I.
        coord must be 'G' or 'C'.
        """
        coord = str(stage_cfg.get("input_coordinate_system") or full_cfg.get("vars", {}).get("coords") or "G").upper()

        if bundle.map is not None:
            m = bundle.map
        else:
            inpath = (stage_cfg.get("sky_map") or full_cfg.get("input", {}).get("map"))
            if not inpath or not os.path.exists(inpath):
                raise RuntimeError(f"[{self.name}] input map not found: {inpath}")
            m = hp.read_map(inpath, field=(0,1,2), verbose=False)  # returns ndarray (npix,) or (ncomp,npix)
            bundle.map = m
            covp = (full_cfg.get("input", {}) or {}).get("cov")
            cov = hp.read_map(covp, field=(0,1,2,5), verbose=False)
            bundle.cov = cov
        m = np.asarray(m)

        if m.ndim == 1:
            I = m
        elif m.ndim == 2:
            I = m[0].copy()
        else:
            raise RuntimeError(f"[{self.name}] unexpected map shape: {m.shape}")

        return I, coord

    def _merge_back(self, I_new: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Return map array with I component replaced, preserving Q/U if present."""
        if original.ndim == 1:
            return I_new
        out = np.asarray(original).copy()
        out[0] = I_new
        return out

    def run(self, bundle: MapBundle, stage_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> Tuple[MapBundle, StageReport]:
        # ---- read config
        remove_cmb = bool(stage_cfg.get("remove_cmb_dipole", True))
        offset_mK  = float(stage_cfg.get("offset_mK", 42.0))
        offset_err = float(stage_cfg.get("offset_mK_err", 6.5))  # stored for headers only
        dipole_mK  = float(stage_cfg.get("dipole_amp_mK", 3.36208))
        min_dec    = float(stage_cfg.get("min_dec_deg", 0.0))    # allow override; default was 0 in your code
        fig_dir    = stage_cfg.get("fig_dir") or None

        # ---- load input
        # Keep the full original so we can reinsert Q/U later if needed
        I_in, coord = self._load_input_I(stage_cfg, full_cfg, bundle)
        nside = hp.get_nside(I_in)
        original_map = bundle.map

        # ---- do the work (pure function)
        res = run_zero_level(
            input_map=I_in, coord=coord, remove_cmb=remove_cmb,
            dipole_amp_mK=dipole_mK, offset_mK=offset_mK,
            min_dec_deg=min_dec, fig_dir=fig_dir
        )

        # ---- merge back and update bundle
        if original_map is None:
            out_map = res.map_I
        else:
            out_map = self._merge_back(res.map_I, original_map)

        bundle.map = out_map
        print('MONO',np.max(bundle.map[1]),bundle.map.shape)
        bundle.nside = nside
        bundle.coords = coord

        # header-ish info (FinalMap stage can write these)
        hdr = bundle.headers
        hdr["MD_MONO"] = True
        hdr["MD_DIPO"] = bool(remove_cmb)
        hdr["MD_OFST"] = float(offset_mK)
        hdr["MD_EOFS"] = float(offset_err)
        hdr["MD_MINV"] = float(res.metrics.get("min_above0dec_mK", np.nan))
        hdr["MD_DPAP"] = float(res.metrics.get("dipole_amp_mK", dipole_mK))
        hdr["MD_DGLN"] = 264.021
        hdr["MD_DGLT"] = 48.253

        # report
        rep = StageReport(
            name=self.name,
            summary="monopole set and CMB dipole subtracted" if remove_cmb else "monopole set (dipole kept)",
            metrics={
                "coord": coord,
                "nside": nside,
                **res.metrics,
            },
            figures=list(res.figures),
        )

        return bundle, rep
