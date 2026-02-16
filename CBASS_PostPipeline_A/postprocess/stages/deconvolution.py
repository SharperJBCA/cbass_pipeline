# postprocess/stages/deconvolution.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import os, numpy as np, healpy as hp, logging
from . import Stage
from ..types import MapBundle, StageReport
from matplotlib import pyplot
from ..deconv.core import (
    build_transfer_functions, apply_transfer_to_maps,
    apply_transfer_to_cov, dec_mask
)

class Deconvolution(Stage):
    name = "Deconvolution"

    def run(self, bundle: MapBundle, cfg: Dict[str, Any], full: Dict[str, Any]) -> Tuple[MapBundle, StageReport]:
        fig_dir = cfg.get("fig_dir") or None
        if fig_dir: os.makedirs(fig_dir, exist_ok=True)

        # Load maps
        if bundle.map is None:
            path = (full.get("input", {}) or {}).get("map")
            covp = (full.get("input", {}) or {}).get("cov")
            if not path or not covp: raise RuntimeError("[Deconvolution] need input.map and input.cov")
            arr = hp.read_map(path, field=(0,1,2), verbose=False)
            cov = hp.read_map(covp, field=(0,1,2,5), verbose=False)
        else:
            arr = bundle.map
            cov = bundle.cov
        if bundle.cov is None:
            covp = (full.get("input", {}) or {}).get("cov")
            cov = hp.read_map(covp, field=(0,1,2,5), verbose=False)
            bundle.cov = cov 
        I,Q,U = (arr if arr.ndim==2 else np.vstack([arr, np.zeros_like(arr), np.zeros_like(arr)]))
        nside = hp.get_nside(I)
        nside_out = int(cfg.get("nside_out", nside))
        lmax = int(cfg.get("beam_function_lmax", 3*nside - 1))
        apodise_inpaint_flag = bool(cfg.get("use_edge_inpainting", False))
        apply_transfer_function = bool(cfg.get("apply_transfer_function", True))

        # Build transfer
        beam_file = cfg.get("beam_filename")
        if (isinstance(beam_file, str) and beam_file.lower() == "none") or not beam_file:
            beam_file = None
        R0, R2, pixwin = build_transfer_functions(
            beam_filename=beam_file,
            output_fwhm_deg=float(cfg.get("output_fwhm", 1.0)),
            nside_in=nside, nside_out=nside_out, lmax=lmax,
            beam_format=cfg.get("beam_format", "THETA"),
            beam_normalise=bool(cfg.get("beam_normalise", False)),
            apply_transfer_function=apply_transfer_function
        )

        # plot R0 
        pyplot.plot(R0)
        pyplot.plot(R2)
        pyplot.xscale('log')
        pyplot.savefig(f'{fig_dir}/beam_transfer_function.png')
        pyplot.close()
        # Apply
        hp.mollview(Q,norm='hist')
        pyplot.savefig(f'{fig_dir}/input_Q.png')
        pyplot.close()
        hp.mollview(U,norm='hist')
        pyplot.savefig(f'{fig_dir}/input_U.png')
        pyplot.close()

        dI, dQ, dU = apply_transfer_to_maps(I,Q,U, bundle.coords, R0,R2, pixwin, lmax=lmax, nside_out=nside_out, apodise_inpaint=apodise_inpaint_flag)

        hp.mollview(dI,norm='hist')
        pyplot.savefig(f'{fig_dir}/deconvolved_I.png')
        pyplot.close()
        hp.mollview(dQ,norm='hist')
        pyplot.savefig(f'{fig_dir}/deconvolved_Q.png')
        pyplot.close()
        hp.mollview(dU,norm='hist')
        pyplot.savefig(f'{fig_dir}/deconvolved_U.png')
        pyplot.close()


        covII,covQQ,covUU,covQU = cov
        dII,dQQ,dUU,dQU = apply_transfer_to_cov(
            covII,covQQ,covUU,covQU, R0,R2, pixwin, lmax=lmax, nside_out=nside_out
        )

        # Dec mask: always follow the coordinate system of the in-memory map.
        # A stale/incorrect stage config here can otherwise apply a Galactic
        # declination cut to Celestial maps (or vice versa).
        map_coord = str(bundle.coords or cfg.get("map_coord") or full.get("vars", {}).get("coords") or "G").upper()
        m = dec_mask(nside_out, coord=map_coord, min_dec=float(cfg.get("min_dec", -13)))
        for mapp in (dI,dQ,dU,dII,dQQ,dUU,dQU):
            mapp[m==0] = hp.UNSEEN

        # Update bundle
        bundle.map = np.asarray([dI,dQ,dU])
        bundle.nside = nside_out
        bundle.cov = np.asarray([dII,dQQ,dUU,dQU])


        # Header-ish flags for FinalMap
        bundle.headers["DCONV"] = bool(beam_file is not None)
        bundle.headers["BL_FILE"] = os.path.basename(beam_file) if beam_file else "none"
        bundle.headers["NSIDE_IN"] = nside
        bundle.headers["NSIDE_OUT"] = nside_out
        bundle.headers["FWHM_OUT"] = float(cfg.get("output_fwhm", 1.0))
        bundle.headers["L_MAX"] = lmax
        bundle.headers["PIXWIN_APPLIED"] = True
        bundle.headers["DECMIN"] = float(cfg.get("min_dec", -13))

        rep = StageReport(
            name=self.name,
            summary=f"deconvolved→{bundle.headers['FWHM_OUT']}°; ns {nside}→{nside_out}; beam={'yes' if beam_file else 'no'}",
            metrics=dict(
                nside_in=nside, nside_out=nside_out, lmax=lmax,
                beam=bool(beam_file), beam_file=bundle.headers["BL_FILE"],
            ),
            figures=[] if not fig_dir else [  # add some optional plots you already had
                os.path.join(fig_dir, "deconvolved_I.png"),
                os.path.join(fig_dir, "deconvolved_Q.png"),
                os.path.join(fig_dir, "deconvolved_U.png"),
            ],
        )
        return bundle, rep
