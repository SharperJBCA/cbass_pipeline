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
    name = "DeconvAndPixwin"

    def update_bundle(self, bundle, dI, dQ, dU, dII, dQQ, dUU, dQU, did_beam_deconv, beam_file, nside, nside_out, lmax,fig_dir, cfg):
        # Update bundle
        bundle.map = np.asarray([dI,dQ,dU])
        bundle.nside = nside_out
        bundle.cov = np.asarray([dII,dQQ,dUU,dQU])


        # Header-ish flags for FinalMap
        bundle.headers["DCONV"] = did_beam_deconv
        if did_beam_deconv:
            bundle.headers["BL_FILE"] = os.path.basename(beam_file)
            bundle.headers["FWHM_OUT"] = float(cfg.get("output_fwhm", 1.0))
        else:
            bundle.headers.pop("BL_FILE", None)
            bundle.headers.pop("FWHM_OUT", None)
        bundle.headers["NSIDE_IN"] = nside
        bundle.headers["NSIDEOUT"] = nside_out
        bundle.headers["PIXWIN"] = True
        bundle.headers["DECMIN"] = float(cfg.get("min_dec", -13)) if did_beam_deconv else -15.6

        summary_mode = (
            f"deconvolved:{bundle.headers['FWHM_OUT']}°"
            if did_beam_deconv
            else "pixel-window only (no beam deconvolution)"
        )
        beam_file_metric = bundle.headers.get("BL_FILE")

        rep = StageReport(
            name=self.name,
            summary=f"{summary_mode}; ns {nside}→{nside_out}; beam={'yes' if did_beam_deconv else 'no'}",
            metrics=dict(
                nside_in=nside, nside_out=nside_out, lmax=lmax,
                beam=did_beam_deconv, beam_file=beam_file_metric,
            ),
            figures=[] if not fig_dir else [  # add some optional plots you already had
                os.path.join(fig_dir, "deconvolved_I.png"),
                os.path.join(fig_dir, "deconvolved_Q.png"),
                os.path.join(fig_dir, "deconvolved_U.png"),
            ],
        )
        return bundle, rep 

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
        beam_file = cfg.get("beam_filename")

        # If no beam and no change in nside, skip
        if (nside == nside_out) and (apply_transfer_function == False):
            covII,covQQ,covUU,covQU = cov
            did_beam_deconv = bool(apply_transfer_function and (beam_file is not None))
            bundle, rep = self.update_bundle(bundle, I,Q,U, covII,covQQ,covUU,covQU, did_beam_deconv, beam_file, nside, nside_out, lmax,fig_dir, cfg)
            return bundle, rep

        # Build transfer
        if (isinstance(beam_file, str) and beam_file.lower() == "none") or not beam_file:
            beam_file = None
        did_beam_deconv = bool(apply_transfer_function and (beam_file is not None))
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
        if did_beam_deconv:
            map_coord = str(bundle.coords or cfg.get("map_coord") or full.get("vars", {}).get("coords") or "G").upper()
            m = dec_mask(nside_out, coord=map_coord, min_dec=float(cfg.get("min_dec", -13)))
            for mapp in (dI,dQ,dU,dII,dQQ,dUU,dQU):
                mapp[m==0] = hp.UNSEEN

        bundle, rep = self.update_bundle(bundle, dI, dQ, dU, dII, dQQ, dUU, dQU, did_beam_deconv, beam_file, nside, nside_out, lmax,fig_dir, cfg)

        return bundle, rep
