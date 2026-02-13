# postprocess/modules/source_subtraction.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import os, json, logging
import numpy as np
import healpy as hp
from matplotlib import pyplot 
from . import Stage
from ..types import MapBundle, StageReport

log = logging.getLogger("SourceSubtraction")

# Try to use your existing helpers; fall back to a clear error if missing.
from ..source_subtraction import Catalogues, Mapper


class SourceSubtraction(Stage):
    name = "SourceSubtraction"

    # ---------- small helpers ----------

    def _load_input_IQU(
        self, stage_cfg: Dict[str, Any], full_cfg: Dict[str, Any], bundle: MapBundle
    ) -> Tuple[np.ndarray, str]:
        """
        Returns (IQU, coord). If bundle has a map, use it; otherwise load from disk.
        coord is 'G' or 'C' (uppercased).
        """
        coord = str(stage_cfg.get("coords") or full_cfg.get("vars", {}).get("coords") or "G").upper()

        if bundle.map is not None:
            m = np.asarray(bundle.map)
        else:
            inpath = stage_cfg.get("sky_map") or full_cfg.get("input", {}).get("map")
            if not inpath or not os.path.exists(inpath):
                raise RuntimeError(f"[{self.name}] input map not found: {inpath}")
            m = hp.read_map(inpath, field=(0,1,2), verbose=False)
            covp = (full_cfg.get("input", {}) or {}).get("cov")
            cov = hp.read_map(covp, field=(0,1,2,5), verbose=False)
            bundle.map = m
            bundle.cov = cov
        # Normalize to 3×N (I,Q,U) array internally
        if m.ndim == 1:
            I = m
            Q = np.full_like(I, hp.UNSEEN)
            U = np.full_like(I, hp.UNSEEN)
            m3 = np.vstack([I, Q, U])
        elif m.ndim == 2 and m.shape[0] in (3,):
            m3 = m
        elif m.ndim == 2 and m.shape[0] == 1:
            m3 = np.vstack([m[0], np.full_like(m[0], hp.UNSEEN), np.full_like(m[0], hp.UNSEEN)])
        else:
            raise RuntimeError(f"[{self.name}] unexpected map shape: {m.shape}")

        print(m.ndim,bundle.map.shape)
        print('SOURCE SUB',np.max(m3[1]))

        return m3, coord

    def _merge_back_I(self, new_I: np.ndarray, original_iqu: np.ndarray) -> np.ndarray:
        """Return IQU array with I replaced by new_I, preserving Q/U."""
        out = np.asarray(original_iqu).copy()
        out[0] = new_I
        return out

    @staticmethod
    def _ensure_dir(p: str) -> None:
        os.makedirs(p, exist_ok=True)

    @staticmethod
    def _read_mask_bool(path: str) -> np.ndarray:
        m = hp.read_map(path)
        return (m.astype(float) > 0.5)

    @staticmethod
    def _masked_subtract(I: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
        out = I.copy()
        valid = (I != hp.UNSEEN) & np.isfinite(I)
        out[valid] = I[valid] - tmpl[valid]
        out[~valid] = hp.UNSEEN
        return out

    @staticmethod
    def _apply_mask_rule(
        cat,
        rule: Dict[str, Any],
        mask_bool: np.ndarray,
        *,
        mask_coord: str = "G",
        source_coord: str = "G",
    ) -> Dict[str, Any]:
        """Apply a single threshold-mask rule to one catalogue and return stats."""
        before = int(cat.size)
        mode = rule.get("mode")
        limit = None
        if mode == "all":
            cat.mask_map(mask_bool, map_coord=mask_coord, source_coord=source_coord)
        elif mode == "min_flux":
            limit = float(rule["limit"])
            cat.mask_map(mask_bool, flux=cat.flux, lower_limit=limit, map_coord=mask_coord, source_coord=source_coord)
        after = int(cat.size)
        return {
            "mode": mode,
            "limit": limit,
            "before": before,
            "after": after,
            "removed": before - after,
        }

    @staticmethod
    def _plot_mask_tier_diagnostics(records: List[Dict[str, Any]], out_path: str) -> None:
        """Create a compact diagnostic table plot of tier-mask removals."""
        if len(records) == 0:
            return
        rows = []
        for r in records:
            lim = "-" if r.get("limit") is None else f"{float(r['limit']):g}"
            rows.append([
                r.get("mask", ""),
                r.get("catalogue", ""),
                r.get("mode", ""),
                lim,
                str(r.get("before", "")),
                str(r.get("removed", "")),
                str(r.get("after", "")),
            ])

        fig_h = max(3.0, 0.36 * len(rows) + 1.6)
        fig, ax = pyplot.subplots(figsize=(13, fig_h))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["Mask", "Catalogue", "Mode", "Limit [Jy]", "Before", "Removed", "After"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.2)
        ax.set_title("Source subtraction mask-tier diagnostics", fontsize=11, pad=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        pyplot.close(fig)

    @staticmethod
    def _haversine_theta(glat1, glon1, glat2, glon2):
        """Great-circle distance for arrays, inputs in degrees."""
        t1 = np.radians(glat1); p1 = np.radians(glon1)
        t2 = np.radians(glat2); p2 = np.radians(glon2)
        dt = t2 - t1
        dp = p2 - p1
        a = np.sin(dt/2)**2 + np.cos(t1)*np.cos(t2)*np.sin(dp/2)**2
        return 2*np.arcsin(np.sqrt(a))

    @classmethod
    def _common_sources(cls, cat1, cat2, nside_low=64):
        """Your 'common_sources' logic, lightly adapted."""
        m = np.zeros(12 * nside_low**2)
        p1 = hp.ang2pix(nside_low, *cat1.thetaphi)
        p2 = hp.ang2pix(nside_low, *cat2.thetaphi)
        m[p1] += 1; m[p2] += 1
        common = np.where(m > 1)[0]
        i1 = np.where(np.in1d(p1, common))[0]
        i2 = np.where(np.in1d(p2, common))[0]

        mask1 = np.ones(cat1.size, dtype=bool)
        mask2 = np.ones(cat2.size, dtype=bool)
        f2 = np.zeros(cat2.size); ef2 = np.zeros(cat2.size)

        for k, (gl1, gb1, fl1, efl1) in enumerate(zip(cat1.glon[i1], cat1.glat[i1], cat1.flux[i1], cat1.eflux[i1])):
            dist = cls._haversine_theta(gb1, gl1, cat2.glat[i2], cat2.glon[i2])
            if np.min(dist) * 60 > 1:
                continue
            j = i2[np.argmin(dist)]
            fl2 = cat2.flux[j]; efl2 = cat2.eflux[j]
            mask1[i1[k]] = False
            mask2[j] = False
            w = 1/efl1**2 + 1/efl2**2
            f2[j] = (fl1/efl1**2 + fl2/efl2**2)/w
            ef2[j] = np.sqrt(1/w)

        cat1.remove_sources(mask1)
        cat2.update_sources(~mask2, f2[~mask2], ef2[~mask2])
        return cat1, cat2

    @staticmethod
    def _merge_catalogues(cbass_cat, other_cat, fwhm_deg=1.0, nside=1024):
        """Your 'merge_catalogues' weight-down around C-BASS detections."""
        sigma = np.radians(fwhm_deg) / 2.355
        for i in range(cbass_cat.flux.size):
            dist = SourceSubtraction._haversine_theta(
                cbass_cat.glat[i], cbass_cat.glon[i],
                other_cat.glat,     other_cat.glon
            )
            w = np.exp(-0.5 * (dist**2) / sigma**2)
            other_cat.flux = other_cat.flux * (1 - w)
        return cbass_cat, other_cat

    # ---------- main run ----------

    def run(self, bundle: MapBundle, stage_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> Tuple[MapBundle, StageReport]:
        # --- config (pull from stage, then fall back to [vars]/defaults)
        coords = (stage_cfg.get("coords") or full_cfg.get("vars", {}).get("coords") or "G").upper()
        nside  = int(stage_cfg.get("nside") or bundle.nside or full_cfg.get("vars", {}).get("nside", 1024))

        out_dir = stage_cfg.get("output_dir") or os.path.join(full_cfg.get("vars", {}).get("output_root", "out"), "SourceSubtraction")
        fig_dir = stage_cfg.get("fig_dir") or None
        self._ensure_dir(out_dir)
        if fig_dir:
            self._ensure_dir(fig_dir)

        # beam/template knobs
        cbass_fwhm_deg = float(stage_cfg.get("cbass_fwhm_deg", 1.0))
        use_beam_model = bool(stage_cfg.get("use_beam_model", True))
        beam_model_filename = stage_cfg.get("beam_model_filename") or full_cfg.get("vars", {}).get("beam_model_filename")

        # catalogues
        cbass_cat = stage_cfg.get("cbass_catalogue", "")
        gb6_cat   = stage_cfg.get("gb6_catalogue", "")
        pmn_cat   = stage_cfg.get("pmn_catalogue", "")
        pmnt_cat  = stage_cfg.get("pmnt_catalogue", "")
        ming_cat  = stage_cfg.get("mingaliev_catalogue", "")

        # flux cuts
        cbass_max_flux = float(stage_cfg.get("cbass_max_flux", 10.0))
        cbass_min_flux = float(stage_cfg.get("cbass_min_flux", 0.61))
        gb6_min_flux   = float(stage_cfg.get("gb6_min_flux", 0.1))
        pmn_min_flux   = float(stage_cfg.get("pmn_min_flux", 0.1))
        pmnt_min_flux  = float(stage_cfg.get("pmnt_min_flux", 0.1))
        ming_min_flux  = float(stage_cfg.get("mingaliev_min_flux", 0.0))

        # sky cuts
        cbass_min_lat  = float(stage_cfg.get("cbass_min_latitude", 10.0))
        cbass_min_dec  = float(stage_cfg.get("cbass_min_declination", -6.0))

        # masks & regions
        exclude_regions: List[Dict[str, Any]] = stage_cfg.get("exclude_regions") or []
        threshold_mask_maps = stage_cfg.get("threshold_mask_maps") or []
        threshold_masks_coord = str(stage_cfg.get("threshold_masks_coord") or coords).upper()

        # naming/caching
        cat_stub   = stage_cfg.get("final_output_catalogue_name_stub", "cbass_dr1_ss_catalogue")
        tmpl_name  = stage_cfg.get("output_source_map_name", "source_map_merged_gb6_pmn_pmnt_mingaliev_cbass.fits")
        do_res_fit = bool(stage_cfg.get("do_residual_fits", True))

        # --- load input map
        iqu_in, coord_in = self._load_input_IQU(stage_cfg, full_cfg, bundle)
        if coord_in != coords:
            log.warning(f"[{self.name}] input coord {coord_in} != configured {coords}; proceeding without rotation.")

        nside_in = hp.get_nside(iqu_in[0])
        if nside_in != nside:
            log.warning(f"[{self.name}] bundle NSIDE {nside_in} != configured {nside}; using {nside_in}.")
            nside = nside_in

        # --- build/restore combined catalogue
        cat_path = os.path.join(out_dir, f"{cat_stub}.hdf5")
        if os.path.exists(cat_path):
            total = Catalogues.CBASS()
            total.load_file(cat_path, name="CBASS_GB6_PMN_PMN_Mingaliev")
        else:
            # build fresh
            # threshold masks (boolean True==masked)
            tms = []
            for tm in threshold_mask_maps:
                mf = str(tm["mask_filename"])
                m = self._read_mask_bool(mf)
                tms.append({"mask": m, "minimum_flux": float(tm.get("minimum_flux", 0.0))})

            # exclude holes (galactic)
            excl = np.zeros(12 * nside**2, dtype=bool)
            for reg in exclude_regions:
                pix = hp.query_disc(
                    nside,
                    hp.ang2vec((90.0 - reg["lat"]) * np.pi/180.0, reg["lon"] * np.pi/180.0),
                    reg["radius"] * np.pi/180.0
                )
                excl[pix] = True

            cb = Catalogues.CBASS(min_flux=cbass_min_flux); cb(cbass_cat); cb.mask_map(excl)
            # clamp very bright off-plane spots
            hi = (cb.flux > cbass_max_flux)
            if np.any(hi):
                mask_hi = np.ones(12 * nside**2, dtype=bool)
                for lon, lat in zip(cb.glon[hi], cb.glat[hi]):
                    if abs(lat) < cbass_min_lat:
                        continue
                    pix = hp.query_disc(nside, hp.ang2vec((90-lat)*np.pi/180., lon*np.pi/180.), 2*np.pi/180.)
                    mask_hi[pix] = False
                    for tm in tms:
                        tm["mask"][pix] = True
                cb.mask_map(mask_hi)

            cb.mask_declinations(declination_min=cbass_min_dec, declination_max=90)
            for tm in tms:
                if "minimum_flux" in tm:
                    cb.mask_map(tm["mask"], cb.flux, lower_limit=tm["minimum_flux"])
                else:
                    cb.mask_map(tm["mask"])

            mg = Catalogues.Mingaliev(min_flux=ming_min_flux); mg(ming_cat)
            g6 = Catalogues.GB6(min_flux=gb6_min_flux);       g6(gb6_cat)
            p1 = Catalogues.PMN(min_flux=pmn_min_flux);       p1(pmn_cat)
            p2 = Catalogues.PMN(min_flux=pmnt_min_flux);      p2(pmnt_cat)

            # ---- load Boolean mask for each entry and apply in sequence
            mask_tier_records: List[Dict[str, Any]] = []
            for tm in threshold_mask_maps:
                m = self._read_mask_bool(str(tm["mask_filename"]))
                mask_name = os.path.basename(str(tm["mask_filename"]))
                # Per-catalogue knobs (optional keys)
                if "cbass" in tm:
                    s = self._apply_mask_rule(cb, tm["cbass"], m, mask_coord=threshold_masks_coord)
                    mask_tier_records.append({"mask": mask_name, "catalogue": "cbass", **s})
                if "gb6" in tm:
                    s = self._apply_mask_rule(g6, tm["gb6"], m, mask_coord=threshold_masks_coord)
                    mask_tier_records.append({"mask": mask_name, "catalogue": "gb6", **s})
                if "pmn" in tm:
                    s = self._apply_mask_rule(p1, tm["pmn"], m, mask_coord=threshold_masks_coord)
                    mask_tier_records.append({"mask": mask_name, "catalogue": "pmn", **s})
                if "pmnt" in tm:
                    s = self._apply_mask_rule(p2, tm["pmnt"], m, mask_coord=threshold_masks_coord)
                    mask_tier_records.append({"mask": mask_name, "catalogue": "pmnt", **s})
                if "ming" in tm:
                    s = self._apply_mask_rule(mg, tm["ming"], m, mask_coord=threshold_masks_coord)
                    mask_tier_records.append({"mask": mask_name, "catalogue": "ming", **s})

            diag_plot = os.path.join(fig_dir or out_dir, "source_mask_tier_diagnostics.png")
            self._plot_mask_tier_diagnostics(mask_tier_records, diag_plot)

            # de-duplicate & merge weighting
            mg, g6 = self._common_sources(mg, g6)
            p1, g6 = self._common_sources(p1, g6)

            catalogue_sources = g6 + p1 + p2 + mg
            _ = self._merge_catalogues(cb,catalogue_sources , fwhm_deg=cbass_fwhm_deg, nside=nside)
            total = cb + catalogue_sources
            hp.mollview(excl,title='Exclude Mask')
            figure_path = os.path.join(out_dir, 'ss_exclude_mask.png')
            pyplot.savefig(figure_path)
            pyplot.close() 
            total.mask_map(excl); 
            total.clean_nan()
            total.write_file(cat_path)
            print("N sources:", total.size)
            print("Total flux [Jy]:", float(total.flux.sum()))
            print("Min/Med/Max Jy:", np.min(total.flux), np.median(total.flux), np.max(total.flux))

        # --- build/restore source template map
        tmpl_path = os.path.join(out_dir, tmpl_name)
        if os.path.exists(tmpl_path):
            src_map = hp.read_map(tmpl_path)
        else:
            beam_model = None
            if use_beam_model and beam_model_filename:
                beam_model = np.loadtxt(beam_model_filename)
            if coords == "C":
                map_lon, map_lat = Catalogues.rotate(total.glon, total.glat, coord=("G", "C"))
            else:
                map_lon, map_lat = total.glon, total.glat

            src_map = Mapper.pixel_space(
                total.flux, map_lon, map_lat,
                nside=nside_in, fwhm_deg=cbass_fwhm_deg,
                use_beam_model=bool(use_beam_model and (beam_model is not None)),
                beam_model=beam_model
            )
            hp.write_map(tmpl_path, src_map, overwrite=True)
            print('HELLO',tmpl_path.replace('.fits','.png'))
            hp.mollview(src_map,title='Source_map',norm='hist')
            pyplot.savefig(tmpl_path.replace('.fits','.png'))
            pyplot.close()

        # ensure NSIDE matches input
        if hp.get_nside(src_map) != nside_in:
            src_map = hp.ud_grade(src_map, nside_in, pess=False)
        src_map[iqu_in[0] == hp.UNSEEN] = hp.UNSEEN

        # --- subtract on I, preserve Q/U
        I_out = self._masked_subtract(iqu_in[0], src_map)
        m_out = self._merge_back_I(I_out, iqu_in)

        # --- manifest for provenance
        manifest = {
            "module": "SourceSubtraction",
            "nside": nside_in,
            "coords": coords,
            "cbass_fwhm_deg": cbass_fwhm_deg,
            "use_beam_model": use_beam_model,
            "beam_model_filename": beam_model_filename,
            "catalogue_hdf5": cat_path,
            "source_template_map": tmpl_path,
            "flux_cuts": {
                "cbass_max_flux": cbass_max_flux,
                "cbass_min_flux": cbass_min_flux,
                "gb6_min_flux": gb6_min_flux,
                "pmn_min_flux": pmn_min_flux,
                "pmnt_min_flux": pmnt_min_flux,
                "mingaliev_min_flux": ming_min_flux,
            },
            "sky_cuts": {
                "cbass_min_latitude": cbass_min_lat,
                "cbass_min_declination": cbass_min_dec,
            },
            "exclude_regions": exclude_regions,
            "threshold_masks": [tm.get("mask_filename", "") for tm in threshold_mask_maps],
        }
        with open(os.path.join(out_dir, "source_subtraction.manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # --- update bundle
        bundle.map = m_out
        bundle.nside = nside_in
        bundle.coords = coords

        # header cards (FinalMap will write them)
        hdr = bundle.headers
        hdr["SS_SSUB"] = True
        hdr["SS_MDEC"] = cbass_min_dec
        hdr["SS_CMAX"] = cbass_max_flux
        hdr["SS_CMIN"] = cbass_min_flux
        hdr["SS_GB6"]  = gb6_min_flux
        hdr["SS_PMN"]  = pmn_min_flux
        hdr["SS_MLV"]  = ming_min_flux
        hdr["SS_CATLG"] = os.path.basename(cat_path)

        # report
        nsrc = getattr(total, "size", np.nan)
        rep = StageReport(
            name=self.name,
            summary="point-source template subtracted from I; Q/U preserved",
            metrics={
                "coord": coords,
                "nside": nside_in,
                "num_sources": float(nsrc) if np.isfinite(nsrc) else None,
                "template_map": os.path.basename(tmpl_path),
            },
            figures=[],
        )
        return bundle, rep
