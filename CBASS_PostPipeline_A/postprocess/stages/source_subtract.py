# postprocess/modules/source_subtraction.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import os, json, logging
import numpy as np
import healpy as hp
from matplotlib import pyplot
from scipy.spatial import cKDTree
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

    # ---------- residual assessment ----------

    def _assess_residuals(
        self,
        I_residual: np.ndarray,
        src_map: np.ndarray,
        catalogue,
        nside: int,
        fwhm_deg: float,
        coords: str,
        out_dir: str,
        fig_dir: Optional[str],
        plane_mask: Optional[np.ndarray] = None,
        mask_coord: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess subtraction quality per beam-scale source cluster.

        Uses aperture photometry on the residual map, reporting fluxes in Jy.
        Negative S_resid = over-subtraction; positive = under-subtraction.

        Parameters
        ----------
        plane_mask : bool array (True = masked/galactic plane).
            Clusters whose centroid falls in a masked pixel are skipped.
        """
        import h5py

        # K -> Jy conversion at 4.76 GHz
        KB = 1.380649e-23
        C = 2.99792458e8
        nu = 4.76e9
        jy_per_sr_per_K = 2 * KB * (nu / C) ** 2 * 1e26
        pix_sr = hp.nside2pixarea(nside)
        K_to_Jy_per_pix = jy_per_sr_per_K * pix_sr

        fwhm_rad = np.radians(fwhm_deg)
        inner_rad = 0.5 * fwhm_rad
        annulus_inner = 0.75 * fwhm_rad
        annulus_outer = 1.5 * fwhm_rad

        # --- 1. Group sources into beam-scale clusters via cKDTree ---
        if coords == "C":
            src_lon, src_lat = Catalogues.rotate(catalogue.glon, catalogue.glat, coord=("G", "C"))
        else:
            src_lon, src_lat = catalogue.glon, catalogue.glat

        theta = np.radians(90.0 - src_lat)
        phi = np.radians(src_lon)
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        uvecs = np.column_stack((st * cp, st * sp, ct))

        chord_link = 2 * np.sin(fwhm_rad / 2)
        tree = cKDTree(uvecs)
        pairs = tree.query_pairs(r=chord_link, output_type="ndarray")

        # Union-find for clustering
        n_src = catalogue.size
        parent = np.arange(n_src)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in pairs:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        roots = np.array([find(i) for i in range(n_src)])
        unique_roots = np.unique(roots)

        # --- 2-3. Per-cluster aperture photometry ---
        valid = (
            (I_residual != hp.UNSEEN) & np.isfinite(I_residual) &
            (src_map != hp.UNSEEN) & np.isfinite(src_map)
        )
        n_skipped_plane = 0
        cluster_records = []
        for root in unique_roots:
            members = np.where(roots == root)[0]
            fluxes = catalogue.flux[members]
            brightest_idx = members[np.argmax(fluxes)]

            # Flux-weighted centroid
            w = fluxes / fluxes.sum()
            cx = np.sum(w * uvecs[members, 0])
            cy = np.sum(w * uvecs[members, 1])
            cz = np.sum(w * uvecs[members, 2])
            norm = np.sqrt(cx**2 + cy**2 + cz**2)
            centroid_vec = np.array([cx / norm, cy / norm, cz / norm])

            # Skip clusters on the galactic plane
            if plane_mask is not None:
                if mask_coord and mask_coord != coords:
                    # Rotate centroid from working frame to mask frame
                    c_theta = np.arccos(np.clip(centroid_vec[2], -1, 1))
                    c_phi = np.arctan2(centroid_vec[1], centroid_vec[0])
                    rot = hp.rotator.Rotator(coord=[coords, mask_coord])
                    c_theta_m, c_phi_m = rot(c_theta, c_phi)
                    cpix = hp.ang2pix(nside, c_theta_m, c_phi_m)
                else:
                    cpix = hp.vec2pix(nside, *centroid_vec)
                if plane_mask[cpix]:
                    n_skipped_plane += 1
                    continue

            # Inner disc and annulus pixels
            inner_pix = hp.query_disc(nside, centroid_vec, inner_rad)
            outer_pix = hp.query_disc(nside, centroid_vec, annulus_outer)
            inner_set = set(inner_pix)
            annulus_pix = np.array([p for p in outer_pix if p not in inner_set])
            # Filter to annulus inner boundary
            if annulus_pix.size > 0:
                inner_boundary_pix = hp.query_disc(nside, centroid_vec, annulus_inner)
                inner_boundary_set = set(inner_boundary_pix)
                annulus_pix = np.array([p for p in annulus_pix if p not in inner_boundary_set])

            # Valid pixels only
            inner_valid = inner_pix[valid[inner_pix]] if inner_pix.size > 0 else np.array([], dtype=int)
            annulus_valid = annulus_pix[valid[annulus_pix]] if annulus_pix.size > 0 else np.array([], dtype=int)

            if inner_valid.size == 0:
                continue

            # Aperture photometry on residual
            res_inner = I_residual[inner_valid]
            res_annulus = I_residual[annulus_valid]
            bg = np.median(res_annulus) if annulus_valid.size > 0 else 0.0
            S_resid = float((np.sum(res_inner) - len(res_inner) * bg) * K_to_Jy_per_pix)

            # Aperture photometry on template (normalisation reference)
            tmpl_inner = src_map[inner_valid]
            tmpl_annulus = src_map[annulus_valid]
            tmpl_bg = np.median(tmpl_annulus) if annulus_valid.size > 0 else 0.0
            S_tmpl = float((np.sum(tmpl_inner) - len(tmpl_inner) * tmpl_bg) * K_to_Jy_per_pix)

            # Fractional residual (dimensionless, 0 = perfect)
            frac_resid = S_resid / S_tmpl if abs(S_tmpl) > 0 else np.nan

            # Reference metrics
            peak_residual = float(np.max(np.abs(res_inner)))
            rms_annulus = float(np.sqrt(np.mean(res_annulus ** 2))) if annulus_valid.size > 0 else np.nan

            src_name = catalogue.source[brightest_idx]
            if isinstance(src_name, bytes):
                src_name = src_name.decode("ascii", errors="replace")

            cluster_records.append({
                "centroid_vec": centroid_vec,
                "member_indices": members,
                "S_resid": S_resid,
                "S_tmpl": S_tmpl,
                "frac_resid": frac_resid,
                "peak_residual": peak_residual,
                "rms_annulus": rms_annulus,
                "n_sources": int(len(members)),
                "brightest_flux": float(fluxes.max()),
                "brightest_source": src_name,
                "total_flux": float(fluxes.sum()),
            })

        if n_skipped_plane > 0:
            log.info("Residual assessment: skipped %d clusters inside galactic plane mask", n_skipped_plane)

        if not cluster_records:
            log.warning("No valid clusters for residual assessment.")
            return {}, []

        # --- 4. Save residual catalogue ---
        res_path = os.path.join(out_dir, "residual_assessment.hdf5")
        scalar_keys = [
            "S_resid", "S_tmpl", "frac_resid",
            "peak_residual", "rms_annulus", "n_sources",
            "brightest_flux", "total_flux",
        ]
        with h5py.File(res_path, "w") as hf:
            for key in scalar_keys:
                hf.create_dataset(key, data=np.array([r[key] for r in cluster_records]))
            hf.create_dataset(
                "brightest_source",
                data=np.array([r["brightest_source"] for r in cluster_records], dtype="S20"),
            )
        log.info("Residual assessment saved to %s (%d clusters)", res_path, len(cluster_records))

        # --- 5. Diagnostic plots ---
        plot_dir = fig_dir or out_dir
        total_flux = np.array([r["total_flux"] for r in cluster_records])
        frac_res = np.array([r["frac_resid"] for r in cluster_records])
        S_resid_arr = np.array([r["S_resid"] for r in cluster_records])

        # Scatter: frac_resid vs total cluster flux
        fig, ax = pyplot.subplots(figsize=(8, 5))
        finite = np.isfinite(frac_res) & np.isfinite(total_flux)
        ax.scatter(total_flux[finite], frac_res[finite], s=12, alpha=0.6, edgecolors="none")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Total cluster flux [Jy]")
        ax.set_ylabel("S_resid / S_tmpl")
        ax.set_title("Source subtraction: fractional aperture residual vs flux")
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "residual_frac_vs_flux.png"), dpi=150)
        pyplot.close(fig)

        # Histogram: frac_resid distribution (clipped to +/- 1)
        fig, ax = pyplot.subplots(figsize=(7, 4.5))
        finite_f = np.isfinite(frac_res)
        clipped = frac_res[finite_f]
        clipped = clipped[(clipped >= -1) & (clipped <= 1)]
        ax.hist(clipped, bins=50, edgecolor="k", linewidth=0.4)
        ax.axvline(0.0, color="r", lw=1.2, ls="--", label="ideal (0)")
        ax.set_xlabel("S_resid / S_tmpl")
        ax.set_ylabel("Count")
        ax.set_xlim(-1, 1)
        ax.set_title("Distribution of fractional aperture residual")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "residual_frac_histogram.png"), dpi=150)
        pyplot.close(fig)

        # Worst-12 cutouts: gnomonic patches (~4 deg across)
        # Rank purely by |S_resid| (Jy) — largest residual flux first
        abs_S = np.abs(S_resid_arr)
        abs_S[~np.isfinite(abs_S)] = -1
        worst_idx = np.argsort(abs_S)[::-1][:12]
        n_worst = len(worst_idx)
        if n_worst > 0:
            from matplotlib.patches import Circle

            # Prepare sanitised maps for plotting (NaN instead of UNSEEN)
            I_orig = I_residual + src_map
            unseen_thresh = hp.UNSEEN / 2  # UNSEEN ~ -1.6e30
            I_orig_plot = np.where((I_orig < unseen_thresh) | ~np.isfinite(I_orig), np.nan, I_orig)
            src_map_plot = np.where((src_map < unseen_thresh) | ~np.isfinite(src_map), np.nan, src_map)
            I_res_plot = np.where((I_residual < unseen_thresh) | ~np.isfinite(I_residual), np.nan, I_residual)

            cutout_size_deg = 4.0
            xsize = 200
            reso_arcmin = cutout_size_deg * 60.0 / xsize
            # Aperture radii in pixels for overlay circles
            pix_per_deg = xsize / cutout_size_deg
            inner_rad_pix = np.degrees(inner_rad) * pix_per_deg
            ann_inner_pix = np.degrees(annulus_inner) * pix_per_deg
            ann_outer_pix = np.degrees(annulus_outer) * pix_per_deg
            centre_pix = xsize / 2.0

            fig, axes = pyplot.subplots(n_worst, 3, figsize=(14, 3.2 * n_worst))
            if n_worst == 1:
                axes = axes[np.newaxis, :]
            for row, wi in enumerate(worst_idx):
                rec = cluster_records[wi]
                cv = rec["centroid_vec"]
                c_theta = np.arccos(np.clip(cv[2], -1, 1))
                c_phi = np.arctan2(cv[1], cv[0])
                rot = (np.degrees(c_phi), 90.0 - np.degrees(c_theta))

                for col, (data, title) in enumerate([
                    (I_orig_plot, "Original I"),
                    (src_map_plot, "Template"),
                    (I_res_plot, "Residual"),
                ]):
                    ax = axes[row, col]
                    try:
                        patch = hp.gnomview(
                            data, rot=rot, reso=reso_arcmin,
                            xsize=xsize, return_projected_map=True, no_plot=True,
                        )
                        # Belt-and-braces: catch any remaining sentinels
                        patch = np.where(
                            (patch < unseen_thresh) | ~np.isfinite(patch),
                            np.nan,
                            patch,
                        )
                        im = ax.imshow(patch, origin="lower", cmap="RdBu_r")
                        pyplot.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        # Overlay aperture and annulus circles
                        ax.add_patch(Circle((centre_pix, centre_pix), inner_rad_pix,
                                            fill=False, ec="lime", lw=1.0, ls="-"))
                        ax.add_patch(Circle((centre_pix, centre_pix), ann_inner_pix,
                                            fill=False, ec="cyan", lw=0.7, ls="--"))
                        ax.add_patch(Circle((centre_pix, centre_pix), ann_outer_pix,
                                            fill=False, ec="cyan", lw=0.7, ls="--"))
                    except Exception:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                    if col == 0:
                        ov = "OVER" if rec["S_resid"] < 0 else "under"
                        label = (f"{rec['brightest_source']} ({rec['n_sources']}src)\n"
                                 f"S_resid={rec['S_resid']:.3f}Jy  "
                                 f"frac={rec['frac_resid']:.2f}  [{ov}]")
                        ax.set_ylabel(label, fontsize=7)
                    if row == 0:
                        ax.set_title(title, fontsize=9)
                    ax.tick_params(labelsize=5)

            fig.suptitle("Worst-12 clusters by |S_resid|", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(os.path.join(plot_dir, "residual_worst12_cutouts.png"), dpi=150)
            pyplot.close(fig)

        return {
            "n_clusters": len(cluster_records),
            "median_frac_resid": float(np.nanmedian(frac_res)),
            "residual_catalogue": res_path,
        }, cluster_records

    # ---------- residual correction ----------

    def _correct_oversubtracted(
        self,
        I_residual: np.ndarray,
        src_map: np.ndarray,
        catalogue,
        cluster_records: list,
        correct_n: int,
        correct_threshold: float,
        nside: int,
        fwhm_deg: float,
        coords: str,
        out_dir: str,
        fig_dir: Optional[str],
        beam_model,
        use_beam_model: bool,
        nside_source_map: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct the worst over-subtracted clusters by adding flux back.

        Returns (corrected_I_map, flux_correction_per_source) where
        flux_correction_per_source has the same length as catalogue and
        contains the delta flux (Jy) applied to each source (0 for uncorrected).
        """
        from matplotlib.patches import Circle

        flux_correction = np.zeros(catalogue.size, dtype=float)

        # --- 1. Select clusters to correct ---
        over = [
            (i, r) for i, r in enumerate(cluster_records)
            if r["S_resid"] < 0 and abs(r["S_resid"]) > correct_threshold
        ]
        if not over:
            log.info("Residual correction: no over-subtracted clusters above threshold.")
            return I_residual, flux_correction

        # Sort by |S_resid| descending
        over.sort(key=lambda x: abs(x[1]["S_resid"]), reverse=True)
        corrected = over[:correct_n]
        uncorrected = over[correct_n:correct_n + 3]

        log.info(
            "Residual correction: correcting %d / %d over-subtracted clusters (threshold=%.4f Jy)",
            len(corrected), len(over), correct_threshold,
        )

        # --- 2. Build correction template ---
        # Get source positions in the working coordinate frame
        if coords == "C":
            src_lon, src_lat = Catalogues.rotate(catalogue.glon, catalogue.glat, coord=("G", "C"))
        else:
            src_lon, src_lat = catalogue.glon, catalogue.glat

        corr_fluxes = []
        corr_lons = []
        corr_lats = []
        for _, rec in corrected:
            members = rec["member_indices"]
            frac = rec["frac_resid"]  # negative for over-subtracted
            for j in members:
                delta_flux = -frac * catalogue.flux[j]
                corr_fluxes.append(delta_flux)
                corr_lons.append(src_lon[j])
                corr_lats.append(src_lat[j])
                flux_correction[j] = delta_flux

        corr_fluxes = np.array(corr_fluxes)
        corr_lons = np.array(corr_lons)
        corr_lats = np.array(corr_lats)

        correction_template = Mapper.pixel_space(
            corr_fluxes, corr_lons, corr_lats,
            nside=nside_source_map, fwhm_deg=fwhm_deg,
            use_beam_model=bool(use_beam_model and beam_model is not None),
            beam_model=beam_model,
        )
        if nside_source_map != nside:
            correction_template = hp.ud_grade(correction_template, nside, pess=False)

        # --- 3. Apply correction ---
        I_corrected = I_residual.copy()
        valid = (I_residual != hp.UNSEEN) & np.isfinite(I_residual)
        I_corrected[valid] = I_residual[valid] + correction_template[valid]

        log.info(
            "Residual correction: template peak=%.4e K, applied to %d pixels",
            float(np.max(np.abs(correction_template[valid]))),
            int(np.sum(valid)),
        )

        # --- 4. Diagnostic cutouts ---
        plot_dir = fig_dir or out_dir
        show_clusters = [(rec, True) for _, rec in corrected]
        show_clusters += [(rec, False) for _, rec in uncorrected]
        n_show = len(show_clusters)

        if n_show > 0:
            fwhm_rad = np.radians(fwhm_deg)
            inner_rad = 0.5 * fwhm_rad
            annulus_inner = 0.75 * fwhm_rad
            annulus_outer = 1.5 * fwhm_rad

            cutout_size_deg = 4.0
            xsize = 200
            reso_arcmin = cutout_size_deg * 60.0 / xsize
            pix_per_deg = xsize / cutout_size_deg
            inner_rad_pix = np.degrees(inner_rad) * pix_per_deg
            ann_inner_pix = np.degrees(annulus_inner) * pix_per_deg
            ann_outer_pix = np.degrees(annulus_outer) * pix_per_deg
            centre_pix = xsize / 2.0

            # Prepare maps for plotting
            I_orig = I_residual + src_map
            unseen_thresh = hp.UNSEEN / 2
            I_orig_plot = np.where((I_orig < unseen_thresh) | ~np.isfinite(I_orig), np.nan, I_orig)
            src_map_plot = np.where((src_map < unseen_thresh) | ~np.isfinite(src_map), np.nan, src_map)
            I_before_plot = np.where((I_residual < unseen_thresh) | ~np.isfinite(I_residual), np.nan, I_residual)
            I_after_plot = np.where((I_corrected < unseen_thresh) | ~np.isfinite(I_corrected), np.nan, I_corrected)

            fig, axes = pyplot.subplots(n_show, 4, figsize=(18, 3.2 * n_show))
            if n_show == 1:
                axes = axes[np.newaxis, :]

            for row, (rec, was_corrected) in enumerate(show_clusters):
                cv = rec["centroid_vec"]
                c_theta = np.arccos(np.clip(cv[2], -1, 1))
                c_phi = np.arctan2(cv[1], cv[0])
                rot = (np.degrees(c_phi), 90.0 - np.degrees(c_theta))

                panels = [
                    (I_orig_plot, "Original I"),
                    (src_map_plot, "Template"),
                    (I_before_plot, "Residual Before"),
                    (I_after_plot, "Residual After"),
                ]

                for col, (data, title) in enumerate(panels):
                    ax = axes[row, col]
                    try:
                        patch = hp.gnomview(
                            data, rot=rot, reso=reso_arcmin,
                            xsize=xsize, return_projected_map=True, no_plot=True,
                        )
                        patch = np.where(
                            (patch < unseen_thresh) | ~np.isfinite(patch),
                            np.nan, patch,
                        )
                        im = ax.imshow(patch, origin="lower", cmap="RdBu_r")
                        pyplot.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.add_patch(Circle((centre_pix, centre_pix), inner_rad_pix,
                                            fill=False, ec="lime", lw=1.0, ls="-"))
                        ax.add_patch(Circle((centre_pix, centre_pix), ann_inner_pix,
                                            fill=False, ec="cyan", lw=0.7, ls="--"))
                        ax.add_patch(Circle((centre_pix, centre_pix), ann_outer_pix,
                                            fill=False, ec="cyan", lw=0.7, ls="--"))
                    except Exception:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                    if col == 0:
                        tag = "[CORRECTED]" if was_corrected else "[uncorrected]"
                        label = (
                            f"{rec['brightest_source']} ({rec['n_sources']}src)\n"
                            f"S_resid={rec['S_resid']:.3f}Jy  "
                            f"frac={rec['frac_resid']:.2f}  {tag}"
                        )
                        ax.set_ylabel(label, fontsize=7)
                    if row == 0:
                        ax.set_title(title, fontsize=9)
                    ax.tick_params(labelsize=5)

            fig.suptitle("Residual correction cutouts", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(os.path.join(plot_dir, "residual_correction_cutouts.png"), dpi=150)
            pyplot.close(fig)

        return I_corrected, flux_correction

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
        nside_source_map = int(stage_cfg.get("nside_source_map", 2048))

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
        cat_path = stage_cfg.get("run_catalogue") or os.path.join(out_dir, f"{cat_stub}.hdf5")
        if os.path.exists(cat_path):
            log.info("Loading cached catalogue from %s", cat_path)
            total = Catalogues.CBASS()
            total.load_file(cat_path, name="CBASS_GB6_PMN_PMN_Mingaliev")
        else:
            log.info("Building fresh catalogue (no cache at %s)", cat_path)
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
                cb.mask_map(mask_hi)

            cb.mask_declinations(declination_min=cbass_min_dec, declination_max=90)

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
            mg, g6 = Catalogues.common_sources(mg, g6)
            p1, g6 = Catalogues.common_sources(p1, g6)

            catalogue_sources = g6 + p1 + p2 + mg
            Catalogues.merge_catalogues(cb, catalogue_sources, fwhm_deg=cbass_fwhm_deg)
            total = cb + catalogue_sources
            total.mask_map(excl)
            total.clean_nan()
            total.write_file(cat_path)
            log.info("N sources: %d | Total flux: %.2f Jy | Min/Med/Max: %.3f / %.3f / %.3f Jy",
                     total.size, float(total.flux.sum()),
                     float(np.min(total.flux)), float(np.median(total.flux)), float(np.max(total.flux)))

        # --- build/restore source template map
        beam_model = None
        tmpl_path = stage_cfg.get("run_source_map") or os.path.join(out_dir, tmpl_name)
        if os.path.exists(tmpl_path):
            log.info("Loading cached source template from %s", tmpl_path)
            src_map = hp.read_map(tmpl_path)
        else:
            log.info("Building fresh source template map (no cache at %s)", tmpl_path)
            if use_beam_model and beam_model_filename:
                beam_model = np.loadtxt(beam_model_filename)
            if coords == "C":
                map_lon, map_lat = Catalogues.rotate(total.glon, total.glat, coord=("G", "C"))
            else:
                map_lon, map_lat = total.glon, total.glat

            log.info("Generating source map at NSIDE=%d (will downgrade to %d)", nside_source_map, nside_in)
            src_map = Mapper.pixel_space(
                total.flux, map_lon, map_lat,
                nside=nside_source_map, fwhm_deg=cbass_fwhm_deg,
                use_beam_model=bool(use_beam_model and (beam_model is not None)),
                beam_model=beam_model
            )
            if nside_source_map != nside_in:
                src_map = hp.ud_grade(src_map, nside_in, pess=False)
            hp.write_map(tmpl_path, src_map, overwrite=True)

        # ensure NSIDE matches input
        if hp.get_nside(src_map) != nside_in:
            src_map = hp.ud_grade(src_map, nside_in, pess=False)
        src_map[iqu_in[0] == hp.UNSEEN] = 0.0  # zero, not UNSEEN — no source flux there

        # --- subtract on I, preserve Q/U
        I_out = self._masked_subtract(iqu_in[0], src_map)
        m_out = self._merge_back_I(I_out, iqu_in)

        # --- residual assessment (diagnostic)
        residual_info = {}
        cluster_records = []
        if do_res_fit:
            # Build a galactic plane mask from threshold masks to exclude
            # diffuse-dominated regions.  Use the 80% mask if available,
            # otherwise fall back to any mask that exists.
            plane_mask = None
            residual_mask_index = int(stage_cfg.get("residual_mask_index", -2))
            if threshold_mask_maps:
                idx = residual_mask_index % len(threshold_mask_maps)
                rmf = str(threshold_mask_maps[idx].get("mask_filename", ""))
                if rmf and os.path.exists(rmf):
                    plane_mask = self._read_mask_bool(rmf)
                    log.info("Residual assessment: excluding clusters inside %s", os.path.basename(rmf))

            residual_info, cluster_records = self._assess_residuals(
                I_out, src_map, total, nside_in, cbass_fwhm_deg, coords, out_dir, fig_dir,
                plane_mask=plane_mask,
                mask_coord=threshold_masks_coord,
            )

        # --- residual correction for over-subtracted sources
        correct_n = int(stage_cfg.get("residual_correct_n", 0))
        correct_threshold = float(stage_cfg.get("residual_correct_threshold", 0.0))
        if correct_n > 0 and cluster_records:
            # Ensure beam model is loaded (may be None if template was cached)
            if beam_model is None and use_beam_model and beam_model_filename:
                beam_model = np.loadtxt(beam_model_filename)
            I_out, flux_corr = self._correct_oversubtracted(
                I_out, src_map, total, cluster_records,
                correct_n, correct_threshold,
                nside_in, cbass_fwhm_deg, coords,
                out_dir, fig_dir,
                beam_model, use_beam_model, nside_source_map,
            )
            m_out = self._merge_back_I(I_out, iqu_in)
            # Store per-source corrections in catalogue and re-write
            total.flux_correction = flux_corr
            total.write_file(cat_path)
            n_corrected = int(np.count_nonzero(flux_corr))
            log.info("Catalogue re-written with FLUX_CORRECTION column (%d sources modified)", n_corrected)

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
