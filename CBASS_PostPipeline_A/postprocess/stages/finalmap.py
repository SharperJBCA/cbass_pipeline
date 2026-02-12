from __future__ import annotations
from typing import Dict, Any, Tuple
import os
import numpy as np
from astropy.io import fits

from . import Stage
from ..types import MapBundle, StageReport

PRESERVE_KEYS = {"AEFF", "CALDATE", "POLCCONV", "BAD_DATA", "TELESCOP", "FREQ", "COORDSYS"}
TABLE_STRUCTURE_KEYS = {
    "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME"
}

class FinalMap(Stage):
    name = "FinalMap"

    def _read_preserved_cards(self, src_path: str, preserve_all: bool, preserve_in_ext1: bool = True) -> Dict[str, Any]:
        out = {}
        if not (src_path and os.path.exists(src_path)):
            return out
        with fits.open(src_path, memmap=False) as hdul:
            hdu_index = 1 if (preserve_in_ext1 and len(hdul) > 1) else 0
            hdr = hdul[hdu_index].header

            if preserve_all:
                for card in hdr.cards:
                    k = card.keyword
                    if k in ("", "COMMENT", "HISTORY", "END"):
                        continue
                    if k in TABLE_STRUCTURE_KEYS or k.startswith("TTYPE") or k.startswith("TFORM") or k.startswith("TUNIT"):
                        continue
                    out[k] = card.value
            else:
                for k in PRESERVE_KEYS:
                    if k in hdr:
                        out[k] = hdr[k]
        return out

    def _merge_header(self,
                      hdr: fits.Header,
                      preserved: Dict[str, Any],
                      pipeline_cards: Dict[str, Any],
                      history_lines):
        # 1) carry preserved cards if not already set
        for k, v in preserved.items():
            if k not in hdr:
                hdr[k] = v
        # 2) add/overwrite pipeline cards
        for k, v in pipeline_cards.items():
            try:
                hdr[k] = v
            except Exception:
                hdr[k] = str(v)
        # 3) provenance
        for line in history_lines:
            hdr["HISTORY"] = line

    def _apply_template_order(self, hdr: fits.Header, template_path: str) -> None:
        """
        Reorder `hdr` to match the order of `template_path` exactly.
        - For keywords in both template and hdr: keep template order, but use
          the VALUE from hdr and the COMMENT from the template (if present).
        - For keywords only in template: copy the template card.
        - Any extra keywords (only in hdr) are appended at the end.
        - All HISTORY cards from hdr are moved to the very end.
        """
        # Make a working copy of the current header
        base_hdr = hdr.copy()

        # Extract all HISTORY lines from current header (pipeline provenance)
        history_lines = [c.value for c in base_hdr.cards if c.keyword == "HISTORY"]
        # Remove them from the working header so we can re-append at the end
        while "HISTORY" in base_hdr:
            del base_hdr["HISTORY"]

        # Load template header (this defines order + default comments)
        tmpl_hdr = fits.Header.fromtextfile(template_path)

        new_hdr = fits.Header()

        # Pass 1: follow template order exactly
        for card in tmpl_hdr.cards:
            key = card.keyword

            # Don't copy END; astropy will add it at write time
            if key == "END":
                continue

            # We handle HISTORY separately at the end
            if key == "HISTORY":
                continue

            # Copy COMMENT / blank cards exactly from template
            if key in ("COMMENT", ""):
                new_hdr.append(card,bottom=True)
                continue

            # For "normal" keywords: prefer value from base_hdr if present
            if key in base_hdr:
                value = base_hdr[key]
                # Prefer template comment if available, else use base_hdr comment
                comment = card.comment if (card.comment not in (None, "")) else base_hdr.comments[key]
                new_hdr.append((key,value, comment),bottom=True)
            else:
                # Not set in base_hdr – keep template card as-is
                new_hdr.append(card,bottom=True)
        # Pass 2: append any extra keywords that are in base_hdr but not in template
        tmpl_keys = {c.keyword for c in tmpl_hdr.cards}
        for card in base_hdr.cards:
            key = card.keyword
            if key in ("COMMENT", "HISTORY", "END", ""):
                continue
            if key not in tmpl_keys:
                new_hdr.append(card, bottom=True)

        # Pass 3: append HISTORY cards at the very end
        for line in history_lines:
            new_hdr["HISTORY"] = line

        # Replace original header contents with the new ordered header
        #hdr.clear()
        #hdr.extend(new_hdr.cards)
        return new_hdr


    def _resolve_template_path(self, template_path: str | None) -> str | None:
        if not template_path:
            return None
        if os.path.exists(template_path):
            return template_path

        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate = os.path.join(pkg_root, template_path)
        if os.path.exists(candidate):
            return candidate
        return None

    def _infer_nside_from_npix(self, npix: int) -> int:
        if npix <= 0:
            raise ValueError(f"Invalid npix={npix}")
        nside = int(round((npix / 12.0) ** 0.5))
        if 12 * nside * nside != npix:
            raise ValueError(f"npix={npix} is not a valid HEALPix map size")
        return nside

    def _update_output_geometry_cards(self, hdr: fits.Header, npix: int, nside: int | None, coords: str | None) -> None:
        out_nside = int(nside) if nside is not None else self._infer_nside_from_npix(npix)
        hdr["FIRSTPIX"] = 0
        hdr["LASTPIX"] = int(npix - 1)
        hdr["NSIDE"] = out_nside
        if coords:
            hdr["COORDSYS"] = str(coords).upper()

    def _update_beam_cards(self, hdr: fits.Header, pipeline_cards: Dict[str, Any]) -> None:
        dconv = bool(pipeline_cards.get("DCONV", False))
        if not dconv:
            return

        fwhm_out = pipeline_cards.get("FWHM_OUT")
        if fwhm_out is None:
            return

        try:
            fwhm_deg = float(fwhm_out)
        except Exception:
            return

        hdr["BMAJ"] = fwhm_deg
        hdr["BMIN"] = fwhm_deg

    def run(self,
            bundle: MapBundle,
            stage_cfg: Dict[str, Any],
            full_cfg: Dict[str, Any]) -> Tuple[MapBundle, StageReport]:

        out = (full_cfg.get("FinalMap") or {}).get("output")
        if not out:
            return bundle, StageReport(self.name, summary="no output path")

        os.makedirs(os.path.dirname(out), exist_ok=True)

        # ---- arrange data into 7 components: I, Q, U, II, QQ, UU, QU ----
        # Expect bundle.map shape ~ (3, Npix), bundle.cov shape ~ (4, Npix)
        data = np.concatenate((bundle.map, bundle.cov), axis=0).astype("f4")
        ncomp, npix = data.shape
        if ncomp != 7:
            raise ValueError(f"Expected 7 components (I,Q,U,II,QQ,UU,QU), got {ncomp}")

        # ---- build BINTABLE columns ----
        colnames = [
            "I_STOKES", "Q_STOKES", "U_STOKES",
            "II_COV", "QQ_COV", "UU_COV", "QU_COV",
        ]
        units = [
            "K_RJ", "K_RJ", "K_RJ",
            "K_RJ^2", "K_RJ^2", "K_RJ^2", "K_RJ^2",
        ]

        cols = []
        for i, (name, unit) in enumerate(zip(colnames, units)):
            cols.append(
                fits.Column(
                    name=name,
                    array=data[i],
                    format="E",   # 32-bit float
                    unit=unit,
                )
            )

        tbhdu = fits.BinTableHDU.from_columns(cols, name="xtension")

        preserve_all = bool(stage_cfg.get("preserve_all_headers", False))
        preserve_in_ext1 = bool(stage_cfg.get("preserve_in_ext1", True))

        # Read preserved cards
        src_path = bundle.source_path or (full_cfg.get("input", {})).get("map")
        preserved = self._read_preserved_cards(src_path, preserve_all=preserve_all, preserve_in_ext1=preserve_in_ext1)

        # Merge into table header (values first)
        self._merge_header(tbhdu.header, preserved, bundle.headers, bundle.history)

        # Ensure geometry cards describe the output map (not preserved input values)
        self._update_output_geometry_cards(tbhdu.header, npix=npix, nside=bundle.nside, coords=bundle.coords)

        # If deconvolution ran, update beam metadata to output resolution.
        self._update_beam_cards(tbhdu.header, bundle.headers)

        # Apply template ordering
        template_path = self._resolve_template_path(
            stage_cfg.get("header_template")
            or (full_cfg.get("global", {}) or {}).get("default_header_file")
        )
        if template_path:
            tbhdu.header = self._apply_template_order(tbhdu.header, template_path)

        primary = fits.PrimaryHDU()
        hdul = fits.HDUList([primary, tbhdu])
        hdul.writeto(out, overwrite=True)

        rep = StageReport(self.name, summary=f"wrote {out}")
        return bundle, rep