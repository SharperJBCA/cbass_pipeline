from __future__ import annotations
from typing import Dict, Any, Tuple
import os
import numpy as np
import healpy as hp
from astropy.io import fits
from datetime import datetime

from . import Stage
from ..types import MapBundle, StageReport

PRESERVE_KEYS = {"AEFF","CALDATE","POLCCONV","BAD_DATA","TELESCOP","FREQ","COORDSYS"}

class FinalMap(Stage):
    name = "FinalMap"

    def _read_preserved_cards(self, src_path: str) -> Dict[str, Any]:
        out = {}
        if not (src_path and os.path.exists(src_path)):
            return out
        with fits.open(src_path, memmap=False) as hdul:
            hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
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
            print(card)
        print('---')
        # Pass 2: append any extra keywords that are in base_hdr but not in template
        tmpl_keys = {c.keyword for c in tmpl_hdr.cards}
        for card in base_hdr.cards:
            key = card.keyword
            if key in ("COMMENT", "HISTORY", "END", ""):
                continue
            if key not in tmpl_keys:
                new_hdr.append(card,bottom=True)
        for k,v in new_hdr.items():
            print(k,v)
        print('---') 

        # Pass 3: append HISTORY cards at the very end
        for line in history_lines:
            new_hdr["HISTORY"] = line

        # Replace original header contents with the new ordered header
        #hdr.clear()
        #hdr.extend(new_hdr.cards)
        return new_hdr
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

        # Read preserved cards
        src_path = bundle.source_path or (full_cfg.get("input", {})).get("map")
        preserved = self._read_preserved_cards(src_path)

        # Merge into table header (values first)
        self._merge_header(tbhdu.header, preserved, bundle.headers, bundle.history)

        # Apply template ordering
        template_path = '/scratch/nas_cbassarc/sharper/work/CBASS_MagneticDust/cbass_post_pipeline/CBASS_PostPipeline_A/ancillary_data/cbass_dr1_header_template.hdr'
        tbhdu.header = self._apply_template_order(tbhdu.header, template_path)

        for k,c in tbhdu.header.items():
            print(k,c)

        primary = fits.PrimaryHDU()
        hdul = fits.HDUList([primary, tbhdu])
        hdul.writeto(out, overwrite=True)

        rep = StageReport(self.name, summary=f"wrote {out}")
        return bundle, rep
