# postprocess/plotting/stage.py
from __future__ import annotations
from typing import Dict, Any, Tuple
from ..stages import Stage
from ..types import MapBundle, StageReport
from .quicklook import run_quicklook

class PlotQuicklook(Stage):
    name = "PlotQuicklook"

    def run(self, bundle: MapBundle, stage_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> Tuple[MapBundle, StageReport]:
        fig_root = stage_cfg.get("fig_root") or full_cfg.get("vars", {}).get("fig_root") or "figures"
        tag = stage_cfg.get("tag") or "quicklook"

        res = run_quicklook(bundle, fig_root=fig_root, tag=tag)

        rep = StageReport(
            name=self.name,
            summary="Saved full-sky IQU, covariance, and zoom panels.",
            metrics={"num_fullsky": len(res.fullsky_figs), "num_zooms": len(res.zoom_figs)},
            figures=list(res.fullsky_figs.values()) + list(res.zoom_figs.values()),
        )
        return bundle, rep
