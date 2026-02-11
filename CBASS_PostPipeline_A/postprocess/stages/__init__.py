from __future__ import annotations
from typing import Tuple, Dict, Any
from ..types import MapBundle, StageReport

class Stage:
    name = "Stage"
    def run(self, bundle: MapBundle, stage_cfg: Dict[str, Any], full_cfg: Dict[str, Any]) -> Tuple[MapBundle, StageReport]:
        # must return (updated_bundle, StageReport)
        return bundle, StageReport(name=self.name, summary="noop")
