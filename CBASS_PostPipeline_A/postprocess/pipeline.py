# postprocess/pipeline.py
from __future__ import annotations
import os, json, time
from typing import Dict, List, Tuple
from .types import MapBundle, StageReport
from . import registry as reg

def run_pipeline(cfg: Dict) -> Tuple[MapBundle, List[StageReport]]:
    modules = cfg.get("stages", {}).get("order") or cfg.get("modules") or []
    if not modules:
        raise RuntimeError("No modules specified (stages.order or modules).")

    # Minimal bundle. You’ll load real maps in Masks or an explicit Initialise stage later.
    bundle = MapBundle(
        map=None,
        nside=cfg.get("vars", {}).get("nside"),
        coords=cfg.get("vars", {}).get("coords"),
        history=[],
    )

    reports: List[StageReport] = []
    t0 = time.time()
    print('MODULES',modules)
    for name in modules:
        print(f'Running {name}')
        stage = reg.get_stage(name)
        stage_cfg = cfg.get(name, {})
        t1 = time.time()
        bundle, report = stage.run(bundle, stage_cfg, cfg)
        report.metrics["time_sec"] = round(time.time() - t1, 3)
        reports.append(report)
        bundle.history.append(f"{name} completed")

    # Save a tiny manifest next to the final output path if present
    out_path = (cfg.get("FinalMap") or {}).get("output")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        manifest = {
            "modules": modules,
            "vars": cfg.get("vars", {}),
            "outputs": {"map": out_path},
            "reports": [r.__dict__ for r in reports],
            "runtime_sec": round(time.time() - t0, 3),
        }
        with open(out_path + ".manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    return bundle, reports
