from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class MapBundle:
    map: Optional[np.ndarray] = None        # (npix,) or (ncomp, npix)
    cov: Optional[np.ndarray] = None 
    nside: Optional[int] = None
    coords: Optional[str] = None            # "G" | "C"
    units: Optional[str] = None
    beam_fwhm_arcmin: Optional[float] = None
    freq_ghz: Optional[float] = None
    mask: Optional[np.ndarray] = None       # bool or UNSEEN-like
    headers: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    source_path: Optional[str] = None

@dataclass
class StageReport:
    name: str
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    figures: List[str] = field(default_factory=list)  # paths to saved figs (optional)
