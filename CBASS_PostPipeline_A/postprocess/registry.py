from __future__ import annotations
from typing import Dict
from .stages.masks import Masks
from .stages.monopole_dipole import MonopoleDipole
from .stages.deconvolution import Deconvolution
from .stages.finalmap import FinalMap
from .stages.source_subtract import SourceSubtraction
from .stages.plotting import PlotQuicklook

_REGISTRY: Dict[str, object] = {
    "Masks": Masks(),
    "MonoDipoleSub": MonopoleDipole(),
    "Deconvolution": Deconvolution(),
    "SourceSubtraction": SourceSubtraction(),
    "PlotQuicklook": PlotQuicklook(),
    "FinalMap": FinalMap(),
}

def get_stage(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown stage '{name}'. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]
