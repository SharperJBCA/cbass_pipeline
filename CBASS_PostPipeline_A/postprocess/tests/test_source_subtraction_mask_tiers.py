from __future__ import annotations

import numpy as np
import healpy as hp

from postprocess.source_subtraction.Catalogues import Catalogue
from postprocess.stages.source_subtract import SourceSubtraction


def _make_catalogue(glon, glat, flux, name="TEST"):
    n = len(flux)
    return Catalogue(
        name=name,
        glon=np.asarray(glon, dtype=float),
        glat=np.asarray(glat, dtype=float),
        flux=np.asarray(flux, dtype=float),
        eflux=np.ones(n, dtype=float),
        flag=np.zeros(n, dtype=bool),
        source=np.array([name] * n, dtype="S20"),
    )


def _pix_to_lonlat(nside: int, pix: int) -> tuple[float, float]:
    theta, phi = hp.pix2ang(nside, pix)
    return float(np.degrees(phi)), float(90.0 - np.degrees(theta))


def test_mask_map_min_flux_applies_only_inside_mask():
    nside = 1
    lon0, lat0 = _pix_to_lonlat(nside, 0)
    lon1, lat1 = _pix_to_lonlat(nside, 1)

    cat = _make_catalogue(
        glon=[lon0, lon1],
        glat=[lat0, lat1],
        flux=[1.5, 1.5],
        name="CBASS",
    )

    mask = np.zeros(12 * nside**2, dtype=bool)
    mask[0] = True

    cat.mask_map(mask, flux=cat.flux, lower_limit=2.0)

    # Masked + below threshold removed; unmasked source is preserved.
    assert cat.size == 1
    assert np.isclose(cat.flux[0], 1.5)
    assert np.isclose(cat.glon[0], lon1)


def test_tiered_mask_rules_and_cg05_all_catalogues_masking():
    nside = 1

    # four sources, one placed in each tier-only region
    lon_lat = [_pix_to_lonlat(nside, p) for p in [0, 1, 2, 3]]
    glon = [p[0] for p in lon_lat]
    glat = [p[1] for p in lon_lat]

    cbass = _make_catalogue(glon, glat, flux=[20.0, 1.5, 0.8, 0.5], name="CBASS")
    gb6 = _make_catalogue(glon, glat, flux=[8.0, 8.0, 8.0, 8.0], name="GB6")

    cg05 = np.zeros(12 * nside**2, dtype=bool)
    cg10 = np.zeros(12 * nside**2, dtype=bool)
    cg20 = np.zeros(12 * nside**2, dtype=bool)
    cg30 = np.zeros(12 * nside**2, dtype=bool)

    # nested masks
    cg05[0] = True
    cg10[[0, 1]] = True
    cg20[[0, 1, 2]] = True
    cg30[[0, 1, 2, 3]] = True

    # C-BASS tier thresholds at CG05/10/20/30
    SourceSubtraction._apply_mask_rule(cbass, {"mode": "min_flux", "limit": 10.0}, cg05)
    SourceSubtraction._apply_mask_rule(cbass, {"mode": "min_flux", "limit": 2.0}, cg10)
    SourceSubtraction._apply_mask_rule(cbass, {"mode": "min_flux", "limit": 1.0}, cg20)
    SourceSubtraction._apply_mask_rule(cbass, {"mode": "min_flux", "limit": 0.61}, cg30)

    # all catalogues removed in CG05
    SourceSubtraction._apply_mask_rule(cbass, {"mode": "all"}, cg05)
    SourceSubtraction._apply_mask_rule(gb6, {"mode": "all"}, cg05)

    # CBASS: pixel 0 removed by CG05 all-catalogue mask; pixels 1/2/3 removed by tier limits.
    assert cbass.size == 0

    # Non-CBASS catalogue: only CG05 source removed by the all-catalogue rule.
    assert gb6.size == 3
