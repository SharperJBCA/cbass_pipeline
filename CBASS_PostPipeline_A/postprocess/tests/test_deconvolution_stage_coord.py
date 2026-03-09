import numpy as np
import healpy as hp

from postprocess.stages.deconvolution import Deconvolution


def test_deconvolution_declination_mask_uses_bundle_coords(monkeypatch, tmp_path):
    nside = 8
    npix = hp.nside2npix(nside)

    # Minimal in-memory bundle stub
    class Bundle:
        def __init__(self):
            self.map = np.vstack([
                np.ones(npix, dtype=float),
                np.zeros(npix, dtype=float),
                np.zeros(npix, dtype=float),
            ])
            self.cov = np.zeros((4, npix), dtype=float)
            self.coords = "C"
            self.headers = {}
            self.nside = nside
            self.history = []

    bundle = Bundle()

    # Keep run lightweight by bypassing heavy transforms.
    monkeypatch.setattr(
        "postprocess.stages.deconvolution.build_transfer_functions",
        lambda **kwargs: (np.ones(4), np.ones(4), np.ones(4)),
    )
    monkeypatch.setattr(
        "postprocess.stages.deconvolution.apply_transfer_to_maps",
        lambda *args, **kwargs: (bundle.map[0].copy(), bundle.map[1].copy(), bundle.map[2].copy()),
    )
    monkeypatch.setattr(
        "postprocess.stages.deconvolution.apply_transfer_to_cov",
        lambda *args, **kwargs: tuple(bundle.cov[i].copy() for i in range(4)),
    )

    seen = {}

    def fake_dec_mask(nside_out, coord, min_dec):
        seen["coord"] = coord
        return np.ones(hp.nside2npix(nside_out), dtype=bool)

    monkeypatch.setattr("postprocess.stages.deconvolution.dec_mask", fake_dec_mask)

    stage = Deconvolution()
    cfg = {
        "fig_dir": str(tmp_path),
        "nside_out": nside,
        "beam_function_lmax": 3,
        "map_coord": "G",  # intentionally wrong/stale: bundle coord must win
        "min_dec": -14.6,
    }

    stage.run(bundle, cfg, {"vars": {"coords": "G"}})

    assert seen["coord"] == "C"
    assert bundle.headers["DECMIN"] == -14.6


def test_deconvolution_sets_header_flags_for_pixwin_only(monkeypatch, tmp_path):
    nside = 8
    npix = hp.nside2npix(nside)

    class Bundle:
        def __init__(self):
            self.map = np.vstack([
                np.ones(npix, dtype=float),
                np.zeros(npix, dtype=float),
                np.zeros(npix, dtype=float),
            ])
            self.cov = np.zeros((4, npix), dtype=float)
            self.coords = "G"
            self.headers = {}
            self.nside = nside
            self.history = []

    bundle = Bundle()

    monkeypatch.setattr(
        "postprocess.stages.deconvolution.build_transfer_functions",
        lambda **kwargs: (np.ones(4), np.ones(4), np.ones(4)),
    )
    monkeypatch.setattr(
        "postprocess.stages.deconvolution.apply_transfer_to_maps",
        lambda *args, **kwargs: (bundle.map[0].copy(), bundle.map[1].copy(), bundle.map[2].copy()),
    )
    monkeypatch.setattr(
        "postprocess.stages.deconvolution.apply_transfer_to_cov",
        lambda *args, **kwargs: tuple(bundle.cov[i].copy() for i in range(4)),
    )

    seen = {"called": False}

    def fake_dec_mask(nside_out, coord, min_dec):
        seen["called"] = True
        return np.ones(hp.nside2npix(nside_out), dtype=bool)

    monkeypatch.setattr("postprocess.stages.deconvolution.dec_mask", fake_dec_mask)

    stage = Deconvolution()
    cfg = {
        "fig_dir": str(tmp_path),
        "nside_out": nside,
        "beam_function_lmax": 3,
        "map_coord": "G",
        "min_dec": -14.6,
        "beam_filename": "some_beam.txt",
        "apply_transfer_function": False,
    }

    stage.run(bundle, cfg, {})

    assert seen["called"] is False
    assert bundle.headers["DCONV"] is False
    assert "BL_FILE" not in bundle.headers
    assert "FWHM_OUT" not in bundle.headers
    assert bundle.headers["DECMIN"] == -15.6
