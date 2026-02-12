import numpy as np
import healpy as hp

from postprocess.deconv import core, inpainting


def test_inpaint_and_apodise_uses_seen_median_for_fill_value():
    nside = 8
    npix = hp.nside2npix(nside)
    hpmap = np.full(npix, hp.UNSEEN, dtype=float)

    seen_pix = np.arange(10)
    hpmap[seen_pix] = 5.0

    out = inpainting.inpaint_and_apodise(
        hpmap,
        decapo=-90.0,  # no apodised cut: keep full map
        smapo=0,
        n_iter=0,
        set_0=False,
    )

    assert np.all(np.isfinite(out))
    assert np.all(out == 5.0)


def test_apply_transfer_inpainting_receives_unseen_pixels(monkeypatch):
    nside = 8
    npix = hp.nside2npix(nside)

    I = np.ones(npix, dtype=float)
    Q = np.zeros(npix, dtype=float)
    U = np.zeros(npix, dtype=float)

    masked = np.arange(50)
    I[masked] = hp.UNSEEN
    Q[masked] = hp.UNSEEN
    U[masked] = hp.UNSEEN

    calls = []

    def fake_inpaint(arr, *args, **kwargs):
        calls.append(np.count_nonzero(arr == hp.UNSEEN))
        out = arr.copy()
        out[out == hp.UNSEEN] = 0.0
        return out

    monkeypatch.setattr(core.inpainting, "inpaint_and_apodise", fake_inpaint)

    lmax = 3
    ones = np.ones(lmax + 1)
    core.apply_transfer_to_maps(
        I,
        Q,
        U,
        R0=ones,
        R2=ones,
        pixwin=ones,
        lmax=lmax,
        nside_out=nside,
        apodise_inpaint=True,
    )

    assert len(calls) == 3
    assert all(count > 0 for count in calls)
