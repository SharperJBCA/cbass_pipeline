from pathlib import Path

import numpy as np
from astropy.io import fits

from postprocess.stages.finalmap import FinalMap
from postprocess.types import MapBundle


def _build_bundle(nside: int = 8) -> MapBundle:
    npix = 12 * nside * nside
    data = np.zeros((3, npix), dtype=np.float32)
    cov = np.zeros((4, npix), dtype=np.float32)
    return MapBundle(
        map=data,
        cov=cov,
        nside=nside,
        coords="G",
        headers={"PIPEKEY": "set_by_pipeline"},
        history=["history line"],
    )


def _write_source_with_header(path: Path) -> None:
    cols = [fits.Column(name="I_STOKES", array=np.zeros(12, dtype=np.float32), format="E", unit="K_RJ")]
    ext = fits.BinTableHDU.from_columns(cols)
    ext.header["TELESCOP"] = "CBASS"
    ext.header["CALDATE"] = "2025-02-01"
    ext.header["CUSTOMK"] = "custom_value"
    fits.HDUList([fits.PrimaryHDU(), ext]).writeto(path)


def test_finalmap_preserves_all_headers_by_default(tmp_path: Path):
    src = tmp_path / "input.fits"
    out = tmp_path / "output.fits"
    _write_source_with_header(src)

    stage = FinalMap()
    bundle = _build_bundle()
    bundle.source_path = str(src)

    stage.run(bundle, stage_cfg={}, full_cfg={"FinalMap": {"output": str(out)}})

    with fits.open(out, memmap=False) as hdul:
        hdr = hdul[1].header
        assert hdr["TELESCOP"] == "CBASS"
        assert hdr["CALDATE"] == "2025-02-01"
        assert hdr["CUSTOMK"] == "custom_value"
        assert hdr["PIPEKEY"] == "set_by_pipeline"


def test_finalmap_can_limit_to_curated_header_keys(tmp_path: Path):
    src = tmp_path / "input.fits"
    out = tmp_path / "output.fits"
    _write_source_with_header(src)

    stage = FinalMap()
    bundle = _build_bundle()
    bundle.source_path = str(src)

    stage.run(
        bundle,
        stage_cfg={"preserve_all_headers": False},
        full_cfg={"FinalMap": {"output": str(out)}},
    )

    with fits.open(out, memmap=False) as hdul:
        hdr = hdul[1].header
        assert hdr["TELESCOP"] == "CBASS"
        assert hdr["CALDATE"] == "2025-02-01"
        assert "CUSTOMK" not in hdr
        assert hdr["PIPEKEY"] == "set_by_pipeline"
