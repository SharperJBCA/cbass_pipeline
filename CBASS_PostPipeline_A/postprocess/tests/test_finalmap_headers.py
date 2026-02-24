from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyError

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


def test_finalmap_defaults_to_curated_header_keys(tmp_path: Path):
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
        assert "CUSTOMK" not in hdr
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


def test_finalmap_can_preserve_all_non_structural_headers(tmp_path: Path):
    src = tmp_path / "input.fits"
    out = tmp_path / "output.fits"
    _write_source_with_header(src)

    stage = FinalMap()
    bundle = _build_bundle()
    bundle.source_path = str(src)

    stage.run(
        bundle,
        stage_cfg={"preserve_all_headers": True},
        full_cfg={"FinalMap": {"output": str(out)}},
    )

    with fits.open(out, memmap=False) as hdul:
        hdr = hdul[1].header
        assert hdr["TELESCOP"] == "CBASS"
        assert hdr["CALDATE"] == "2025-02-01"
        assert hdr["CUSTOMK"] == "custom_value"
        assert hdr["PIPEKEY"] == "set_by_pipeline"


def test_finalmap_updates_geometry_cards_from_output_bundle(tmp_path: Path):
    out = tmp_path / "output.fits"

    stage = FinalMap()
    bundle = _build_bundle(nside=4)
    bundle.coords = "C"

    stage.run(bundle, stage_cfg={}, full_cfg={"FinalMap": {"output": str(out)}})

    with fits.open(out, memmap=False) as hdul:
        hdr = hdul[1].header
        assert hdr["NSIDE"] == 4
        assert hdr["FIRSTPIX"] == 0
        assert hdr["LASTPIX"] == 12 * 4 * 4 - 1
        assert hdr["COORDSYS"] == "C"


def test_finalmap_updates_beam_cards_when_deconvolved(tmp_path: Path):
    out = tmp_path / "output.fits"

    stage = FinalMap()
    bundle = _build_bundle(nside=4)
    bundle.headers.update({"DCONV": True, "FWHM_OUT": 1.0})

    stage.run(bundle, stage_cfg={}, full_cfg={"FinalMap": {"output": str(out)}})

    with fits.open(out, memmap=False) as hdul:
        hdr = hdul[1].header
        assert float(hdr["BMAJ"]) == 1.0
        assert float(hdr["BMIN"]) == 1.0


def test_finalmap_keeps_input_beam_cards_without_deconvolution(tmp_path: Path):
    src = tmp_path / "input.fits"
    out = tmp_path / "output.fits"
    _write_source_with_header(src)

    with fits.open(src, mode="update", memmap=False) as hdul:
        hdul[1].header["BMAJ"] = 0.75
        hdul[1].header["BMIN"] = 0.75

    stage = FinalMap()
    bundle = _build_bundle(nside=4)
    bundle.source_path = str(src)
    bundle.headers.update({"DCONV": False})

    stage.run(bundle, stage_cfg={"preserve_all_headers": True}, full_cfg={"FinalMap": {"output": str(out)}})

    with fits.open(out, memmap=False) as hdul:
        hdr = hdul[1].header
        assert float(hdr["BMAJ"]) == 0.75
        assert float(hdr["BMIN"]) == 0.75


def test_finalmap_skips_optional_template_cards_when_stage_not_run(tmp_path: Path):
    out = tmp_path / "output.fits"
    template = tmp_path / "template.hdr"
    template.write_text(
        "\n".join(
            [
                "XTENSION= 'BINTABLE'",
                "BITPIX  = 8",
                "NAXIS   = 2",
                "NAXIS1  = 4",
                "NAXIS2  = 0",
                "PCOUNT  = 0",
                "GCOUNT  = 1",
                "TFIELDS = 1",
                "TTYPE1  = 'I_STOKES'",
                "TFORM1  = 'E'",
                "TUNIT1  = 'K_RJ'",
                "DCONV   = T / deconvolution flag",
                "BL_FILE = 'none' / optional beam file",
                "SS_SSUB = T / source subtraction flag",
            ]
        )
        + "\n"
    )

    stage = FinalMap()
    bundle = _build_bundle(nside=4)

    stage.run(
        bundle,
        stage_cfg={"header_template": str(template)},
        full_cfg={"FinalMap": {"output": str(out)}},
    )

    with fits.open(out, memmap=False) as hdul:
        hdr = hdul[1].header
        assert "DCONV" not in hdr
        assert "BL_FILE" not in hdr
        assert "SS_SSUB" not in hdr


def test_safe_card_comment_handles_verify_errors():
    stage = FinalMap()
    card = fits.Card("TESTKEY", 1)

    def _raise_verify_error():
        raise VerifyError("CONTINUE cards must have string values.")

    card._parse_comment = _raise_verify_error
    card._comment = None

    assert stage._safe_card_comment(card) == ""
