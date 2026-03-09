#!/usr/bin/env python3
"""Update ADS header value in the template and/or generated FITS files."""

from __future__ import annotations

import argparse
from pathlib import Path

from astropy.io import fits


DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "postprocess" / "FinalMapHeader.hdr"


def update_template(template_path: Path, ads_value: str) -> None:
    hdr = fits.Header.fromtextfile(str(template_path))
    hdr["ADS"] = ads_value
    hdr.totextfile(str(template_path), overwrite=True)


def update_fits(path: Path, ads_value: str, ext: int) -> None:
    with fits.open(path, mode="update", memmap=False) as hdul:
        if ext >= len(hdul):
            raise IndexError(f"{path}: extension {ext} not found")
        hdul[ext].header["ADS"] = ads_value
        hdul.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Update ADS reference string in C-BASS headers")
    parser.add_argument("ads_value", help="New ADS reference string")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Path to FinalMapHeader.hdr")
    parser.add_argument("--fits", nargs="*", default=[], help="Optional FITS files to patch")
    parser.add_argument("--ext", type=int, default=1, help="Header extension for FITS files (default: 1)")
    args = parser.parse_args()

    update_template(args.template, args.ads_value)
    print(f"[updated] template {args.template}")

    for f in args.fits:
        p = Path(f)
        update_fits(p, args.ads_value, args.ext)
        print(f"[updated] {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
