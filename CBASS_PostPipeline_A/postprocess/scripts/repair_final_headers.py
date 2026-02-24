#!/usr/bin/env python3
"""Safely repair FinalMap FITS headers in-place or to new files.

Use this for previously generated maps that may carry inconsistent deconvolution
metadata (e.g. DCONV=False but BL_FILE/FWHM_OUT still present).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from astropy.io import fits

REMOVE_IF_NOT_DECONV = ("BL_FILE", "FWHM_OUT")


def repair_header(hdr: fits.Header, native_fwhm_deg: float | None = None) -> bool:
    """Repair one header; returns True if modified."""
    changed = False

    dconv = bool(hdr.get("DCONV", False))
    if not dconv:
        for key in REMOVE_IF_NOT_DECONV:
            if key in hdr:
                del hdr[key]
                changed = True

    if (not dconv) and native_fwhm_deg is not None:
        for beam_key in ("BMAJ", "BMIN"):
            if beam_key not in hdr or float(hdr[beam_key]) != float(native_fwhm_deg):
                hdr[beam_key] = float(native_fwhm_deg)
                changed = True

    return changed


def process_file(path: Path, output: Path | None, ext: int, native_fwhm_deg: float | None) -> bool:
    mode = "readonly" if output else "update"
    with fits.open(path, mode=mode, memmap=False) as hdul:
        if ext >= len(hdul):
            raise IndexError(f"{path}: extension {ext} not found")
        changed = repair_header(hdul[ext].header, native_fwhm_deg=native_fwhm_deg)
        if output:
            hdul.writeto(output, overwrite=True)
        elif changed:
            hdul.flush()
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair C-BASS final-map FITS headers")
    parser.add_argument("files", nargs="+", help="FITS files to process")
    parser.add_argument("--ext", type=int, default=1, help="Header extension index (default: 1)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If set, input files are not modified in-place.",
    )
    parser.add_argument(
        "--native-fwhm-deg",
        type=float,
        default=0.75,
        help="Native beam FWHM in degrees to restore when DCONV=False (default: 0.75).",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in args.files:
        src = Path(file_name)
        dst = args.output_dir / src.name if args.output_dir else None
        changed = process_file(src, dst, args.ext, args.native_fwhm_deg)
        action = "updated" if changed else "no-change"
        target = dst if dst else src
        print(f"[{action}] {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
