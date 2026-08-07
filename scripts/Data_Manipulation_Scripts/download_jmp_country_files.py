"""Download JMP household country files from washdata.org.

Files are saved to data/original_data/jmp_country_files/{ISO3}.xlsx.
Skips downloads when a valid xlsx already exists locally or in the sibling
well-coverage repository (copied rather than re-downloaded).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pycountry

URL = "https://washdata.org/data/country/{iso3}/household/download"


def _repo_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    local_dir = repo_root / "data" / "original_data" / "jmp_country_files"
    sibling_dir = repo_root.parent / "well-coverage" / "data" / "raw"
    return local_dir, sibling_dir


def _is_valid_xlsx(path: Path) -> bool:
    if not path.is_file():
        return False
    with open(path, "rb") as handle:
        return handle.read(2) == b"PK"


def _copy_from_sibling(local_dir: Path, sibling_dir: Path) -> int:
    if not sibling_dir.is_dir():
        return 0
    copied = 0
    local_dir.mkdir(parents=True, exist_ok=True)
    for src in sibling_dir.glob("*.xlsx"):
        if not _is_valid_xlsx(src):
            continue
        dest = local_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1
    return copied


def main() -> int:
    local_dir, sibling_dir = _repo_paths()
    local_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_from_sibling(local_dir, sibling_dir)
    if copied:
        print(f"copied {copied} files from {sibling_dir}")

    codes = sorted(country.alpha_3 for country in pycountry.countries)
    to_download = [
        iso3
        for iso3 in codes
        if not _is_valid_xlsx(local_dir / f"{iso3}.xlsx")
    ]
    print(f"{len(codes)} ISO3 codes; {len(to_download)} to download")

    for index in range(0, len(to_download), 8):
        batch = to_download[index : index + 8]
        processes = []
        for iso3 in batch:
            out = local_dir / f"{iso3}.xlsx"
            processes.append(
                subprocess.Popen(
                    [
                        "curl",
                        "-sL",
                        "--max-time",
                        "120",
                        "-o",
                        str(out),
                        URL.format(iso3=iso3),
                    ]
                )
            )
        for process in processes:
            process.wait()
        print(f"  downloaded {min(index + 8, len(to_download))}/{len(to_download)}")

    kept, removed = 0, 0
    for path in sorted(local_dir.glob("*.xlsx")):
        if _is_valid_xlsx(path):
            kept += 1
        else:
            path.unlink(missing_ok=True)
            removed += 1

    print(f"kept {kept} valid files, removed {removed} invalid responses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
