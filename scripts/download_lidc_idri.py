#!/usr/bin/env python3
"""Download and extract the CT series in TCIA's LIDC-IDRI collection.

The script is safe to run again: completed series are skipped and partial ZIP
downloads are resumed when the NBIA endpoint supports HTTP range requests.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests


API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"


def ct_series(metadata: Path) -> list[tuple[str, str]]:
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        return [
            (row["Subject ID"], row["Series ID"])
            for row in rows
            if row["Modality"].strip().upper() == "CT"
        ]


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe path in ZIP: {member.filename}")
    archive.extractall(destination)


def download_one(item: tuple[str, str], root: Path, cache: Path) -> tuple[str, int]:
    patient, series_uid = item
    final_dir = root / patient / series_uid
    marker = final_dir / ".complete"
    if marker.exists():
        return "skipped", 0

    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / f"{series_uid}.zip.part"
    url = f"{API}?SeriesInstanceUID={quote(series_uid)}"

    for attempt in range(1, 7):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            with zipfile.ZipFile(partial) as archive:
                bad_file = archive.testzip()
                if bad_file:
                    raise zipfile.BadZipFile(f"CRC failure: {bad_file}")
                temporary = final_dir.with_name(f".extracting-{series_uid}")
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir(parents=True)
                safe_extract(archive, temporary)

            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temporary, final_dir)
            marker.touch()
            size = partial.stat().st_size
            partial.unlink()
            return "downloaded", size
        except Exception:
            if attempt == 6:
                raise
            time.sleep(min(60, 2**attempt))

    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    items = ct_series(args.metadata)
    cache = args.output.parent / ".download-cache"
    print(f"Found {len(items)} CT series; output={args.output}; workers={args.workers}", flush=True)

    downloaded = skipped = failed = bytes_received = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, item, args.output, cache): item for item in items}
        for finished, future in enumerate(as_completed(futures), start=1):
            patient, series_uid = futures[future]
            try:
                status, size = future.result()
                downloaded += status == "downloaded"
                skipped += status == "skipped"
                bytes_received += size
                print(
                    f"[{finished}/{len(items)}] {status}: {patient}/{series_uid} "
                    f"(session download {bytes_received / 2**30:.2f} GiB)",
                    flush=True,
                )
            except Exception as error:
                failed += 1
                print(f"[{finished}/{len(items)}] FAILED: {patient}/{series_uid}: {error}", flush=True)

    print(f"Complete: downloaded={downloaded}, skipped={skipped}, failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
