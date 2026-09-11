#!/usr/bin/env python3
"""Materialise the shared BTC 5m datasets into a local ``data/`` directory.

The datasets live in the private ``btc-5m-data`` repository as gzipped CSVs so
every model repository can stay small. This script clones (or refreshes) that
repository and unpacks the archives it needs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path


DEFAULT_SOURCE_REPO = "bangxiao0927/btc-5m-data"
DEFAULT_CACHE = Path.home() / ".cache" / "btc-5m-data"


def _run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"command failed: {' '.join(command)}\n{detail}")


def sync_cache(source_repo: str, cache: Path, refresh: bool) -> Path:
    """Clone or update the private dataset repository."""

    if (cache / ".git").is_dir():
        if refresh:
            _run(["git", "fetch", "--depth", "1", "origin", "HEAD"], cwd=cache)
            _run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=cache)
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = (
        source_repo
        if "://" in source_repo or source_repo.startswith("git@")
        else f"https://github.com/{source_repo}.git"
    )
    _run(["git", "clone", "--depth", "1", url, str(cache)])
    return cache


def load_manifest(cache: Path) -> dict:
    path = cache / "MANIFEST.json"
    if not path.is_file():
        raise SystemExit(f"manifest not found in {cache}; is this the dataset repo?")
    return json.loads(path.read_text())


def verify(archive: Path, expected: str | None) -> None:
    if not expected:
        return
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"checksum mismatch for {archive.name}")


def unpack(archive: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive, "rb") as source, target.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=1024 * 1024)
    return target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unpack the shared BTC 5m CSVs into ./data"
    )
    parser.add_argument("--dest", type=Path, default=Path("data"))
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="use an existing checkout of the dataset repo instead of cloning",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="glob of file names to unpack; repeatable",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="update an existing cache checkout before unpacking",
    )
    parser.add_argument("--list", action="store_true", help="list contents and exit")
    parser.add_argument("--no-verify", action="store_true", help="skip sha256 checks")
    args = parser.parse_args()

    if args.from_dir is not None:
        cache = args.from_dir.resolve()
    elif args.list and not (args.cache / "MANIFEST.json").is_file():
        cache = sync_cache(args.source_repo, args.cache, refresh=False)
    else:
        cache = sync_cache(args.source_repo, args.cache, refresh=args.refresh)

    manifest = load_manifest(cache)
    entries = manifest.get("files", [])
    if args.only:
        entries = [
            entry
            for entry in entries
            if any(fnmatch(entry["file"], pattern) for pattern in args.only)
        ]

    if args.list:
        print(f"source: {cache}")
        for entry in entries:
            print(
                f"  {entry['file']:38} rows={entry['rows']:>8} "
                f"gz={entry['bytes_gz'] / 1e6:6.2f}MB raw={entry['bytes_raw'] / 1e6:7.2f}MB"
            )
        print(f"total: {len(entries)} files")
        return

    written = skipped = 0
    for entry in entries:
        archive = cache / entry["archive"]
        target = args.dest / entry["file"]
        if target.exists() and not args.force:
            skipped += 1
            continue
        if not archive.is_file():
            print(f"missing archive: {archive}", file=sys.stderr)
            continue
        if not args.no_verify:
            verify(archive, entry.get("sha256_gz"))
        size = unpack(archive, target)
        written += 1
        print(f"wrote {target} ({size / 1e6:.2f}MB)")
    print(f"done: {written} written, {skipped} already present, dest={args.dest.resolve()}")


if __name__ == "__main__":
    main()
