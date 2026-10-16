#!/usr/bin/env python3
"""
verify_frozen.py — Verify that a dataset frozen in GCS is byte-for-byte intact
and consistent with its MANIFEST.

This is the enforcement half of MLOps rung 1: bootstrap_fsl_data.py *records*
the checksums; this script *checks* them. It answers "are these the exact bytes
I froze?" — not "are these data good for training" (that is a separate concern).

What it checks, in order:
  1. Downloads MANIFEST.json from gs://<bucket>/raw/<dataset>/.
  2. Downloads the archive and verifies its SHA256 against the manifest.
  3. (default) Unpacks the archive and verifies every file's SHA256 against the
     per-file map in the manifest — catches a single corrupted file inside.

Exit code 0 on success, 1 on any mismatch — so it can gate a CI pipeline:
"do not train if the data does not match the manifest."

Designed for Vertex AI Workbench (Application Default Credentials). No secrets.

Usage:
    python verify_frozen.py --project my-proj --dataset omniglot
    python verify_frozen.py --project my-proj --dataset omniglot --archive-only
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import NoReturn

LOG = logging.getLogger("verify_frozen")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def die(msg: str, exc: Exception | None = None) -> NoReturn:
    LOG.error(msg)
    if exc is not None:
        LOG.debug("cause: %r", exc)
    sys.exit(1)


def resolve_project(arg: str | None) -> str:
    if arg:
        return arg
    env = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if env:
        return env
    die("No --project given and GOOGLE_CLOUD_PROJECT is not set.")


def get_bucket(project: str, bucket_name: str):
    try:
        from google.cloud import storage
    except ImportError as e:
        die("google-cloud-storage not installed. Run: pip install google-cloud-storage", e)
    from google.cloud.exceptions import NotFound
    client = storage.Client(project=project)
    try:
        return client.get_bucket(bucket_name)
    except NotFound as e:
        die(f"Bucket gs://{bucket_name} not found.", e)


def download_blob_bytes(bucket, name: str) -> bytes:
    blob = bucket.blob(name)
    if not blob.exists():
        die(f"Missing object: gs://{bucket.name}/{name}")
    return blob.download_as_bytes()


def verify_per_file(archive_path: Path, expected: dict[str, str]) -> tuple[int, list[str]]:
    """Unpack the archive in memory and check each member's SHA256.

    Returns (number_checked, list_of_problems).
    """
    problems: list[str] = []
    seen: set[str] = set()
    # The archive was written as an uncompressed tar inside a gzip stream.
    with open(archive_path, "rb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            with tarfile.open(fileobj=gz, mode="r") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = member.name
                    seen.add(name)
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        problems.append(f"unreadable member: {name}")
                        continue
                    digest = sha256_bytes(fobj.read())
                    exp = expected.get(name)
                    if exp is None:
                        problems.append(f"file not in manifest: {name}")
                    elif exp != digest:
                        problems.append(f"checksum mismatch: {name}")
    # files in manifest but absent from archive
    for name in expected:
        if name not in seen:
            problems.append(f"file missing from archive: {name}")
    return len(seen), problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify a GCS-frozen dataset against its MANIFEST (MLOps rung 1)."
    )
    ap.add_argument("--dataset", default="omniglot")
    ap.add_argument("--project", default=None, help="GCP project (else $GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--bucket", default=None, help="Bucket name (default: <project>-fsl-data)")
    ap.add_argument("--archive-only", action="store_true",
                    help="Only verify the archive checksum; skip per-file checks.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    project = resolve_project(args.project)
    bucket_name = args.bucket or f"{project}-fsl-data"
    prefix = f"raw/{args.dataset}"
    archive_blob = f"{prefix}/{args.dataset}.tar.gz"
    manifest_blob = f"{prefix}/MANIFEST.json"

    bucket = get_bucket(project, bucket_name)

    # 1. Manifest.
    LOG.info("Reading manifest: gs://%s/%s", bucket_name, manifest_blob)
    manifest = json.loads(download_blob_bytes(bucket, manifest_blob).decode("utf-8"))
    expected_archive_sha = manifest.get("archive_sha256")
    expected_files = manifest.get("files_sha256", {})
    if not expected_archive_sha:
        die("Manifest has no 'archive_sha256'; cannot verify.")

    # 2. Archive checksum.
    LOG.info("Downloading archive: gs://%s/%s", bucket_name, archive_blob)
    with tempfile.TemporaryDirectory(prefix="verify-") as tmp:
        archive_local = Path(tmp) / f"{args.dataset}.tar.gz"
        archive_local.write_bytes(download_blob_bytes(bucket, archive_blob))

        actual_archive_sha = sha256_file(archive_local)
        if actual_archive_sha != expected_archive_sha:
            LOG.error("Archive SHA256 MISMATCH")
            LOG.error("  expected: %s", expected_archive_sha)
            LOG.error("  actual  : %s", actual_archive_sha)
            die("Archive does not match manifest — data is NOT intact.")
        LOG.info("Archive SHA256 OK: %s", actual_archive_sha)

        # 3. Per-file checks.
        if args.archive_only:
            LOG.info("Skipping per-file checks (--archive-only).")
        elif not expected_files:
            LOG.warning("Manifest has no per-file checksums; archive-level check only.")
        else:
            LOG.info("Verifying %d files inside the archive ...", len(expected_files))
            checked, problems = verify_per_file(archive_local, expected_files)
            if problems:
                for p in problems[:20]:
                    LOG.error("  %s", p)
                if len(problems) > 20:
                    LOG.error("  ... and %d more", len(problems) - 20)
                die(f"{len(problems)} file-level problem(s) found — data is NOT intact.")
            LOG.info("All %d files match the manifest.", checked)

    LOG.info("VERIFIED: gs://%s/%s is intact and consistent with its manifest. OK", bucket_name, prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
