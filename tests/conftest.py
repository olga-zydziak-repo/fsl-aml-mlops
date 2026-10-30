"""Shared fixtures: a synthetic SAML-D freeze, the loader pointed at it, and GCP stubbed out.

No real data and no network anywhere in the test suite. `frozen` runs the ORIGINAL
freeze code (scripts/bootstrap_fsl_data.py) on synthetic transactions once per
session; `local_gcs` makes the loader read that freeze instead of GCS;
`fake_gcp` replaces google.cloud.aiplatform / google.cloud.storage with recorders,
so the Experiments handoff (train -> gate -> active learning) can be checked call
by call without a project.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = "samld-case"


def import_script(name: str):
    """Import scripts/<name>.py as a module (the entry points are scripts, not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def frozen(tmp_path_factory):
    """Run the ORIGINAL freeze code on synthetic transactions."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("networkx")
    pytest.importorskip("scipy")
    pytest.importorskip("pyarrow")
    from tests import samld_synth

    out = tmp_path_factory.mktemp("samld")
    csv = samld_synth.write_csv(out / "SAML-D.csv", seed=0)
    bootstrap = import_script("bootstrap_fsl_data")
    mp = pytest.MonkeyPatch()
    mp.setenv("SAMLD_CSV", str(csv))
    root = out / "frozen"
    root.mkdir()
    bootstrap._build_samld_case(str(root))
    mp.undo()
    sha = hashlib.sha256(b"".join((root / n).read_bytes() for n in sorted(
        ("cases.parquet", "schema.json", "folds.json", "stats.json")))).hexdigest()
    return {"dir": root, "sha": sha,
            "schema": json.loads((root / "schema.json").read_text()),
            "table": pd.read_parquet(root / "cases.parquet"),
            "folds": json.loads((root / "folds.json").read_text()),
            "stats": json.loads((root / "stats.json").read_text())}


@pytest.fixture
def local_gcs(frozen, monkeypatch):
    """The loader reads the manifest and the archive from GCS; point both at the local freeze."""
    from fsl.data import samld

    monkeypatch.setattr(samld, "fetch_manifest", lambda project, bucket, name: {"archive_sha256": frozen["sha"]})
    monkeypatch.setattr(samld, "_ensure_local", lambda cfg, manifest, name=None: frozen["dir"])
    return frozen


class FakeAiplatform:
    """google.cloud.aiplatform as the code under test sees it: every call is recorded in order.

    `fail_on` names a function that raises (to check that logging never takes the result down).
    """

    def __init__(self, fail_on: str = ""):
        self.calls: list = []
        self.fail_on = fail_on

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def f(*a, **kw):
            if name == self.fail_on:
                raise RuntimeError(f"stubbed failure in {name}")
            self.calls.append((name, a, kw))
        return f

    def names(self) -> list:
        return [c[0] for c in self.calls]

    def first(self, name: str):
        """Positional args of the first call to `name` (its dict for log_* calls)."""
        for n, a, kw in self.calls:
            if n == name:
                return a[0] if len(a) == 1 else (a, kw)
        raise KeyError(name)


class FakeStorage:
    """google.cloud.storage with one bucket: blobs download what `files` holds, uploads are recorded."""

    def __init__(self, files: dict | None = None):
        self.files = dict(files or {})   # blob name -> bytes
        self.uploaded: dict = {}         # blob name -> local path

    def Client(self, project=None, **kw):   # noqa: N802 - mirrors the SDK
        store = self

        class _Blob:
            def __init__(self, name):
                self.name = name

            def download_as_bytes(self):
                if self.name not in store.files:
                    raise FileNotFoundError(self.name)
                return store.files[self.name]

            def exists(self):
                return self.name in store.files

            def upload_from_filename(self, path):
                store.uploaded[self.name] = str(path)
                store.files[self.name] = Path(path).read_bytes()

            def upload_from_string(self, body, content_type=None):
                store.files[self.name] = body.encode() if isinstance(body, str) else body

        class _Bucket:
            def __init__(self, name):
                self.name = name

            def blob(self, name):
                return _Blob(name)

        class _Client:
            def bucket(self, name):
                return _Bucket(name)

        return _Client()


@pytest.fixture
def fake_gcp(monkeypatch):
    """Install stub google.cloud.aiplatform / google.cloud.storage modules for one test."""
    ai, st = FakeAiplatform(), FakeStorage()
    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    cloud.aiplatform, cloud.storage = ai, st
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", ai)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", st)
    return {"aiplatform": ai, "storage": st}
