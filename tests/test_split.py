"""Experiment design in the pipeline (fsl.data.split + prepare_experiment_entry).

The contract under test: moving folds, standardisation and floors from the
freeze into a pipeline step must not change a single fold, statistic or floor.
The freeze itself (scripts/bootstrap_fsl_data.py, unchanged) is run on
synthetic SAML-D-shaped transactions and the new code is compared with what it
wrote. No GCS, no real data: the loader's two GCS calls are replaced by a local
directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("networkx")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from tests.conftest import DATASET, import_script  # noqa: E402  (fixtures frozen, local_gcs live there too)
from fsl.data import samld  # noqa: E402
from fsl.data import split as S  # noqa: E402


def test_split_matches_the_freeze(frozen):
    sp = S.make_split(frozen["table"], frozen["schema"], pool=4, k=5, q=5, seed=0)
    cmp = S.compare_with_frozen(sp, frozen["folds"], frozen["stats"], frozen["schema"]["log1p_features"])
    assert cmp == {"same_folds": True, "same_log1p": True, "same_stat_keys": True, "stats_max_abs_diff": 0.0}
    assert len(sp["folds"]) == frozen["schema"]["n_folds"] == 5
    assert sp["leftover_tested_in_extra_fold"] == frozen["schema"]["leftover_tested_in_extra_fold"]


def test_snapshot_with_infeasible_class_and_shuffled_rows(frozen):
    """An Airflow snapshot need not be pre-filtered or ordered like the freeze."""
    C = frozen["table"]
    extra = C[C["label"] == C["label"].iloc[0]].head(3).copy()
    extra["label"], extra["grp"], extra["case"] = "Tiny_typology", -1, -extra["case"]
    C2 = pd.concat([C, extra], ignore_index=True).sample(frac=1.0, random_state=1).reset_index(drop=True)
    sp = S.make_split(C2, frozen["schema"], pool=4, k=5, q=5, seed=0)
    cmp = S.compare_with_frozen(sp, frozen["folds"], frozen["stats"], frozen["schema"]["log1p_features"])
    assert cmp["same_folds"] and cmp["same_log1p"] and cmp["stats_max_abs_diff"] < 1e-12
    assert sp["dropped_classes"] == ["Tiny_typology"]


def test_number_of_folds_follows_the_data():
    names = [f"T{i:02d}" for i in range(16)]
    assert len(S.make_folds(names, 4, 0)[0]) == 4                 # 16 = 4 blocks, no leftover
    folds, left = S.make_folds(names + ["T16", "T17"], 4, 0)
    assert len(folds) == 5 and len(left) == 2 and len(folds[-1]["test"]) == 4
    for f in folds:
        assert not set(f["test"]) & set(f["val"]) and not set(f["test"]) & set(f["train"])


def test_floors_match_the_freeze_notebook(frozen):
    """Reference = cell 3 of samld_case_freeze.ipynb, procedure verbatim."""
    from fsl.config import TrainConfig
    from fsl.models.mlp import MLP
    from fsl.training.loop import evaluate

    bundle = {"windows": frozen["table"], "schema": frozen["schema"], "folds": frozen["folds"],
              "stats": frozen["stats"], "name": DATASET}
    ref = {}
    for k in range(frozen["schema"]["n_folds"]):
        accs = []
        for seed in (0, 1, 2):
            c = TrainConfig(project="p", bucket="b", dataset=DATASET, seed=seed, fold=k,
                            n_way=4, k_shot=5, query=5, log_to_vertex=False)
            _, _, te = samld.build_task_samplers(bundle, c)
            torch.manual_seed(seed)
            m = MLP(in_features=len(frozen["schema"]["feature_names"]), hid=64)
            accs.append(float(evaluate(m, te, c, 60, torch.device("cpu")).mean()))
        ref[str(k)] = round(float(np.mean(accs)), 4)
    got = S.compute_floors(bundle, n_way=4, k_shot=5, query=5, episodes=60)
    assert got["floors"] == ref
    assert got["thresholds"] == {k: round(min(f + 0.10, f + 0.5 * (1 - f)), 4) for k, f in ref.items()}


def _run_prepare(tmp_path, extra=()):
    prep = import_script("prepare_experiment_entry")
    out = tmp_path / "kfp"
    argv = ["prepare_experiment_entry.py", "--project", "p", "--bucket", "b", "--dataset", DATASET,
            "--split-root", str(tmp_path / "bucket"), "--floor-episodes", "40",
            "--split-uri-output-path", str(out / "uri"), "--split-sha-output-path", str(out / "sha"),
            "--fold-ids-output-path", str(out / "ids"), "--report-html-path", str(out / "report.html"),
            "--summary-output-path", str(out / "summary"), *extra]
    old = sys.argv
    sys.argv = argv
    try:
        prep.main()
    finally:
        sys.argv = old
    return {n: (out / n).read_text() for n in ("uri", "sha", "ids", "summary")}


def test_prepare_step_writes_a_verifiable_split(local_gcs, tmp_path):
    res = _run_prepare(tmp_path)
    assert json.loads(res["ids"]) == ["0", "1", "2", "3", "4"]
    doc = S.load_split(res["uri"])
    assert doc["split_sha"] == res["sha"] and doc["data_sha"] == local_gcs["sha"]
    assert res["uri"].endswith(S.split_path(DATASET, doc["split_sha"]))
    assert "folds True" in res["summary"]
    # same input -> same address and content (rerunning the step is idempotent)
    assert _run_prepare(tmp_path)["sha"] == res["sha"]
    # an edited file no longer matches its own hash
    tampered = json.loads(Path(res["uri"]).read_text())
    tampered["thresholds"]["0"] = 0.5
    Path(res["uri"]).write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="split_sha"):
        S.load_split(res["uri"])


def test_prepare_step_stops_when_the_freeze_disagrees(local_gcs, tmp_path, monkeypatch):
    bad = [dict(f) for f in local_gcs["folds"]]
    bad[0], bad[1] = {**bad[0], "test": bad[1]["test"]}, {**bad[1], "test": bad[0]["test"]}
    real_load = samld.load_frozen_samld

    def load_with_wrong_folds(cfg):
        bundle, sha = real_load(cfg)
        return {**bundle, "folds": bad}, sha

    monkeypatch.setattr(samld, "load_frozen_samld", load_with_wrong_folds)
    with pytest.raises(SystemExit, match="differs from the frozen"):
        _run_prepare(tmp_path)


def test_loader_with_split_uri_equals_the_freeze(local_gcs, tmp_path):
    from fsl.config import TrainConfig

    uri = _run_prepare(tmp_path)["uri"]
    for fold in (0, 4):
        base = dict(project="p", bucket="b", dataset=DATASET, seed=3, fold=fold,
                    n_way=4, k_shot=5, query=5, log_to_vertex=False)
        legacy, _ = samld.load_frozen_samld(TrainConfig(**base))
        viasplit, _ = samld.load_frozen_samld(TrainConfig(**base, split_uri=uri))
        assert legacy["split_sha"] == "" and viasplit["split_sha"] == S.load_split(uri)["split_sha"]
        a = samld.build_task_samplers(legacy, TrainConfig(**base))
        b = samld.build_task_samplers(viasplit, TrainConfig(**base, split_uri=uri))
        for sa, sb in zip(a, b):
            assert np.array_equal(sa.X, sb.X) and sa.classes == sb.classes
            for _ in range(5):                     # same seed -> same episodes
                (xa, ya), (xb, yb) = sa.sample(), sb.sample()
                assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_loader_refuses_a_split_from_other_data(local_gcs, tmp_path, monkeypatch):
    from fsl.config import TrainConfig

    uri = _run_prepare(tmp_path)["uri"]
    monkeypatch.setattr(samld, "fetch_manifest", lambda project, bucket, name: {"archive_sha256": "0" * 64})
    with pytest.raises(ValueError, match="refusing to mix"):
        samld.load_frozen_samld(TrainConfig(project="p", bucket="b", dataset=DATASET, split_uri=uri))


def test_training_is_identical_under_the_split(local_gcs, tmp_path):
    """The model trained from split.json is the model trained from the freeze."""
    from fsl.config import TrainConfig
    from fsl.training.loop import train

    uri = _run_prepare(tmp_path)["uri"]
    base = dict(project="p", bucket="b", dataset=DATASET, seed=0, fold=3, n_way=4, k_shot=5, query=5,
                encoder="mlp", in_channels=31, image_size=1, embedding_hid=16, train_iters=20,
                eval_every=10, val_episodes=5, test_episodes=30, log_to_vertex=False)
    a = train(TrainConfig(**base))
    b = train(TrainConfig(**base, split_uri=uri))
    assert a["test_accuracy_mean"] == b["test_accuracy_mean"]
    for (ka, va), (kb, vb) in zip(a["model"].state_dict().items(), b["model"].state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


def test_prepare_compares_floors_with_the_workbench_file(local_gcs, tmp_path):
    first = S.load_split(_run_prepare(tmp_path)["uri"])
    legacy = json.dumps({"sha": local_gcs["sha"], "floors": first["floors"], "thresholds": first["thresholds"]})
    _run_prepare(tmp_path, extra=("--legacy-floors-json", legacy))
    report = (tmp_path / "kfp" / "report.html").read_text()
    assert "folds identical" in report and "max |diff| 0.0000" in report
