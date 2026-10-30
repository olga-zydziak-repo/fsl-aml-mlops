"""One Experiments run per training (v1.2), checked without a GCP project.

The contract: the code that trains opens the run (parameters, the train/val
curve as time series, the test result), hands its name on, and later steps
resume that run and add their numbers. Logging must never cost a training its
result. google.cloud.aiplatform is a recorder here (tests/conftest.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("networkx")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from tests.conftest import DATASET, import_script  # noqa: E402

SMALL = dict(n_way=4, k_shot=5, query=5, encoder="mlp", in_channels=31, image_size=1,
             embedding_hid=16, train_iters=20, eval_every=10, val_episodes=5, test_episodes=30)


def _cfg(**over):
    from fsl.config import TrainConfig

    base = dict(project="p", bucket="b", dataset=DATASET, seed=0, fold=3, experiment_name="exp",
                log_to_vertex=True, **SMALL)
    base.update(over)
    return TrainConfig(**base)


def test_training_opens_the_run_logs_the_curve_and_the_result(local_gcs, fake_gcp):
    from fsl.training.loop import train

    ai = fake_gcp["aiplatform"]
    res = train(_cfg(), run_name="job-123-f3", extra_params={"run_type": "campaign", "split_sha": "abc"})
    assert res["experiment_run"] == "job-123-f3"
    names = ai.names()
    assert names[:3] == ["init", "start_run", "log_params"]
    assert names[-2:] == ["log_metrics", "end_run"]
    assert ai.first("init")[1] == {"project": "p", "location": "us-central1", "experiment": "exp"}
    assert ai.first("start_run") == "job-123-f3"
    params = ai.first("log_params")
    assert params["fold"] == 3 and params["train_iters"] == 20 and params["encoder"] == "mlp"
    assert params["dataset_archive_sha256"] == local_gcs["sha"]
    assert params["run_type"] == "campaign" and params["split_sha"] == "abc"
    # the curve: iteration 1 and every eval_every, as TensorBoard time series
    steps = [kw["step"] for n, a, kw in ai.calls if n == "log_time_series_metrics"]
    assert steps == [1, 10, 20]
    point = ai.first("log_time_series_metrics")
    assert set(point) == {"train_loss", "train_acc", "val_acc"}
    final = ai.first("log_metrics")
    assert final["test_accuracy_mean"] == res["test_accuracy_mean"]
    assert set(final) == {"test_accuracy_mean", "test_accuracy_ci95", "test_accuracy_std", "train_time_seconds"}


def test_training_without_logging_touches_nothing(local_gcs, fake_gcp):
    from fsl.training.loop import train

    res = train(_cfg(log_to_vertex=False))
    assert res["experiment_run"] == "" and fake_gcp["aiplatform"].calls == []


def test_a_failed_log_never_costs_the_training_its_result(local_gcs, fake_gcp, capsys):
    """start_run fails: training completes, returns no run name, and stops calling the SDK."""
    from fsl.training.loop import train

    ai = fake_gcp["aiplatform"]
    ai.fail_on = "start_run"
    ok = train(_cfg(log_to_vertex=False))
    res = train(_cfg(), run_name="job-123-f3")
    assert res["experiment_run"] == "" and res["test_accuracy_mean"] == ok["test_accuracy_mean"]
    assert ai.names() == ["init"]
    assert "WARNING: Vertex Experiments logging stopped at start_run" in capsys.readouterr().out
    # a failure in the middle keeps the run name (the run exists, later steps may still resume it)
    ai.calls, ai.fail_on = [], "log_time_series_metrics"
    res = train(_cfg(), run_name="job-124-f3")
    assert res["experiment_run"] == "job-124-f3"
    assert ai.names() == ["init", "start_run", "log_params"]


def _run_prepare(tmp_path):
    prep = import_script("prepare_experiment_entry")
    out = tmp_path / "kfp"
    argv = ["prepare_experiment_entry.py", "--project", "p", "--bucket", "b", "--dataset", DATASET,
            "--split-root", str(tmp_path / "bucket"), "--floor-episodes", "40",
            "--split-uri-output-path", str(out / "uri"), "--split-sha-output-path", str(out / "sha"),
            "--fold-ids-output-path", str(out / "ids"), "--report-html-path", str(out / "report.html"),
            "--summary-output-path", str(out / "summary")]
    old, sys.argv = sys.argv, argv
    try:
        prep.main()
    finally:
        sys.argv = old
    return (out / "uri").read_text(), (out / "sha").read_text()


def test_pipeline_entry_names_the_run_after_the_job_and_hands_it_on(local_gcs, fake_gcp, tmp_path):
    """scripts/train_pipeline_entry.py end to end with GCS stubbed: the run is <job>-f<fold>, its
    parameters say where the model went and under which split, and the name reaches the output."""
    ai, st = fake_gcp["aiplatform"], fake_gcp["storage"]
    uri, split_sha = _run_prepare(tmp_path)
    n_features = len(local_gcs["schema"]["feature_names"])
    st.files[f"raw/{DATASET}/MANIFEST.json"] = json.dumps(
        {"archive_sha256": local_gcs["sha"], "geometry": {"kind": "tabular", "n_features": n_features}}).encode()
    out = tmp_path / "out"
    argv = ["train_pipeline_entry.py", "--project", "p", "--bucket", "b", "--dataset", DATASET,
            "--encoder", "mlp", "--fold", "3", "--split-uri", uri,
            "--n-way", "4", "--k-shot", "5", "--query", "5", "--seed", "0", "--train-iters", "20",
            "--embedding-hid", "16", "--lr", "0.001", "--eval-every", "10", "--val-episodes", "5",
            "--test-episodes", "30",
            "--experiment-name", "exp", "--run-prefix", "fsl-fold-campaign-20261016221157",
            "--run-type", "campaign",
            "--accuracy-mean-output-path", str(out / "mean"), "--accuracy-ci95-output-path", str(out / "ci"),
            "--accuracy-std-output-path", str(out / "std"), "--train-time-output-path", str(out / "t"),
            "--model-dir-output-path", str(out / "model_dir"), "--data-sha-output-path", str(out / "sha"),
            "--experiment-run-output-path", str(out / "run")]
    entry = import_script("train_pipeline_entry")
    old, sys.argv = sys.argv, argv
    try:
        entry.main()
    finally:
        sys.argv = old

    run = (out / "run").read_text()
    assert run == "fsl-fold-campaign-20261016221157-f3"
    assert ai.first("start_run") == run
    params = ai.first("log_params")
    model_dir = (out / "model_dir").read_text()
    assert params["model_dir"] == model_dir and model_dir.startswith("gs://b/models/samld-case/protonet-f3-")
    assert params["pipeline_job"] == "fsl-fold-campaign-20261016221157"
    assert params["split_sha"] == split_sha and params["run_type"] == "campaign"
    # the artifact knows its run, and the model really went where the run says
    arch = json.loads(st.files[model_dir[len("gs://b/"):] + "/architecture.json"])
    assert arch["experiment_run"] == run and arch["split_sha"] == split_sha
    assert model_dir[len("gs://b/"):] + "/model.pt" in st.uploaded
    assert float((out / "mean").read_text()) == ai.first("log_metrics")["test_accuracy_mean"]


def test_pipeline_entry_run_names_are_valid_vertex_ids():
    entry = import_script("train_pipeline_entry")
    assert entry.run_name_for("fsl-fold-campaign-20261016221157", 3, True) == "fsl-fold-campaign-20261016221157-f3"
    assert entry.run_name_for("fsl-hello-pipeline-20261012", 0, False) == "fsl-hello-pipeline-20261012"
    assert entry.run_name_for("My Job_x", 1, True) == "my-job-x-f1"
    assert entry.run_name_for("", 1, True) == ""
    assert len(entry.run_name_for("a" * 200, 1, True)) == 128


def test_active_learning_adds_its_numbers_to_the_same_run(fake_gcp, capsys):
    from types import SimpleNamespace

    al = import_script("al_simulation_entry")
    ai = fake_gcp["aiplatform"]
    args = SimpleNamespace(project="p", region="r", experiment_name="exp", experiment_run="job-f3",
                           reps=30, rounds=20, queue_size=8)
    rm = {"delta_8_16": 0.071, "ci95": 0.02, "labels_to_match_random_max": 10}
    rr = {"margin": [0.5, 0.9], "random": [0.5, 0.85]}
    effect = {"before": 0.6, "after_queue": 0.8, "after_random": 0.7, "batch": 8}
    al.log_to_experiment(args, rm, rr, effect, 24)
    assert ai.names() == ["init", "start_run", "log_params", "log_metrics", "end_run"]
    assert [c for c in ai.calls if c[0] == "start_run"][0][1:] == (("job-f3",), {"resume": True})
    assert ai.first("log_params")["al_budget"] == 24
    m = ai.first("log_metrics")
    assert m["al_margin_vs_random_8_16"] == 0.071 and m["al_labels_to_match_random"] == 10
    assert abs(m["al_queue_effect_vs_random"] - 0.1) < 1e-12 and m["al_acc_margin_at_budget"] == 0.9
    # never reached random's accuracy: no number is invented
    rm["labels_to_match_random_max"] = None
    _, m2 = al.al_experiment_record(args, rm, rr, effect, 24)
    assert "al_labels_to_match_random" not in m2
    # no run handed in -> nothing logged; a failing SDK -> the report goes on
    ai.calls = []
    al.log_to_experiment(SimpleNamespace(**{**vars(args), "experiment_run": ""}), rm, rr, effect, 24)
    assert ai.calls == []
    ai.fail_on = "start_run"
    al.log_to_experiment(args, rm, rr, effect, 24)
    assert "WARNING: active learning not logged" in capsys.readouterr().out
