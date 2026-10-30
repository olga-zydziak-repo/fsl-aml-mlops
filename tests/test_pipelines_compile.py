"""Both Vertex pipelines compile from rung9_hpo.ipynb with GCP stubbed.

The notebook is the only place the pipelines are defined, so a broken cell
(a component signature that no longer matches its caller, a parameter that the
campaign passes but train_container does not accept) is otherwise found on
Vertex, after a submit. Here the definition cells run with aiplatform stubbed,
the compiler produces both YAMLs, and the compiled pipelines are checked for
the shapes v1.1 and v1.2 introduced: the fold loop iterates over the output of
the prepare_experiment step, training receives the split, and the Experiments
run opened by the training container is handed to the gate and to active
learning (which runs after the gate, never beside it). The gate's logging code
is also exercised directly with the SDK stubbed.

Skipped when kfp / google-cloud-pipeline-components are not installed (the
plain unit-test job); the CI "pipelines" job installs them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("kfp")
pytest.importorskip("google_cloud_pipeline_components")
aiplatform = pytest.importorskip("google.cloud.aiplatform")

from tests.conftest import FakeAiplatform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "rung9_hpo.ipynb"
CAMPAIGN_MARKER = "KAMPANIA FOLDOW W JEDNYM PRZEBIEGU"
SINGLE_RUN_MARKER = "FOLD = 3"        # submits one run and reads a Workbench-only floors file


class _FakeJob:
    submitted: list = []

    def __init__(self, **kw):
        self.kw = kw
        self.name = "fake-job-20260101000000"
        self.resource_name = f"projects/ci/locations/ci/pipelineJobs/{self.name}"

    def submit(self, **kw):
        _FakeJob.submitted.append(self.kw)


def _code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture
def compiled(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("HOME", str(tmp_path))          # no ~/data_cache/samld_case_floors.json
    monkeypatch.setattr(aiplatform, "init", lambda **kw: None)
    monkeypatch.setattr(aiplatform, "PipelineJob", _FakeJob)
    _FakeJob.submitted = []
    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()
    g = {"__name__": "__main__"}
    ran = 0
    for i, src in enumerate(_code_cells()):
        body = src.strip()
        if not body or body.startswith(("%", "!")) or SINGLE_RUN_MARKER in src:
            continue
        f = cells_dir / f"cell{i}.py"                  # a real file: kfp reads component source with inspect
        f.write_text(src)
        # dont_inherit: this module's `from __future__ import annotations` must not leak into
        # the cells, or kfp sees every annotation as a string and refuses the components
        exec(compile(src, str(f), "exec", dont_inherit=True), g)
        ran += 1
        if CAMPAIGN_MARKER in src:
            break
    assert ran >= 10, "the notebook changed shape: too few definition cells ran"
    import yaml
    from kfp import compiler

    training_yaml = tmp_path / "training_pipeline.yaml"
    compiler.Compiler().compile(g["training_pipeline"], str(training_yaml))
    campaign_yaml = ROOT / "campaign_pipeline.yaml"    # the campaign cell writes it next to the notebook
    yield {"training": training_yaml.read_text(), "campaign": campaign_yaml.read_text(),
           "training_spec": yaml.safe_load(training_yaml.read_text()),
           "campaign_spec": yaml.safe_load(campaign_yaml.read_text()),
           "submitted": _FakeJob.submitted, "g": g}
    campaign_yaml.unlink(missing_ok=True)


def _tasks(spec: dict) -> dict:
    """Every task of a compiled pipeline, flattened across nested DAGs (loops, conditions, exit handlers)."""
    out = {}
    for comp in spec["components"].values():
        for name, task in comp.get("dag", {}).get("tasks", {}).items():
            out[name] = task
    for name, task in spec["root"]["dag"]["tasks"].items():
        out[name] = task
    return out


def _param_source(task: dict, param: str) -> dict:
    return task["inputs"]["parameters"][param]


def test_training_pipeline_compiles(compiled):
    t = compiled["training"]
    assert "gate-passed" in t and "gate-failed" in t
    assert "--split-uri" in t                          # train_container carries the v1.1 parameter
    assert "--experiment-run-output-path" in t         # ...and opens the Experiments run (v1.2)
    assert "log-experiment-component" not in t         # the separate logging step is gone


def _comes_from(task: dict, param: str, producer: str, key: str) -> bool:
    """True when the task's input is `producer`'s output `key`, directly (same DAG) or passed
    down into a nested DAG (condition / loop) as a pipelinechannel-- component input."""
    src = _param_source(task, param)
    if "taskOutputParameter" in src:
        t = src["taskOutputParameter"]
        return t["producerTask"] == producer and t["outputParameterKey"] == key
    return src.get("componentInputParameter", "").endswith(f"{producer}-{key}")


def _assert_run_handoff(tasks: dict, trainer: str):
    """The gate and active learning resume the run the trainer opened: their experiment_run
    input is the trainer's experiment_run output, and the trainer's run prefix is the job name."""
    assert _comes_from(tasks["evaluate-gate-component"], "experiment_run", trainer, "experiment_run")
    al = [n for n in tasks if n.startswith("al-container")]
    assert al, "no active-learning task compiled"
    for name in al:
        assert _comes_from(tasks[name], "experiment_run", trainer, "experiment_run"), name
    train = tasks[trainer]
    prefix = _param_source(train, "run_prefix")["runtimeValue"]["constant"]
    assert prefix == "{{$.pipeline_job_name}}"


def test_training_pipeline_hands_the_run_to_gate_and_active_learning(compiled):
    tasks = _tasks(compiled["training_spec"])
    _assert_run_handoff(tasks, "train-container")
    assert _param_source(tasks["evaluate-gate-component"], "floor")["componentInputParameter"].endswith("floor")


def test_campaign_loops_over_the_prepare_step(compiled):
    c = compiled["campaign"]
    assert "prepare-experiment-container" in c
    assert "parameterIterator" in c and "fold_ids" in c
    assert c.count("--split-uri") >= 1 and "split_uri" in c
    assert "fold_ids:\n          parameterType: LIST" in c
    assert "log-experiment-component" not in c


def test_campaign_writes_one_run_per_fold_in_order(compiled):
    tasks = _tasks(compiled["campaign_spec"])
    _assert_run_handoff(tasks, "train-container")
    # the gate logs the fold's floor (lift) from the prepare step's split.json
    floor = _param_source(tasks["evaluate-gate-component"], "floor")["taskOutputParameter"]
    assert floor["producerTask"] == "fold-config-component" and floor["outputParameterKey"] == "floor"
    # active learning waits for the gate: two writers never touch the run at the same time
    assert "evaluate-gate-component" in tasks["al-container-quiet"]["dependentTasks"]
    # the campaign-level run carries the split and the hyperparameters of the campaign
    summary = tasks["campaign-summary-component"]
    assert _param_source(summary, "split_sha")["taskOutputParameter"]["producerTask"] == "prepare-experiment-container"
    assert _param_source(summary, "train_iters")["taskOutputParameter"]["producerTask"] == "resolve-hyperparameters"


def test_campaign_cell_submits_once_with_the_right_parameters(compiled):
    sub = compiled["submitted"]
    assert len(sub) == 1
    pv = sub[0]["parameter_values"]
    for k in ("project", "bucket", "dataset", "n_way", "k_shot", "query", "floor_episodes", "legacy_floors_json"):
        assert k in pv, k
    assert pv["legacy_floors_json"] == ""              # no Workbench floors file in CI: comparison skipped
    assert sub[0].get("enable_caching") is False
    assert pv["experiment_name"]                       # the runs land in the config's experiment


class _FakeMetrics:
    def __init__(self):
        self.logged = {}

    def log_metric(self, k, v):
        self.logged[k] = v


def _run_gate(compiled, monkeypatch, fake, **kw):
    import types
    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    cloud.aiplatform = fake
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", fake)
    gm = _FakeMetrics()
    out = compiled["g"]["evaluate_gate_component"].python_func(
        accuracy_mean=0.9132, accuracy_ci95=0.012, accuracy_std=0.11,
        accuracy_threshold=0.9305, max_std_threshold=0.15, gate_metrics=gm, **kw)
    return out, gm


def test_gate_resumes_the_training_run_and_logs_the_verdict(compiled, monkeypatch):
    fake = FakeAiplatform()
    out, gm = _run_gate(compiled, monkeypatch, fake, project="p", region="r", experiment_name="exp",
                        experiment_run="job-f3", floor=0.8611)
    assert out.passed is False and "accuracy_lower_ci: FAIL" in out.reason
    names = [c[0] for c in fake.calls]
    assert names == ["init", "start_run", "log_params", "log_metrics", "end_run"]
    assert fake.calls[1][1] == ("job-f3",) and fake.calls[1][2] == {"resume": True}
    params, metrics = fake.calls[2][1][0], fake.calls[3][1][0]
    assert params["gate_threshold"] == 0.9305 and params["floor"] == 0.8611 and "FAIL" in params["gate_reason"]
    assert metrics["gate_passed"] == 0.0 and abs(metrics["lift"] - (0.9132 - 0.8611)) < 1e-12
    assert abs(metrics["accuracy_lower_ci"] - gm.logged["accuracy_lower_ci"]) < 1e-12


def test_gate_without_a_run_logs_nothing_and_a_failed_log_keeps_the_verdict(compiled, monkeypatch):
    fake = FakeAiplatform()
    out, _ = _run_gate(compiled, monkeypatch, fake, experiment_name="exp", experiment_run="")
    assert fake.calls == [] and out.passed is False
    fake = FakeAiplatform(fail_on="start_run")
    out, _ = _run_gate(compiled, monkeypatch, fake, project="p", experiment_name="exp", experiment_run="job-f3")
    assert out.passed is False and [c[0] for c in fake.calls] == ["init"]
