"""Training loop, episodic evaluation, and Vertex AI Experiments logging.

One training = one Experiments run, opened HERE, by whoever trains (the
notebook's scripts/train.py or the pipeline's training container):
parameters at the start, the train/val curve as time series (Vertex
TensorBoard) at every evaluation, the test result at the end. A pipeline
passes the run name in and gets it back in the result, so its later steps
(gate, active learning) resume the same run and add their numbers to it.
Logging is optional (cfg.log_to_vertex) so the training logic runs without GCP,
and it is never allowed to cost a training its result: after the first failure
the logger turns itself off and says so once.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from fsl.config import TrainConfig
from fsl.data.registry import get_loaders
from fsl.models.registry import get_encoder
from fsl.models.protonet import fast_adapt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, tasks, cfg: TrainConfig, n_episodes: int, device) -> np.ndarray:
    """Few-shot accuracy over `n_episodes` sampled episodes."""
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            _, acc = fast_adapt(
                model, tasks.sample(), cfg.n_way, cfg.k_shot, cfg.query, device
            )
            accs.append(acc.item())
    model.train()
    return np.array(accs)


class ExperimentLog:
    """The Vertex Experiments run of one training (params, curve, result).

    `run_name` is the name actually opened, "" when nothing was (disabled, or
    start failed): a pipeline writes it to an output so downstream steps know
    whether there is a run to resume. Every method is a no-op once disabled.
    """

    def __init__(self, cfg: TrainConfig, *, run_name: str = "", extra_params: dict | None = None,
                 enabled: bool | None = None):
        self.cfg = cfg
        self.enabled = bool(cfg.log_to_vertex) if enabled is None else bool(enabled)
        self.requested = run_name or f"protonet-{cfg.n_way}w{cfg.k_shot}s-seed{cfg.seed}-{int(time.time())}"
        self.extra = dict(extra_params or {})
        self.run_name = ""

    def _guarded(self, what: str, fn) -> None:
        if not self.enabled:
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - logging must never fail the training
            self.enabled = False
            print(f"WARNING: Vertex Experiments logging stopped at {what}: {e!r}")

    def start(self, archive_sha: str) -> None:
        def go():
            from google.cloud import aiplatform

            # init(experiment=...) attaches the project's default Vertex TensorBoard to the
            # experiment (creating it if needed, SDK >= 1.25), which time-series logging needs.
            aiplatform.init(project=self.cfg.project, location=self.cfg.region,
                            experiment=self.cfg.experiment_name)
            try:
                aiplatform.start_run(self.requested)
            except Exception as e:  # noqa: BLE001
                # a retried pipeline task finds the run its first attempt created: take it over
                if "already exist" not in str(e).lower() and type(e).__name__ != "AlreadyExists":
                    raise
                print(f"run {self.requested} already exists: resuming it")
                aiplatform.start_run(self.requested, resume=True)
            self.run_name = self.requested          # from here on there is a run to resume
            params = self.cfg.as_params()
            params["dataset_archive_sha256"] = archive_sha  # provenance
            params.update(self.extra)
            aiplatform.log_params(params)
            print(f"Run started: {self.run_name} (experiment {self.cfg.experiment_name})")

        self._guarded("start_run", go)

    def curve(self, metrics: dict, step: int) -> None:
        def go():
            from google.cloud import aiplatform

            aiplatform.log_time_series_metrics(metrics, step=step)

        self._guarded(f"log_time_series_metrics(step={step})", go)

    def finish(self, metrics: dict) -> None:
        def go():
            from google.cloud import aiplatform

            aiplatform.log_metrics(metrics)
            aiplatform.end_run()

        self._guarded("log_metrics", go)


def train(cfg: TrainConfig, *, run_name: str = "", extra_params: dict | None = None) -> dict:
    """Run the full rung-2 training + evaluation; return a results dict.

    run_name / extra_params: the Experiments run to open and what to log with
    the config's parameters (a pipeline passes its job name, split_sha, model
    location). Empty run_name = a timestamped name, as the notebook always had.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    load_frozen, build_task_samplers = get_loaders(cfg)
    dataset, archive_sha = load_frozen(cfg)
    train_tasks, val_tasks, test_tasks = build_task_samplers(dataset, cfg)

    model = get_encoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    log = ExperimentLog(cfg, run_name=run_name, extra_params=extra_params)
    log.start(archive_sha)

    set_seed(cfg.seed)  # re-seed right before the loop for reproducibility
    t0 = time.time()
    run_loss, run_acc, seen = 0.0, 0.0, 0  # rolling train stats between evals
    for it in range(1, cfg.train_iters + 1):
        opt.zero_grad()
        loss, acc = fast_adapt(
            model, train_tasks.sample(), cfg.n_way, cfg.k_shot, cfg.query, device
        )
        loss.backward()
        opt.step()
        run_loss += loss.item()
        run_acc += acc.item()
        seen += 1

        if it % cfg.eval_every == 0 or it == 1:
            val = evaluate(model, val_tasks, cfg, cfg.val_episodes, device)
            train_loss = run_loss / seen
            train_acc = run_acc / seen
            run_loss, run_acc, seen = 0.0, 0.0, 0
            print(
                f"iter {it:5d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
                f"| val acc {val.mean():.3f} +/- "
                f"{1.96 * val.std() / np.sqrt(len(val)):.3f}"
            )
            log.curve(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": float(val.mean()),
                },
                step=it,
            )
    train_time = time.time() - t0

    test_accs = evaluate(model, test_tasks, cfg, cfg.test_episodes, device)
    mean = float(test_accs.mean())
    ci95 = float(1.96 * test_accs.std() / np.sqrt(len(test_accs)))
    print(
        f"TEST {cfg.n_way}-way {cfg.k_shot}-shot: {mean * 100:.2f}% +/- "
        f"{ci95 * 100:.2f}% (over {cfg.test_episodes} episodes)"
    )

    log.finish({
        "test_accuracy_mean": mean,
        "test_accuracy_ci95": ci95,
        "test_accuracy_std": float(test_accs.std()),
        "train_time_seconds": train_time,
    })

    return {
        "test_accuracy_mean": mean,
        "test_accuracy_ci95": ci95,
        "test_accuracy_std": float(test_accs.std()),
        "train_time_seconds": train_time,
        "model": model,
        "experiment_run": log.run_name,
    }
