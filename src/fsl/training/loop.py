"""Training loop, episodic evaluation, and Vertex AI Experiments logging.

Adds time-series metrics (train/val curves per eval step) logged to Vertex
TensorBoard, on top of the params + summary metrics. Experiment/TensorBoard
logging is optional (cfg.log_to_vertex) so the training logic runs without GCP.
Mirrors sections 7-9 of the rung-2 notebook, plus time-series curves.
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


def train(cfg: TrainConfig) -> dict:
    """Run the full rung-2 training + evaluation; return a results dict."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    load_frozen, build_task_samplers = get_loaders(cfg)
    dataset, archive_sha = load_frozen(cfg)
    train_tasks, val_tasks, test_tasks = build_task_samplers(dataset, cfg)

    model = get_encoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.log_to_vertex:
        _start_experiment(cfg, archive_sha)

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
            if cfg.log_to_vertex:
                _log_time_series(
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

    if cfg.log_to_vertex:
        _log_results(mean, ci95, float(test_accs.std()), train_time)

    return {
        "test_accuracy_mean": mean,
        "test_accuracy_ci95": ci95,
        "test_accuracy_std": float(test_accs.std()),
        "train_time_seconds": train_time,
        "model": model,
    }


def _start_experiment(cfg: TrainConfig, archive_sha: str) -> None:
    from google.cloud import aiplatform

    # init(experiment=...) auto-creates and attaches a default Vertex TensorBoard
    # instance if none exists (SDK >= 1.25), enabling time-series logging.
    aiplatform.init(project=cfg.project, location=cfg.region, experiment=cfg.experiment_name)
    run_name = f"protonet-{cfg.n_way}w{cfg.k_shot}s-seed{cfg.seed}-{int(time.time())}"
    aiplatform.start_run(run_name)
    params = cfg.as_params()
    params["dataset_archive_sha256"] = archive_sha  # provenance
    aiplatform.log_params(params)
    print("Run started:", run_name)


def _log_time_series(metrics: dict, step: int) -> None:
    from google.cloud import aiplatform

    aiplatform.log_time_series_metrics(metrics, step=step)


def _log_results(mean: float, ci95: float, std: float, train_time: float) -> None:
    from google.cloud import aiplatform

    aiplatform.log_metrics({
        "test_accuracy_mean": mean,
        "test_accuracy_ci95": ci95,
        "test_accuracy_std": std,
        "train_time_seconds": train_time,
    })
    aiplatform.end_run()