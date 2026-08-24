"""One training loop, one prediction path, two models.

The comparison this repository is about only means anything if the two models
are treated identically.  So both go through the code in this file: same loss
(GEARS' autofocus + direction-aware loss), same optimiser and schedule, same
dataloaders, same split, same seeds.  The only difference is which module was
handed in.

The output of a run is not a score -- it is a ``predictions.npz`` holding the
condition-mean predicted and true profiles on the held-out perturbations.  Every
readout is computed afterwards from that file, by
:mod:`pertreadout.readouts`, so the same predictions can be scored every way the
literature scores them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from pertreadout import data as data_mod

MODELS = ("gears", "hgnn")


def _log(message: str) -> None:
    """Progress goes out unbuffered: these runs are long and usually redirected."""
    print(message, flush=True)


def build_model(model_name: str, pert_data, device: str, cfg: dict, use_priors: bool = True):
    """Dispatch to a model builder, both of which return ``(module, context)``."""
    common = dict(
        device=device,
        hidden_size=int(cfg.get("hidden_size", 64)),
        decoder_hidden_size=int(cfg.get("decoder_hidden_size", 16)),
        use_priors=use_priors,
    )
    if model_name == "gears":
        from pertreadout.models import gears_baseline

        return gears_baseline.build(pert_data, **common)
    if model_name == "hgnn":
        from pertreadout.models import bipartite_hgnn

        return bipartite_hgnn.build(
            pert_data, bipartite_rank=int(cfg.get("bipartite_rank", 32)), **common
        )
    raise ValueError(f"unknown model {model_name!r}; expected one of {MODELS}")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model,
    context: dict,
    pert_data,
    epochs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    max_steps_per_epoch: Optional[int] = None,
    log: Callable[[str], None] = _log,
) -> list[float]:
    """GEARS' training recipe, applied to whichever module was passed in."""
    from gears.utils import loss_fct
    from torch.optim.lr_scheduler import StepLR

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    train_loader = pert_data.dataloader["train_loader"]

    history = []
    for epoch in range(epochs):
        model.train()
        total, steps = 0.0, 0
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fct(
                pred,
                batch.y,
                batch.pert,
                ctrl=context["ctrl_expression"],
                direction_lambda=context["direction_lambda"],
                dict_filter=context["dict_filter"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1
            if max_steps_per_epoch and steps >= max_steps_per_epoch:
                break
        scheduler.step()
        history.append(total / max(steps, 1))
        log(f"    epoch {epoch + 1}/{epochs}  train loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def collect_predictions(model, loader, device: str) -> dict:
    """Run a loader and return per-cell predictions, truths and condition labels."""
    model.eval()
    preds, truths, conds = [], [], []
    for batch in loader:
        batch.to(device)
        pred = model(batch)
        preds.append(pred.detach().cpu().numpy())
        truths.append(batch.y.detach().cpu().numpy())
        conds.extend(list(batch.pert))
    return {
        "pred": np.concatenate(preds, axis=0),
        "true": np.concatenate(truths, axis=0),
        "condition": np.asarray(conds, dtype=object),
    }


def condition_means(per_cell: dict) -> dict:
    """Average per-cell predictions and truths within each held-out condition.

    This is what GEARS' own ``compute_metrics`` scores, and it is where the
    per-condition variance needed for the noise floor comes from.
    """
    conds = np.asarray([str(c) for c in per_cell["condition"]])
    keep = conds != "ctrl"
    pred, true, conds = per_cell["pred"][keep], per_cell["true"][keep], conds[keep]

    order = sorted(set(conds.tolist()))
    pm, tm, tv, n = [], [], [], []
    for c in order:
        idx = conds == c
        pm.append(pred[idx].mean(axis=0))
        tm.append(true[idx].mean(axis=0))
        tv.append(true[idx].var(axis=0))
        n.append(int(idx.sum()))
    return {
        "conditions": np.asarray(order, dtype=object),
        "pred": np.asarray(pm, dtype=np.float64),
        "true": np.asarray(tm, dtype=np.float64),
        "true_var": np.asarray(tv, dtype=np.float64),
        "n_cells": np.asarray(n, dtype=np.int64),
    }


def run(
    dataset: str,
    model_name: str,
    seed: int,
    cfg: dict,
    out_root: Path,
    use_priors: bool = True,
    device: str = "cpu",
    log: Callable[[str], None] = _log,
) -> Path:
    """Train one (dataset, model, seed) and save its held-out predictions.

    Returns the path to ``predictions.npz``.  Re-running a completed run is a
    no-op unless ``cfg['force']`` is set, so a long ``reproduce`` can be resumed.
    """
    tag = f"{dataset}/{model_name}{'' if use_priors else '_noprior'}/seed{seed}"
    out_dir = out_root / tag
    out_file = out_dir / "predictions.npz"
    if out_file.exists() and not cfg.get("force"):
        log(f"  {tag}: cached")
        return out_file

    started = time.time()
    set_seed(seed)
    log(f"  {tag}: loading data")
    pert_data = data_mod.load_pert_data(
        dataset,
        seed=seed,
        max_cells_per_condition=int(cfg.get("max_cells_per_condition", 32)),
        batch_size=int(cfg.get("batch_size", 16)),
        split=str(cfg.get("split", "simulation")),
        train_gene_set_size=float(cfg.get("train_gene_set_size", 0.75)),
    )

    model, context = build_model(model_name, pert_data, device, cfg, use_priors=use_priors)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  {tag}: {n_params/1e6:.2f}M parameters")

    history = train_model(
        model,
        context,
        pert_data,
        epochs=int(cfg.get("epochs", 8)),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
        device=device,
        max_steps_per_epoch=cfg.get("max_steps_per_epoch"),
        log=log,
    )

    loader_key = "test_loader" if "test_loader" in pert_data.dataloader else "val_loader"
    per_cell = collect_predictions(model, pert_data.dataloader[loader_key], device)
    means = condition_means(per_cell)

    adata = pert_data.adata
    ctrl_mean, ctrl_var, n_ctrl = data_mod.control_statistics(adata)
    de_idx = data_mod.de_indices(adata, means["conditions"], k=int(cfg.get("de_k", 20)))

    # standard error of each measured delta: control mean and condition mean are
    # both finite-sample averages.
    delta_se = np.sqrt(
        ctrl_var[None, :] / max(n_ctrl, 1) + means["true_var"] / np.maximum(means["n_cells"], 1)[:, None]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        pred=means["pred"],
        true=means["true"],
        conditions=means["conditions"].astype(str),
        de_idx=de_idx,
        ctrl_mean=ctrl_mean,
        delta_se=delta_se,
        n_cells=means["n_cells"],
        gene_names=np.asarray(list(adata.var_names), dtype=str),
    )
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "use_priors": use_priors,
                "split_loader": loader_key,
                "n_held_out_conditions": int(len(means["conditions"])),
                "n_parameters": int(n_params),
                "train_loss": history,
                "seconds": round(time.time() - started, 1),
                "config": {k: v for k, v in cfg.items() if k != "force"},
            },
            indent=2,
        )
        + "\n"
    )
    log(f"  {tag}: {len(means['conditions'])} held-out conditions, {time.time() - started:.0f}s")
    del pert_data, model
    return out_file


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
