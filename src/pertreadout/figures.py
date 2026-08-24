"""Figures, rendered from the tables -- never from predictions directly.

Honest-axis rule, applied here rather than left to discipline:
:func:`honest_ylim` refuses to start a bar axis anywhere but zero, and refuses a
free axis on a correlation panel (those are pinned to ``[-1, 1]`` or ``[0, 1]``).
If an effect is only visible zoomed in, the caption has to say so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pertreadout.evaluate import MODEL_LABELS

PRIMARY = {"gears": "#3B6EA5", "hgnn": "#C1553B",
           "gears_noprior": "#8FA9C4", "hgnn_noprior": "#DDA08E"}
DATASET_ORDER = ["norman", "adamson", "dixit"]


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _order(values: Iterable[str]) -> list[str]:
    vals = list(dict.fromkeys(values))
    return [d for d in DATASET_ORDER if d in vals] + [d for d in vals if d not in DATASET_ORDER]


def honest_ylim(ax, kind: str = "bar", values: Optional[np.ndarray] = None) -> None:
    """Set y limits under the repo's axis rule.

    ``bar``        -- must start at zero.  A bar chart whose axis starts at 0.9
                      makes a 1% difference look like a doubling; that is the
                      exact trick this repository exists to point at.
    ``correlation``-- pinned to ``[0, 1]`` (or ``[-1, 1]`` if anything is
                      negative), never auto-scaled.
    ``ratio``      -- starts at zero on a log scale is meaningless, so a log axis
                      is allowed here and the caption says the axis is log.
    """
    if kind == "bar":
        top = float(np.nanmax(values)) if values is not None and len(values) else ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
    elif kind == "correlation":
        lo = -1.0 if values is not None and len(values) and np.nanmin(values) < 0 else 0.0
        ax.set_ylim(lo, 1.0)
    elif kind == "ratio":
        ax.set_yscale("log")
    else:
        raise ValueError(f"unknown axis kind {kind!r}")


def _bars_with_ci(ax, df: pd.DataFrame, readout: str, models=("gears", "hgnn")):
    datasets = _order(df["dataset"])
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        sub = df[(df["readout"] == readout) & (df["model"] == model)].set_index("dataset")
        means = np.array([sub["mean"].get(d, np.nan) for d in datasets])
        lo = np.array([sub["ci_lo"].get(d, np.nan) for d in datasets])
        hi = np.array([sub["ci_hi"].get(d, np.nan) for d in datasets])
        x = np.arange(len(datasets)) + (i - (len(models) - 1) / 2) * width
        ax.bar(x, means, width=width, label=_label(model), color=PRIMARY.get(model, None))
        ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt="none", ecolor="black",
                    capsize=3, lw=1)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([d.capitalize() for d in datasets])
    return datasets


# --------------------------------------------------------------------------- #
def fig_like_for_like(agg: pd.DataFrame, out: Path) -> Path:
    """Both models on the same readout, three panels, all axes from zero."""
    readouts = [("mse_genome_wide", "Genome-wide MSE"),
                ("mse_top20_de", "Top-20 DE MSE"),
                ("pearson_delta", "Pearson on the delta")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (readout, title) in zip(axes, readouts):
        sub = agg[agg["readout"] == readout]
        _bars_with_ci(ax, sub, readout)
        ax.set_title(title)
        kind = "correlation" if readout.startswith("pearson") else "bar"
        honest_ylim(ax, kind, sub["ci_hi"].to_numpy())
    axes[0].set_ylabel("mean over 5 seeds (95% CI)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Like for like: the same readout on both sides of every comparison", fontsize=11)
    fig.tight_layout()
    return _save(fig, out / "01_like_for_like.png")


def fig_ratio_panel(ratio: pd.DataFrame, out: Path) -> Path:
    """The metric-pair ratio, per model, on a log axis.

    Drawn as points with intervals rather than bars.  A bar on a log axis invites
    the reader to compare *areas*, and area on a log scale means nothing -- the
    bar's baseline is arbitrary.  Given what this repository is about, drawing
    that particular chart would be poor form.

    If the two markers sit on top of each other within a dataset, the ratio
    belongs to the metric pair and the data, not to the model.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    datasets = _order(ratio["dataset"])
    models = [m for m in ["gears", "hgnn"] if m in set(ratio["model"])]
    offset = 0.12
    for i, model in enumerate(models):
        sub = ratio[ratio["model"] == model].set_index("dataset")
        means = np.array([sub["mean"].get(d, np.nan) for d in datasets])
        lo = np.array([sub["ci_lo"].get(d, np.nan) for d in datasets])
        hi = np.array([sub["ci_hi"].get(d, np.nan) for d in datasets])
        x = np.arange(len(datasets)) + (i - (len(models) - 1) / 2) * offset
        ax.errorbar(
            x, means, yerr=[means - lo, hi - means], fmt="o", markersize=7,
            color=PRIMARY.get(model), ecolor=PRIMARY.get(model), capsize=4,
            lw=1.5, label=_label(model),
        )
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylabel("top-20-DE MSE / genome-wide MSE")
    ax.set_title("The ratio between the two MSEs is a property of the metric pair")
    honest_ylim(ax, "ratio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.5, -0.02, "Log y axis, and intervals are computed on log(ratio); 95% CI over 5 seeds.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    return _save(fig, out / "02_metric_ratio.png")


def fig_mismatch(agg: pd.DataFrame, out: Path) -> Path:
    """Left: the error. Right: the same numbers compared correctly."""
    datasets = _order(agg["dataset"])
    g = agg.set_index(["dataset", "model", "readout"])["mean"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    x = np.arange(len(datasets))
    left = [g.get((d, "gears", "mse_top20_de"), np.nan) for d in datasets]
    right = [g.get((d, "hgnn", "mse_genome_wide"), np.nan) for d in datasets]
    axes[0].bar(x - 0.2, left, width=0.4, color=PRIMARY["gears"], label="GEARS -- top-20-DE MSE")
    axes[0].bar(x + 0.2, right, width=0.4, color=PRIMARY["hgnn"], label="HGNN -- genome-wide MSE")
    axes[0].set_title("Two different metrics (this is the error)")
    honest_ylim(axes[0], "bar", np.array(left + right, dtype=float))

    a = [g.get((d, "gears", "mse_genome_wide"), np.nan) for d in datasets]
    b = [g.get((d, "hgnn", "mse_genome_wide"), np.nan) for d in datasets]
    axes[1].bar(x - 0.2, a, width=0.4, color=PRIMARY["gears"], label="GEARS -- genome-wide MSE")
    axes[1].bar(x + 0.2, b, width=0.4, color=PRIMARY["hgnn"], label="HGNN -- genome-wide MSE")
    axes[1].set_title("The same metric on both sides")
    honest_ylim(axes[1], "bar", np.array(a + b, dtype=float))

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in datasets])
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "The left panel is what you get if the two bars are different metrics.", fontsize=11
    )
    fig.tight_layout()
    return _save(fig, out / "03_mismatched_comparison.png")


def fig_pearson_family(agg: pd.DataFrame, out: Path) -> Path:
    """Absolute vs delta vs DE-delta, with the do-nothing baseline drawn in."""
    readouts = [("pearson_absolute", "Pearson, absolute expression"),
                ("pearson_delta", "Pearson, delta"),
                ("pearson_delta_de", "Pearson, delta on top-20 DE")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (readout, title) in zip(axes, readouts):
        sub = agg[agg["readout"] == readout]
        datasets = _bars_with_ci(ax, sub, readout)
        ax.set_title(title, fontsize=10)
        honest_ylim(ax, "correlation", sub["ci_lo"].to_numpy())
        if readout == "pearson_absolute":
            base = agg[agg["readout"] == "pearson_absolute_ctrl_baseline"].groupby("dataset")["mean"].mean()
            for i, d in enumerate(datasets):
                if d in base.index:
                    ax.hlines(base[d], i - 0.45, i + 0.45, color="black", ls="--", lw=1.2,
                              label="control mean profile" if i == 0 else None)
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("mean over 5 seeds (95% CI)")
    fig.suptitle("Same predictions, three correlation readouts. Axes fixed to [0, 1].", fontsize=11)
    fig.tight_layout()
    return _save(fig, out / "04_pearson_family.png")


def fig_direction_sweep(sweep: pd.DataFrame, out: Path) -> Path:
    """Direction accuracy against effect-size threshold, surviving genes behind it."""
    datasets = _order(sweep["dataset"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = sweep[sweep["dataset"] == dataset]
        twin = ax.twinx()
        for model in [m for m in ["gears", "hgnn"] if m in set(sub["model"])]:
            s = sub[sub["model"] == model].sort_values("percentile")
            ax.plot(s["percentile"], s["direction_accuracy"], color=PRIMARY[model], label=_label(model))
            ax.fill_between(s["percentile"], s["ci_lo"], s["ci_hi"], color=PRIMARY[model], alpha=0.18)
            twin.plot(s["percentile"], s["n_genes_surviving"], color="grey", ls=":", lw=1)
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("percentile of |true delta| used as min_abs_delta")
        ax.set_title(dataset.capitalize())
        twin.set_yscale("log")
        twin.set_ylabel("genes surviving (dotted, log)", fontsize=8)
    axes[0][0].set_ylabel("direction accuracy")
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Direction accuracy vs effect size. Dashed line is chance for a sign call.", fontsize=11
    )
    fig.tight_layout()
    return _save(fig, out / "05_direction_vs_effect_size.png")


def fig_prior_ablation(agg: pd.DataFrame, out: Path) -> Path:
    """GO and co-expression priors against identity graphs, same architectures."""
    models = [m for m in ["gears", "gears_noprior", "hgnn", "hgnn_noprior"] if m in set(agg["model"])]
    if len(models) < 3:
        return None
    sub = agg[agg["readout"] == "mse_genome_wide"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _bars_with_ci(ax, sub, "mse_genome_wide", models=models)
    ax.set_ylabel("genome-wide MSE, mean over 5 seeds (95% CI)")
    ax.set_title("Identity-graph ablation: how much do the priors contribute?")
    honest_ylim(ax, "bar", sub["ci_hi"].to_numpy())
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out / "06_prior_ablation.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all(tables_dir: Path, out_dir: Path) -> list[Path]:
    tables_dir, out_dir = Path(tables_dir), Path(out_dir)
    agg = pd.read_csv(tables_dir / "readouts_by_model.csv")
    ratio = pd.read_csv(tables_dir / "metric_ratio_band.csv")
    sweep = pd.read_csv(tables_dir / "direction_sweep.csv")
    written = [
        fig_like_for_like(agg, out_dir),
        fig_ratio_panel(ratio, out_dir),
        fig_mismatch(agg, out_dir),
        fig_pearson_family(agg, out_dir),
        fig_direction_sweep(sweep, out_dir),
        fig_prior_ablation(agg, out_dir),
    ]
    return [w for w in written if w is not None]
