"""Turn saved predictions into the tables the README and the notebooks quote.

Nothing in here computes a metric: every number comes from
:func:`pertreadout.readouts.readout_table`, applied to the ``predictions.npz``
files that :mod:`pertreadout.train` wrote.  Aggregation across seeds is a mean
with a 95% t-interval -- a headline number without one is not shipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from pertreadout.readouts import (
    READOUT_COLUMNS,
    direction_accuracy_sweep,
    fraction_below_noise_floor,
    mean_ci,
    readout_table,
)
from pertreadout.train import load_predictions

MODEL_LABELS = {
    "gears": "GEARS",
    "hgnn": "Bipartite HGNN",
    "gears_noprior": "GEARS (identity graphs)",
    "hgnn_noprior": "Bipartite HGNN (identity graphs)",
}


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def find_runs(root: Path) -> pd.DataFrame:
    """Locate every ``predictions.npz`` under ``root`` and parse its identity."""
    rows = []
    for path in sorted(Path(root).glob("*/*/seed*/predictions.npz")):
        seed_dir, model_dir, dataset_dir = path.parent, path.parent.parent, path.parent.parent.parent
        rows.append(
            {
                "dataset": dataset_dir.name,
                "model": model_dir.name,
                "seed": int(seed_dir.name.replace("seed", "")),
                "path": path,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# per-run and aggregated readouts
# --------------------------------------------------------------------------- #
def per_run_table(root: Path, k: int = 20) -> pd.DataFrame:
    """One row per (dataset, model, seed): every readout, from the same call."""
    runs = find_runs(root)
    if runs.empty:
        raise FileNotFoundError(f"no predictions under {root}; run `pertreadout reproduce` first")
    frames = []
    for row in runs.itertuples():
        d = load_predictions(row.path)
        frames.append(
            readout_table(
                d["pred"], d["true"], d["ctrl_mean"], d["de_idx"], k=k,
                dataset=row.dataset, model=row.model, seed=row.seed,
            ).assign(
                frac_below_noise_floor=fraction_below_noise_floor(
                    d["true"], d["ctrl_mean"], d["delta_se"]
                ),
                pearson_absolute_ctrl_baseline=_ctrl_baseline_pearson(d),
                frac_true_delta_exactly_zero=float(np.mean(d["true"] == d["ctrl_mean"])),
            )
        )
    return pd.concat(frames, ignore_index=True)


def _ctrl_baseline_pearson(d) -> float:
    """Pearson-absolute of the do-nothing predictor, on this exact test set.

    Drawn as a horizontal line on the Pearson-absolute panel in ``notebooks/03``:
    it is the score to beat, and it is very close to 1.
    """
    from pertreadout.readouts import pearson_absolute

    trivial = np.tile(d["ctrl_mean"], (d["true"].shape[0], 1))
    return pearson_absolute(trivial, d["true"])


def aggregate(per_run: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Mean and 95% CI over seeds, for each (dataset, model, readout)."""
    columns = list(columns or READOUT_COLUMNS + [
        "frac_below_noise_floor", "frac_true_delta_exactly_zero", "pearson_absolute_ctrl_baseline"])
    rows = []
    for (dataset, model), grp in per_run.groupby(["dataset", "model"], sort=False):
        for col in columns:
            if col not in grp:
                continue
            m, lo, hi = mean_ci(grp[col].to_numpy())
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "readout": col,
                    "mean": m,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_seeds": int(grp[col].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the like-for-like table -- the repository's headline
# --------------------------------------------------------------------------- #
def like_for_like(agg: pd.DataFrame, readout: str = "mse_genome_wide") -> pd.DataFrame:
    """Both models scored on the *same* readout, side by side, with CIs.

    The winner column is decided by the mean; ``overlapping_ci`` says whether the
    two intervals overlap, which is the only honest way to read a gap this size.
    """
    sub = agg[agg["readout"] == readout]
    rows = []
    for dataset, grp in sub.groupby("dataset", sort=False):
        g = grp.set_index("model")
        if not {"gears", "hgnn"}.issubset(g.index):
            continue
        a, b = g.loc["gears"], g.loc["hgnn"]
        lower_is_better = readout.startswith("mse")
        gears_better = (a["mean"] < b["mean"]) if lower_is_better else (a["mean"] > b["mean"])
        big, small = (max(a["mean"], b["mean"]), min(a["mean"], b["mean"]))
        rows.append(
            {
                "dataset": dataset,
                "readout": readout,
                "GEARS": a["mean"],
                "GEARS_ci": (a["ci_lo"], a["ci_hi"]),
                "HGNN": b["mean"],
                "HGNN_ci": (b["ci_lo"], b["ci_hi"]),
                "winner": "GEARS" if gears_better else "Bipartite HGNN",
                "relative_gap_pct": 100.0 * (big - small) / small if small else float("nan"),
                "overlapping_ci": not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"]),
                "n_seeds": int(min(a["n_seeds"], b["n_seeds"])),
            }
        )
    return pd.DataFrame(rows)


def mismatched_comparison(agg: pd.DataFrame) -> pd.DataFrame:
    """The error, made explicit: model A on top-20 DE against model B genome-wide.

    Both metrics are named on both sides.  This table exists to be labelled as a
    demonstration of what metric pairing produces -- it is not a result.
    """
    rows = []
    for dataset, grp in agg[agg["readout"].isin(["mse_genome_wide", "mse_top20_de"])].groupby(
        "dataset", sort=False
    ):
        g = grp.set_index(["model", "readout"])["mean"]
        if not {("gears", "mse_top20_de"), ("hgnn", "mse_genome_wide")}.issubset(g.index):
            continue
        rows.append(
            {
                "dataset": dataset,
                "GEARS top-20-DE MSE": g[("gears", "mse_top20_de")],
                "HGNN genome-wide MSE": g[("hgnn", "mse_genome_wide")],
                "apparent_ratio": g[("gears", "mse_top20_de")] / g[("hgnn", "mse_genome_wide")],
                "like-for-like genome-wide ratio (GEARS/HGNN)": g[("gears", "mse_genome_wide")]
                / g[("hgnn", "mse_genome_wide")],
                "like-for-like top-20-DE ratio (GEARS/HGNN)": g[("gears", "mse_top20_de")]
                / g[("hgnn", "mse_top20_de")],
            }
        )
    return pd.DataFrame(rows)


def ratio_band(per_run: pd.DataFrame) -> pd.DataFrame:
    """The metric-pair ratio ``mse_top20_de / mse_genome_wide``, per model.

    If this is similar for both models on a dataset, the ratio is a property of
    the metric pair and the data, not of the model -- which is the claim.

    The interval is computed on ``log(ratio)`` and exponentiated, so it is a
    multiplicative interval that cannot extend below zero.  A t-interval taken
    directly on a ratio can, and a lower bound of "-62x" on a strictly positive
    quantity is a presentation error, not a wide interval.
    """
    rows = []
    for (dataset, model), grp in per_run.groupby(["dataset", "model"], sort=False):
        values = grp["mse_ratio_de_over_genome"].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            continue
        m, lo, hi = mean_ci(np.log(values))
        rows.append(
            {"dataset": dataset, "model": model, "mean": float(np.exp(m)),
             "ci_lo": float(np.exp(lo)), "ci_hi": float(np.exp(hi)),
             "n_seeds": int(values.size)}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the direction-accuracy sweep
# --------------------------------------------------------------------------- #
def direction_sweep(root: Path, n_points: int = 25, max_percentile: float = 99.0) -> pd.DataFrame:
    """Direction accuracy against ``min_abs_delta``, for every run.

    Thresholds are laid out on each run's *own* percentiles of ``|true delta|``,
    from 0 to the 99th, so the sweep is comparable across datasets with different
    effect-size scales.  The grid is fixed before any accuracy is computed: the
    threshold is never chosen to improve the curve.
    """
    runs = find_runs(root)
    frames = []
    for row in runs.itertuples():
        d = load_predictions(row.path)
        true_delta = np.abs(d["true"] - d["ctrl_mean"])
        qs = np.linspace(0.0, max_percentile, n_points)
        thresholds = np.percentile(true_delta, qs)
        sweep = direction_accuracy_sweep(d["pred"], d["true"], d["ctrl_mean"], thresholds)
        sweep["percentile"] = qs
        sweep["dataset"], sweep["model"], sweep["seed"] = row.dataset, row.model, row.seed
        frames.append(sweep)
    if not frames:
        raise FileNotFoundError(f"no predictions under {root}")
    return pd.concat(frames, ignore_index=True)


def aggregate_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI of the sweep across seeds, at each percentile."""
    rows = []
    for (dataset, model, pct), grp in sweep.groupby(["dataset", "model", "percentile"], sort=True):
        m, lo, hi = mean_ci(grp["direction_accuracy"].to_numpy())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "percentile": pct,
                "direction_accuracy": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_genes_surviving": grp["n_genes_surviving"].mean(),
                "min_abs_delta": grp["min_abs_delta"].mean(),
            }
        )
    return pd.DataFrame(rows)


def sweep_verdict(agg_sweep: pd.DataFrame) -> pd.DataFrame:
    """Did accuracy climb with effect size?  One row per (dataset, model).

    Reported whichever way it comes out.  A flat or falling curve is the more
    interesting finding and must not be hidden.
    """
    rows = []
    for (dataset, model), grp in agg_sweep.groupby(["dataset", "model"], sort=False):
        g = grp.sort_values("percentile")
        lo, hi = g.iloc[0], g.iloc[-1]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "accuracy_at_all_genes": lo["direction_accuracy"],
                "accuracy_at_p99": hi["direction_accuracy"],
                "max_accuracy": g["direction_accuracy"].max(),
                "climbed": bool(hi["direction_accuracy"] > lo["direction_accuracy"]),
                "reaches_chance": bool(g["direction_accuracy"].max() >= 0.5),
                "genes_at_p99": hi["n_genes_surviving"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def write_tables(root: Path, out_dir: Path, k: int = 20) -> dict[str, pd.DataFrame]:
    """Compute and write every table the README and notebooks read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run = per_run_table(root, k=k)
    agg = aggregate(per_run)
    sweep = direction_sweep(root)
    agg_sweep = aggregate_sweep(sweep)

    tables = {
        "per_run_readouts": per_run.drop(columns=[c for c in ["path"] if c in per_run]),
        "readouts_by_model": agg,
        "like_for_like_genome_wide": like_for_like(agg, "mse_genome_wide"),
        "like_for_like_top20_de": like_for_like(agg, "mse_top20_de"),
        "like_for_like_pearson_delta": like_for_like(agg, "pearson_delta"),
        "mismatched_comparison": mismatched_comparison(agg),
        "metric_ratio_band": ratio_band(per_run),
        "direction_sweep": agg_sweep,
        "direction_sweep_verdict": sweep_verdict(agg_sweep),
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


# --------------------------------------------------------------------------- #
# markdown, so the README's numbers are generated rather than transcribed
# --------------------------------------------------------------------------- #
def _fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.{digits}f}" if abs(x) < 1000 else f"{x:.0f}"


def markdown_report(root: Path, k: int = 20) -> str:
    """The README's results section, rendered from the saved predictions.

    Run ``pertreadout report`` and paste. Every number in the README comes from
    here, so none of them is typed by hand.
    """
    per_run = per_run_table(root, k=k)
    agg = aggregate(per_run)
    lfl = like_for_like(agg, "mse_genome_wide")
    lfl_de = like_for_like(agg, "mse_top20_de")
    lfl_pd = like_for_like(agg, "pearson_delta")
    ratio = ratio_band(per_run)
    mism = mismatched_comparison(agg)
    verdict = sweep_verdict(aggregate_sweep(direction_sweep(root)))

    out: list[str] = []

    def ci(row, side):
        lo, hi = row[f"{side}_ci"]
        return f"{_fmt(row[side], 5)} [{_fmt(lo, 5)}, {_fmt(hi, 5)}]"

    out.append("### Like for like: genome-wide MSE (lower is better)\n")
    out.append("| Dataset | GEARS | Bipartite HGNN | Winner | Gap | CIs overlap? |")
    out.append("|---|---|---|---|---|---|")
    for r in lfl.itertuples():
        row = r._asdict()
        out.append(
            f"| {row['dataset'].capitalize()} | {ci(row, 'GEARS')} | {ci(row, 'HGNN')} | "
            f"{row['winner']} | {row['relative_gap_pct']:.0f}% | "
            f"{'yes' if row['overlapping_ci'] else 'no'} |"
        )

    out.append("\n### Like for like: top-20-DE MSE (lower is better)\n")
    out.append("| Dataset | GEARS | Bipartite HGNN | Winner | Gap | CIs overlap? |")
    out.append("|---|---|---|---|---|---|")
    for r in lfl_de.itertuples():
        row = r._asdict()
        out.append(
            f"| {row['dataset'].capitalize()} | {ci(row, 'GEARS')} | {ci(row, 'HGNN')} | "
            f"{row['winner']} | {row['relative_gap_pct']:.0f}% | "
            f"{'yes' if row['overlapping_ci'] else 'no'} |"
        )

    out.append("\n### Like for like: Pearson on the delta (higher is better)\n")
    out.append("| Dataset | GEARS | Bipartite HGNN | Winner | CIs overlap? |")
    out.append("|---|---|---|---|---|")
    for r in lfl_pd.itertuples():
        row = r._asdict()
        out.append(
            f"| {row['dataset'].capitalize()} | {ci(row, 'GEARS')} | {ci(row, 'HGNN')} | "
            f"{row['winner']} | {'yes' if row['overlapping_ci'] else 'no'} |"
        )

    out.append("\n### The metric-pair ratio: top-20-DE MSE / genome-wide MSE\n")
    out.append("| Dataset | GEARS | Bipartite HGNN |")
    out.append("|---|---|---|")
    piv = ratio[ratio["model"].isin(["gears", "hgnn"])].pivot_table(
        index="dataset", columns="model", values=["mean", "ci_lo", "ci_hi"])
    for dataset in piv.index:
        cells = []
        for m in ["gears", "hgnn"]:
            cells.append(
                f"{piv[('mean', m)][dataset]:.1f}x "
                f"[{piv[('ci_lo', m)][dataset]:.1f}, {piv[('ci_hi', m)][dataset]:.1f}]"
            )
        out.append(f"| {dataset.capitalize()} | " + " | ".join(cells) + " |")
    band = ratio[ratio["model"].isin(["gears", "hgnn"])]["mean"]
    out.append(f"\nMeasured band across datasets and models: **{band.min():.0f}x - {band.max():.0f}x**.")

    out.append("\n### What the mismatched pairing produces\n")
    out.append("| Dataset | GEARS top-20-DE MSE | HGNN genome-wide MSE | Apparent ratio | Genome-wide, both sides | Top-20-DE, both sides |")
    out.append("|---|---|---|---|---|---|")
    for _, row in mism.iterrows():   # column names have spaces; itertuples renames them
        out.append(
            f"| {row['dataset'].capitalize()} | {_fmt(row['GEARS top-20-DE MSE'], 4)} | "
            f"{_fmt(row['HGNN genome-wide MSE'], 4)} | **{row['apparent_ratio']:.1f}x** | "
            f"{row['like-for-like genome-wide ratio (GEARS/HGNN)']:.2f}x | "
            f"{row['like-for-like top-20-DE ratio (GEARS/HGNN)']:.2f}x |"
        )

    out.append("\n### Pearson: what the readout choice is worth\n")
    out.append("| Dataset | Model | Pearson absolute | Control mean profile | Pearson delta | Pearson DE delta |")
    out.append("|---|---|---|---|---|---|")
    piv = agg[agg["model"].isin(["gears", "hgnn"])].pivot_table(
        index=["dataset", "model"], columns="readout", values="mean")
    for (dataset, model) in piv.index:
        r = piv.loc[(dataset, model)]
        out.append(
            f"| {dataset.capitalize()} | {MODEL_LABELS.get(model, model)} | "
            f"{_fmt(r.get('pearson_absolute'))} | "
            f"{_fmt(r.get('pearson_absolute_ctrl_baseline'))} | "
            f"{_fmt(r.get('pearson_delta'))} | {_fmt(r.get('pearson_delta_de'))} |"
        )

    out.append("\n### Direction accuracy against effect size\n")
    out.append("| Dataset | Model | All genes | At the 99th percentile of \\|true delta\\| | Climbed? | Reaches chance? |")
    out.append("|---|---|---|---|---|---|")
    for r in verdict[verdict["model"].isin(["gears", "hgnn"])].itertuples():
        row = r._asdict()
        out.append(
            f"| {row['dataset'].capitalize()} | {MODEL_LABELS.get(row['model'], row['model'])} | "
            f"{_fmt(row['accuracy_at_all_genes'], 3)} | {_fmt(row['accuracy_at_p99'], 3)} | "
            f"{'yes' if row['climbed'] else 'no'} | {'yes' if row['reaches_chance'] else 'no'} |"
        )
    noise = per_run.groupby("dataset")["frac_below_noise_floor"].mean()
    zeros = per_run.groupby("dataset")["frac_true_delta_exactly_zero"].mean()
    out.append("\n| Dataset | True delta within 1.96 SE of zero | True delta exactly zero |")
    out.append("|---|---|---|")
    for d in noise.index:
        out.append(f"| {d.capitalize()} | {noise[d]:.1%} | {zeros.get(d, float('nan')):.1%} |")

    out.append("\n### Identity-graph ablation: what the priors are worth\n")
    abl = agg[agg["readout"] == "mse_genome_wide"].pivot_table(
        index="dataset", columns="model", values="mean")
    out.append("| Dataset | Model | With GO + co-expression priors | With identity graphs | Change |")
    out.append("|---|---|---|---|---|")
    for dataset in abl.index:
        for base in ["gears", "hgnn"]:
            if base in abl.columns and base + "_noprior" in abl.columns:
                a, b = abl[base][dataset], abl[base + "_noprior"][dataset]
                out.append(
                    f"| {dataset.capitalize()} | {MODEL_LABELS[base]} | {_fmt(a, 5)} | {_fmt(b, 5)} | "
                    f"{100 * (b - a) / a:+.1f}% |"
                )
    return "\n".join(out) + "\n"
