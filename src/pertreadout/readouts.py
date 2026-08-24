"""Every readout this literature reports, computed from one place.

The argument of this repository is that the choice of readout, not the model,
drives most of the reported differences between perturbation-response models.
That argument is only checkable if every number in the README, every figure and
every table comes from the same code path applied to the same predictions.  So:
nothing computes a metric inline.  Everything goes through :func:`readout_table`.

Conventions
-----------
``pred`` and ``true`` are ``(n_conditions, n_genes)`` arrays of *condition-mean
expression profiles* (post-perturbation, in the same log-normalised space GEARS
trains in).  This matches ``gears.inference.compute_metrics``, which averages
over the cells of a condition before scoring.

``ctrl_mean`` is the ``(n_genes,)`` mean unperturbed profile.  A *delta* is
``profile - ctrl_mean``.

``de_idx`` is ``(n_conditions, k)`` integer indices into the gene axis, the
top-k differentially expressed genes for each condition, taken from
``adata.uns['rank_genes_groups_cov_all']`` exactly as GEARS does.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "mse_genome_wide",
    "mse_top_k_de",
    "pearson_absolute",
    "pearson_delta",
    "pearson_delta_de",
    "direction_accuracy",
    "direction_accuracy_sweep",
    "fraction_below_noise_floor",
    "readout_table",
    "mean_ci",
    "READOUT_COLUMNS",
]

READOUT_COLUMNS = [
    "mse_genome_wide",
    "mse_top20_de",
    "mse_ratio_de_over_genome",
    "pearson_absolute",
    "pearson_delta",
    "pearson_delta_de",
    "direction_accuracy",
]


# --------------------------------------------------------------------------- #
# shaping helpers
# --------------------------------------------------------------------------- #
def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"expected (n_conditions, n_genes), got shape {a.shape}")
    return a


def _check_pair(pred, true):
    pred, true = _as_2d(pred), _as_2d(true)
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} disagree")
    return pred, true


def _as_de_idx(de_idx, n_conditions: int, k: int) -> np.ndarray:
    de = np.asarray(de_idx)
    if de.ndim == 1:
        de = np.tile(de[None, :], (n_conditions, 1))
    if de.shape[0] != n_conditions:
        raise ValueError(f"de_idx has {de.shape[0]} rows, expected {n_conditions}")
    if de.shape[1] < k:
        raise ValueError(f"de_idx supplies {de.shape[1]} genes, need k={k}")
    return de[:, :k].astype(int)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with GEARS' nan convention.

    ``gears.inference.compute_metrics`` replaces a nan correlation with 0.  A nan
    arises exactly when one of the two vectors is constant, which is not an edge
    case here: the trivial "predict the control mean" baseline has a *constant
    zero* delta, so ``pearson_delta`` on it is undefined and reported as 0.  That
    is the honest answer -- an uninformative predictor earns no correlation --
    and :mod:`tests.test_readouts` pins it.
    """
    if x.size < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    r = stats.pearsonr(x, y)[0]
    return 0.0 if np.isnan(r) else float(r)


# --------------------------------------------------------------------------- #
# the readouts
# --------------------------------------------------------------------------- #
def mse_genome_wide(pred, true, mask: Optional[Sequence[int]] = None) -> float:
    """Mean squared error over *all* genes, averaged across conditions.

    This is GEARS' ``mse`` -- the number its training logs print as validation
    MSE.  On these datasets most genes barely move under perturbation, so this
    average is dominated by genes whose true delta is ~0, and it is small.
    """
    pred, true = _check_pair(pred, true)
    if mask is not None:
        mask = np.asarray(mask)
        pred, true = pred[:, mask], true[:, mask]
    return float(np.mean(np.mean((pred - true) ** 2, axis=1)))


def mse_top_k_de(pred, true, de_idx, k: int = 20) -> float:
    """MSE restricted to the top-k differentially expressed genes per condition.

    This is GEARS' ``mse_de`` -- the number reported in the paper's tables.  It
    is scored on exactly the genes selected for having the largest response, so
    it is far larger than :func:`mse_genome_wide` on the same predictions.  The
    size of that gap is the subject of ``notebooks/03``.
    """
    pred, true = _check_pair(pred, true)
    de = _as_de_idx(de_idx, pred.shape[0], k)
    rows = np.arange(pred.shape[0])[:, None]
    return float(np.mean(np.mean((pred[rows, de] - true[rows, de]) ** 2, axis=1)))


def pearson_absolute(pred, true) -> float:
    """Pearson r between predicted and true *absolute* expression profiles.

    Reported widely, and close to uninformative: the unperturbed mean profile
    alone scores ~0.98 here, because the correlation is dominated by the shared
    baseline expression level of every gene rather than by the response.  Pinned
    as a property in the tests, not treated as a bug.
    """
    pred, true = _check_pair(pred, true)
    return float(np.mean([_corr(p, t) for p, t in zip(pred, true)]))


def pearson_delta(pred, true, ctrl_mean) -> float:
    """Pearson r on the delta -- prediction minus control, truth minus control.

    The informative version of the above.  See Ahlmann-Eltze, Huber & Anders,
    Nature Methods 2025.
    """
    pred, true = _check_pair(pred, true)
    c = np.asarray(ctrl_mean, dtype=np.float64).reshape(-1)
    return float(np.mean([_corr(p - c, t - c) for p, t in zip(pred, true)]))


def pearson_delta_de(pred, true, ctrl_mean, de_idx, k: int = 20) -> float:
    """Pearson r on the delta, restricted to the top-k DE genes ("Pearson DE Delta")."""
    pred, true = _check_pair(pred, true)
    c = np.asarray(ctrl_mean, dtype=np.float64).reshape(-1)
    de = _as_de_idx(de_idx, pred.shape[0], k)
    vals = [_corr(p[d] - c[d], t[d] - c[d]) for p, t, d in zip(pred, true, de)]
    return float(np.mean(vals))


def direction_accuracy(pred, true, ctrl_mean, min_abs_delta: float = 0.0) -> float:
    """Fraction of genes whose predicted change has the same sign as the true change.

    Follows ``gears.inference.deeper_analysis``'s ``frac_correct_direction_all``:
    a gene counts as correct when ``sign(pred - ctrl) == sign(true - ctrl)``,
    with ``sign(0)`` treated as its own third value.

    ``min_abs_delta`` restricts scoring to genes whose *true* absolute delta is at
    least that large.  At 0 -- the published setting -- the score is taken over
    every gene, including the large majority whose true delta is measurement
    noise around zero.  Sign is then being graded on coin flips, and any small
    systematic bias in the predicted delta pushes the whole population one way,
    which is how a binary-sounding metric lands *below* 50%.  ``notebooks/04``
    sweeps this threshold to test that explanation.
    """
    pred, true = _check_pair(pred, true)
    c = np.asarray(ctrl_mean, dtype=np.float64).reshape(-1)
    accs = []
    for p, t in zip(pred, true):
        td = t - c
        keep = np.abs(td) >= min_abs_delta if min_abs_delta > 0 else np.ones_like(td, bool)
        if not keep.any():
            continue
        accs.append(float(np.mean(np.sign(p[keep] - c[keep]) == np.sign(td[keep]))))
    return float(np.mean(accs)) if accs else float("nan")


def direction_accuracy_sweep(pred, true, ctrl_mean, thresholds) -> pd.DataFrame:
    """Direction accuracy as a function of ``min_abs_delta``, with survivor counts."""
    pred, true = _check_pair(pred, true)
    c = np.asarray(ctrl_mean, dtype=np.float64).reshape(-1)
    true_delta = true - c
    rows = []
    for thr in np.asarray(thresholds, dtype=np.float64):
        keep = np.abs(true_delta) >= thr
        rows.append(
            {
                "min_abs_delta": float(thr),
                "direction_accuracy": direction_accuracy(pred, true, c, min_abs_delta=thr),
                "n_genes_surviving": float(keep.sum(axis=1).mean()),
                "frac_genes_surviving": float(keep.mean()),
            }
        )
    return pd.DataFrame(rows)


def fraction_below_noise_floor(true, ctrl_mean, delta_se, z: float = 1.96) -> float:
    """Fraction of (condition, gene) true deltas indistinguishable from zero.

    ``delta_se`` is the per-gene standard error of the *measured delta*: the
    control mean and the condition mean are each averages over a finite number of
    cells, so their difference carries a standard error of
    ``sqrt(var_ctrl / n_ctrl + var_cond / n_cond)``.  A true delta smaller than
    ``z`` of those standard errors is not separable from no change at all, so its
    sign carries no signal -- yet :func:`direction_accuracy` at
    ``min_abs_delta=0`` scores it anyway.  This number is the explanation for the
    sub-chance direction accuracy, so it is reported alongside it.

    The comparison is ``<=``, not ``<``, and that is not a rounding choice.  A
    gene with no detected expression in either arm has a delta of exactly zero
    *and* a standard error of exactly zero; under a strict ``<`` it would be
    excluded from the count, which is backwards -- it is the least distinguishable
    from no change of any gene in the panel.
    """
    true = _as_2d(true)
    c = np.asarray(ctrl_mean, dtype=np.float64).reshape(-1)
    se = np.asarray(delta_se, dtype=np.float64)
    if se.ndim == 1:
        se = se[None, :]
    return float(np.mean(np.abs(true - c) <= z * se))


# --------------------------------------------------------------------------- #
# the table everything else reads from
# --------------------------------------------------------------------------- #
def readout_table(pred, true, ctrl_mean, de_idx, k: int = 20, **meta) -> pd.DataFrame:
    """Every readout for one set of predictions, as a single-row DataFrame.

    Extra keyword arguments (``dataset=``, ``model=``, ``seed=``) are carried
    through as leading columns so rows from many runs concatenate directly.
    """
    pred, true = _check_pair(pred, true)
    mse_gw = mse_genome_wide(pred, true)
    mse_de = mse_top_k_de(pred, true, de_idx, k=k)
    row = dict(meta)
    row.update(
        {
            "n_conditions": int(pred.shape[0]),
            "n_genes": int(pred.shape[1]),
            "mse_genome_wide": mse_gw,
            "mse_top20_de": mse_de,
            "mse_ratio_de_over_genome": float(mse_de / mse_gw) if mse_gw > 0 else float("nan"),
            "pearson_absolute": pearson_absolute(pred, true),
            "pearson_delta": pearson_delta(pred, true, ctrl_mean),
            "pearson_delta_de": pearson_delta_de(pred, true, ctrl_mean, de_idx, k=k),
            "direction_accuracy": direction_accuracy(pred, true, ctrl_mean),
        }
    )
    return pd.DataFrame([row])


def mean_ci(values, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and a two-sided ``1-alpha`` t-interval over seeds.

    Returns ``(mean, lo, hi)``.  With a single seed the interval is degenerate
    and both bounds equal the mean -- which is the point: no CI, no headline.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    m = float(v.mean())
    if v.size < 2:
        return m, m, m
    half = stats.t.ppf(1 - alpha / 2, v.size - 1) * float(v.std(ddof=1)) / np.sqrt(v.size)
    return m, m - float(half), m + float(half)
