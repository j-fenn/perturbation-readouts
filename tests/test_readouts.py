"""These tests are the repository's argument, written so it can be executed.

Each one pins a property of a readout that, taken together, explains how two
models with near-identical predictive accuracy can be reported as differing by
an order of magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from pertreadout.readouts import (
    direction_accuracy,
    direction_accuracy_sweep,
    fraction_below_noise_floor,
    mean_ci,
    mse_genome_wide,
    mse_top_k_de,
    pearson_absolute,
    pearson_delta,
    pearson_delta_de,
    readout_table,
)


@pytest.fixture
def toy():
    """A small synthetic perturbation experiment with realistic structure.

    Baseline expression varies a lot between genes (that is what inflates
    Pearson on absolute expression); the perturbation response is confined to a
    handful of genes and everything else moves by noise only (that is what
    deflates genome-wide MSE and direction accuracy).
    """
    rng = np.random.default_rng(0)
    n_cond, n_genes, n_de = 12, 500, 20
    ctrl_mean = rng.gamma(2.0, 1.0, size=n_genes)

    true = np.tile(ctrl_mean, (n_cond, 1))
    de_idx = np.stack([rng.choice(n_genes, size=n_de, replace=False) for _ in range(n_cond)])
    rows = np.arange(n_cond)[:, None]
    true[rows, de_idx] += rng.normal(0, 2.0, size=(n_cond, n_de))   # real response
    true += rng.normal(0, 0.05, size=true.shape)                    # measurement noise

    pred = ctrl_mean + 0.6 * (true - ctrl_mean) + rng.normal(0, 0.05, size=true.shape)
    # standard error of the measured delta: `true` above carries one draw of
    # measurement noise at sd 0.05, so that is the scale a delta must beat.
    delta_se = np.full(n_genes, 0.05)
    return dict(pred=pred, true=true, ctrl_mean=ctrl_mean, de_idx=de_idx, delta_se=delta_se)


# --------------------------------------------------------------------------- #
# 1. Pearson on absolute expression is nearly free
# --------------------------------------------------------------------------- #
def test_control_mean_profile_scores_high_pearson_absolute(toy):
    """The trivial predictor -- "assume nothing changed" -- scores >= 0.95.

    Not a bug in the metric implementation: a documented property.  Any paper
    reporting Pearson ~0.99 on absolute expression is reporting a number the
    unperturbed mean profile almost reaches on its own.
    """
    trivial = np.tile(toy["ctrl_mean"], (toy["true"].shape[0], 1))
    assert pearson_absolute(trivial, toy["true"]) >= 0.95


def test_control_mean_profile_scores_zero_pearson_delta(toy):
    """The same trivial predictor scores ~0 once the baseline is subtracted.

    Its delta is identically zero, so the correlation is undefined; following
    GEARS' own nan-to-zero convention we report 0.  Adding a whisker of noise so
    the vector is no longer constant gives the same answer the honest way.
    """
    n = toy["true"].shape[0]
    trivial = np.tile(toy["ctrl_mean"], (n, 1))
    assert abs(pearson_delta(trivial, toy["true"], toy["ctrl_mean"])) < 1e-12

    rng = np.random.default_rng(1)
    jittered = trivial + rng.normal(0, 1e-6, size=trivial.shape)
    assert abs(pearson_delta(jittered, toy["true"], toy["ctrl_mean"])) < 0.1


def test_pearson_delta_rewards_a_real_predictor(toy):
    """Sanity check in the other direction: a predictor with signal scores well."""
    assert pearson_delta(toy["pred"], toy["true"], toy["ctrl_mean"]) > 0.8
    assert pearson_delta_de(toy["pred"], toy["true"], toy["ctrl_mean"], toy["de_idx"]) > 0.8


# --------------------------------------------------------------------------- #
# 2. Direction accuracy on noise is a coin flip
# --------------------------------------------------------------------------- #
def test_direction_accuracy_on_pure_noise_is_chance():
    """With no signal anywhere, sign agreement is ~0.5 -- by construction.

    This is the control for ``notebooks/04``.  A published direction accuracy of
    7-20% is not "near chance": it is far *below* the number this test produces
    from pure noise, which is what makes it worth explaining.
    """
    rng = np.random.default_rng(7)
    n_cond, n_genes = 40, 2000
    ctrl_mean = np.zeros(n_genes)
    true = rng.normal(0, 1, size=(n_cond, n_genes))
    pred = rng.normal(0, 1, size=(n_cond, n_genes))
    assert direction_accuracy(pred, true, ctrl_mean, min_abs_delta=0.0) == pytest.approx(0.5, abs=0.02)


def test_direction_accuracy_climbs_when_the_signal_is_real(toy):
    """Thresholding on effect size must raise accuracy when signal exists.

    If the sweep in ``notebooks/04`` does *not* climb on real data, this test
    says the sweep itself is sound and the flat curve is a finding, not a bug.
    """
    at_zero = direction_accuracy(toy["pred"], toy["true"], toy["ctrl_mean"], 0.0)
    at_high = direction_accuracy(toy["pred"], toy["true"], toy["ctrl_mean"], 1.0)
    assert at_high > at_zero


def test_direction_sweep_is_monotone_in_surviving_genes(toy):
    sweep = direction_accuracy_sweep(toy["pred"], toy["true"], toy["ctrl_mean"], [0.0, 0.5, 1.0, 2.0])
    counts = sweep["n_genes_surviving"].to_numpy()
    assert np.all(np.diff(counts) <= 0)
    assert counts[0] == toy["true"].shape[1]


def test_fraction_below_noise_floor_is_most_genes(toy):
    """Most genes in a perturbation experiment do not measurably move."""
    frac = fraction_below_noise_floor(toy["true"], toy["ctrl_mean"], toy["delta_se"])
    assert 0.0 <= frac <= 1.0
    assert frac > 0.5


# --------------------------------------------------------------------------- #
# 3. The MSE ratio -- the source of every large "improvement factor"
# --------------------------------------------------------------------------- #
def test_genome_wide_mse_is_smaller_than_top20_de_mse(toy):
    """Genome-wide MSE <= top-20-DE MSE whenever DE genes carry larger variance.

    This inequality is the whole mechanism.  DE genes are *selected* for having
    the largest response, so the residual on them is larger than the residual on
    an average gene.  Pairing one model's top-20-DE MSE against another's
    genome-wide MSE therefore produces a large "improvement factor" from two
    identical models -- the ratio measured in ``notebooks/03``.
    """
    gw = mse_genome_wide(toy["pred"], toy["true"])
    de = mse_top_k_de(toy["pred"], toy["true"], toy["de_idx"])
    assert gw <= de


def test_the_ratio_is_a_property_of_the_metric_pair_not_the_model(toy):
    """Two very different predictors give similar DE/genome-wide ratios.

    Because the ratio tracks the variance structure of the data rather than the
    quality of the fit, it survives changing the model outright.
    """
    rng = np.random.default_rng(3)
    weak = toy["ctrl_mean"] + 0.1 * (toy["true"] - toy["ctrl_mean"]) + rng.normal(0, 0.05, toy["true"].shape)
    ratios = []
    for p in (toy["pred"], weak):
        ratios.append(
            mse_top_k_de(p, toy["true"], toy["de_idx"]) / mse_genome_wide(p, toy["true"])
        )
    assert min(ratios) > 5.0
    assert max(ratios) / min(ratios) < 3.0


def test_mismatched_pairing_manufactures_an_improvement(toy):
    """Scoring one model on DE genes and another genome-wide invents a win.

    Here both "models" are the *same predictions*.  Any ratio this produces is
    pure metric mismatch -- which is why ``notebooks/03`` prints both metrics on
    both sides of every comparison.
    """
    honest = mse_genome_wide(toy["pred"], toy["true"]) / mse_genome_wide(toy["pred"], toy["true"])
    mismatched = mse_top_k_de(toy["pred"], toy["true"], toy["de_idx"]) / mse_genome_wide(
        toy["pred"], toy["true"]
    )
    assert honest == pytest.approx(1.0)
    assert mismatched > 5.0


# --------------------------------------------------------------------------- #
# 4. plumbing
# --------------------------------------------------------------------------- #
def test_readout_table_has_every_column_and_carries_metadata(toy):
    from pertreadout.readouts import READOUT_COLUMNS

    tbl = readout_table(
        toy["pred"], toy["true"], toy["ctrl_mean"], toy["de_idx"],
        dataset="toy", model="toy_model", seed=0,
    )
    assert len(tbl) == 1
    for col in READOUT_COLUMNS:
        assert col in tbl.columns
        assert np.isfinite(tbl[col].iloc[0])
    assert tbl["dataset"].iloc[0] == "toy"
    assert tbl["seed"].iloc[0] == 0


def test_readout_table_matches_the_standalone_functions(toy):
    """Nothing may compute a metric inline; the table must agree with the parts."""
    tbl = readout_table(toy["pred"], toy["true"], toy["ctrl_mean"], toy["de_idx"]).iloc[0]
    assert tbl["mse_genome_wide"] == pytest.approx(mse_genome_wide(toy["pred"], toy["true"]))
    assert tbl["mse_top20_de"] == pytest.approx(mse_top_k_de(toy["pred"], toy["true"], toy["de_idx"]))
    assert tbl["pearson_delta"] == pytest.approx(
        pearson_delta(toy["pred"], toy["true"], toy["ctrl_mean"])
    )


def test_mean_ci_needs_more_than_one_seed():
    m, lo, hi = mean_ci([0.5])
    assert lo == hi == m
    m, lo, hi = mean_ci([0.40, 0.42, 0.44, 0.46, 0.48])
    assert lo < m < hi
    assert m == pytest.approx(0.44)


def test_readouts_accept_a_single_condition(toy):
    tbl = readout_table(
        toy["pred"][0], toy["true"][0], toy["ctrl_mean"], toy["de_idx"][0]
    )
    assert tbl["n_conditions"].iloc[0] == 1
