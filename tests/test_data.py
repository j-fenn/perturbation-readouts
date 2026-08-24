"""One test per condition-normalisation rewrite rule.

These rules decide which cells are controls and how a perturbation is named.
Every delta readout in the repo subtracts the control mean, so a mistake here
does not raise -- it quietly shifts every number.  Hence a test each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pertreadout.data import (
    normalize_condition,
    normalize_obs,
    normalize_var,
    subsample_by_condition,
)


# --------------------------------------------------------------------------- #
# rewrite rules, one test each
# --------------------------------------------------------------------------- #
def test_rule_double_underscore_becomes_plus():
    assert normalize_condition(["AHR__KLF1"]).tolist() == ["AHR+KLF1"]


def test_rule_whitespace_is_stripped():
    assert normalize_condition(["  AHR + KLF1  "]).tolist() == ["AHR+KLF1"]


def test_rule_empty_label_is_a_control():
    assert normalize_condition(["", "   "]).tolist() == ["ctrl", "ctrl"]


def test_rule_double_control_collapses_to_one():
    assert normalize_condition(["ctrl+ctrl"]).tolist() == ["ctrl"]
    assert normalize_condition(["ctrl__ctrl"]).tolist() == ["ctrl"]


def test_rule_single_gene_keeps_its_ctrl_partner():
    """``GENE+ctrl`` is GEARS' spelling of a single-gene perturbation; keep it."""
    assert normalize_condition(["AHR+ctrl"]).tolist() == ["AHR+ctrl"]


def test_rule_condition_falls_back_to_guide_merged():
    obs = pd.DataFrame({"guide_merged": ["AHR__KLF1", "ctrl"]})
    assert normalize_obs(obs)["condition"].tolist() == ["AHR+KLF1", "ctrl"]


def test_rule_missing_condition_column_is_an_error():
    with pytest.raises(ValueError, match="condition-like"):
        normalize_obs(pd.DataFrame({"something_else": ["x"]}))


def test_rule_dose_val_is_one_plus_one_for_two_part_labels():
    obs = normalize_obs(pd.DataFrame({"condition": ["AHR+KLF1", "AHR+ctrl", "ctrl"]}))
    assert obs["dose_val"].tolist() == ["1+1", "1+1", "1"]


def test_rule_control_flag_marks_only_the_control():
    """The deliberate deviation from the source project's rule.

    Its ``ensure_obs_columns`` set ``control = 1`` for any label with fewer than
    two parts, which in the GEARS-distributed files is only ``'ctrl'`` -- but on
    a file where single perturbations are written ``'AHR'`` rather than
    ``'AHR+ctrl'`` it would mark every single-gene perturbation as a control and
    fold perturbed cells into the control mean.
    """
    obs = normalize_obs(pd.DataFrame({"condition": ["AHR+KLF1", "AHR+ctrl", "AHR", "ctrl"]}))
    assert obs["control"].tolist() == [0, 0, 0, 1]


def test_rule_cell_type_and_condition_name_are_filled_in():
    obs = normalize_obs(pd.DataFrame({"condition": ["AHR+ctrl", "ctrl"]}))
    assert obs["cell_type"].tolist() == ["K562", "K562"]
    assert obs["condition_name"].tolist() == ["K562_AHR+ctrl_1+1", "K562_ctrl_1"]


def test_rule_existing_condition_name_is_respected():
    obs = normalize_obs(
        pd.DataFrame({"condition": ["AHR+ctrl"], "condition_name": ["K562_AHR+ctrl_1+1"]})
    )
    assert obs["condition_name"].tolist() == ["K562_AHR+ctrl_1+1"]


def test_var_names_are_made_unique_and_gene_name_added():
    var, names = normalize_var(pd.DataFrame(index=["A", "A", "B"]), ["A", "A", "B"])
    assert names.tolist() == ["A", "A-1", "B"]
    assert var["gene_name"].tolist() == ["A", "A-1", "B"]


# --------------------------------------------------------------------------- #
# subsampling
# --------------------------------------------------------------------------- #
def test_subsample_caps_each_condition_and_is_deterministic():
    conditions = ["a"] * 100 + ["b"] * 3 + ["ctrl"] * 50
    idx = subsample_by_condition(conditions, max_cells_per_condition=10, seed=0)
    again = subsample_by_condition(conditions, max_cells_per_condition=10, seed=0)
    assert np.array_equal(idx, again)
    assert np.array_equal(idx, np.sort(idx))
    picked = pd.Series(np.asarray(conditions)[idx]).value_counts().to_dict()
    assert picked == {"a": 10, "ctrl": 10, "b": 3}


def test_subsample_changes_with_the_seed():
    conditions = ["a"] * 100
    assert not np.array_equal(
        subsample_by_condition(conditions, 10, seed=0), subsample_by_condition(conditions, 10, seed=1)
    )


# --------------------------------------------------------------------------- #
# the shipped configs
# --------------------------------------------------------------------------- #
def test_every_shipped_config_is_loadable_and_complete():
    """Each config must name its dataset and carry at least five seeds.

    Not a formality: two of these configs were once derived from the third by a
    text edit that dropped the ``dataset`` key, and nothing noticed until a
    command that happened to read it failed several steps in.
    """
    from pathlib import Path

    from pertreadout.cli import load_config
    from pertreadout.data import DATASETS

    configs = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    assert {p.stem for p in configs} == set(DATASETS)
    for path in configs:
        cfg = load_config(path)
        assert cfg["dataset"] == path.stem
        assert len(cfg["seeds"]) >= 5
        for key in ["split", "epochs", "max_cells_per_condition", "batch_size", "de_k"]:
            assert key in cfg, f"{path.name} is missing {key}"


def test_a_config_without_a_dataset_is_refused(tmp_path):
    from pertreadout.cli import load_config

    bad = tmp_path / "bad.yaml"
    bad.write_text("seeds: [0, 1, 2, 3, 4]\nepochs: 6\n")
    with pytest.raises(ValueError, match="expected one of"):
        load_config(bad)


def test_a_config_with_too_few_seeds_is_refused(tmp_path):
    """Multi-seed or it does not ship: the refusal is in code, not in a README."""
    from pertreadout.cli import load_config

    bad = tmp_path / "thin.yaml"
    bad.write_text("dataset: norman\nseeds: [0, 1]\n")
    with pytest.raises(ValueError, match="95% CI"):
        load_config(bad)
