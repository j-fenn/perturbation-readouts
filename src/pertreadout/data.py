"""Dataset access for the three public CRISPR perturbation datasets.

Nothing here ships data.  All three datasets are public and reachable through
GEARS' own ``PertData`` loader, which pulls them from the Harvard Dataverse; we
cache them under ``$PERTREADOUT_CACHE`` (default ``~/.cache/pertreadout``) and
commit a checksum manifest so a reader can confirm they got the same bytes.

Two things happen on top of the download:

1. **Condition normalisation** (:func:`normalize_obs`, :func:`normalize_var`).
   The GEARS-distributed files are already normalised, so these are no-ops on
   them -- they exist so a reader can point this pipeline at their own ``.h5ad``.
   The rules are ported from the original project's ``ensure_obs_columns`` /
   ``ensure_var_columns``; each rule has a test in ``tests/test_data.py``.  One
   rule is deliberately *not* ported faithfully: see :func:`normalize_obs`.

2. **Cell subsampling** (:func:`prepare`).  A cap of N cells per condition, which
   is what makes these runs finish on a laptop CPU.  It is a real limitation and
   it is stated in the README and carried in every results table as
   ``max_cells_per_condition``.

What is emphatically *not* ported: the original project's ``ensure_minimal_uns``,
which filled ``adata.uns['rank_genes_groups_cov_all']`` with "the first 20 genes
in the file" when the key was missing.  Every DE-based readout in this repo would
be meaningless under that substitution, so a missing DE key is a hard error here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DATASETS = ("norman", "adamson", "dixit")

#: DE genes per condition, as GEARS ranks them.  ``k=20`` throughout the repo.
DE_UNS_KEY = "rank_genes_groups_cov_all"

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "checksums.json"


# --------------------------------------------------------------------------- #
# cache locations
# --------------------------------------------------------------------------- #
def cache_root() -> Path:
    root = Path(os.environ.get("PERTREADOUT_CACHE", Path.home() / ".cache" / "pertreadout"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def raw_dir(name: str) -> Path:
    return cache_root() / name


def prepared_dir(name: str, max_cells_per_condition: int, seed: int) -> Path:
    """Where a subsampled copy lives.

    Directly under the cache root, not in a subdirectory: GEARS' ``PertData``
    derives its dataset name from the last path component and then writes its
    processed cell graphs to ``<data_path>/<dataset_name>``, so a nested layout
    sends that cache somewhere that does not exist.
    """
    return cache_root() / f"{name}_cells{max_cells_per_condition}_seed{seed}"


# --------------------------------------------------------------------------- #
# normalisation rules (one test each in tests/test_data.py)
# --------------------------------------------------------------------------- #
def normalize_condition(values: Iterable[str]) -> pd.Series:
    """Apply the condition-string rewrite rules, in order.

    ``guide_merged``-style labels arrive in several dialects.  The rules:

    ==============  =========  =============================================
    input           output     rule
    ==============  =========  =============================================
    ``' A + B '``   ``'A+B'``  strip outer and inner whitespace
    ``'A__B'``      ``'A+B'``  ``__`` is the double-guide separator
    ``''``          ``'ctrl'`` an empty guide label is an unperturbed cell
    ``'ctrl+ctrl'`` ``'ctrl'`` a double non-targeting guide is still a control
    ==============  =========  =============================================
    """
    s = pd.Series(list(values), dtype="object").astype(str)
    s = s.str.strip()
    s = s.str.replace("__", "+", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.mask(s.eq("") | s.eq("nan"), "ctrl")
    s = s.replace({"ctrl+ctrl": "ctrl"})
    return s


def normalize_obs(obs: pd.DataFrame, default_cell_type: str = "K562") -> pd.DataFrame:
    """Fill in the obs columns GEARS' ``PertData`` requires.

    ``condition`` comes from ``condition`` or, failing that, ``guide_merged``.
    ``dose_val`` is ``'1+1'`` for a two-part label and ``'1'`` otherwise, matching
    the GEARS-distributed files, where a single-gene perturbation is written
    ``'GENE+ctrl'`` and therefore carries ``'1+1'``.

    **Deviation from the source project, on purpose.**  Its ``ensure_obs_columns``
    set ``control = 1`` for every condition whose label had fewer than two parts,
    which marks single-gene perturbations as controls.  In the GEARS-distributed
    files ``control == 1`` holds for ``condition == 'ctrl'`` and nothing else --
    verified on all three datasets -- and that is what is implemented here.  The
    original rule would have folded perturbed cells into the control mean, which
    is the denominator of every delta readout in this repo.
    """
    obs = obs.copy()

    if "condition" in obs.columns:
        source = obs["condition"]
    elif "guide_merged" in obs.columns:
        source = obs["guide_merged"]
    else:
        raise ValueError("no condition-like column (expected `condition` or `guide_merged`)")

    condition = normalize_condition(source.to_numpy())
    condition.index = obs.index
    obs["condition"] = condition

    if "cell_type" not in obs.columns:
        obs["cell_type"] = default_cell_type
    obs["cell_type"] = obs["cell_type"].astype(str).fillna(default_cell_type)

    n_parts = condition.str.count(r"\+") + 1
    if "dose_val" not in obs.columns:
        obs["dose_val"] = np.where(n_parts == 2, "1+1", "1")
    obs["dose_val"] = obs["dose_val"].astype(str)

    # control iff the condition is the control -- see the docstring.
    obs["control"] = (condition == "ctrl").astype(int)

    if "condition_name" not in obs.columns:
        obs["condition_name"] = (
            obs["cell_type"].astype(str) + "_" + condition + "_" + obs["dose_val"].astype(str)
        )
    obs["condition_name"] = obs["condition_name"].astype(str)
    return obs


def normalize_var(var: pd.DataFrame, var_names: Iterable[str]) -> tuple[pd.DataFrame, pd.Index]:
    """Ensure unique string gene identifiers and a ``gene_name`` column."""
    names = pd.Index([str(v) for v in var_names])
    if names.has_duplicates:
        counts: dict[str, int] = {}
        out = []
        for n in names:
            if n in counts:
                counts[n] += 1
                out.append(f"{n}-{counts[n]}")
            else:
                counts[n] = 0
                out.append(n)
        names = pd.Index(out)
    var = var.copy()
    var.index = names
    if "gene_name" not in var.columns:
        var["gene_name"] = names
    var["gene_name"] = var["gene_name"].astype(str)
    return var, names


# --------------------------------------------------------------------------- #
# download / prepare
# --------------------------------------------------------------------------- #
def download(name: str) -> Path:
    """Fetch a dataset through GEARS' loader if it is not already cached."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    target = raw_dir(name) / "perturb_processed.h5ad"
    if target.exists():
        return raw_dir(name)
    from gears import PertData  # imported lazily: only the full extra needs it

    PertData(str(cache_root())).load(data_name=name)
    if not target.exists():
        raise RuntimeError(f"download of {name} did not produce {target}")
    return raw_dir(name)


def subsample_by_condition(conditions, max_cells_per_condition: int, seed: int) -> np.ndarray:
    """Indices of at most N cells per condition, chosen without replacement.

    Deterministic given ``seed``, and sorted so the resulting AnnData keeps the
    file's cell order.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(list(conditions)).astype(str)
    keep: list[int] = []
    for cond in np.unique(values):
        idx = np.flatnonzero(values == cond)
        take = min(len(idx), max_cells_per_condition)
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    return np.sort(np.asarray(keep, dtype=np.int64))


def prepare(name: str, max_cells_per_condition: int, seed: int, force: bool = False) -> Path:
    """Materialise a subsampled, normalised copy of a dataset for GEARS to load.

    Returns the directory holding ``perturb_processed.h5ad``.  The DE gene lists
    and non-zero/non-dropout index sets in ``adata.uns`` are carried through
    untouched -- they are computed on the *full* dataset by GEARS' own
    preprocessing, which is what makes the top-20-DE readout comparable to the
    published one even though we train on a subsample.
    """
    out_dir = prepared_dir(name, max_cells_per_condition, seed)
    out_file = out_dir / "perturb_processed.h5ad"
    if out_file.exists() and not force:
        return out_dir

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    src = download(name) / "perturb_processed.h5ad"
    adata = sc.read_h5ad(src)

    if DE_UNS_KEY not in adata.uns:
        raise ValueError(
            f"{src} has no `uns[{DE_UNS_KEY!r}]`. Every DE readout in this repo needs "
            "real differential-expression rankings; fabricating them would silently "
            "invalidate the analysis."
        )

    obs = normalize_obs(adata.obs)
    var, names = normalize_var(adata.var, adata.var_names)
    adata.obs, adata.var = obs, var
    adata.var_names = names

    keep = subsample_by_condition(adata.obs["condition"], max_cells_per_condition, seed)
    sub = adata[keep, :].to_memory() if adata.isbacked else adata[keep, :].copy()
    sub.uns = dict(adata.uns)  # DE rankings computed on the full data, preserved

    if not sparse.issparse(sub.X):
        sub.X = sparse.csr_matrix(np.asarray(sub.X))
    else:
        sub.X = sub.X.tocsr()

    out_dir.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(out_file.as_posix())
    del adata, sub
    return out_dir


# --------------------------------------------------------------------------- #
# the loader the training code actually calls
# --------------------------------------------------------------------------- #
def load_pert_data(
    name: str,
    seed: int,
    max_cells_per_condition: int = 32,
    batch_size: int = 16,
    split: str = "simulation",
    train_gene_set_size: float = 0.75,
    default_pert_graph: bool = True,
):
    """Prepared data, split, and dataloaders -- one object both models consume.

    ``split`` defaults to GEARS' ``simulation`` split, which holds out whole
    *perturbations whose genes were never seen in training*.  That is the claim
    under test: generalisation to novel perturbations, not to novel cells.  The
    original project used ``no_test``, which is also available here, but note
    that GEARS' ``no_test`` branch ignores ``train_gene_set_size`` entirely and
    holds out a random 10% of perturbations without any unseen-gene guarantee --
    a weaker test than the one this repo claims to run.
    """
    from gears import PertData

    data_dir = prepare(name, max_cells_per_condition, seed)
    pert_data = PertData(str(cache_root()), default_pert_graph=default_pert_graph)
    pert_data.load(data_path=str(data_dir))
    pert_data.prepare_split(split=split, seed=seed, train_gene_set_size=train_gene_set_size)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)
    return pert_data


def control_statistics(adata) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean, variance and cell count of the unperturbed profile.

    The mean is the baseline every delta readout subtracts; the variance and
    count give the standard error that :func:`~pertreadout.readouts.
    fraction_below_noise_floor` compares true deltas against.
    """
    mask = (adata.obs["condition"].astype(str) == "ctrl").to_numpy()
    if not mask.any():
        raise ValueError("no control cells (`condition == 'ctrl'`) in this dataset")
    X = adata[mask].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    return X.mean(axis=0), X.var(axis=0), int(mask.sum())


def de_indices(adata, conditions: Iterable[str], k: int = 20) -> np.ndarray:
    """``(len(conditions), k)`` top-k DE gene indices, exactly as GEARS selects them."""
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    cond_to_full = dict(zip(adata.obs["condition"].astype(str), adata.obs["condition_name"].astype(str)))
    rows = []
    for cond in conditions:
        full = cond_to_full[cond]
        genes = adata.uns[DE_UNS_KEY][full][:k]
        rows.append([gene_to_idx[g] for g in genes])
    return np.asarray(rows, dtype=np.int64)


# --------------------------------------------------------------------------- #
# checksum manifest
# --------------------------------------------------------------------------- #
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(names: Optional[Iterable[str]] = None, path: Path = MANIFEST_PATH) -> dict:
    """Record size and SHA-256 of each raw ``perturb_processed.h5ad``."""
    manifest = {}
    for name in names or DATASETS:
        f = raw_dir(name) / "perturb_processed.h5ad"
        if not f.exists():
            continue
        manifest[name] = {
            "file": f"{name}/perturb_processed.h5ad",
            "bytes": f.stat().st_size,
            "sha256": sha256(f),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Check the cached files against the committed manifest."""
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}; run `pertreadout manifest` first")
    manifest = json.loads(path.read_text())
    report = {}
    for name, entry in manifest.items():
        f = raw_dir(name) / "perturb_processed.h5ad"
        if not f.exists():
            report[name] = "missing"
        elif f.stat().st_size != entry["bytes"]:
            report[name] = "size mismatch"
        else:
            report[name] = "ok" if sha256(f) == entry["sha256"] else "checksum mismatch"
    return report
