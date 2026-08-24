"""Command line entry point.

    pertreadout reproduce --config configs/norman.yaml

runs every seed for both models, writes the predictions, computes every readout
from them, and renders the figures.  It is resumable: a run whose
``predictions.npz`` already exists is skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from pertreadout import data as data_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPO_ROOT / "runs"
DEFAULT_RESULTS = REPO_ROOT / "results"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("dataset") not in data_mod.DATASETS:
        raise ValueError(
            f"{path} names dataset {cfg.get('dataset')!r}; expected one of "
            f"{data_mod.DATASETS}. A config without a dataset is refused here rather "
            "than failing several steps later with a KeyError."
        )
    cfg.setdefault("seeds", [0, 1, 2, 3, 4])
    if len(cfg["seeds"]) < 5:
        raise ValueError(
            f"{path} lists {len(cfg['seeds'])} seeds. Every headline number in this "
            "repo carries a 95% CI over at least 5 seeds; a config with fewer is "
            "refused rather than quietly reported without one."
        )
    return cfg


# --------------------------------------------------------------------------- #
def cmd_download(args) -> int:
    for name in args.datasets or data_mod.DATASETS:
        print(f"{name}: {data_mod.download(name)}")
    return 0


def cmd_manifest(args) -> int:
    manifest = data_mod.write_manifest()
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_verify(args) -> int:
    report = data_mod.verify_manifest()
    for name, status in report.items():
        print(f"{name}: {status}")
    return 0 if all(v == "ok" for v in report.values()) else 1


def cmd_train(args) -> int:
    from pertreadout import train

    cfg = load_config(args.config)
    cfg["force"] = args.force
    train.run(
        cfg["dataset"], args.model, args.seed, cfg,
        Path(args.runs), use_priors=not args.no_priors, device=args.device,
    )
    return 0


def cmd_evaluate(args) -> int:
    from pertreadout import evaluate

    tables = evaluate.write_tables(Path(args.runs), Path(args.results) / "tables")
    for name in ("like_for_like_genome_wide", "metric_ratio_band", "direction_sweep_verdict"):
        print(f"\n### {name}")
        print(tables[name].to_string(index=False))
    return 0


def cmd_figures(args) -> int:
    from pertreadout import figures

    written = figures.render_all(Path(args.results) / "tables", Path(args.results) / "figures")
    for p in written:
        print(p)
    return 0


def cmd_report(args) -> int:
    from pertreadout import evaluate

    print(evaluate.markdown_report(Path(args.runs)))
    return 0


def cmd_reproduce(args) -> int:
    from pertreadout import evaluate, figures, train

    cfg = load_config(args.config)
    cfg["force"] = args.force
    dataset = cfg["dataset"]
    runs_root, results_root = Path(args.runs), Path(args.results)

    variants = [(m, True) for m in train.MODELS]
    if cfg.get("run_identity_ablation", True):
        variants += [(m, False) for m in train.MODELS]

    for model_name, use_priors in variants:
        for seed in cfg["seeds"]:
            train.run(dataset, model_name, seed, cfg, runs_root,
                      use_priors=use_priors, device=args.device)

    tables = evaluate.write_tables(runs_root, results_root / "tables")
    figures.render_all(results_root / "tables", results_root / "figures")

    print("\n### like-for-like, genome-wide MSE (the honest comparison)")
    print(tables["like_for_like_genome_wide"].to_string(index=False))
    print("\n### metric-pair ratio: top-20-DE MSE / genome-wide MSE, per model")
    print(tables["metric_ratio_band"].to_string(index=False))
    print("\n### direction accuracy vs effect size")
    print(tables["direction_sweep_verdict"].to_string(index=False))
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pertreadout", description=__doc__)
    parser.add_argument("--runs", default=str(DEFAULT_RUNS), help="where predictions are written")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="where tables and figures go")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the public datasets into the cache")
    p.add_argument("datasets", nargs="*", choices=list(data_mod.DATASETS) + [])
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("manifest", help="write data/checksums.json from the cache")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("verify", help="check the cache against data/checksums.json")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("train", help="train one (model, seed) on one dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, choices=["gears", "hgnn"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--no-priors", action="store_true", help="identity-graph ablation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="recompute every table from saved predictions")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("figures", help="render the figures from the tables")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("report", help="render the README's results section as markdown")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("reproduce", help="all seeds, both models, tables and figures")
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_reproduce)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
