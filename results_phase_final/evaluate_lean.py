"""
[LEAN ENTRYPOINT] Canonical script for the consolidated workflow.

Phase-2 evaluation — K-fold ensemble on the labelled dev-test split.

For a given `(mode, arch)` we automatically discover every
`checkpoints_phase2/{mode}_{arch}_fold{k}.pt` file, score each fold on
the dev-test split, and report:

 1. Per-fold AUC / pAUC / DCASE hmean
 2. **Ensemble AUC / pAUC / DCASE hmean** — scores averaged across folds

The ensemble is the primary number used downstream in
`collect_results_lean.py`. The
per-fold numbers are kept only for transparency and variance analysis.

IMPORTANT:
  - eval-test (sections 03-05, unlabelled) is NOT touched by this
    script. A separate submission-time script will score it once a
    best DG model has been selected.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BearingDataset
from evaluate_lean_utils import (
    compute_metrics,
    plot_overall_summary,
    plot_roc,
    plot_score_histograms,
    plot_section_bars,
    print_report,
    score_dataset,
)
from model import ARCHES, build_model

MODE_ALIASES = {
    "adversarial": "domain_regularized",
}
MODE_CHOICES = ["baseline", "mixed", "domain_regularized", "contrastive"]


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def mode_variants(mode: str):
    canonical = normalize_mode(mode)
    aliases = [k for k, v in MODE_ALIASES.items() if v == canonical]
    out = [canonical] + aliases
    seen = set()
    uniq = []
    for m in out:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


# --------------------------------------------------------------------------- #
# checkpoint discovery
# --------------------------------------------------------------------------- #

def discover_fold_ckpts(ckpt_dir: str, mode: str, arch: str) -> List[str]:
    """Return the sorted list of `{mode}_{arch}_fold{k}.pt` paths."""
    paths: List[str] = []
    for mode_name in mode_variants(mode):
        pattern = os.path.join(ckpt_dir, f"{mode_name}_{arch}_fold*.pt")
        paths.extend(
            p for p in glob.glob(pattern)
            if not p.endswith("_last.pt") and "_history" not in p
        )
    paths = sorted(set(paths))
    return paths


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def score_checkpoint_on_loader(
    ckpt_path: str, loader, device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load a single-fold checkpoint and score the loader."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    model = build_model(arch).to(device)
    model.load_state_dict(ckpt["model_state"])

    scores, domains, sections, labels = score_dataset(model, loader, device)
    return scores, domains, sections, labels, arch


def save_run_artifacts(
    out_dir: str, scores, domains, sections, labels,
    samples: List[dict], rows, agg, overall, hmean, extra: dict,
):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "per_clip_scores.csv")
    with open(csv_path, "w") as f:
        f.write("filename,section,domain,label,score\n")
        for i, s in enumerate(samples):
            f.write(f"{s['filename']},{s['section']},{s['domain']},"
                    f"{s['label']},{scores[i]:.6f}\n")

    summary = {
        "phase": 2,
        "per_section": rows,
        "per_domain_overall": agg,
        "overall": overall,
        "harmonic_mean": hmean,
    }
    summary.update(extra)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_score_histograms(scores, domains, labels,
                          os.path.join(out_dir, "score_histograms.png"))
    plot_roc(scores, domains, labels, os.path.join(out_dir, "roc.png"))
    plot_section_bars(rows, os.path.join(out_dir, "per_section_auc.png"))
    plot_overall_summary(agg, overall, hmean,
                         os.path.join(out_dir, "overall_summary.png"))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="mixed",
                   choices=MODE_CHOICES + list(MODE_ALIASES.keys()),
                   help="Evaluation mode key. 'adversarial' is kept as a backward-compatible "
                        "alias for 'domain_regularized'.")
    p.add_argument("--arch", type=str, default="ae",
                   choices=list(ARCHES.keys()))

    p.add_argument("--dev_test_dir", type=str,
                   default=os.environ.get("BEARING_DEV_TEST_DIR",
                                          "data/dcase_bearing_dev/bearing/test"))
    p.add_argument("--dev_test_cache", type=str,
                   default=os.environ.get("BEARING_DEV_TEST_CACHE", ""))

    p.add_argument("--ckpt_dir", type=str,
                   default=os.environ.get("BEARING_PHASE2_CKPT_DIR",
                                          "checkpoints_phase2"))
    p.add_argument("--out_dir", type=str, default="eval_results_phase2")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_fpr", type=float, default=0.1)
    p.add_argument("--skip_per_fold_plots", action="store_true",
                   help="Skip producing plots for each individual fold "
                        "(keeps only the ensemble plots; faster).")
    return p.parse_args()


def main():
    args = parse_args()
    args.mode = normalize_mode(args.mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_ckpts = discover_fold_ckpts(args.ckpt_dir, args.mode, args.arch)
    if not fold_ckpts:
        raise FileNotFoundError(
            f"No fold checkpoints matching "
            f"{args.mode}_{args.arch}_fold*.pt in {args.ckpt_dir}.\n"
            f"Run train_lean.py --mode {args.mode} --arch {args.arch} first."
        )

    print(f"=== phase-2 evaluation: {args.mode}/{args.arch} ===")
    print(f"Discovered {len(fold_ckpts)} fold checkpoints:")
    for p in fold_ckpts:
        print(f"  - {os.path.basename(p)}")

    # --- build dev-test loader ONCE (shared across folds) ---
    dev_ds = BearingDataset(
        args.dev_test_dir,
        cache_dir=(args.dev_test_cache or None),
        return_label=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    run_root = os.path.join(args.out_dir, f"{args.mode}_{args.arch}")
    os.makedirs(run_root, exist_ok=True)

    # --- score each fold, collect per-clip scores for the ensemble ---
    per_fold_scores: List[np.ndarray] = []
    per_fold_metrics: List[dict] = []
    domains_ref: Optional[np.ndarray] = None
    sections_ref: Optional[np.ndarray] = None
    labels_ref: Optional[np.ndarray] = None
    resolved_arch = args.arch

    for k, ckpt_path in enumerate(fold_ckpts):
        print(f"\n--- Fold {k}: {os.path.basename(ckpt_path)} ---")
        scores, domains, sections, labels, resolved_arch = \
            score_checkpoint_on_loader(ckpt_path, dev_loader, device)

        if domains_ref is None:
            domains_ref, sections_ref, labels_ref = domains, sections, labels
        else:
            assert np.array_equal(domains_ref, domains)
            assert np.array_equal(sections_ref, sections)
            assert np.array_equal(labels_ref, labels)

        rows, agg, overall, hmean = compute_metrics(
            scores, domains, sections, labels, args.max_fpr
        )
        print_report(
            f"phase2/{args.mode}/fold{k}",
            rows, agg, overall, hmean, args.max_fpr,
        )

        per_fold_scores.append(scores)
        per_fold_metrics.append({
            "fold": k,
            "ckpt": ckpt_path,
            "per_section": rows,
            "per_domain_overall": agg,
            "overall": overall,
            "harmonic_mean": hmean,
        })

        if not args.skip_per_fold_plots:
            fold_dir = os.path.join(run_root, f"fold{k}")
            save_run_artifacts(
                fold_dir, scores, domains, sections, labels,
                dev_ds.samples, rows, agg, overall, hmean,
                extra={
                    "mode": args.mode, "arch": resolved_arch,
                    "fold": k, "ckpt": ckpt_path,
                    "n_clips": int(len(scores)), "max_fpr": args.max_fpr,
                },
            )

    # --- ensemble: per-clip mean score across folds ---
    print(f"\n=== Ensemble across {len(fold_ckpts)} folds ===")
    ens_scores = np.mean(np.stack(per_fold_scores, axis=0), axis=0)
    ens_rows, ens_agg, ens_overall, ens_hmean = compute_metrics(
        ens_scores, domains_ref, sections_ref, labels_ref, args.max_fpr
    )
    print_report(
        f"phase2/{args.mode}/ENSEMBLE",
        ens_rows, ens_agg, ens_overall, ens_hmean, args.max_fpr,
    )

    ens_dir = os.path.join(run_root, "ensemble")
    save_run_artifacts(
        ens_dir, ens_scores, domains_ref, sections_ref, labels_ref,
        dev_ds.samples, ens_rows, ens_agg, ens_overall, ens_hmean,
        extra={
            "mode": args.mode, "arch": resolved_arch,
            "ensemble_of": [p for p in fold_ckpts],
            "n_folds": len(fold_ckpts),
            "n_clips": int(len(ens_scores)), "max_fpr": args.max_fpr,
        },
    )

    # --- per-fold / ensemble comparison chart ---
    fig_path = os.path.join(run_root, "fold_vs_ensemble.png")
    plot_fold_vs_ensemble(per_fold_metrics, ens_agg, ens_overall, ens_hmean, fig_path)
    print(f"Fold-vs-ensemble chart -> {fig_path}")

    # --- top-level run summary (primary leaderboard row) ---
    run_summary = {
        "phase": 2,
        "mode": args.mode, "arch": resolved_arch,
        "n_folds": len(fold_ckpts),
        "ckpts": fold_ckpts,
        "ensemble": {
            "per_domain_overall": ens_agg,
            "overall": ens_overall,
            "harmonic_mean": ens_hmean,
        },
        "per_fold": [
            {
                "fold": m["fold"],
                "ckpt": m["ckpt"],
                "source_auc": m["per_domain_overall"]["source"]["auc"],
                "target_auc": m["per_domain_overall"]["target"]["auc"],
                "overall_auc": m["overall"]["auc_pooled"],
                "harmonic_mean": m["harmonic_mean"],
            }
            for m in per_fold_metrics
        ],
    }
    with open(os.path.join(run_root, "summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Run summary -> {run_root}/summary.json")


def plot_fold_vs_ensemble(per_fold_metrics, ens_agg, ens_overall, ens_hmean,
                          out_path):
    folds = [m["fold"] for m in per_fold_metrics] + ["ens"]
    src_aucs = [m["per_domain_overall"]["source"]["auc"] for m in per_fold_metrics] \
        + [ens_agg["source"]["auc"]]
    tgt_aucs = [m["per_domain_overall"]["target"]["auc"] for m in per_fold_metrics] \
        + [ens_agg["target"]["auc"]]
    ovr_aucs = [m["overall"]["auc_pooled"] for m in per_fold_metrics] \
        + [ens_overall["auc_pooled"]]
    hmeans = [m["harmonic_mean"] for m in per_fold_metrics] + [ens_hmean]

    x = np.arange(len(folds))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(folds) + 3), 4.2))
    ax.bar(x - 1.5 * w, src_aucs, w, label="source AUC", color="#1f77b4")
    ax.bar(x - 0.5 * w, tgt_aucs, w, label="target AUC", color="#ff7f0e")
    ax.bar(x + 0.5 * w, ovr_aucs, w, label="overall AUC", color="#2ca02c")
    ax.bar(x + 1.5 * w, hmeans,   w, label="DCASE hmean", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in folds])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("fold")
    ax.set_ylabel("score")
    ax.set_title("Per-fold vs ensemble (dev-test, labelled)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
