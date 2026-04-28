"""
[LEAN ENTRYPOINT] Canonical script for the consolidated workflow.

Phase-2 training with K-fold cross-validation on the combined training
pool (dev-train sections 00-02  +  eval-train sections 03-05).

Why K-fold?
-----------
Phase 1 held out a single fixed 5% slice of source clips as validation.
With only ~150 val clips from one 2999-clip pool, val-loss estimates were
noisy and the "best epoch" signal depended on one arbitrary split.

K-fold addresses both issues AND gives the val set a target-domain
signal the phase-1 val never had:

 - Source clips (5940) AND target clips (59) are each independently
   partitioned into K disjoint folds.
 - For each fold k ∈ {0, ..., K-1}:
       train on source AND target clips in the OTHER (K-1) folds
       validate on (source_fold[k]) ∪ (target_fold[k])
         => val contains both domains, all clips are normal
       save the best-val checkpoint as `{mode}_{arch}_fold{k}.pt`
 - Best-epoch selection is driven by the combined-domain val loss;
   per-domain val losses are logged separately for transparency.
 - Final model used at evaluation time is the ensemble of the K
   fold-best checkpoints (anomaly scores are averaged across folds).

Eval-test (sections 03-05, unlabelled) is NEVER loaded by this script.
It is reserved for the final prediction pass after model selection.
"""

import argparse
import json
import math
import os
import random
import time
from typing import List, Sequence, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import parse_filename
from dataset_phase2 import CombinedBearingDataset
from model import ARCHES, DomainClassifier, build_model, contrastive_loss
from sampler import BalancedDomainSampler

MODE_ALIASES = {
    "adversarial": "domain_regularized",
}
MODE_CHOICES = ["baseline", "mixed", "domain_regularized", "contrastive"]


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #

def build_train_loader(args, dataset):
    common = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    if args.mode == "baseline":
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            drop_last=True, **common,
        )
    sampler = BalancedDomainSampler(
        dataset, batch_size=args.batch_size, target_ratio=args.target_ratio,
        verbose=False,
    )
    return DataLoader(dataset, batch_sampler=sampler, **common)


def build_val_loader(args, val_dataset):
    return DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def list_files_by_domain(
    data_dirs: Sequence[str], domain: str,
) -> List[str]:
    """Return every WAV filename of the given domain ('source' / 'target')
    across all training roots."""
    assert domain in ("source", "target")
    out: List[str] = []
    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        out.extend(
            f for f in os.listdir(d)
            if f.endswith(".wav") and domain in f
        )
    return sorted(set(out))


def make_stratified_kfold_splits(
    files: Sequence[str],
    n_folds: int,
    seed: int,
    strata_fn,
) -> List[Set[str]]:
    """Stratified round-robin K-fold split.

    Groups `files` into buckets keyed by `strata_fn(filename)`, shuffles
    every bucket independently (seeded so the split is reproducible),
    then round-robin-assigns the items of each bucket to the K folds.
    The result is that every fold ends up with roughly the same number
    of items from every stratum, which is exactly what we want when the
    natural strata (here: section × domain) are small.

    Returns a list of `n_folds` disjoint sets whose union is `files`.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(files) < n_folds:
        raise ValueError(
            f"Cannot build {n_folds} folds from only {len(files)} files."
        )

    buckets: dict = {}
    for f in files:
        buckets.setdefault(strata_fn(f), []).append(f)

    folds: List[Set[str]] = [set() for _ in range(n_folds)]
    for key in sorted(buckets.keys(), key=lambda k: str(k)):
        items = sorted(buckets[key])  # deterministic pre-shuffle order
        # Per-bucket offset so the "first" fold to receive an item
        # varies from bucket to bucket (otherwise small buckets would
        # systematically pile up in fold 0).
        rng = random.Random(hash((seed, str(key))) & 0xFFFFFFFF)
        rng.shuffle(items)
        offset = rng.randrange(n_folds)
        for i, f in enumerate(items):
            folds[(i + offset) % n_folds].add(f)
    return folds


def _section_domain_stratum(fname: str):
    """Stratum key used for stratified K-fold: (section, domain)."""
    domain, section, _label = parse_filename(fname)
    return (section, domain)


@torch.no_grad()
def eval_val_loss(model, loader, device) -> dict:
    """Return per-domain and combined reconstruction MSE for the fold
    val loader (which contains both source and target clips)."""
    model.eval()
    totals = {"source": 0.0, "target": 0.0, "all": 0.0}
    counts = {"source": 0, "target": 0, "all": 0}

    for batch in loader:
        x, domain, section = batch[:3]
        x = x.to(device, non_blocking=True)
        recon, _ = model(x)
        m = min(recon.shape[-1], x.shape[-1])
        per_clip = ((recon[..., :m] - x[..., :m]) ** 2).mean(dim=[1, 2, 3])
        per_clip_cpu = per_clip.detach().cpu().tolist()

        for err, dom in zip(per_clip_cpu, domain):
            key = dom if dom in ("source", "target") else "source"
            totals[key] += err
            counts[key] += 1
            totals["all"] += err
            counts["all"] += 1

    model.train()
    return {
        "val_loss":     totals["all"]    / max(counts["all"], 1),
        "val_loss_src": totals["source"] / max(counts["source"], 1),
        "val_loss_tgt": totals["target"] / max(counts["target"], 1),
        "n_val_src":    counts["source"],
        "n_val_tgt":    counts["target"],
    }


def save_ckpt(path, mode, arch, fold_idx, n_folds, model, domain_clf,
              args, extra=None):
    payload = {
        "phase": 2,
        "mode": mode,
        "arch": arch,
        "fold": fold_idx,
        "n_folds": n_folds,
        "embed_dim": model.embed_dim,
        "model_state": model.state_dict(),
        "domain_clf_state": domain_clf.state_dict(),
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def plot_fold_history(history, out_path, title):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history], "-o",
            label="train", color="#1f77b4")
    ax.plot(epochs, [h["val_loss"] for h in history], "-o",
            label="val (all)", color="#d62728")
    if all("val_loss_src" in h for h in history):
        ax.plot(epochs, [h["val_loss_src"] for h in history], "--s",
                label="val (source)", color="#2ca02c", markersize=4)
        ax.plot(epochs, [h["val_loss_tgt"] for h in history], "--^",
                label="val (target)", color="#ff7f0e", markersize=4)
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    best_val = min(h["val_loss"] for h in history)
    ax.axvline(best_epoch, color="k", ls="--", alpha=0.4,
               label=f"best @ epoch {best_epoch} (val={best_val:.4f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_kfold_overview(per_fold_history, out_path, title):
    """One panel per fold, shared y-axis."""
    k = len(per_fold_history)
    fig, axes = plt.subplots(1, k, figsize=(3.8 * k, 3.8), sharey=True)
    if k == 1:
        axes = [axes]
    for i, (hist, ax) in enumerate(zip(per_fold_history, axes)):
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["train_loss"] for h in hist], "-o",
                label="train", color="#1f77b4", markersize=3)
        ax.plot(epochs, [h["val_loss"] for h in hist], "-o",
                label="val", color="#d62728", markersize=3)
        best = min(hist, key=lambda h: h["val_loss"])
        ax.axvline(best["epoch"], color="k", ls="--", alpha=0.3)
        ax.set_title(f"fold {i}\n(best val={best['val_loss']:.4f} @ ep {best['epoch']})",
                     fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("loss")
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# single-fold trainer
# --------------------------------------------------------------------------- #

def train_one_fold(
    args, fold_idx, val_src_files, val_tgt_files,
    data_dirs, cache_dirs, device,
) -> dict:
    print(f"\n========= Fold {fold_idx + 1}/{args.n_folds} =========")
    val_files = set(val_src_files) | set(val_tgt_files)
    print(f"  val: {len(val_src_files)} source + {len(val_tgt_files)} target "
          f"= {len(val_files)} clips (both drawn from both roots, all normal)")

    if args.mode == "baseline":
        train_ds = CombinedBearingDataset(
            root_dirs=data_dirs, cache_dirs=cache_dirs,
            domain_filter="source", exclude_files=val_files, verbose=False,
        )
    else:
        train_ds = CombinedBearingDataset(
            root_dirs=data_dirs, cache_dirs=cache_dirs,
            exclude_files=val_files, verbose=False,
        )
    val_ds = CombinedBearingDataset(
        root_dirs=data_dirs, cache_dirs=cache_dirs,
        include_only_files=val_files, verbose=False,
    )
    train_src = sum(1 for s in train_ds.samples if s["domain"] == "source")
    train_tgt = sum(1 for s in train_ds.samples if s["domain"] == "target")
    print(f"  train={len(train_ds)} (src={train_src}, tgt={train_tgt})  "
          f"val={len(val_ds)}")

    train_loader = build_train_loader(args, train_ds)
    val_loader = build_val_loader(args, val_ds)

    model = build_model(args.arch).to(device)
    domain_clf = DomainClassifier(model.embed_dim).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(domain_clf.parameters()), lr=args.lr
    )
    criterion = nn.MSELoss()

    best_path = os.path.join(
        args.ckpt_dir, f"{args.mode}_{args.arch}_fold{fold_idx}.pt"
    )
    history_json = os.path.join(
        args.ckpt_dir, f"{args.mode}_{args.arch}_fold{fold_idx}_history.json"
    )
    history_png = os.path.join(
        args.ckpt_dir, f"{args.mode}_{args.arch}_fold{fold_idx}_history.png"
    )

    best_val = math.inf
    best_epoch = -1
    history: List[dict] = []
    first_batch_logged = False

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        steps = 0
        for x, domain, section in train_loader:
            if not first_batch_logged:
                n_src = sum(1 for d in domain if d == "source")
                n_tgt = sum(1 for d in domain if d == "target")
                print(f"  [batch-check] first batch: {n_src} source + "
                      f"{n_tgt} target = {len(domain)} "
                      f"(target share = {n_tgt / len(domain):.0%})")
                first_batch_logged = True

            x = x.to(device, non_blocking=True)
            recon, z = model(x)
            min_t = min(recon.shape[-1], x.shape[-1])
            recon = recon[..., :min_t]
            x = x[..., :min_t]
            loss = criterion(recon, x)

            if args.mode == "domain_regularized":
                labels = torch.tensor(
                    [0 if d == "source" else 1 for d in domain], device=device
                )
                loss = loss + 0.1 * nn.CrossEntropyLoss()(domain_clf(z), labels)
            if args.mode == "contrastive":
                z_flat = z.mean(dim=[2, 3])
                loss = loss + 0.1 * contrastive_loss(z_flat, z_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1

        train_loss = running / max(steps, 1)
        val = eval_val_loss(model, val_loader, device)
        dt = time.time() - t0

        tag = ""
        if val["val_loss"] < best_val:
            best_val = val["val_loss"]
            best_epoch = epoch
            save_ckpt(best_path, args.mode, args.arch, fold_idx, args.n_folds,
                      model, domain_clf, args,
                      extra={"epoch": epoch,
                             "val_loss":     val["val_loss"],
                             "val_loss_src": val["val_loss_src"],
                             "val_loss_tgt": val["val_loss_tgt"],
                             "train_loss":   train_loss,
                             "val_src_files": sorted(val_src_files),
                             "val_tgt_files": sorted(val_tgt_files)})
            tag = "  <- new best, saved"

        print(f"  fold{fold_idx} epoch {epoch}  "
              f"train={train_loss:.4f}  "
              f"val={val['val_loss']:.4f} "
              f"(src={val['val_loss_src']:.4f}, "
              f"tgt={val['val_loss_tgt']:.4f})  "
              f"steps={steps}  time={dt:.1f}s{tag}")
        history.append({
            "epoch": epoch,
            "train_loss":   train_loss,
            "val_loss":     val["val_loss"],
            "val_loss_src": val["val_loss_src"],
            "val_loss_tgt": val["val_loss_tgt"],
        })

    with open(history_json, "w") as f:
        json.dump({
            "phase": 2, "mode": args.mode, "arch": args.arch,
            "fold": fold_idx, "n_folds": args.n_folds,
            "best_epoch": best_epoch, "best_val_loss": best_val,
            "history": history,
        }, f, indent=2)
    plot_fold_history(
        history, history_png,
        title=f"[phase2] {args.mode}/{args.arch} fold {fold_idx} "
              f"(best val={best_val:.4f} @ epoch {best_epoch})",
    )

    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "ckpt": best_path,
        "history": history,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="mixed",
                   choices=MODE_CHOICES + list(MODE_ALIASES.keys()),
                   help="Training mode. 'adversarial' is kept as a backward-compatible alias "
                        "for 'domain_regularized'.")
    p.add_argument("--arch", type=str, default="ae",
                   choices=list(ARCHES.keys()),
                   help="Backbone: ae (tiny), unet, or mobilenet (pretrained)")

    p.add_argument("--dev_train_dir", type=str,
                   default=os.environ.get("BEARING_DEV_TRAIN_DIR",
                                          "data/dcase_bearing_dev/bearing/train"))
    p.add_argument("--eval_train_dir", type=str,
                   default=os.environ.get("BEARING_EVAL_TRAIN_DIR",
                                          "data/dcase_bearing_eval/bearing/train"))

    p.add_argument("--dev_train_cache", type=str,
                   default=os.environ.get("BEARING_DEV_TRAIN_CACHE", ""))
    p.add_argument("--eval_train_cache", type=str,
                   default=os.environ.get("BEARING_EVAL_TRAIN_CACHE", ""))

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--target_ratio", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--n_folds", type=int, default=5,
                   help="Number of K-fold CV folds (K>=2). Each fold is a "
                        "disjoint source-clip val set.")
    p.add_argument("--folds_to_run", type=str, default="",
                   help="Comma-separated list of fold indices to run, e.g. "
                        "'0,3'. Empty = run all folds.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--ckpt_dir", type=str,
                   default=os.environ.get("BEARING_PHASE2_CKPT_DIR",
                                          "checkpoints_phase2"))
    args = p.parse_args()
    args.mode = normalize_mode(args.mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    data_dirs = [args.dev_train_dir, args.eval_train_dir]
    cache_dirs = [args.dev_train_cache or None, args.eval_train_cache or None]

    print("=== Phase-2 K-fold training ===")
    for d in data_dirs:
        exists = "ok" if os.path.isdir(d) else "MISSING"
        print(f"  train root: {d}  [{exists}]")

    # --- build STRATIFIED K-fold splits over (section, domain) ---
    # Source and target are split separately so that per-fold domain
    # composition matches the global 5940:59 source:target ratio; each
    # is additionally stratified by SECTION so every fold sees roughly
    # the same number of clips from every (section, domain) bucket.
    source_files = list_files_by_domain(data_dirs, "source")
    target_files = list_files_by_domain(data_dirs, "target")
    if not source_files:
        raise RuntimeError(
            f"No source WAVs found under {data_dirs}. "
            f"Run prepare_cache_lean.py (or stage data) first."
        )
    if not target_files:
        raise RuntimeError(
            f"No target WAVs found under {data_dirs}. "
            f"Run prepare_cache_lean.py (or stage data) first."
        )
    if len(target_files) < args.n_folds:
        raise ValueError(
            f"Only {len(target_files)} target clips; need at least "
            f"{args.n_folds} for K={args.n_folds}-fold CV. "
            f"Reduce --n_folds or add more target clips."
        )
    source_folds = make_stratified_kfold_splits(
        source_files, n_folds=args.n_folds, seed=args.seed,
        strata_fn=_section_domain_stratum,
    )
    target_folds = make_stratified_kfold_splits(
        target_files, n_folds=args.n_folds, seed=args.seed + 1,
        strata_fn=_section_domain_stratum,
    )
    print(
        f"\nStratified K-fold config: K={args.n_folds}  "
        f"total_source={len(source_files)}  total_target={len(target_files)}\n"
        f"  strata = (section, domain)  [12 buckets: 6 sections × 2 domains]\n"
        f"  source fold sizes: {[len(f) for f in source_folds]}\n"
        f"  target fold sizes: {[len(f) for f in target_folds]}\n"
        f"  val set per fold  = (source fold) ∪ (target fold), both domains"
    )

    # --- dump per-(section, domain) counts per fold so we can verify
    # stratification worked (and for the run's log / README) ---
    def _bucket_counts(fold_sets):
        matrix: List[dict] = []
        for fs in fold_sets:
            counts: dict = {}
            for f in fs:
                dom, sec, _ = parse_filename(f)
                counts[(sec, dom)] = counts.get((sec, dom), 0) + 1
            matrix.append(counts)
        return matrix

    src_matrix = _bucket_counts(source_folds)
    tgt_matrix = _bucket_counts(target_folds)
    all_keys = sorted({k for m in src_matrix + tgt_matrix for k in m})
    print("\n  per-fold (section, domain) composition:")
    header = "    " + " ".join(f"f{i:>3}" for i in range(args.n_folds))
    print("    section/domain    " + header[4:])
    for sec, dom in all_keys:
        row = " ".join(
            f"{(src_matrix if dom == 'source' else tgt_matrix)[i].get((sec, dom), 0):>4d}"
            for i in range(args.n_folds)
        )
        print(f"    sec{sec:02d} {dom:<6}    {row}")

    # Optional subset for iterative debugging.
    if args.folds_to_run:
        indices = [int(x) for x in args.folds_to_run.split(",") if x.strip()]
    else:
        indices = list(range(args.n_folds))
    print(f"Will run folds: {indices}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # --- announce composition of the full (pre-split) training pool ---
    peek = CombinedBearingDataset(
        root_dirs=data_dirs, cache_dirs=cache_dirs,
        verbose=False,
    )
    by_section: dict = {}
    for s in peek.samples:
        by_section[(s["section"], s["domain"])] = \
            by_section.get((s["section"], s["domain"]), 0) + 1
    print("\nFull training pool composition (before K-fold split):")
    for key in sorted(by_section):
        print(f"  section_{key[0]:02d} {key[1]:>6} = {by_section[key]}")

    # --- run selected folds ---
    fold_results: List[dict] = []
    for k in indices:
        result = train_one_fold(
            args, fold_idx=k,
            val_src_files=source_folds[k],
            val_tgt_files=target_folds[k],
            data_dirs=data_dirs, cache_dirs=cache_dirs, device=device,
        )
        fold_results.append(result)

    # --- kfold summary ---
    summary_path = os.path.join(
        args.ckpt_dir, f"{args.mode}_{args.arch}_kfold_summary.json"
    )
    summary_png = os.path.join(
        args.ckpt_dir, f"{args.mode}_{args.arch}_kfold_overview.png"
    )
    best_vals = [r["best_val_loss"] for r in fold_results]
    mean_best = sum(best_vals) / len(best_vals)
    std_best = (
        (sum((v - mean_best) ** 2 for v in best_vals) / len(best_vals)) ** 0.5
    )
    best_fold_idx = min(fold_results, key=lambda r: r["best_val_loss"])["fold"]

    with open(summary_path, "w") as f:
        json.dump({
            "phase": 2,
            "mode": args.mode, "arch": args.arch,
            "n_folds": args.n_folds,
            "folds_run": indices,
            "mean_best_val_loss": mean_best,
            "std_best_val_loss": std_best,
            "best_fold": best_fold_idx,
            "fold_results": [
                {k: v for k, v in r.items() if k != "history"}
                for r in fold_results
            ],
        }, f, indent=2)

    plot_kfold_overview(
        [r["history"] for r in fold_results], summary_png,
        title=f"[phase2] {args.mode}/{args.arch} — K={args.n_folds}-fold CV\n"
              f"mean best val = {mean_best:.4f}  (±{std_best:.4f})  "
              f"| best fold = {best_fold_idx}",
    )

    print(
        f"\nK-fold summary:\n"
        f"  mean best val loss = {mean_best:.4f}  (std = {std_best:.4f})\n"
        f"  best fold          = {best_fold_idx}\n"
        f"  per-fold best val  = "
        + ", ".join(f"{r['fold']}:{r['best_val_loss']:.4f}" for r in fold_results)
        + f"\n  summary JSON -> {summary_path}"
        + f"\n  overview PNG -> {summary_png}"
    )


if __name__ == "__main__":
    main()
