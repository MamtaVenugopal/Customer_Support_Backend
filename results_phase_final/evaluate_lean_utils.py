"""
[LEAN UTILS] Shared evaluation helpers for evaluate_lean.py
"""



import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from dataset import BearingDataset
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


DOMAINS = ["source", "target"]
SECTIONS_DEFAULT = [0, 1, 2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="baseline",
                   choices=MODE_CHOICES + list(MODE_ALIASES.keys()),
                   help="Evaluation mode key. 'adversarial' is kept as a backward-compatible "
                        "alias for 'domain_regularized'.")
    p.add_argument("--arch", type=str, default="ae",
                   choices=list(ARCHES.keys()),
                   help="Backbone to look up. Ignored if --ckpt is explicit "
                        "or if the checkpoint contains an 'arch' field.")
    p.add_argument("--ckpt", type=str, default="",
                   help="Path to .pt checkpoint. "
                        "Defaults to checkpoints/{mode}_{arch}.pt (the BEST by val loss). "
                        "Pass the *_last.pt path explicitly to evaluate the final epoch instead.")
    p.add_argument("--use_last", action="store_true",
                   help="Use {mode}_{arch}_last.pt instead of the best checkpoint.")
    p.add_argument("--test_dir", type=str,
                   default=os.environ.get("BEARING_TEST_DIR",
                                          "data/dcase_bearing_dev/bearing/test"))
    p.add_argument("--cache_dir", type=str,
                   default=os.environ.get("BEARING_TEST_CACHE_DIR", ""))
    p.add_argument("--out_dir", type=str, default="eval_results")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_fpr", type=float, default=0.1,
                   help="FPR ceiling for pAUC (DCASE uses 0.1)")
    return p.parse_args()


def score_dataset(model, loader, device):
    """Return per-clip reconstruction MSE + metadata for every test clip."""
    model.eval()
    scores, domains, sections, labels = [], [], [], []
    with torch.no_grad():
        for x, dom, sec, lab in loader:
            x = x.to(device, non_blocking=True)
            recon, _ = model(x)
            m = min(recon.shape[-1], x.shape[-1])
            err = (recon[..., :m] - x[..., :m]) ** 2
            per_clip = err.mean(dim=[1, 2, 3]).cpu().numpy()

            scores.append(per_clip)
            domains.extend(list(dom))
            sections.extend([int(s) for s in sec])
            labels.extend(list(lab))
    scores = np.concatenate(scores)
    return scores, np.array(domains), np.array(sections), np.array(labels)


def _auc_pauc(y_true, y_score, max_fpr):
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    auc = roc_auc_score(y_true, y_score)
    pauc = roc_auc_score(y_true, y_score, max_fpr=max_fpr)
    return auc, pauc


def compute_metrics(scores, domains, sections, labels, max_fpr):
    y_true = (labels == "anomaly").astype(int)

    rows = []
    per_domain_auc = {d: [] for d in DOMAINS}
    per_domain_pauc = {d: [] for d in DOMAINS}

    for sec in sorted(set(sections.tolist())):
        for dom in DOMAINS:
            mask = (sections == sec) & (domains == dom)
            if mask.sum() == 0:
                continue
            auc, pauc = _auc_pauc(y_true[mask], scores[mask], max_fpr)
            rows.append({
                "section": sec, "domain": dom, "n": int(mask.sum()),
                "auc": auc, "pauc": pauc,
            })
            per_domain_auc[dom].append(auc)
            per_domain_pauc[dom].append(pauc)

    # Aggregated (all sections) per domain
    agg = {}
    for dom in DOMAINS:
        mask = domains == dom
        auc, pauc = _auc_pauc(y_true[mask], scores[mask], max_fpr)
        agg[dom] = {
            "auc": auc,
            "pauc": pauc,
            "mean_section_auc": float(np.nanmean(per_domain_auc[dom])) if per_domain_auc[dom] else float("nan"),
            "mean_section_pauc": float(np.nanmean(per_domain_pauc[dom])) if per_domain_pauc[dom] else float("nan"),
            "n": int((domains == dom).sum()),
        }

    # Overall (pool ALL sections AND domains together — the "global" metric)
    overall_auc, overall_pauc = _auc_pauc(y_true, scores, max_fpr)
    all_section_aucs  = [r["auc"]  for r in rows]
    all_section_paucs = [r["pauc"] for r in rows]
    overall = {
        "n": int(len(scores)),
        "auc_pooled":  overall_auc,          # AUC on all 600 clips at once
        "pauc_pooled": overall_pauc,
        "mean_section_domain_auc":  float(np.nanmean(all_section_aucs))  if all_section_aucs  else float("nan"),
        "mean_section_domain_pauc": float(np.nanmean(all_section_paucs)) if all_section_paucs else float("nan"),
    }

    # DCASE-style harmonic mean of (mean-section source AUC, mean-section target AUC, mean-section pAUC)
    vals = [agg["source"]["mean_section_auc"],
            agg["target"]["mean_section_auc"],
            float(np.nanmean(per_domain_pauc["source"] + per_domain_pauc["target"]))]
    vals = [v for v in vals if not np.isnan(v) and v > 0]
    hmean = len(vals) / sum(1.0 / v for v in vals) if vals else float("nan")

    return rows, agg, overall, hmean


def print_report(mode, rows, agg, overall, hmean, max_fpr):
    print(f"\n=== Evaluation report [{mode}] ===")
    print(f"{'section':>8} {'domain':>8} {'n':>5} {'AUC':>8} {'pAUC@'+str(max_fpr):>10}")
    for r in rows:
        print(f"{r['section']:>8} {r['domain']:>8} {r['n']:>5} "
              f"{r['auc']:>8.4f} {r['pauc']:>10.4f}")

    print("-" * 60)
    print("  Per-domain (pooled across sections):")
    for dom in DOMAINS:
        a = agg[dom]
        print(f"    {dom:>6} [n={a['n']:>3}]: AUC={a['auc']:.4f}  pAUC={a['pauc']:.4f}  "
              f"mean-section AUC={a['mean_section_auc']:.4f}")

    print("-" * 60)
    print("  Overall (pooled across ALL sections AND domains):")
    print(f"    n={overall['n']}  pooled AUC={overall['auc_pooled']:.4f}  "
          f"pooled pAUC={overall['pauc_pooled']:.4f}")
    print(f"    mean of the 6 (section x domain) AUCs  = {overall['mean_section_domain_auc']:.4f}")
    print(f"    mean of the 6 (section x domain) pAUCs = {overall['mean_section_domain_pauc']:.4f}")
    print(f"    DCASE harmonic mean (src AUC, tgt AUC, pAUC) = {hmean:.4f}")


def plot_score_histograms(scores, domains, labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        mask = domains == dom
        s_norm = scores[mask & (labels == "normal")]
        s_anom = scores[mask & (labels == "anomaly")]
        bins = np.linspace(scores.min(), scores.max(), 40)
        ax.hist(s_norm, bins=bins, alpha=0.6, label=f"normal (n={len(s_norm)})", color="#1f77b4")
        ax.hist(s_anom, bins=bins, alpha=0.6, label=f"anomaly (n={len(s_anom)})", color="#d62728")
        ax.set_title(f"{dom} — reconstruction MSE")
        ax.set_xlabel("anomaly score")
        ax.legend()
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_roc(scores, domains, labels, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    y_true = (labels == "anomaly").astype(int)

    # Per-domain ROCs
    for dom, color in zip(DOMAINS, ["#1f77b4", "#ff7f0e"]):
        mask = domains == dom
        if len(np.unique(y_true[mask])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[mask], scores[mask])
        auc = roc_auc_score(y_true[mask], scores[mask])
        ax.plot(fpr, tpr, color=color, label=f"{dom} (AUC={auc:.3f})")

    # Overall ROC (all clips pooled)
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color="#2ca02c", linewidth=2.2,
                label=f"overall (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — source vs target vs overall")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_overall_summary(agg, overall, hmean, out_path):
    """One-glance bar chart: per-domain vs overall AUC/pAUC."""
    groups = ["source", "target", "overall"]
    aucs  = [agg["source"]["auc"],  agg["target"]["auc"],  overall["auc_pooled"]]
    paucs = [agg["source"]["pauc"], agg["target"]["pauc"], overall["pauc_pooled"]]

    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w/2, aucs,  w, label="AUC",        color="#1f77b4")
    b2 = ax.bar(x + w/2, paucs, w, label="pAUC@0.1",   color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax.set_ylabel("score")
    ax.set_title(f"Overall evaluation summary  (DCASE hmean = {hmean:.3f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_height() + 0.01,
                    f"{rect.get_height():.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_section_bars(rows, out_path):
    sections = sorted({r["section"] for r in rows})
    x = np.arange(len(sections))
    w = 0.35
    src_auc = [next((r["auc"] for r in rows if r["section"] == s and r["domain"] == "source"), np.nan) for s in sections]
    tgt_auc = [next((r["auc"] for r in rows if r["section"] == s and r["domain"] == "target"), np.nan) for s in sections]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, src_auc, w, label="source", color="#1f77b4")
    ax.bar(x + w/2, tgt_auc, w, label="target", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"section_{s:02d}" for s in sections])
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
    ax.set_ylabel("AUC")
    ax.set_title("Per-section AUC by domain")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(src_auc):
        ax.text(x[i] - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(tgt_auc):
        ax.text(x[i] + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    args.mode = normalize_mode(args.mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = os.environ.get("BEARING_CKPT_DIR", "checkpoints")
    suffix = "_last" if args.use_last else ""
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = ""
        for mode_name in mode_variants(args.mode):
            candidate = os.path.join(ckpt_dir, f"{mode_name}_{args.arch}{suffix}.pt")
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
        # Legacy fallback: old checkpoints were saved as just {mode}.pt
        if not ckpt_path:
            for mode_name in mode_variants(args.mode):
                legacy = os.path.join(ckpt_dir, f"{mode_name}.pt")
                if os.path.exists(legacy):
                    ckpt_path = legacy
                    break
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. "
            f"Run `train.py --mode {args.mode} --arch {args.arch}` first."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", args.arch)

    out_dir = os.path.join(args.out_dir, f"{args.mode}_{arch}")
    os.makedirs(out_dir, exist_ok=True)

    dataset = BearingDataset(
        args.test_dir, cache_dir=(args.cache_dir or None), return_label=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(arch).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {ckpt_path}  (arch={arch}, mode={args.mode})")

    scores, domains, sections, labels = score_dataset(model, loader, device)
    rows, agg, overall, hmean = compute_metrics(
        scores, domains, sections, labels, args.max_fpr
    )
    print_report(args.mode, rows, agg, overall, hmean, args.max_fpr)

    # ---- save per-clip CSV ----
    csv_path = os.path.join(out_dir, "per_clip_scores.csv")
    with open(csv_path, "w") as f:
        f.write("filename,section,domain,label,score\n")
        for i, s in enumerate(dataset.samples):
            f.write(f"{s['filename']},{s['section']},{s['domain']},"
                    f"{s['label']},{scores[i]:.6f}\n")

    # ---- save summary JSON ----
    summary = {
        "mode": args.mode,
        "arch": arch,
        "ckpt": ckpt_path,
        "n_clips": int(len(scores)),
        "max_fpr": args.max_fpr,
        "per_section": rows,
        "per_domain_overall": agg,
        "overall": overall,
        "harmonic_mean": hmean,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots ----
    plot_score_histograms(scores, domains, labels,
                          os.path.join(out_dir, "score_histograms.png"))
    plot_roc(scores, domains, labels, os.path.join(out_dir, "roc.png"))
    plot_section_bars(rows, os.path.join(out_dir, "per_section_auc.png"))
    plot_overall_summary(agg, overall, hmean,
                         os.path.join(out_dir, "overall_summary.png"))

    print(f"\nSaved results to {out_dir}/")
    for fn in ["per_clip_scores.csv", "summary.json",
               "score_histograms.png", "roc.png",
               "per_section_auc.png", "overall_summary.png"]:
        print(f"  - {fn}")


if __name__ == "__main__":
    main()
