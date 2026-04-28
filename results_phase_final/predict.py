"""
Standalone inference script for winner-model deployment.

Loads a phase-2 winner ensemble from:
  <model_dir>/best_model/WINNER.json
  <model_dir>/best_model/{key}_fold*.pt

Scores an unlabeled eval-style folder (section_XX_YYYY.wav) with
reconstruction MSE and writes:
  - anomaly_score_section_XX.csv
  - decision_result_section_XX.csv (if threshold provided/found)
  - predictions_all.csv
  - predict_summary.json
"""

import argparse
import glob
import json
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_phase2 import BearingTestDataset
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing best_model/WINNER.json (for example /content/results_phase_final).",
    )
    p.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder with unlabeled eval-style WAV files (section_XX_YYYY.wav).",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional mel-cache directory for input_dir. Default: auto-derived by BearingTestDataset.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="predict_outputs",
        help="Output folder for prediction CSV/JSON files.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Inference batch size.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed threshold. If omitted, script tries "
             "<model_dir>/threshold_metrics.json; if still missing, only scores are written.",
    )
    return p.parse_args()


def load_winner_ckpts(model_dir: str):
    winner_path = os.path.join(model_dir, "best_model", "WINNER.json")
    if not os.path.exists(winner_path):
        raise FileNotFoundError(f"Missing {winner_path}")
    with open(winner_path) as f:
        winner = json.load(f)

    key = winner["key"]
    arch = winner["arch"]
    ckpts = sorted(glob.glob(os.path.join(model_dir, "best_model", f"{key}_fold*.pt")))
    if not ckpts:
        raise FileNotFoundError(
            f"No fold checkpoints found for key={key} under "
            f"{os.path.join(model_dir, 'best_model')}"
        )
    return winner, arch, ckpts


def resolve_threshold(args_threshold: Optional[float], model_dir: str) -> Optional[float]:
    if args_threshold is not None:
        return float(args_threshold)
    auto = os.path.join(model_dir, "threshold_metrics.json")
    if os.path.exists(auto):
        with open(auto) as f:
            payload = json.load(f)
        return float(payload["metrics"]["threshold"])
    return None


@torch.no_grad()
def score_ensemble(loader, arch: str, ckpts: List[str], device: str):
    models = []
    for p in ckpts:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        m = build_model(arch).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)

    all_scores = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        per_fold = []
        for m in models:
            recon, _ = m(x)
            t = min(recon.shape[-1], x.shape[-1])
            err = ((recon[..., :t] - x[..., :t]) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            per_fold.append(err)
        ens = np.mean(np.stack(per_fold, axis=0), axis=0)
        all_scores.append(ens)
    return np.concatenate(all_scores)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    winner, arch, ckpts = load_winner_ckpts(args.model_dir)
    threshold = resolve_threshold(args.threshold, args.model_dir)

    ds = BearingTestDataset(
        args.input_dir,
        cache_dir=(args.cache_dir or None),
        verbose=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    scores = score_ensemble(loader, arch=arch, ckpts=ckpts, device=device)
    decisions = (scores >= threshold).astype(int) if threshold is not None else None

    rows = []
    for i, s in enumerate(ds.samples):
        row = {
            "filename": s["filename"],
            "section": int(s["section"]),
            "score": float(scores[i]),
        }
        if decisions is not None:
            row["decision"] = int(decisions[i])
        rows.append(row)

    all_csv = os.path.join(args.out_dir, "predictions_all.csv")
    with open(all_csv, "w") as f:
        if decisions is None:
            f.write("filename,section,score\n")
            for r in rows:
                f.write(f"{r['filename']},{r['section']},{r['score']:.10f}\n")
        else:
            f.write("filename,section,score,decision\n")
            for r in rows:
                f.write(
                    f"{r['filename']},{r['section']},{r['score']:.10f},{r['decision']}\n"
                )

    sections = sorted({r["section"] for r in rows})
    for sec in sections:
        sec_rows = sorted((r for r in rows if r["section"] == sec), key=lambda x: x["filename"])

        score_path = os.path.join(args.out_dir, f"anomaly_score_section_{sec:02d}.csv")
        with open(score_path, "w") as f:
            for r in sec_rows:
                f.write(f"{r['filename']},{r['score']:.10f}\n")

        if decisions is not None:
            dec_path = os.path.join(args.out_dir, f"decision_result_section_{sec:02d}.csv")
            with open(dec_path, "w") as f:
                for r in sec_rows:
                    f.write(f"{r['filename']},{r['decision']}\n")

    summary = {
        "winner": winner,
        "device": device,
        "model_dir": args.model_dir,
        "input_dir": args.input_dir,
        "n_folds": len(ckpts),
        "n_clips": len(rows),
        "sections": sections,
        "threshold": threshold,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }
    if decisions is not None:
        summary["predicted_anomaly_rate"] = float(np.mean(decisions))

    summary_path = os.path.join(args.out_dir, "predict_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Prediction complete ===")
    print(f"winner: {winner['mode']}/{winner['arch']}  key={winner['key']}")
    print(f"fold checkpoints: {len(ckpts)}")
    print(f"clips scored: {len(rows)}")
    print(f"threshold: {threshold if threshold is not None else 'None (scores only)'}")
    print(f"outputs: {args.out_dir}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
