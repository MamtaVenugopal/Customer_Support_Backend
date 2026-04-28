# README 03 - Results, Overfitting Check, Model Comparison, and Thresholding

This document is grounded on artifacts in:

- `results_phase_final/comparison.csv`
- `results_phase_final/README_final_phase.md`
- `phase_final_outputs/threshold_metrics.json`
- `predict_outputs/predict_summary.json`
- training histories in `checkpoints_phase2/*_history.json`

## 1) What plots to review

Global comparison plots:

- `results_phase_final/leaderboard.png`
- `results_phase_final/source_vs_target.png`
- `results_phase_final/kfold_variance.png`

Per-run eval plots:

- `results_phase_final/eval_results/<mode_arch>/ensemble/overall_summary.png`
- `results_phase_final/eval_results/<mode_arch>/ensemble/roc.png`
- `results_phase_final/eval_results/<mode_arch>/ensemble/score_histograms.png`
- `results_phase_final/eval_results/<mode_arch>/ensemble/per_section_auc.png`
- `results_phase_final/eval_results/<mode_arch>/fold_vs_ensemble.png`

Train-vs-val trajectories:

- `results_phase_final/checkpoints/<mode_arch>_fold*_history.png`
- `results_phase_final/checkpoints/<mode_arch>_kfold_overview.png`

## 2) Best model selection summary

From `comparison.csv`, winner is:

- **`mixed/mobilenet`**
- Ensemble metrics:
  - source AUC: **0.6311**
  - target AUC: **0.6707**
  - overall AUC: **0.6512**
  - DCASE hmean: **0.5904**
  - fold target-AUC std: **0.0088**

Selection rationale in bundle:

- Highest hmean, then tie-break on target AUC, then lower fold variance.

## 3) Winner vs UNet ("trained UNet" comparison)

Using the trained `mixed_unet` run from `comparison.csv`:

- `mixed_unet`:
  - source AUC: 0.5993
  - target AUC: 0.6127
  - overall AUC: 0.6051
  - hmean: 0.5690
- `mixed_mobilenet` (winner):
  - source AUC: 0.6311
  - target AUC: 0.6707
  - overall AUC: 0.6512
  - hmean: 0.5904

Interpretation:

- Winner exceeds trained UNet on source/target/overall AUC and hmean.
- Biggest practical gain is on target domain.

## 4) Overfitting check (train vs val)

### Winner (`mixed_mobilenet`)

From fold histories:

- Fold0 train loss: 0.3088 -> 0.0575, val loss: 0.1903 -> 0.0570
- Fold1 train loss: 0.3301 -> 0.0568, val loss: 0.1942 -> 0.0562
- Fold2 train loss: 0.3760 -> 0.0562, val loss: 0.2008 -> 0.0582

Pattern:

- Train and val both improve steadily.
- Small oscillations mid-training, but no severe divergence.
- Mild late-stage overfitting risk in some epochs, controlled by best-epoch checkpointing.

### Trained UNet (`mixed_unet`)

Example fold0:

- train loss: 0.0358 -> 0.000402
- val loss reaches best at epoch 13: 0.000403, then worsens at epoch 14: 0.000608

Pattern:

- Very fast fit with low reconstruction val loss.
- Some late-epoch rebound suggests overfit tendency.
- Despite tiny reconstruction losses, anomaly ranking metrics are below winner.

Key takeaway:

- Lowest reconstruction val loss does not necessarily imply best anomaly AUC/hmean under domain shift.

## 5) Threshold calibration and confusion matrix

From `phase_final_outputs/threshold_metrics.json` (calibrated on labeled dev-test):

- threshold: **0.0543546490**
- precision: **0.5615**
- recall: **0.9433**
- f1: **0.7040**
- roc_auc: **0.6512**
- confusion matrix:
  - TN = 79
  - FP = 221
  - FN = 17
  - TP = 283

Interpretation of FP/FN:

- **High FP (221)**: model is conservative toward misses and flags many normals as anomalies.
- **Low FN (17)**: most anomalies are caught (high recall), good for safety-critical settings.

## 6) Important caveat for unlabeled eval-test

For `predict_outputs/` run:

- clips scored: 600
- sections: 03/04/05
- threshold in this run: `None` (scores only)

Because eval-test is unlabeled:

- true confusion matrix **cannot** be computed there.
- only score statistics and predicted anomaly rate (if threshold applied) are available.

So confusion-matrix discussion must rely on labeled dev-test calibration unless official labels are released.
