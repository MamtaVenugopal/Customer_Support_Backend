# Phase-2 Results Bundle
_Generated 2026-04-28 19:35_

Total phase-2 runs: **9**

Phase-2 models are trained on the **combined** dev-train (sections 00-02) + eval-train (sections 03-05) training pool with **stratified K-fold cross-validation** over (section × domain) and with source+target clips present in every val fold. Every number below is the **fold ensemble** (anomaly scores averaged across the K fold-best checkpoints) scored on the labelled dev-test split (sections 00-02).
Eval-test (sections 03-05, unlabelled) is held out entirely — no script in this pipeline reads it.

## Recommended model

**`mixed/mobilenet`** — this is the model to ship.

Picked by highest ensemble hmean (0.5904); tie-broken by target AUC (0.6707) and lowest fold-to-fold target-AUC σ (0.0088).

| metric | value |
|---|---|
| ensemble source AUC | 0.6311 |
| ensemble target AUC | 0.6707 |
| ensemble overall AUC | 0.6512 |
| ensemble DCASE hmean | 0.5904 |
| source AUC − target AUC (domain gap) | -0.0396 |
| per-fold target AUC σ | 0.0088 |
| K-fold mean best val loss | 0.0571 (±0.0008) |
| fold count | 3 |

The K fold checkpoints for this model are packaged under `best_model/mixed_mobilenet_fold*.pt` for direct use by a future eval-test submission script.

## Leaderboard (by ensemble DCASE harmonic mean)

| rank | mode | arch | K | src AUC | tgt AUC | ovr AUC | hmean | fold tgt σ | src−tgt gap | mean val loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ← recommended | mixed | mobilenet | 3 | 0.6311 | 0.6707 | 0.6512 | 0.5904 | 0.0088 | -0.0396 | 0.0571 |
| 2 | domain_regularized | mobilenet | 3 | 0.6099 | 0.6485 | 0.6292 | 0.5740 | 0.0069 | -0.0386 | 0.0594 |
| 3 | contrastive | mobilenet | 3 | 0.6077 | 0.6426 | 0.6259 | 0.5690 | 0.0095 | -0.0349 | 0.0581 |
| 4 | mixed | unet | 3 | 0.5993 | 0.6127 | 0.6051 | 0.5690 | 0.0034 | -0.0134 | 0.0004 |
| 5 | mixed | ae | 3 | 0.6146 | 0.5546 | 0.5853 | 0.5641 | 0.0064 | 0.0600 | 0.0517 |
| 6 | contrastive | ae | 3 | 0.6218 | 0.5503 | 0.5869 | 0.5613 | 0.0194 | 0.0715 | 0.0514 |
| 7 | domain_regularized | unet | 3 | 0.5884 | 0.5912 | 0.5892 | 0.5585 | 0.0115 | -0.0029 | 0.0006 |
| 8 | contrastive | unet | 3 | 0.5752 | 0.5534 | 0.5639 | 0.5497 | 0.0146 | 0.0218 | 0.0011 |
| 9 | domain_regularized | ae | 3 | 0.5972 | 0.5335 | 0.5662 | 0.5449 | 0.0109 | 0.0636 | 0.0518 |

## Leaderboard (by ensemble target AUC — DG focus)

| rank | mode | arch | K | src AUC | tgt AUC | ovr AUC | hmean | fold tgt σ | src−tgt gap | mean val loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ← recommended | mixed | mobilenet | 3 | 0.6311 | 0.6707 | 0.6512 | 0.5904 | 0.0088 | -0.0396 | 0.0571 |
| 2 | domain_regularized | mobilenet | 3 | 0.6099 | 0.6485 | 0.6292 | 0.5740 | 0.0069 | -0.0386 | 0.0594 |
| 3 | contrastive | mobilenet | 3 | 0.6077 | 0.6426 | 0.6259 | 0.5690 | 0.0095 | -0.0349 | 0.0581 |
| 4 | mixed | unet | 3 | 0.5993 | 0.6127 | 0.6051 | 0.5690 | 0.0034 | -0.0134 | 0.0004 |
| 5 | domain_regularized | unet | 3 | 0.5884 | 0.5912 | 0.5892 | 0.5585 | 0.0115 | -0.0029 | 0.0006 |
| 6 | mixed | ae | 3 | 0.6146 | 0.5546 | 0.5853 | 0.5641 | 0.0064 | 0.0600 | 0.0517 |
| 7 | contrastive | unet | 3 | 0.5752 | 0.5534 | 0.5639 | 0.5497 | 0.0146 | 0.0218 | 0.0011 |
| 8 | contrastive | ae | 3 | 0.6218 | 0.5503 | 0.5869 | 0.5613 | 0.0194 | 0.0715 | 0.0514 |
| 9 | domain_regularized | ae | 3 | 0.5972 | 0.5335 | 0.5662 | 0.5449 | 0.0109 | 0.0636 | 0.0518 |

## Figures

- `leaderboard.png` — all runs' ensemble AUCs and DCASE hmean (target bars carry ±fold-σ error bars).
- `source_vs_target.png` — ensemble src-vs-tgt AUC scatter; distance below the y=x line = domain gap.
- `kfold_variance.png` — per-run dot-plot of per-fold target AUC with the ensemble shown as a red star.
- `checkpoints/{run}_kfold_overview.png` — one train/val curve per fold for every (mode, arch).
- `eval_results/{run}/ensemble/` — full ensemble plots (ROC, score histograms, per-section AUC, overall summary).
- `eval_results/{run}/fold{k}/` — same plots per fold.

## Scope

- Every number above is computed on the **labelled dev-test split** (sections 00-02, 600 clips).
- **Eval-test (sections 03-05, unlabelled) was NOT touched** by any script in this bundle. The `best_model/` directory contains the K fold checkpoints ready for a future submission script.

## Threshold policy (what is calibrated vs applied)

- Threshold is **calibrated on labelled dev-test** using a precision-recall sweep and F1-max selection (`best_thr`).
- That calibrated threshold is then **applied to eval-test** scores only to produce binary decisions (`decision_result_section_XX.csv`).
- There is **no threshold calibration on eval-test** itself because eval-test is unlabelled.
- Eval-train (sections 03-05 train split) is used as part of model training with dev-train; it is **not used for threshold tuning**.

### How `best_thr` is computed

1. Run the winner fold-ensemble on labelled dev-test and get one anomaly score per clip (`dev_scores`).
2. Convert labels to binary ground truth: anomaly = 1, normal = 0 (`y_true`).
3. Sweep all candidate score cutoffs using `precision_recall_curve(y_true, dev_scores)`.
4. For each candidate threshold, compute:
   `F1 = 2 * precision * recall / (precision + recall)`.
5. Select the threshold index with maximum F1 (`best_i = argmax(F1)`), then set:
   `best_thr = thr[best_i]`.
6. Use this fixed `best_thr` for downstream decision files on eval-test:
   `decision = 1 if score >= best_thr else 0`.

In short: the threshold is chosen on the labelled dev-test set to optimize F1, then reused unchanged on eval-test.

## Reproducing any phase-2 row

```
python prepare_cache_lean.py                                   # dev-train + eval-train + dev-test caches
python train_lean.py    --mode <mode> --arch <arch> --n_folds 5
python evaluate_lean.py --mode <mode> --arch <arch>
python collect_results_lean.py --rank_by hmean
```