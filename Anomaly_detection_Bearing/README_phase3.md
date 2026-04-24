# README — Phase 3 (Thresholding + Submission)

This phase starts after Phase 2 picks a winner under `results_phase2/best_model/`.

- Winner used here: `adversarial/mobilenet` (`adversarial_mobilenet`, 3 folds).
- Goal: choose a domain-agnostic threshold, keep false negatives (FN) very low, and export eval-test submission CSVs.

## 1) Why FN matters most for bearings

For bearing health monitoring, FN is usually the most expensive error:

- FN = anomaly predicted as normal -> run-to-failure risk, unplanned downtime, and secondary equipment damage.
- FP = normal predicted as anomaly -> maintenance inefficiency and inspection overhead.

So thresholding policy is usually "high recall first, then improve precision as much as possible without letting FN rise too much."

## 2) Current operating point (domain-agnostic)

Using dev-test labels (`normal/anomaly`) while **ignoring source/target domain** and selecting threshold by max F1:

- Threshold: `0.0573349856`
- Precision: `0.5521`
- Recall: `0.9533`
- F1: `0.6993`
- Accuracy: `0.5900`
- ROC-AUC: `0.6285`
- PR-AUC: `0.5973`
- Confusion matrix: `TN=68, FP=232, FN=14, TP=286`

Interpretation:

- This is a high-sensitivity threshold (very low FN, high recall).
- Cost is a large number of FP alerts.

## 3) Raw scores of the 14 false negatives

The 14 FN anomaly clips are close to the threshold. Score range:

- min FN score: `0.0476765`
- max FN score: `0.0561990`
- threshold: `0.0573350`

Meaning: many missed anomalies are only slightly below the current threshold; pushing threshold upward increases FN quickly.

Files (score, margin = threshold - score):

1. `section_00_target_test_anomaly_0043_vel_4.wav` -> `0.0476765` (margin `0.0096584`)
2. `section_01_target_test_anomaly_0004_vel_4_loc_E.wav` -> `0.0494166` (margin `0.0079184`)
3. `section_01_target_test_anomaly_0017_vel_4_loc_E.wav` -> `0.0494790` (margin `0.0078559`)
4. `section_01_target_test_anomaly_0028_vel_4_loc_E.wav` -> `0.0495809` (margin `0.0077541`)
5. `section_01_source_test_anomaly_0037_vel_4_loc_A.wav` -> `0.0502045` (margin `0.0071305`)
6. `section_00_target_test_anomaly_0023_vel_4.wav` -> `0.0513653` (margin `0.0059696`)
7. `section_01_source_test_anomaly_0008_vel_4_loc_B.wav` -> `0.0526026` (margin `0.0047324`)
8. `section_01_source_test_anomaly_0030_vel_4_loc_A.wav` -> `0.0530014` (margin `0.0043336`)
9. `section_01_source_test_anomaly_0024_vel_4_loc_B.wav` -> `0.0550337` (margin `0.0023013`)
10. `section_01_source_test_anomaly_0036_vel_4_loc_B.wav` -> `0.0552954` (margin `0.0020396`)
11. `section_01_source_test_anomaly_0012_vel_4_loc_A.wav` -> `0.0554515` (margin `0.0018835`)
12. `section_02_source_test_anomaly_0007_vel_14_f-n_B.wav` -> `0.0554522` (margin `0.0018828`)
13. `section_01_target_test_anomaly_0032_vel_4_loc_E.wav` -> `0.0560489` (margin `0.0012861`)
14. `section_00_target_test_anomaly_0018_vel_8.wav` -> `0.0561990` (margin `0.0011360`)

## 4) Threshold adjustment trade-off (keeping FN low)

Measured on dev-test labels:

| threshold | precision | recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|
| 0.0573 (current) | 0.5521 | 0.9533 | 0.6993 | 232 | 14 |
| 0.0580 | 0.5505 | 0.9267 | 0.6907 | 227 | 22 |
| 0.0585 | 0.5467 | 0.8967 | 0.6793 | 223 | 31 |
| 0.0590 | 0.5498 | 0.8833 | 0.6777 | 217 | 35 |
| 0.0595 | 0.5551 | 0.8733 | 0.6788 | 210 | 38 |
| 0.0600 | 0.5558 | 0.8633 | 0.6762 | 207 | 41 |
| 0.0610 | 0.5558 | 0.8300 | 0.6658 | 199 | 51 |

Key finding:

- In this model, raising threshold reduces FP only slowly but increases FN quickly.
- If the objective is "FN must stay near 14", the current threshold is already near-optimal.

## 5) Practical policy recommendation

Use two thresholds (triage) rather than one:

- `T_low = 0.0573` -> high sensitivity alarm (catch nearly all anomalies).
- `T_high` (e.g. `0.061`) -> urgent alarm.
- Scores between `[T_low, T_high)` -> "inspect soon" queue.

This keeps FN low while reducing maintenance noise by prioritization, not by sacrificing recall.

## 6) Artifacts produced for this analysis

- `phase3_threshold_analysis.json`: detailed threshold study, FN clip list, and winner metadata.
- `pahse3_notebook.ipynb`: Colab notebook to reproduce threshold calibration, eval-test CSV export, and explainability plots.

## 7) Explainability outputs (written by `pahse3_notebook.ipynb`)

The notebook writes an `explain/` folder under `phase3_outputs/` with:

- `explain_melbin_mean_error.png` — which mel frequency bins carry the most reconstruction error for normals vs anomalies (domain pooled).
- `explain_pca_latent_label.png`, `explain_pca_latent_section.png`, `explain_pca_latent_domain.png` — 2D PCA of pooled bottleneck vectors `z` (ensemble-mean across folds), colored by label / section / domain.
- `explain_examples/*.png` — a few side-by-side panels: input log-mel, ensemble reconstruction, and squared error map.

Important: inference scoring is still **reconstruction MSE only**. The `adversarial` training term shapes the encoder indirectly; the domain classifier head is **not** used at scoring time.

