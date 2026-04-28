# README 01 - Problem Statement and Dataset Story

## 1) Problem statement (DCASE 2022 Task 2)

This project targets **unsupervised anomalous sound detection (ASD)** for machine condition monitoring under **domain generalization** constraints:

- Train mostly on normal data.
- Handle domain shift between train and test.
- Use one threshold across mixed test domains when domain is unknown at inference.

Reference: [DCASE 2022 Task 2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#dataset-overview)

## 2) Source vs target domain in this task

Per DCASE definition:

- **Source domain**: the dominant operating condition (most training clips).
- **Target domain**: shifted operating condition (few training clips).
- Domain shift comes from speed/load/noise/mic position/etc.
- In evaluation test, domain labels are hidden, so inference must be domain-agnostic.

## 3) Why the data is unbalanced

From DCASE setup (per section):

- Source normal train clips: **990**
- Target normal train clips: **10**

That is roughly **99:1 imbalance** by design.

In this bundle's combined bearing training pool (as logged by the training pipeline), the ratio is approximately:

- Source: **5940**
- Target: **59**

So target is scarce and noisier for optimization. This is exactly why mode/loss design and balanced sampling matter.

## 4) Dataset structure used in this project

This project focuses on **Bearing** machine type with the standard split pattern:

- Development train: sections `00, 01, 02`
- Additional/eval train: sections `03, 04, 05`
- Development test (labeled): sections `00, 01, 02`
- Evaluation test (unlabeled): sections `03, 04, 05`

### File naming conventions used by the code

- Train/dev-test style (contains domain and label):  
  `section_XX_{source|target}_{train|test}_{normal|anomaly}_....wav`
- Eval-test style (unlabeled, no domain token):  
  `section_03_0000.wav`, `section_04_0123.wav`, etc.

## 5) How this bundle uses those splits

- `train_lean.py` uses combined train roots (`dev train + eval train`) with K-fold CV.
- `evaluate_lean.py` scores on **labeled dev-test** for model selection.
- Phase-3 inference scores **unlabeled eval-test** and exports submission-style CSVs.

## 6) Domain-generalization implication

Because target training data is tiny and eval-test has hidden domain labels:

- The model must generalize from mostly-source normal data.
- Thresholding should be robust and ideally not rely on hidden eval statistics.
- You should interpret target-domain performance and fold variance as key robustness signals.
