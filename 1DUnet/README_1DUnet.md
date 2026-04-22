# Bearing Anomaly Detection — 1D Waveform U-Net Autoencoder

Unsupervised anomaly detection on **raw waveform audio** from bearing sensors. A **1D U-Net autoencoder** is trained on **normal-only** clips. At inference time, **mean squared reconstruction error (MSE)** per clip is the anomaly score — clips that are harder to reconstruct score higher and are flagged as anomalous.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Bottleneck](#bottleneck)
5. [Optimizer and Training Config](#optimizer-and-training-config)
6. [Evaluation](#evaluation)
7. [Outputs](#outputs)
8. [File Structure](#file-structure)
9. [Colab Setup](#colab-setup)

---

## Problem Statement

- **Unsupervised**: only normal clips are available at training time.
- **Goal**: learn what "normal" sounds like and flag clips that deviate.
- **Anomaly score**: per-clip mean squared reconstruction error (MSE).  
  Higher MSE → model struggled to reconstruct → likely anomalous.

---

## Dataset

This project uses **DCASE 2022 Task 2** style bearing audio.

### Audio Properties

| Property        | Value                  |
|-----------------|------------------------|
| Modality        | Single-channel waveform |
| Duration        | ~10 seconds per clip   |
| Sample rate     | 16,000 Hz              |
| Fixed length    | 65,536 samples (~4.1s used per clip, zero-padded/trimmed) |

### Data Splits Used

| Split            | Source zip                      | Folder after extraction                        | Labels?         | Used for              |
|------------------|---------------------------------|------------------------------------------------|-----------------|-----------------------|
| **Training**     | `eval_data_bearing_train.zip`   | `data/dcase_bearing_eval/bearing/train/`       | No (normal only)| Train autoencoder     |
| **Validation**   | 10% random split from training  | In-memory subset (not a separate folder)        | No              | Select best checkpoint|
| **Eval test**    | `eval_data_bearing_test.zip`    | `data/dcase_bearing_eval/bearing/test/`        | No (unlabeled)  | Export anomaly scores |
| **Dev test**     | `dev_bearing.zip` (optional)    | `data/dcase_bearing_dev/bearing/test/`         | Yes (filename)  | AUC/pAUC + detection  |

### Dataset Notes

- Training files are **sections 03–05** (eval track); dev test files are **sections 00–02**.
- **Normal vs anomaly** label is encoded in the dev filename:
  - `..._test_normal_...` → label 0
  - `..._test_anomaly_...` → label 1
- `TRAIN_FILE_CAP` in section 2 of the notebook caps number of training files for speed (`None` = use all, ~3000 clips).

---

## Model Architecture

**Class:** `UNet1DAutoencoder` in `1DUnet/model_unet_1d.py`

The model is a symmetric 1D U-Net. It encodes the raw waveform progressively, compresses it into a bottleneck, then decodes back to the original length using skip connections (skips preserve fine-grained waveform detail during reconstruction).

### Building Block: `ConvBlock1D`

Each encoder and decoder stage uses a double-convolution block:

```
Conv1d(kernel=3, pad=1) → BatchNorm1d → ReLU
Conv1d(kernel=3, pad=1) → BatchNorm1d → ReLU
```

No bias in Conv1d (BatchNorm handles offset).

### Full Architecture

```
INPUT (B, 1, 65536)
│
├─ enc1: ConvBlock1D(1 → 32)         [65536 samples]
├─ MaxPool1d(2)
├─ enc2: ConvBlock1D(32 → 64)        [32768 samples]
├─ MaxPool1d(2)
├─ enc3: ConvBlock1D(64 → 128)       [16384 samples]
├─ MaxPool1d(2)
│
├─ BOTTLENECK: ConvBlock1D(128 → 256) [8192 samples]  ← most compressed
│
├─ ConvTranspose1d(256 → 128, stride=2)
├─ cat(skip from enc3) → dec3: ConvBlock1D(256 → 128)
├─ ConvTranspose1d(128 → 64, stride=2)
├─ cat(skip from enc2) → dec2: ConvBlock1D(128 → 64)
├─ ConvTranspose1d(64 → 32, stride=2)
├─ cat(skip from enc1) → dec1: ConvBlock1D(64 → 32)
│
└─ Conv1d(32 → 1, kernel=1)          [65536 samples]
OUTPUT (B, 1, 65536)
```

### Parameter Count Per Layer

| Layer         | Parameters  |
|---------------|-------------|
| enc1          |       3,296 |
| enc2          |      18,688 |
| enc3          |      74,240 |
| **bottleneck**| **295,936** |
| up3 + dec3    |     213,632 |
| up2 + dec2    |      53,568 |
| up1 + dec1    |      13,472 |
| out_conv      |          33 |
| **Total**     | **672,865** |

All 672,865 parameters are trainable.

---

## Bottleneck

The bottleneck is the most compressed representation of the input waveform. It captures only the most essential patterns that the model considers "normal."

| Property             | Value                    |
|----------------------|--------------------------|
| Module               | `ConvBlock1D(128 → 256)` |
| Input channels       | 128                      |
| Output channels      | 256                      |
| Temporal length      | 8,192 samples            |
| Compression factor   | 8× (65536 → 8192)        |
| Bottleneck parameters| 295,936                  |

At test time, an anomalous clip produces a bottleneck representation that does not match normal patterns, leading to poor reconstruction and high MSE.

---

## Optimizer and Training Config

| Parameter         | Value                          |
|-------------------|--------------------------------|
| Optimizer         | Adam                           |
| Learning rate     | 0.001                          |
| Loss function     | Mean Squared Error (MSE)       |
| Epochs            | 20 (full run)                  |
| Batch size        | 32 (GPU) / 16 (CPU)            |
| Validation split  | 10% of training files (random) |
| Seed              | 42                             |
| Workers           | 0 (safe for Colab)             |

### Checkpoint Strategy

- Best model is saved at `results/bearing_unet_waveform_1d.pt` whenever **validation loss improves**.
- Checkpoint stores: `model` state dict, `epoch`, `val_loss`.
- After training, the best checkpoint is automatically reloaded into `model`.

### Waveform Preprocessing

Each clip is:
1. Loaded at 16,000 Hz mono.
2. Zero-padded or trimmed to exactly **65,536 samples**.
3. **Normalized per sample**: zero mean, unit variance.

---

## Evaluation

### Ranking Metrics (AUC / pAUC)

Used when labeled dev clips (`dev_bearing.zip`) are available.

| Metric       | Meaning                                                         |
|--------------|-----------------------------------------------------------------|
| AUC          | Area under ROC curve (1.0 = perfect ranking)                    |
| pAUC @ 0.1   | Partial AUC with FPR ≤ 0.1 (DCASE standard; harder to inflate) |

### Binary Detection (Normal vs Anomaly)

A threshold is computed from dev scores:

```
threshold = (median score of normal clips + median score of anomaly clips) / 2
```

Each clip is then labeled:

| Condition            | Predicted Label |
|----------------------|-----------------|
| MSE score ≥ threshold | **anomaly** (1)  |
| MSE score < threshold | **normal** (0)   |

Evaluation outputs:
- Confusion matrix
- Classification report (precision / recall / F1)
- Top-10 lowest-score clips (confidently normal)
- Top-10 highest-score clips (confidently anomalous)

### Unlabeled Eval Scoring

All clips in `eval_data_bearing_test.zip` are scored and saved to:

```
results/anomaly_scores_bearing_eval_test_1dunet.csv
```

Format: `filename, score` (no header, no labels).

---

## Outputs

| File                                              | Content                              |
|---------------------------------------------------|--------------------------------------|
| `results/bearing_unet_waveform_1d.pt`             | Best checkpoint (model weights)      |
| `results/anomaly_scores_bearing_eval_test_1dunet.csv` | Unlabeled eval scores (name, score)|

These are separate from the 2D mel U-Net outputs (`bearing_unet_mel.pt`, `anomaly_scores_bearing_eval_test.csv`).

---

## File Structure

```
project_root/
│
├── bearing_anomaly_detection_1DUnet.ipynb   ← main notebook
│
├── 1DUnet/                                  ← modular pipeline
│   ├── __init__.py
│   ├── config_1d.py                         ← paths, WaveformConfig, TrainConfig
│   ├── data_description_1d.py               ← dataset narrative
│   ├── data_loading_1d.py                   ← extraction, WaveformDataset, loaders
│   ├── data_visualisation_1d.py             ← waveform plots
│   ├── model_unet_1d.py                     ← UNet1DAutoencoder architecture
│   ├── model_summary_1d.py                  ← parameter counts, torchinfo wrapper
│   ├── training_1d.py                       ← fit(), train_one_epoch(), validate()
│   ├── evaluation_1d.py                     ← anomaly_scores(), AUC, save CSV
│   └── explainability_1d.py                 ← saliency, reconstruction plots
│
├── results/
│   ├── bearing_unet_waveform_1d.pt          ← 1D checkpoint
│   └── anomaly_scores_bearing_eval_test_1dunet.csv
│
├── data/
│   ├── dcase_bearing_eval/bearing/train/    ← normal training wavs (sections 03-05)
│   ├── dcase_bearing_eval/bearing/test/     ← unlabeled eval wavs
│   └── dcase_bearing_dev/bearing/test/      ← labeled dev wavs (sections 00-02)
│
├── requirements.txt                         ← shared with 2D notebook
└── README_1DUnet.md                         ← this file
```

---

## Colab Setup

1. Upload to a single Colab folder:
   - `bearing_anomaly_detection_1DUnet.ipynb`
   - `requirements.txt`
   - `1DUnet/` folder (all `.py` files)
   - `eval_data_bearing_train.zip`
   - `eval_data_bearing_test.zip`
   - `dev_bearing.zip` (optional)

2. Enable GPU: **Runtime → Change runtime type → GPU → Save**.

3. Open `bearing_anomaly_detection_1DUnet.ipynb`.

4. If the working directory is wrong, uncomment in section 0:
   ```python
   os.chdir("/content/YourFolderName")
   ```

5. Run all cells in order. Section 0 installs dependencies automatically.

6. After training, download from `results/`:
   - `bearing_unet_waveform_1d.pt`
   - `anomaly_scores_bearing_eval_test_1dunet.csv`

### Smoke Test (quick CPU check)

In section 2, change:
```python
NUM_EPOCHS = 1
FORCE_CPU  = True
TRAIN_FILE_CAP = 128
```

---

## Reference

- [DCASE 2022 Task 2 — Unsupervised Anomalous Sound Detection](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
