# Bearing Anomaly Detection using U-Net Autoencoder

## Overview

This project implements an **unsupervised anomaly detection system** for bearing vibration signals using a **1D U-Net Autoencoder** architecture. The system is designed to identify anomalies in bearing sound waveforms by learning to reconstruct normal bearing signals and detecting deviations (reconstruction errors) as anomalies.

## Project Structure

```
Unsupervised_Anonamlous/
├── dataloader.py              # Audio data loading and preprocessing
├── model.py                   # U-Net autoencoder architecture
├── optimization.py            # 8 loss functions and optimizers
├── main.py                    # Main training script
├── evaluation.py              # Evaluation metrics and plotting
├── train_example.py           # Example training with auto-plotting
├── bearing/                   # Example dataset structure
│   ├── attributes_00.csv      # Bearing attributes metadata
│   ├── train/                 # Training audio files
│   └── test/                  # Test audio files
└── results/                   # Generated plots and reports
    ├── training_history.png
    ├── loss_comparison.png
    └── evaluation_report.txt
```

## Installation

### 1. Create Virtual Environment

```bash
cd /Users/satta/Desktop/Agentic/Unsupervised_Anonamlous
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install torch librosa soundfile pandas numpy matplotlib scikit-learn
```

**Key Dependencies:**
- `torch`: Deep learning framework
- `librosa`: Audio processing
- `soundfile`: Audio I/O
- `matplotlib`: Visualization
- `scikit-learn`: Metrics (ROC, PR, confusion matrix)

## Quick Start

### Basic Training (3 epochs for testing)

```bash
source venv/bin/activate
python train_example.py --bearing_dir /Users/satta/Downloads/bearing --epochs 3
```

### Full Training with Custom Settings

```bash
python main.py \
  --bearing_dir /Users/satta/Downloads/bearing \
  --epochs 50 \
  --batch_size 8 \
  --loss_type hybrid \
  --optimizer_type adam \
  --use_scheduler
```

### Training Options

```
--bearing_dir       Path to bearing dataset (default: ./bearing)
--epochs            Number of training epochs (default: 50)
--batch_size        Batch size for training (default: 16)
--learning_rate     Learning rate (default: 1e-3)
--sr                Audio sampling rate (default: 16000 Hz)
--loss_type         Loss function: mse, mae, hybrid, smooth_l1, huber
--optimizer_type    Optimizer: adam, adamw, sgd
--use_scheduler     Enable learning rate scheduler
--save_dir          Output directory for results (default: ./results)
```

## Project Components

### 1. **dataloader.py** - Data Loading
- Loads audio waveforms from WAV/MP3/FLAC files
- Reads bearing attributes from CSV
- Pads/trims waveforms to fixed length (65536 samples = 4.1 seconds)
- Handles numeric conversion and missing values
- Returns PyTorch DataLoader for training

**Key Classes:**
- `BearingDataset`: Custom PyTorch Dataset
- `create_dataloaders()`: Factory function for train/val loaders

### 2. **model.py** - U-Net Architecture
- 1D U-Net with encoder-decoder structure
- Encoder: 4 levels of downsampling (65536 → 8192 samples)
- Bottleneck: Deep feature extraction
- Decoder: 4 levels of upsampling with skip connections
- Output: Reconstructed waveform (same shape as input)

**Architecture Summary:**
```
Input (1, 65536)
    ↓ Encoder
(512, 8192) - Bottleneck
    ↓ Decoder with skip connections
Output (1, 65536)
```

**Key Classes:**
- `ConvBlock`: Double 1D convolution with BatchNorm
- `UNet1D`: Full U-Net architecture
- `UNetAutoencoder`: Autoencoder wrapper
- `create_model()`: Factory function

### 3. **optimization.py** - Loss Functions & Optimizers

**8 Loss Functions Available:**

| Loss | Formula | Use Case |
|------|---------|----------|
| **MSE** | $(y-\hat{y})^2$ | Baseline, smooth gradients |
| **MAE** | $\|y-\hat{y}\|$ | Robust to outliers |
| **Hybrid** | $\alpha \cdot MSE + \beta \cdot MAE$ | **RECOMMENDED** |
| **Smooth L1** | Quadratic + linear | Best gradient flow |
| **Huber** | Adjustable threshold | Heavy-tailed distributions |
| **Contrastive** | Pair-based loss | Semi-supervised |
| **Triplet** | Anchor/positive/negative | Multi-class faults |
| **Anomaly Detection** | Recon + sparsity | Production use |

**Optimizers:**
- Adam (default, recommended)
- AdamW (with weight decay)
- SGD (classical)

**Learning Rate Schedulers:**
- ReduceLROnPlateau (reduce on plateau)
- CosineAnnealingLR (cosine annealing)

### 4. **main.py** - Training Script

**BearingAnomalyDetector Class:**
- Handles model initialization
- Trains and validates the model
- Tracks metrics with evaluator
- Saves best checkpoint
- Computes anomaly scores via reconstruction error

**Methods:**
- `train()`: Full training loop
- `train_epoch()`: Single epoch training
- `validate()`: Validation on val_loader
- `evaluate_anomaly_detection()`: Anomaly score computation
- `plot_results()`: Generate all evaluation plots

### 5. **evaluation.py** - Metrics & Visualization

**ModelEvaluator Class:**
- Tracks training/validation losses
- Generates publication-quality plots

**Plots Generated:**
1. **training_history.png**: Loss curves with best epoch marker
2. **loss_comparison.png**: Overfitting analysis
3. **reconstruction_errors.png**: Normal vs anomaly error distribution
4. **roc_curve.png**: ROC curve and AUC score
5. **pr_curve.png**: Precision-Recall curve
6. **confusion_matrix.png**: Classification metrics
7. **evaluation_report.txt**: Text summary

### 6. **train_example.py** - Complete Example

Full example showing:
- Data loading
- Model initialization
- Training with 3 epochs
- Automatic plot generation
- Summary printing

## How It Works

### Training Phase
1. **Load Data**: Read bearing audio files and pad to fixed length
2. **Forward Pass**: Input waveform → U-Net → Reconstructed waveform
3. **Loss Calculation**: Compare reconstruction vs original
   - Normal data: Low reconstruction error (model learns it)
   - Anomalous data: High reconstruction error (model hasn't seen it)
4. **Backprop & Update**: Optimizer adjusts weights
5. **Validation**: Test on held-out data

### Inference Phase
1. Load trained model
2. Input test waveform
3. Compute reconstruction error: $\text{MSE} = \frac{1}{n}\sum(y - \hat{y})^2$
4. Compare to threshold:
   - If error < threshold → **Normal bearing** ✓
   - If error > threshold → **Anomalous bearing** ⚠️

## Example Output

```
Loading data...
Loaded attributes: (1200, 3)
Numeric attribute columns: ['d1p', 'd1v']
Found 2999 audio files
Train batches: 375
Val batches: 75

Using device: cpu
Model created with 9,456,385 parameters
Loss function: hybrid
Optimizer: adam

Starting training...
Epoch 1 Batch 10/375 Loss: 0.123456
Epoch 1 Average Loss: 0.089234
Validation Epoch 1 Loss: 0.085678
Saved best model!

...

Generating evaluation plots...
Training history plot saved to ./results/training_history.png
Loss comparison plot saved to ./results/loss_comparison.png
Training summary...
============================================================
TRAINING SUMMARY
============================================================
Total Epochs: 50
Best Train Loss: 0.023456 (Epoch 32)
Best Val Loss: 0.028901 (Epoch 35)
...
```

## Dataset Format

### Expected Structure

```
bearing/
├── attributes_00.csv
├── attributes_01.csv
├── attributes_02.csv
├── train/
│   ├── section_01_source_train_normal_0001.wav
│   ├── section_01_source_train_normal_0002.wav
│   └── ...
└── test/
    ├── section_01_source_test_anomaly_0001.wav
    └── ...
```

### Attributes CSV Format

```
file_name,d1p,d1v
bearing/test/section_00_source_test_anomaly_001.wav,vel,6
bearing/test/section_00_source_test_anomaly_002.wav,vel,6
...
```

Only **numeric columns** are used. Non-numeric columns are automatically filtered.

## Loss Functions Explained

### Recommended: **Hybrid Loss**
$$L = 0.5 \cdot \text{MSE}(y, \hat{y}) + 0.5 \cdot \text{MAE}(y, \hat{y})$$

**Why?**
- Combines smooth gradient of MSE with robustness of MAE
- Better for real-world noisy bearing data
- Doesn't over-penalize large errors

**Usage:**
```bash
python main.py --loss_type hybrid
```

### Advanced: **Anomaly Detection Loss**
$$L = L_{\text{recon}} + \lambda \cdot L_{\text{sparsity}}$$

Encourages sparse bottleneck representations for better anomaly detection.

## Model Architecture Details

### U-Net Benefits for Audio
1. **Skip Connections**: Preserve high-frequency information
2. **Multi-Scale**: Encoder-decoder captures both local and global patterns
3. **Symmetric**: Input and output same shape (ideal for anomaly detection)
4. **Parameter Efficiency**: ~9M parameters (manageable on CPU)

### Audio Preprocessing
- **Length**: 65536 samples = 4.096 seconds at 16kHz
- **Padding**: Zero-pad shorter files
- **Trimming**: Truncate longer files
- **Normalization**: Float32 in [0, 1]

## Troubleshooting

### Issue: "No audio files found"
**Solution**: Check bearing_dir path and ensure audio files are .wav, .mp3, or .flac

### Issue: "TypeError: can't convert np.ndarray"
**Solution**: Attributes are being converted to numeric automatically. Non-numeric columns are filtered.

### Issue: Shape mismatch errors
**Solution**: Fixed length (65536) ensures all data has same shape. Update in dataloader.py if needed.

### Issue: Out of memory
**Solution**: Reduce batch_size or audio length:
```bash
python main.py --batch_size 4
```

## Performance Metrics

**Training Convergence** (on bearing data):
- Epoch 1: Loss ~0.15
- Epoch 10: Loss ~0.08
- Epoch 50: Loss ~0.02-0.03

**Evaluation Metrics** (computed automatically):
- **ROC-AUC**: Measures discrimination between normal/anomaly
- **PR-AUC**: Precision-Recall area under curve
- **Reconstruction Error**: MSE of test samples

## Next Steps

1. **Run Training**: Start with test script
   ```bash
   python train_example.py --bearing_dir /Users/satta/Downloads/bearing --epochs 3
   ```

2. **Examine Plots**: Check `./results/` folder for:
   - Training curves
   - Loss comparison
   - Reconstruction error distribution

3. **Fine-tune**: Modify hyperparameters
   ```bash
   python main.py \
     --epochs 100 \
     --learning_rate 5e-4 \
     --loss_type smooth_l1 \
     --use_scheduler
   ```

4. **Deploy**: Load best model for inference
   ```python
   from main import BearingAnomalyDetector
   detector = BearingAnomalyDetector(config)
   detector.load_model('best_model.pth')
   anomaly_score = detector.compute_anomaly_score(waveform)
   ```

## References

- **U-Net**: Ronneberger et al., 2015 (https://arxiv.org/abs/1505.04597)
- **Anomaly Detection**: Goldstein & Uchida, 2016
- **Audio Processing**: librosa documentation (https://librosa.org)

## Author Notes

- **Status**: Working with 1D U-Net for bearing vibration analysis
- **Framework**: PyTorch 2.0+
- **Python**: 3.12
- **Last Updated**: April 6, 2026

## License

This project is for educational/research purposes.

---

**Quick Command Reference:**

```bash
# Activate environment
source venv/bin/activate

# Install dependencies (one time)
pip install torch librosa soundfile pandas numpy matplotlib scikit-learn

# Basic test run (3 epochs)
python train_example.py --bearing_dir /Users/satta/Downloads/bearing --epochs 3

# Full training
python main.py --bearing_dir /Users/satta/Downloads/bearing --epochs 50 --loss_type hybrid

# With learning rate scheduler
python main.py --epochs 100 --use_scheduler --learning_rate 5e-4

# Different loss functions
python main.py --loss_type mse      # Baseline
python main.py --loss_type mae      # Robust
python main.py --loss_type hybrid   # Recommended
python main.py --loss_type smooth_l1  # Smooth gradients
```
