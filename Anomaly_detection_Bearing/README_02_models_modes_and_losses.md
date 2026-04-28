# README 02 - Model Architectures, Parameters, and Training Modes

This document maps model code to the training/evaluation pipeline in this bundle.

## 1) Architectures implemented

From `model.py`, available backbones:

- `ae` -> `SimpleAE`
- `unet` -> `UNetAE`
- `mobilenet` -> `MobileNetAE`
- `unet_mobilenet_encoder` -> `UNetMobileNetEncoderAE`

All models expose `forward(x) -> (recon, z)` and are trained using reconstruction loss, with optional regularizers depending on mode.

## 2) Parameter counts

Computed from code-instantiated models:

| Model | Parameters |
|---|---:|
| `SimpleAE` (`ae`) | 6,929 |
| `UNetAE` (`unet`) | 467,233 |
| `MobileNetAE` (`mobilenet`) | 682,385 |
| `UNetMobileNetEncoderAE` (`unet_mobilenet_encoder`) | 703,601 |

Notes:

- `pretrained=True` vs `pretrained=False` does not change parameter count, only initialization.
- The auxiliary domain head (`DomainClassifier`) has tiny parameter cost:
  - embed_dim 32 -> 66
  - embed_dim 96 -> 194
  - embed_dim 128 -> 258

## 3) How data is fed into models

Input to model:

- Tensor shape `(B, 1, H, W)` where `H x W` is log-mel representation.

Model output:

- `recon`: reconstruction tensor
- `z`: bottleneck feature map

Scoring:

- Clip anomaly score = mean squared reconstruction error
- Ensemble score = average of fold scores

## 4) Training modes and losses (`train_lean.py`)

Let reconstruction loss be:

- `L_rec = MSE(recon, x)`

Modes:

- `baseline`: source-only training, optimize `L_rec`
- `mixed`: mixed source+target batches, optimize `L_rec`
- `domain_regularized`: mixed batches, optimize  
  `L = L_rec + 0.1 * CrossEntropy(domain_clf(z), domain_label)`
- `contrastive`: mixed batches, optimize  
  `L = L_rec + 0.1 * contrastive_loss(mean_pool(z), mean_pool(z))`

## 5) Cross-validation protocol used

- K-fold stratified split by `(section, domain)`
- For each fold:
  - train on K-1 folds
  - validate on held fold (both source and target clips)
  - save best checkpoint by combined-domain val loss
- Inference uses fold ensemble (mean score across fold checkpoints)

## 6) Clarifying "UNet vs trained UNet"

- `UNet` in code is architecture `unet` (`UNetAE`).
- "Trained UNet" means checkpoint(s) produced after running `train_lean.py --arch unet ...`.
- In this bundle, trained UNet runs are represented by keys like:
  - `mixed_unet`
  - `domain_regularized_unet`
  - `contrastive_unet`
