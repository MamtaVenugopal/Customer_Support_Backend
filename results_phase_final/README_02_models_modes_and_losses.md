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

Per-batch optimization objective:

- `L_total = L_rec + lambda_dom * L_dom + lambda_ctr * L_ctr`
- In this project: `lambda_dom = 0.1` when domain-regularized mode is active, and `lambda_ctr = 0.1` when contrastive mode is active.
- When a term is not active for a mode, its lambda is `0`.

Modes:

- `baseline`: source-only training, optimize only `L_rec`
- `mixed`: source+target batches, optimize only `L_rec`
- `domain_regularized`: source+target batches, optimize  
  `L_total = L_rec + 0.1 * CrossEntropy(domain_clf(z), domain_label)`
- `contrastive`: source+target batches, optimize  
  `L_total = L_rec + 0.1 * contrastive_loss(mean_pool(z), mean_pool(z))`

How to read this:

- **Modes are defined by which extra loss terms are turned on.**
- `mixed` changes data exposure (source+target in training) but does not add an extra auxiliary loss.
- `domain_regularized` and `contrastive` both start from reconstruction learning and then add one regularizer.
- Final checkpoint selection is still based on validation reconstruction behavior through the fold val-loss criterion.

## 5) Cross-validation protocol used

- K-fold stratified split by `(section, domain)`
- For each fold:
  - train on K-1 folds
  - validate on held fold (both source and target clips)
  - save best checkpoint by combined-domain val loss
- Inference uses fold ensemble (mean score across fold checkpoints)

### Interpreting the training logs (your example)

Example lines:

- `val: 1980 source + 21 target = 2001 clips (both drawn from both roots, all normal)`
- `train=3998 (src=3960, tgt=38)  val=2001`
- `[batch-check] first batch: 51 source + 13 target = 64 (target share = 20%)`

What this means:

- **Fold split composition:** one fold's validation subset has 2001 normal clips total, with 1980 source and 21 target.
- **Train subset composition:** the remaining folds contain 3998 clips total, with 3960 source and 38 target.
- **These are dataset-level counts:** they describe the full fold partitions, not one optimizer step.
- **Strong imbalance remains:** even after splitting, source dominates target by roughly 100:1.
- **Batch balancing is intentional:** the sampler builds each mini-batch with about 20% target (`13/64` in this example), which is much higher than the raw dataset ratio.
- **Why this matters:** without this balancing, gradients would be almost entirely source-driven, and target-domain robustness would usually degrade.

In short:

- Fold-level datasets remain naturally imbalanced.
- Batch construction partially corrects imbalance during optimization.
- Loss mode selection determines whether only reconstruction is optimized (`baseline`, `mixed`) or reconstruction plus auxiliary regularization (`domain_regularized`, `contrastive`).

## 6) Clarifying "UNet vs trained UNet"

- `UNet` in code is architecture `unet` (`UNetAE`).
- "Trained UNet" means checkpoint(s) produced after running `train_lean.py --arch unet ...`.
- In this bundle, trained UNet runs are represented by keys like:
  - `mixed_unet`
  - `domain_regularized_unet`
  - `contrastive_unet`
