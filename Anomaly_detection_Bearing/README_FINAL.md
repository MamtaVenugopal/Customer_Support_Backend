# README FINAL - Discussion Pack Index

This folder now includes a discussion-oriented README set for your review/presentation.

## Files created

1. `README_01_problem_and_dataset.md`  
   - DCASE problem statement context
   - source vs target domain explanation
   - imbalance story (99:1 style and observed pool counts)
   - dataset structure used by this project

2. `README_02_models_modes_and_losses.md`  
   - architecture inventory from `model.py`
   - parameter counts for each model
   - data flow into model and anomaly scoring
   - training modes (`baseline`, `mixed`, `domain_regularized`, `contrastive`) and losses

3. `README_03_results_overfitting_and_selection.md`  
   - where to look for all plots
   - model comparison and winner selection logic
   - winner vs trained UNet comparison
   - overfitting analysis from train-vs-val history
   - threshold and confusion matrix interpretation
   - limitation on unlabeled eval-test confusion matrix

## Core final conclusions

- Best model in this bundle: **`mixed/mobilenet`** (3-fold ensemble).
- It beats trained UNet (`mixed_unet`) on source/target/overall AUC and hmean.
- Calibrated threshold (on labeled dev-test): **0.0543546490**.
- Dev-test confusion matrix at that threshold:
  - TN=79, FP=221, FN=17, TP=283
- For unlabeled eval-test, only scores/decisions can be exported; true confusion matrix is not available without labels.

## Optional next enhancement

If you want one single report instead of three readmes, we can merge all content into
`results_phase_final/README_final_phase.md` and include direct links to each plot file used.
