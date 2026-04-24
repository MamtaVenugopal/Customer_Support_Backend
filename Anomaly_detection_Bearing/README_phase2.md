# README — Phase 2 (Domain Generalisation with K-fold CV)

This document describes **every piece of data used in phase 2**, how the
training/validation split is done, and which split is deliberately kept
out of reach until the very last step.

---

## 1. Goal

Train on all 6 machine sections with domain-generalisation-aware
validation, then ship the single best `(mode, arch)` model.

| what | phase 2 |
|---|---|
| training data | `dev_train` **+** `eval_train` (sections 00-05) |
| train/val split | **stratified K-fold CV** (K=5) over `(section × domain)` for both source and target clips |
| per-fold val set | `(source_fold[k]) ∪ (target_fold[k])` — both domains, all normal |
| final model | **K-fold ensemble** (per-clip anomaly score averaged across the K fold-best checkpoints) |
| evaluation | `dev_test` only (sections 00-02, labelled) |
| architectures swept | **3 backbones** — `ae` (SimpleAE, tiny), `unet` (UNetAE, 2-level with skips), `mobilenet` (ImageNet-pretrained MobileNetV2) |
| DG modes swept | **4 modes** — `baseline / mixed / adversarial / contrastive` (see phase-1 README for the DG loss definitions) |
| full matrix | **4 modes × 3 backbones = 12 runs × K=5 folds = 60 fold checkpoints** |
| model selection | automatic — `collect_results_phase2.py` ranks **every** `(mode, arch)` across the full 12-run matrix by ensemble DCASE hmean and stages the winner under `results_phase2/best_model/` |

Phase-1 outputs are left entirely alone — phase-2 is independently
benchmarkable from the report in `results_phase2/RESULTS_phase2.md`.

---

## 2. Data used in phase 2

All counts assume the standard DCASE 2023 Task 2 bearing release.

### 2.1 Training pool — K-fold source split + full target

| source folder | section | source clips | target clips | total |
|---|---|---|---|---|
| `data/dcase_bearing_dev/bearing/train`  | 00 | 990 | 10 | 1000 |
| `data/dcase_bearing_dev/bearing/train`  | 01 | 990 | 9*  | 999  |
| `data/dcase_bearing_dev/bearing/train`  | 02 | 990 | 10 | 1000 |
| `data/dcase_bearing_eval/bearing/train` | 03 | 990 | 10 | 1000 |
| `data/dcase_bearing_eval/bearing/train` | 04 | 990 | 10 | 1000 |
| `data/dcase_bearing_eval/bearing/train` | 05 | 990 | 10 | 1000 |
| **total** | | **5940** | **59** | **5999** |

*(\*) section 01 is one clip short on the target side — a known DCASE
peculiarity inherited from phase 1 (documented in the main `README.md`).*

Both folders are loaded together into a single
`CombinedBearingDataset` (`dataset_phase2.py`) whose samples carry a
`(section, domain, label, filename, path, cache)` record.

### 2.2 Evaluation — `dev_test` only (labelled)

| folder | sections | clips | labelled? |
|---|---|---|---|
| `data/dcase_bearing_dev/bearing/test` | 00-02 | 600 | yes (normal / anomaly, source / target) |

Each section contributes 100 clips per domain: 50 normal + 50 anomaly.
This is the **only** split on which we compute AUC / pAUC / DCASE hmean
in phase 2, which makes phase-1 vs phase-2 numbers directly comparable.

### 2.3 Held out — `eval_test` (UNLABELLED, NOT TOUCHED)

| folder | sections | clips | labelled? | touched in phase 2? |
|---|---|---|---|---|
| `data/dcase_bearing_eval/bearing/test` | 03-05 | 600 | **no — submission-only** | **NO** |

By policy:

- `prepare_cache_phase2.py` does **not** build `.npy` files for this split.
- `train_phase2.py` does **not** reference this split.
- `evaluate_phase2.py` does **not** score this split.
- `collect_results_phase2.py` does **not** include any column, CSV, or
  plot derived from this split.
- The notebook `phase2_notebook.ipynb` does **not** stage or copy its
  zip/WAVs to Colab local disk and explicitly unsets
  `BEARING_EVAL_TEST_DIR` / `BEARING_EVAL_TEST_CACHE` environment
  variables if a previous run set them.

A separate submission script (not part of this repository yet) will be
written to score `eval_test` **only once** a best `(mode, arch)` has
been chosen from the phase-2 leaderboard. Until then the 600 unlabelled
clips stay on disk (or in the zip) untouched.

---

## 3. Stratified K-fold cross-validation strategy

`train_phase2.py`:

1. Gather every **source**-domain WAV filename across both training
   roots → 5940 files, and every **target**-domain filename →
   59 files.
2. Split each list independently into **K=5 stratified folds** where
   the strata are `(section, domain)` — so for source we have 6
   buckets (one per section × `source`) and for target we have 6
   buckets (one per section × `target`). Within each bucket we
   deterministically shuffle (seeded from `(args.seed, bucket_key)`)
   and round-robin-assign clips to the K folds, with a per-bucket
   offset so small buckets do not pile up in fold 0. The resulting
   folds are disjoint and sum to the original file list.

   **Stratification guarantee**: with 990 source clips per section and
   K=5 folds, every fold gets **exactly 198 source clips per section**
   (6 sections × 198 = 1188 clips/fold). For targets with only ~10
   clips/section, each fold gets 1-2 target clips per section instead
   of "maybe 0, maybe 4" that plain random K-fold would produce.
   Empirical fold sizes:
   - **source folds**: [1188, 1188, 1188, 1188, 1188] (exactly 198 per section)
   - **target folds**: ≈ [11, 12, 12, 12, 12] (59 total, section 01 is one clip short)

   A per-`(section, domain)` count matrix is printed at start-up so
   you can confirm the stratification worked.
3. For each fold `k ∈ {0, …, K-1}`:
   - **Train** on source AND target clips in the **other** K-1
     folds. In `baseline` mode training stays source-only; in
     `mixed / adversarial / contrastive` modes the sampler pulls
     target clips from the training partition (≈ 47 available) in
     every batch.
   - **Validate** on `(source_fold[k]) ∪ (target_fold[k])` —
     **both domains present**, all clips normal. This is the key
     upgrade over phase 1: val now sees ≈ 12 target clips per fold
     so the best-epoch signal reflects target reconstruction too,
     not just source.
   - Run `--epochs` epochs (default 20). The epoch with lowest
     **combined-domain** val MSE is saved as
     `checkpoints_phase2/{mode}_{arch}_fold{k}.pt`. Per-domain
     losses (`val_loss_src`, `val_loss_tgt`) are logged each epoch
     for transparency and are stored in the history JSON.
   - Training curves are saved to
     `{mode}_{arch}_fold{k}_history.{json,png}` — the PNG now shows
     four traces (train / val-all / val-source / val-target).
4. A **K-fold summary** is written to
   `checkpoints_phase2/{mode}_{arch}_kfold_summary.json`:

   ```json
   {
     "mean_best_val_loss": 0.0812,
     "std_best_val_loss":  0.0034,
     "best_fold": 3,
     "fold_results": [...]
   }
   ```

5. A grid of train/val curves (one panel per fold) is saved as
   `checkpoints_phase2/{mode}_{arch}_kfold_overview.png`.

### Why stratified K-fold (and why mixed-domain val)?

- **Section-balanced val fold.** Plain random K-fold can hand some
  folds 0 target clips from section `X` and another fold 4. With
  stratification by `(section, domain)` every fold is guaranteed to
  see every section — the per-fold variance you observe later comes
  from the model, not from the draw.
- **Lower-variance val signal.** Every source and target clip ends
  up in exactly one val fold, so the mean best-val-loss across folds
  is a much smoother estimate than any single holdout could give.
- **Target-aware model selection.** The phase-1 convention of
  validating on source-only clips meant "best" was whatever
  reconstructed source best; the target domain never influenced it.
  Including ~12 target val clips per fold makes the combined val
  loss penalise any checkpoint that reconstructs target clips
  poorly — which is exactly the quantity DG cares about.
- **Built-in ensemble at inference time.** Averaging per-clip
  anomaly scores from K independently-trained models reduces
  prediction variance for free (see §5).
- **No label leakage.** Train/val splits partition **normal** source
  and target clips only; anomaly clips never appear in any training
  or validation set — they exist exclusively in the `dev_test` split
  and are only seen at evaluation time.

### Note on target-fold size

Each target val fold has ≈ 12 clips, so the per-fold target val loss
is noisy on its own. The combined val loss (≈ 1200 total clips per
fold, ~99% source) is stable, and the K-fold mean of the target val
loss is a reasonable estimate of target reconstruction quality across
≈ 59 target clips in total.

### Flags exposed by `train_phase2.py` (quick reference)

| flag | default | meaning |
|---|---|---|
| `--mode` | `mixed` | DG training mode — one of `baseline / mixed / adversarial / contrastive` (see §3b) |
| `--arch` | `ae` | autoencoder backbone — one of `ae / unet / mobilenet` (see §3b) |
| `--epochs` | `20` | max epochs per fold (no early-stopping; best-val checkpoint is kept) |
| `--batch_size` | `64` | mini-batch size used during training and val |
| `--lr` | `1e-3` | Adam learning rate for the AE + domain-classifier |
| `--target_ratio` | `0.2` | fraction of target clips per batch when `mode ≠ baseline` (sampler parameter) |
| `--n_folds` | `5` | K — number of stratified folds to build |
| `--folds_to_run` | `""` (all) | comma-separated subset, e.g. `0,3` — handy for quick iteration |
| `--seed` | `42` | controls fold assignment, per-bucket shuffle, and fold offsets |
| `--num_workers` | `2` | DataLoader worker count |
| `--ckpt_dir` | `$BEARING_PHASE2_CKPT_DIR`, else `checkpoints_phase2` | where fold `.pt`/`history.*` files are written |
| `--dev_train_dir`, `--eval_train_dir` | env-var or `data/…` | training roots; both are used at once |
| `--dev_train_cache`, `--eval_train_cache` | env-var or `""` | per-root mel `.npy` cache dirs |

Section §3b below has the deep dive on what each mode / backbone / sampler parameter does mechanically — what code path it activates, what loss term it adds, and what it is (and is not) guaranteed to achieve.

---

## 3b. What each parameter actually does

This section expands every `train_phase2.py` flag that materially changes model behaviour, with the exact code paths each one activates. Use this as the authoritative reference when reading the leaderboard.

### 3b.1 `--mode` — the DG training recipe

Every mode optimises the autoencoder reconstruction MSE. Modes differ in (a) which clips go into a batch and (b) what extra term is added to the loss. Nothing else branches on `--mode`.

| mode | batch composition | extra loss term | design intent | caveat — what it actually does in this codebase |
|---|---|---|---|---|
| `baseline` | random shuffle over **source-only** clips (target never seen in training) | **none** | reproduce the DCASE baseline AE on the combined dev+eval training pool | literally a vanilla AE on source-only data — a faithful reference point |
| `mixed` | `BalancedDomainSampler` → ~(1 − `target_ratio`) source + `target_ratio` target in every batch | **none** | force target clips into every gradient step so the AE's reconstruction loss is computed over both domains | exactly the textbook "balanced-batch" DG baseline — the honest reference for what a domain-balanced AE achieves |
| `adversarial` | balanced batches (same as `mixed`) | `+ 0.1 × CrossEntropy(DomainClassifier(z), domain_label)` | *intended*: DANN-style adversarial alignment — encoder learns domain-invariant `z` | **NOT** textbook DANN: there is no gradient-reversal layer (I checked). Encoder + domain classifier share one optimiser and both descend the same CE. The encoder is therefore pushed to make `z` domain-*separable*, not invariant. Empirically this matches the phase-2 `ae` result where `adversarial` had the **largest** src−tgt gap (0.055). Call it a domain-discriminative regulariser, not adversarial DA. |
| `contrastive` | balanced batches (same as `mixed`) | `+ 0.1 × contrastive_loss(z̄, z̄)` where `z̄` is global-avg-pool of `z` | *intended*: SimCLR-style representation regularisation on the bottleneck | **NOT** SimCLR: `contrastive_loss(z, z)` is called with the same tensor on both sides and labels `= arange(B)`, so there are no positive pairs. The loss penalises off-diagonal cosine similarities → pushes embeddings toward mutual orthogonality. Effectively a batch-uniformity regulariser (Wang & Isola 2020), not contrastive learning. |

The exact branch in code is at ```334:342:train_phase2.py```.

### 3b.2 `--arch` — the autoencoder backbone

All three are reconstruction autoencoders with signature `forward(x) -> (recon, z)`. `z` is the bottleneck feature map the domain / contrastive head attach to.

| arch | params | encoder | bottleneck (`embed_dim`) | decoder | notes |
|---|---|---|---|---|---|
| `ae` (SimpleAE) | ≈ 30 k | Conv(1→16)→MaxPool→Conv(16→32)→MaxPool | **32** channels @ 1/4 resolution | 2× ConvTranspose back to input | tiny, trains in minutes, least likely to overfit on our tiny dataset |
| `unet` (UNetAE) | ≈ 500 k | 2-level U-Net with BatchNorm + skip connections | **128** channels @ 1/4 | symmetric, skip-concatenated | more capacity; risk: skips let the AE reconstruct anomalies too well, shrinking the normal-vs-anomaly error gap |
| `mobilenet` (MobileNetAE) | ≈ 2 M | ImageNet-pretrained MobileNetV2 `features[0:14]` (1-channel log-mel repeated to 3) | **96** channels @ 1/16 | 4× ConvTranspose up-sampling | biggest; pretrained ImageNet features give a head start on the encoder |

Definitions live in ```27:188:model.py```. The head modules (domain classifier, contrastive head) plug onto `z`'s channel count automatically via each backbone's `embed_dim` class attribute.

### 3b.3 `--target_ratio` — what the batch sampler does

Only active when `--mode` is `mixed / adversarial / contrastive` (the baseline mode uses a plain random shuffler). With default `--batch_size 64` and `--target_ratio 0.2`:

- every batch has **exactly** 13 target clips (rounded from 64 × 0.2) and 51 source clips
- across an epoch every target clip is typically visited ~ (51 × steps_per_epoch / 59) ≈ 40× per pass, because target is heavily oversampled
- this is what makes the target domain visible to the AE at all during training — without the sampler, target clips are < 1% of the training pool and the model is effectively source-only

Increasing `--target_ratio` toward 0.5 means more balanced batches but more target-oversampling. 0.2 is the standard choice and matches the phase-1 sweep. The sampler itself is in ```5:60:sampler.py```.

### 3b.4 `--n_folds`, `--folds_to_run`, `--seed` — the stratified K-fold split

Implementation: `make_stratified_kfold_splits(...)` at ```103:142:train_phase2.py``` with stratum `(section, domain)`.

- `--n_folds` (K): the number of disjoint val partitions. The val set for fold `k` is `source_folds[k] ∪ target_folds[k]`. K=5 is the default; K=3 is a defensible speed-up (ensemble variance is empirically already small — phase-2 `ae` had per-fold target-AUC σ of 0.002–0.021).
- `--folds_to_run "0,3"`: train only these folds. Useful for fast iteration; Step 4 of the notebook will still build the ensemble from whatever fold checkpoints it finds on disk.
- `--seed`: controls (a) which files land in which fold via per-`(section, domain)` bucket shuffles and (b) the per-bucket fold offset. The same seed is used for source; `seed + 1` is used for target so source/target splits are independent but both reproducible.

Stratification guarantee (K=5, default seed): source folds are `[1188, 1188, 1188, 1188, 1188]` (exactly 198 per section); target folds are `[11, 12, 12, 12, 12]`. With K=3 the source-per-section is 330 and target is ~20/fold.

### 3b.5 `--epochs`, `--batch_size`, `--lr` — the optimisation loop

- `--epochs`: max number of passes over the training partition per fold. No early stopping is wired in — the fold's **best-val-loss checkpoint** is kept (combined-domain val MSE), everything else is discarded. The val loss is computed on normal clips only (both source and target), so "best val" rewards *reconstruction quality on normals*, NOT separation of anomalies from normals. In practice for AE-based AD, val loss saturates well before the anomaly AUC does; pushing `--epochs` too high can even hurt AUC (over-reconstruction of anomalies). 15 epochs is a good operating point for `unet / mobilenet`; 20 is fine for the tiny `ae`.
- `--batch_size`: 64 is the standard choice. The `BalancedDomainSampler` assumes enough batch size to carry ≥ 1 target clip at the chosen `target_ratio` — keep `batch_size × target_ratio ≥ 1` (64 × 0.2 = 13, comfortably safe).
- `--lr`: Adam learning rate for the single optimiser that trains both the AE and the `DomainClassifier` head (the head is always constructed but only contributes gradient when `mode=adversarial`).

### 3b.6 Data-processing parameters — fixed, not on the CLI

These are hard-coded in `dataset.py` (```12:13:dataset.py```) and are **shared with phase 1**, so any phase-1 vs phase-2 comparison is apples-to-apples on the audio side:

| parameter | value | meaning |
|---|---|---|
| `SR` | 16 000 Hz | audio is resampled to 16 kHz before STFT |
| `N_MELS` | 64 | number of mel frequency bins |
| STFT hop/win/nfft | librosa defaults (`hop=512, n_fft=2048, win=2048`) | not overridden — librosa `melspectrogram` defaults |
| clip length | whatever the WAV is | no trimming or fixed-length padding; decoders align output to input time-length at loss time |

If you ever want to sweep these, change them in `dataset.py` and regenerate the mel caches (`python prepare_cache_phase2.py --force`). They are not notebook-level knobs by design — keeping them fixed is how we keep phase-1 and phase-2 numbers comparable.

### 3b.7 Which parameters actually change behaviour vs which are cosmetic

If you're choosing what to ablate next, here's the effect size I'd expect on target AUC based on the phase-2 `ae` leaderboard and typical AE-AD dynamics:

| knob | impact | worth sweeping? |
|---|---|---|
| `--mode` | **high** (0.527 → 0.547 hmean just from `baseline → mixed`) | yes — already in the 12-run matrix |
| `--arch` | **high** (tiny AE hits a ceiling ~0.55 target AUC; pretrained MobileNet typically +0.05–0.10) | yes — already in the 12-run matrix |
| `--epochs` (within 10–20 range) | medium — too many hurts AUC (over-reconstruction); too few underfits | no — fix at 15–20 per arch |
| `--target_ratio` | medium | maybe — 0.3–0.5 can help on very target-scarce sections |
| `--lr`, `--batch_size` | low | no |
| `--seed` | low (fold-assignment noise only — phase-2 fold σ was already tiny) | no |

---

## 4. Evaluation (dev-test ensemble)

`evaluate_phase2.py` picks a `(mode, arch)` and automatically discovers
every matching fold checkpoint
`checkpoints_phase2/{mode}_{arch}_fold*.pt`. For each fold it:

1. Loads the checkpoint and scores every clip in `dev_test`
   (sections 00-02, 600 labelled clips) with its reconstruction MSE.
2. Records per-fold source / target / overall AUC + pAUC + DCASE
   harmonic mean (for transparency and variance analysis).

It then builds the **ensemble** by averaging the per-clip scores
across all K folds and recomputes metrics on the ensemble scores.
Those ensemble numbers are the primary phase-2 leaderboard values.

Artefacts land under `eval_results_phase2/{mode}_{arch}/`:

```
eval_results_phase2/{mode}_{arch}/
├── summary.json          # ensemble metrics + per-fold table
├── fold_vs_ensemble.png  # per-fold AUCs vs ensemble star plot
├── ensemble/
│   ├── summary.json
│   ├── per_clip_scores.csv
│   ├── overall_summary.png
│   ├── score_histograms.png
│   ├── roc.png
│   └── per_section_auc.png
└── fold0/ ... foldK-1/
    ├── summary.json
    ├── per_clip_scores.csv
    └── the same four plots
```

---

## 5. How the ensemble score is formed

For each clip `i` and each fold model `f_k`, we compute reconstruction
MSE `s_k(i) = mean((x_i − f_k(x_i))²)`. The ensemble score is simply:

```
s_ens(i) = (1/K) · Σ_k s_k(i)
```

No per-fold normalisation is applied — this matches standard DCASE
baseline practice. If fold-to-fold score distributions drift heavily
(check `fold_vs_ensemble.png` and `summary.json`'s `fold_tgt_auc_std`
in `collect_results_phase2.py`'s CSV), per-fold z-score normalisation
before averaging is an obvious next step.

---

## 6. Aggregated leaderboard and recommended model

`collect_results_phase2.py` scans `eval_results_phase2/` and
`checkpoints_phase2/` and produces:

```
results_phase2/
├── RESULTS_phase2.md        # leaderboard + per-fold σ + recommended model + repro
├── comparison.csv           # flat per-run CSV (ensemble metrics)
├── leaderboard.png          # ranked ensemble AUC / hmean bar chart (+ tgt σ bars)
├── source_vs_target.png     # ensemble src-vs-tgt AUC scatter
├── kfold_variance.png       # per-run per-fold AUC dot plot with ens star
├── best_model/              # K fold-checkpoints of the recommended (mode, arch)
│   ├── WINNER.json          # {mode, arch, metrics, n_folds}
│   ├── {mode}_{arch}_fold{0..K-1}.pt
│   ├── {mode}_{arch}_fold{0..K-1}_history.{json,png}
│   ├── {mode}_{arch}_kfold_summary.json
│   └── {mode}_{arch}_kfold_overview.png
├── checkpoints/             # EVERY run's fold ckpts + histories + overview PNGs
└── eval_results/            # full phase-2 eval artefacts per run
```

### Recommendation logic

The collector ranks **every `(mode, arch)` in the 12-run matrix**
(4 modes × 3 backbones) by the metric passed via `--rank_by`
(default: `hmean`, options: `hmean / tgt_auc / ovr_auc`). Ties are
broken first by ensemble **target AUC** (since DG is what we actually
care about) and then by the **lowest per-fold target-AUC σ** (we
prefer stable models over fragile ones). All three backbones compete
head-to-head — there is no per-backbone pre-filter.

The resulting winner is the model to ship. Its K fold-checkpoints are
copied verbatim under `results_phase2/best_model/` along with a
`WINNER.json` stating the choice, so the future eval-test submission
script only needs to load `best_model/*.pt` and average their per-clip
reconstruction MSEs.

The leaderboard's primary ordering is the configured ranking metric
(default: ensemble DCASE harmonic mean). Columns include:

- `src_auc`, `tgt_auc`, `ovr_auc`, `hmean` — ensemble metrics on dev-test
- `fold_tgt_auc_std`, `fold_src_auc_std`, `fold_hmean_std` — spread
  across the K folds (confidence proxy)
- `src_tgt_gap` — `src_auc − tgt_auc`, the remaining domain gap
- `mean_best_val_loss` / `std_best_val_loss` — K-fold training summary

---

## 7. File inventory (phase-2 specific)

| file | role |
|---|---|
| `dataset_phase2.py` | `CombinedBearingDataset` (union of multiple training roots). Also contains `BearingTestDataset` — kept for the future eval-test submission script; **not imported by any active phase-2 script**. |
| `prepare_cache_phase2.py` | builds mel caches for `dev_train`, `eval_train`, `dev_test` only |
| `train_phase2.py` | K-fold CV training loop across the combined pool |
| `evaluate_phase2.py` | per-fold + ensemble evaluation on `dev_test` |
| `collect_results_phase2.py` | aggregates everything, writes `RESULTS_phase2.md`, zips bundle |
| `phase2_notebook.ipynb` | Colab runner wiring all steps together |
| `README_phase2.md` | this file |

Phase-1 files (`train.py`, `evaluate.py`, `collect_results.py`,
`fixed_notebook.ipynb`, `README.md`) are untouched.

---

## 8. Reproducing the full phase-2 sweep end-to-end

Phase 2 now sweeps **all four DG modes × all three backbones = 12 runs**.
Each run is itself a K=5 stratified K-fold ensemble, so the total is
60 fold checkpoints. `collect_results_phase2.py` ranks these 12 runs
in one leaderboard and picks a single winner to ship.

```bash
# one-time caches (dev_train + eval_train + dev_test)
python prepare_cache_phase2.py

# --- Backbone 1/3: SimpleAE (tiny, 20 epochs per fold) ---
python train_phase2.py --mode baseline    --arch ae --epochs 20 --n_folds 5
python train_phase2.py --mode mixed       --arch ae --epochs 20 --n_folds 5
python train_phase2.py --mode adversarial --arch ae --epochs 20 --n_folds 5
python train_phase2.py --mode contrastive --arch ae --epochs 20 --n_folds 5

# --- Backbone 2/3: UNetAE (2-level U-Net w/ skips, 40 epochs per fold) ---
python train_phase2.py --mode baseline    --arch unet --epochs 40 --n_folds 5
python train_phase2.py --mode mixed       --arch unet --epochs 40 --n_folds 5
python train_phase2.py --mode adversarial --arch unet --epochs 40 --n_folds 5
python train_phase2.py --mode contrastive --arch unet --epochs 40 --n_folds 5

# --- Backbone 3/3: MobileNetV2 (ImageNet-pretrained, 40 epochs per fold) ---
python train_phase2.py --mode baseline    --arch mobilenet --epochs 40 --n_folds 5
python train_phase2.py --mode mixed       --arch mobilenet --epochs 40 --n_folds 5
python train_phase2.py --mode adversarial --arch mobilenet --epochs 40 --n_folds 5
python train_phase2.py --mode contrastive --arch mobilenet --epochs 40 --n_folds 5

# ensemble evaluation on dev-test (evaluate_phase2.py is called once per
# (mode, arch); the Step 4 cell in the notebook loops over all 12 runs
# automatically by globbing checkpoints_phase2/*_fold*.pt)
for mode in baseline mixed adversarial contrastive; do
  for arch in ae unet mobilenet; do
    python evaluate_phase2.py --mode $mode --arch $arch
  done
done

# aggregate leaderboard + plots + winner selection (across all 12 runs) + zip bundle
python collect_results_phase2.py --rank_by hmean
```

The collector writes `results_phase2/` plus the zipped sibling
`results_phase2.zip`. The winner is picked over the **full 12-run
matrix** (3 backbones × 4 modes); its K fold-checkpoints end up under
`results_phase2/best_model/`. In Colab the notebook additionally
copies both into your Drive project folder and triggers a local
download of the zip so the bundle survives the runtime and is
available offline.

Epoch budget rationale: SimpleAE has ~0.01M params and saturates by
epoch 20; UNetAE (~0.5M) and MobileNetV2-based AE (~2M) both need
~40 epochs per fold to converge on the combined dev+eval training
pool.

---

## 9. What is deliberately NOT in phase 2 (yet)

- **Scoring `eval_test` (sections 03-05, unlabelled).** No script in
  this bundle reads that split. Once we pick a winning `(mode, arch)`
  from the leaderboard in §6, we will add a dedicated submission
  script that:
  1. Builds a mel cache for `eval_test`.
  2. Loads all K fold checkpoints of the winning `(mode, arch)`.
  3. Averages their per-clip reconstruction MSEs.
  4. Emits DCASE-format `anomaly_score_section_XX.csv` and
     `decision_result_section_XX.csv` files per section.
- **Per-fold score normalisation** before averaging (standard DCASE
  z-score trick). Only worth adding if `fold_tgt_auc_std` turns out to
  be large after the first full run.
- **Checkpoint of the full ensemble.** We don't dump a "merged" model;
  the ensemble is just the K checkpoints plus the average-at-inference
  rule above.
