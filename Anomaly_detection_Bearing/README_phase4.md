# README — Phase 4 (Explainability)

This phase is **analysis only** (no training). It explains what drives the winner model's anomaly score on labelled `dev_test`.

Notebook: `phase4_notebook.ipynb`

## What the anomaly score is

The deployed score is **mean squared reconstruction error (MSE)** between the input log-mel spectrogram and the ensemble-mean reconstruction across K fold checkpoints.

## What explainability means here

Because the score is reconstruction-based, "explainability" answers:

- **Where** the model fails to reconstruct (time-frequency error heatmaps)
- **Which mel bins** carry most error on average for normals vs anomalies (and by section / domain)
- **Whether pooled bottleneck features** `z` separate labels / domains / sections (PCA)
- **Which input pixels most influence the MSE** (Integrated Gradients of MSE w.r.t. input)

## Outputs

The notebook writes everything under `/content/phase4_outputs/` and zips it to `phase4_outputs.zip`.

Key files:

- `explain/melbin_pooled_label.png`
- `explain/melbin_by_section.png`
- `explain/pca_z_{label,domain,section}.png`
- `explain/ig_{anom,norm}_*.png` (integrated gradients panels)
- `phase4_summary.json`

## Runtime notes

Integrated Gradients is the expensive step. Defaults:

- `N_EXAMPLES_HIGH = 6`, `N_EXAMPLES_LOW = 6`
- `IG_STEPS = 32`

If Colab is slow, reduce examples and/or IG steps first.

---

## Discussion — what `phase4_outputs.zip` shows (winner: `adversarial/mobilenet`, K=3 ensemble)

This section is grounded in the artefacts inside `phase4_outputs.zip` (plots under `explain/` plus `phase4_summary.json`). I also recomputed a few pooled statistics from `dev_test` with the same winner checkpoints and saved them as `phase4_readme_stats.json` in the repo root for traceability.

### 1) The headline: anomalies are mostly "more reconstruction error everywhere"

On `dev_test` (600 clips, 300 normal / 300 anomaly), the **per-clip ROC-AUC of the ensemble MSE score** is ~**0.627** (`phase4_readme_stats.json`: `roc_auc_score_mse = 0.6265333`). That matches the Phase 2 winner's pooled AUC (`ovr_auc ≈ 0.628` in `phase4_summary.json`) — explainability is consistent with the leaderboard: the model is **weak-but-real** better than chance, not a sharp separator.

### 2) Mel-bin attribution (what frequency bands explain the score)

File: `explain/melbin_pooled_label.png`

Mechanically, this plot is:

- blue = mean squared reconstruction error per mel bin for **normal** clips
- orange = same for **anomaly** clips
- black = `anomaly − normal` difference

Quantitatively (pooled over sections + domains), the **largest positive gaps** (top mel bins by `anomaly−normal` difference) are concentrated in mid-frequency bands, especially:

- top bins by difference: **37, 41, 11, 36, 35, 18, 34, 38, 12, 20** (`phase4_readme_stats.json`)

Interpretation:

- The model's anomaly signal is **not** a single magic frequency — it is a **broad mid-band excess reconstruction error** on anomalies vs normals.
- That pattern is typical for bearing faults that create narrowband energy changes that still smear across neighboring mel bins after pooling/averaging.

There are also **negative difference bins** (normal > anomaly in that bin, on average), worst:

- **30, 23, 42, 62, 22** (`phase4_readme_stats.json`)

Interpretation:

- Some frequency bins are **ambiguous** or even **anti-correlated** with the global label after averaging. This is exactly why **precision is limited** at a low global threshold: a lot of "healthy but spectro-temporally messy" normal clips can still accumulate error in bins that overlap the anomaly profile.

Per-section view (`explain/melbin_by_section.png`) differs by machine section (different peak locations):

- argmax `(anomaly−normal)` difference bin by section: **sec00 → 41**, **sec01 → 37**, **sec02 → 18** (`phase4_readme_stats.json`)

Interpretation:

- Even on the same mel axis, **section-specific mechanical behavior shifts which bins carry the most extra anomaly error**. That is a strong argument for **per-section calibration** (threshold or normalization) in Phase 3 deployment, not only a single global cutoff.

### 3) Latent PCA (does `z` "explain" labels cleanly?)

Files:

- `explain/pca_z_label.png`
- `explain/pca_z_domain.png`
- `explain/pca_z_section.png`

What you should see qualitatively:

- **Label PCA** shows heavy overlap between normal vs anomaly in 2D — the first two PCs are not a clean decision boundary.
- A crude linear proxy: ROC-AUC of **PC1 alone** as a score is ~**0.596** (`phase4_readme_stats.json`). That is *better than chance*, but still modest, consistent with overlap in the plot.
- **Domain PCA** shows **target** points occupying a wider region along PC1 than **source** — domain shift is still visible in `z` even for the "adversarial" run. This aligns with the Phase 2 observation that the adversarial term here is **not** textbook DANN (no gradient reversal), so it does not guarantee domain-invariant embeddings.

Important nuance:

- At inference, **we do not classify using `z`**. These PCA plots explain **representation geometry**, not the deployed scorepath. The deployed scorepath is reconstruction MSE.

### 4) Integrated Gradients panels (what input pixels move the MSE)

Files: `explain/ig_anom_*.png` and `explain/ig_norm_*.png`

Each figure is a 4-panel strip:

1. input log-mel
2. ensemble reconstruction
3. squared error map
4. **IG(MSE)** — sensitivity of the scalar MSE to each input mel-time pixel

How to read them:

- For **high-MSE anomalies**, IG should highlight the time-frequency regions where small input changes would most increase reconstruction mismatch (often aligned with the error map, but not identical).
- For **low-MSE normals**, IG should be comparatively diffuse / smaller magnitude — consistent with "easy to reconstruct".

### 5) So what did `adversarial` actually buy you, explainability-wise?

Training mode `adversarial` adds a domain term on `z` during training, but **the score you analyze in Phase 4 is still reconstruction error**.

What we can honestly claim from these plots:

- The model's discriminative behavior is dominated by **mid-mel reconstruction error patterns** that differ between normal vs anomaly **on average**, but are **not** perfectly consistent bin-by-bin (negative-diff bins exist).
- The latent space still shows **domain structure** in PCA, meaning the encoder is not fully domain-neutral in `z` — consistent with the known implementation caveat (no gradient reversal).

What we cannot claim from these plots alone:

- That the domain head "caused" a specific mel bin to light up. To show causality you'd need an ablation comparing `mixed` vs `adversarial` side-by-side on the same plots (same seeds, same folds).

### 6) Practical recommendations (based on these outputs)

1. **Do not rely on a single global mel-bin threshold** — section curves differ (peaks at bins **41 / 37 / 18** for sections 00/01/02).
2. If maintenance noise dominates, prefer **triage thresholds** (low/high) rather than raising a single cutoff (Phase 3 discussion): raising a single cutoff tends to increase FN quickly for this model family.
3. If you want stronger "feature explainability" for stakeholders, add a **paired baseline** run (`mixed/mobilenet`, same K) and diff the mel-bin curves — that isolates what the adversarial term changes.

