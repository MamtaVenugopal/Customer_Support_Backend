# Bearing anomaly detection (DCASE-style U-Net on mel spectrograms)

Unsupervised anomaly detection on **bearing** audio: train on **normal-only** clips, score tests with **reconstruction error** from a **2D U-Net autoencoder** on log-mel spectrograms. Works **locally** and on **Google Colab (GPU)**.

---

## Google Colab — what you need to do (checklist)

1. **Copy files** into one folder (see [Files to upload](#files-to-upload-google-colab) and `FILES_FOR_COLAB.txt`). Easiest: zip that folder, upload the zip to Colab, unzip under `/content`.
2. In Colab: **Runtime → Change runtime type → GPU** (T4/L4, etc.) → Save.
3. Open **`bearing_unet_asd.ipynb`** from that folder so **Run → Run all** uses the correct working directory, **or** after opening the notebook set the path:
   - Run **section 0** first.
   - If imports fail, uncomment `os.chdir("/content/YourFolderName")` in section 0 and set the folder that contains `bearing_unet_asd.ipynb`, `bearing_asd/`, `requirements.txt`, and the `.zip` files.
4. **Run all cells in order.** Section **0** runs `pip install -r requirements.txt` (does **not** replace Colab’s PyTorch unless you add `torch` to `requirements.txt` — **don’t**, to keep CUDA working).
5. **Optional — Google Drive:** for large zips or to keep `results/` after disconnect, upload project + zips to Drive, mount Drive, `os.chdir` to that path in section 0, then run all.
6. **After training:** download `results/bearing_unet_mel.pt` and `results/anomaly_scores_bearing_eval_test.csv` (or rely on Drive path).

---

## Files to upload (Google Colab)

| Path | Purpose |
|------|--------|
| `bearing_unet_asd.ipynb` | Main notebook (**section 0 = Colab/local bootstrap**) |
| `requirements.txt` | Pip dependencies (no `torch` pin — Colab supplies CUDA PyTorch) |
| `bearing_asd/__init__.py` | Package |
| `bearing_asd/config.py` | Paths & hyperparameters |
| `bearing_asd/data_description.py` | Dataset / task text |
| `bearing_asd/data_loading.py` | Zip extract, mel features, DataLoaders |
| `bearing_asd/data_visualisation.py` | Plots |
| `bearing_asd/model_unet.py` | U-Net |
| `bearing_asd/model_summary.py` | Model summary |
| `bearing_asd/training.py` | Training loop |
| `bearing_asd/evaluation.py` | Scores, AUC / pAUC, checkpoint load |
| `bearing_asd/explainability.py` | Reconstruction / saliency plots |

**Not needed on Colab:** `venv/`, `__pycache__/`, `scripts/`, `Archive/`, executed notebook copies, stray `bearing/` CSV-only trees unless you use them separately.

**Data archives** (place in the **same folder** as the notebook — `default_paths()` expects this):

| File | Required? | Role |
|------|-----------|------|
| `eval_data_bearing_train.zip` | **Yes** | Normal-only training (sections 03–05) |
| `eval_data_bearing_test.zip` | **Yes** | Unlabeled eval-style test audio |
| `dev_bearing.zip` | No (recommended) | Labeled dev test → AUC / pAUC + explainability demo |

---

## Notebook behavior (Colab-compatible details)

- **Section 0:** `pip install`, `sys.path`, `matplotlib` inline, prints `PROJECT_ROOT` and CUDA status.
- **Batch size:** **32** if CUDA is available, **16** on CPU (`## 2. Setup` cell).
- **`num_workers=0`:** avoids multiprocessing issues on Colab.
- **`TRAIN_FILE_CAP`:** default **512** for quick runs; set to **`None`** for all **3000** training files when using GPU.
- **Outputs:** `data/dcase_bearing_eval/`, `data/dcase_bearing_dev/`, `results/bearing_unet_mel.pt`, `results/anomaly_scores_bearing_eval_test.csv`.

Regenerate the notebook from the template anytime:

```bash
python scripts/build_notebook.py
```

---

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# If needed: install PyTorch from https://pytorch.org for your CUDA/CPU
jupyter lab
```

Open `bearing_unet_asd.ipynb` from the project root and run **section 0** first (same as Colab).

---

## Reference

- [DCASE 2022 Task 2 — dataset overview](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#dataset-overview)

---

## Troubleshooting (Colab)

| Issue | What to do |
|--------|------------|
| `ModuleNotFoundError: bearing_asd` | Run section 0; set `os.chdir` to the folder that **contains** `bearing_asd/`. |
| `FileNotFoundError` for zips | Put zips **next to** `bearing_unet_asd.ipynb` (same as `PROJECT_ROOT` printed in section 0). |
| CUDA `False` | **Runtime → Change runtime type → GPU**, then **Runtime → Restart session**, rerun from section 0. |
| Disk full | Use Drive; extract under `MyDrive/...`; delete `.zip` after extract if needed. |
| `librosa` / WAV errors | `requirements.txt` includes `soundfile`; rerun section 0. |
| Slow training | GPU + `TRAIN_FILE_CAP = None` + higher `epochs`; increase `batch_size` if memory allows. |
