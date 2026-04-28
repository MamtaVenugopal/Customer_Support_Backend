import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


SR = 16000
N_MELS = 64
TARGET_WIDTH = 313


def parse_filename(fname):
    """
    Parse DCASE bearing filenames, e.g.
      section_00_source_train_normal_0000_vel_22.wav
      section_01_target_test_anomaly_0005_vel_12_loc_G.wav

    Returns (domain, section, label) where
      domain in {source, target}
      section is int, -1 if unknown
      label  in {normal, anomaly, unknown}
    """
    domain = "target" if "target" in fname else "source"

    section = -1
    if "section_" in fname:
        try:
            section = int(fname.split("section_")[1][:2])
        except Exception:
            section = -1

    if "_anomaly_" in fname:
        label = "anomaly"
    elif "_normal_" in fname:
        label = "normal"
    else:
        label = "unknown"

    return domain, section, label


def _compute_log_mel(wav_path):
    if librosa is None:
        raise RuntimeError("librosa is required to build the cache")
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel).astype(np.float32)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    if log_mel.shape[1] < TARGET_WIDTH:
        pad = TARGET_WIDTH - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
    else:
        log_mel = log_mel[:, :TARGET_WIDTH]
    return log_mel


class BearingDataset(Dataset):
    """
    Loads bearing clips as pre-computed log-mel tensors.

    Each WAV is decoded once; a `.npy` cache file is stored per clip.
    Pass `return_label=True` to get the normal/anomaly label in each item.
    """

    def __init__(self, root_dir, domain_filter=None, cache_dir=None,
                 return_label=False, verbose=True,
                 exclude_files=None, include_only_files=None):
        """
        exclude_files / include_only_files : iterable of bare WAV filenames
            used to carve out a train/val split without touching the files
            on disk. Pass a set of filenames to keep or drop.
        """
        self.root_dir = root_dir.rstrip("/")
        self.return_label = return_label

        exclude_files = set(exclude_files) if exclude_files else None
        include_only_files = set(include_only_files) if include_only_files else None

        default_cache = os.path.join(
            os.path.dirname(self.root_dir),
            "_mel_cache_" + os.path.basename(self.root_dir),
        )
        self.cache_dir = cache_dir or default_cache
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples = []
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if not f.endswith(".wav"):
                    continue
                if include_only_files is not None and f not in include_only_files:
                    continue
                if exclude_files is not None and f in exclude_files:
                    continue
                domain, section, label = parse_filename(f)
                if domain_filter and domain != domain_filter:
                    continue
                wav_path = os.path.join(root, f)
                rel = os.path.relpath(wav_path, self.root_dir)
                cache_path = os.path.join(self.cache_dir, rel + ".npy")
                self.samples.append({
                    "path": wav_path,
                    "cache": cache_path,
                    "domain": domain,
                    "section": section,
                    "label": label,
                    "filename": f,
                })
        if verbose:
            print(f"Total samples loaded: {len(self.samples)} from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def _atomic_save(self, path, arr):
        """Write to a .tmp next to the target then rename.

        Prevents truncated .npy files when a Colab runtime is killed
        mid-write, which would otherwise trigger
        `ValueError: mmap length is greater than file size` on next load.

        We open an explicit file handle so numpy does NOT silently
        append `.npy` to our temp name (it does that for str/Path
        arguments when the extension isn't already .npy).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)

    def _valid_cache(self, path):
        """Heuristic: a real log-mel .npy for our config is > 8 KB."""
        try:
            return os.path.getsize(path) > 8 * 1024
        except OSError:
            return False

    def build_cache(self, force=False):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        todo = [s for s in self.samples
                if force or not self._valid_cache(s["cache"])]
        if not todo:
            print(f"Cache already complete ({len(self.samples)} files) at {self.cache_dir}")
            return

        print(f"Building mel cache for {len(todo)} files -> {self.cache_dir}")

        def _one(s):
            log_mel = _compute_log_mel(s["path"])
            self._atomic_save(s["cache"], log_mel)

        done = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            for fut in as_completed([ex.submit(_one, s) for s in todo]):
                fut.result()
                done += 1
                if done % 200 == 0 or done == len(todo):
                    print(f"  cached {done}/{len(todo)}")

    def _load_or_rebuild(self, sample):
        cache_path = sample["cache"]
        if self._valid_cache(cache_path):
            try:
                return np.load(cache_path)
            except (ValueError, EOFError, OSError):
                # Truncated or otherwise corrupt; delete and fall through
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        log_mel = _compute_log_mel(sample["path"])
        self._atomic_save(cache_path, log_mel)
        return log_mel

    def __getitem__(self, idx):
        sample = self.samples[idx]
        log_mel = self._load_or_rebuild(sample)

        x = torch.from_numpy(np.ascontiguousarray(log_mel)).unsqueeze(0).float()
        if self.return_label:
            return x, sample["domain"], sample["section"], sample["label"]
        return x, sample["domain"], sample["section"]
