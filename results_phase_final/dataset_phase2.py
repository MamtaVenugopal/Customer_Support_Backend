"""
Phase-2 dataset: combines multiple training roots (dev + eval train) into
a single torch Dataset, while preserving per-root cache directories.

Reuses everything from dataset.py (log-mel computation, atomic caching,
filename parsing, auto-rebuild of corrupt .npy files). The only new piece
is `CombinedBearingDataset`, which walks each root independently and
concatenates their sample lists.

Also provides a thin `BearingTestDataset` for the *unlabelled* DCASE
evaluation test split (sections 03-05) whose filenames do NOT contain
domain or normal/anomaly tokens.
"""

import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

from dataset import (
    BearingDataset,
    _compute_log_mel,  # reused for the unlabelled test split
    parse_filename,
)


class CombinedBearingDataset(torch.utils.data.Dataset):
    """Union of several `BearingDataset` instances, one per root directory.

    Each root gets its own cache dir (either user-supplied or auto-derived
    as `<parent>/_mel_cache_<leaf>`), so dev-train and eval-train caches
    stay separate. Otherwise behaves identically to `BearingDataset`.

    `exclude_files` / `include_only_files` are applied to *every* child,
    so the val-split carver works uniformly across roots.
    """

    def __init__(
        self,
        root_dirs: Sequence[str],
        cache_dirs: Optional[Sequence[Optional[str]]] = None,
        domain_filter: Optional[str] = None,
        return_label: bool = False,
        exclude_files: Optional[Iterable[str]] = None,
        include_only_files: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ):
        assert len(root_dirs) > 0, "root_dirs must contain at least one path"
        if cache_dirs is None:
            cache_dirs = [None] * len(root_dirs)
        assert len(cache_dirs) == len(root_dirs)

        self.children: List[BearingDataset] = []
        self.samples: List[dict] = []
        self._child_offsets: List[int] = [0]

        for root, cache in zip(root_dirs, cache_dirs):
            child = BearingDataset(
                root_dir=root,
                cache_dir=cache,
                domain_filter=domain_filter,
                return_label=return_label,
                exclude_files=exclude_files,
                include_only_files=include_only_files,
                verbose=verbose,
            )
            self.children.append(child)
            self.samples.extend(child.samples)
            self._child_offsets.append(len(self.samples))

        self.return_label = return_label
        if verbose:
            print(
                f"CombinedBearingDataset: {len(self.samples)} samples "
                f"from {len(self.children)} roots"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _which_child(self, idx: int) -> BearingDataset:
        for i in range(len(self.children)):
            lo = self._child_offsets[i]
            hi = self._child_offsets[i + 1]
            if lo <= idx < hi:
                return self.children[i]
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        # Delegate to child so it can handle atomic rebuild / caching.
        child = self._which_child(idx)
        local_idx = idx - self._child_offsets[self.children.index(child)]
        return child[local_idx]

    def build_cache(self, force: bool = False) -> None:
        for i, child in enumerate(self.children):
            print(f"[CombinedBearingDataset] caching root {i + 1}/{len(self.children)}: "
                  f"{child.root_dir}")
            child.build_cache(force=force)


def _parse_eval_test_filename(fname: str):
    """The DCASE eval *test* split uses a stripped-down filename scheme:

        section_03_0000.wav  section_04_0123.wav  section_05_0199.wav

    There is no domain token and no normal/anomaly token: labels and
    source/target mapping are only available server-side for submissions.

    Returns (section:int, clip_idx:int).
    """
    parts = fname.replace(".wav", "").split("_")
    section = int(parts[1])
    clip_idx = int(parts[2])
    return section, clip_idx


class BearingTestDataset(torch.utils.data.Dataset):
    """Dataset for the unlabelled DCASE eval-test split (sections 03-05)."""

    def __init__(self, root_dir: str, cache_dir: Optional[str] = None,
                 verbose: bool = True):
        self.root_dir = root_dir.rstrip("/")
        default_cache = os.path.join(
            os.path.dirname(self.root_dir),
            "_mel_cache_" + os.path.basename(self.root_dir),
        )
        self.cache_dir = cache_dir or default_cache
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples: List[dict] = []
        for f in sorted(os.listdir(self.root_dir)):
            if not f.endswith(".wav"):
                continue
            section, clip_idx = _parse_eval_test_filename(f)
            wav_path = os.path.join(self.root_dir, f)
            self.samples.append({
                "filename": f,
                "path": wav_path,
                "cache": os.path.join(self.cache_dir, f + ".npy"),
                "section": section,
                "clip_idx": clip_idx,
                "domain": "unknown",
                "label": "unknown",
            })
        if verbose:
            print(f"BearingTestDataset: {len(self.samples)} clips from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _atomic_save(self, path: str, arr: np.ndarray) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)

    def _valid_cache(self, path: str) -> bool:
        try:
            return os.path.getsize(path) > 8 * 1024
        except OSError:
            return False

    def build_cache(self, force: bool = False) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        todo = [s for s in self.samples
                if force or not self._valid_cache(s["cache"])]
        if not todo:
            print(f"Eval-test cache already complete ({len(self.samples)} files) "
                  f"at {self.cache_dir}")
            return

        print(f"Building eval-test cache for {len(todo)} files -> {self.cache_dir}")

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

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        cache_path = s["cache"]
        if self._valid_cache(cache_path):
            try:
                log_mel = np.load(cache_path)
            except (ValueError, EOFError, OSError):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                log_mel = _compute_log_mel(s["path"])
                self._atomic_save(cache_path, log_mel)
        else:
            log_mel = _compute_log_mel(s["path"])
            self._atomic_save(cache_path, log_mel)

        x = torch.from_numpy(np.ascontiguousarray(log_mel)).unsqueeze(0).float()
        return x, s["section"], s["clip_idx"], s["filename"]
