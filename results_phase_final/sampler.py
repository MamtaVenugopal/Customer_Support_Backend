import random
from torch.utils.data import Sampler


class BalancedDomainSampler(Sampler):
    """
    Controlled domain sampler for domain-generalization training.

    Every yielded batch is guaranteed to contain BOTH domains in a fixed
    ratio. Default = 80% source / 20% target.

        n_target = max(1, round(batch_size * target_ratio))
        n_source = batch_size - n_target

    - Source indices are drawn WITHOUT replacement from a per-epoch
      shuffled pool, so the model sees each source clip once per epoch.
    - Target indices are OVERSAMPLED (drawn with replacement) from the
      small target pool, so even a pool of ~10 clips keeps appearing in
      every batch.
    - Indices are shuffled within each batch.

    Pass this to DataLoader via `batch_sampler=`.
    """

    def __init__(self, dataset, batch_size=32, target_ratio=0.2, seed=None, verbose=True):
        if not 0.0 < target_ratio < 1.0:
            raise ValueError("target_ratio must be in (0, 1)")

        self.batch_size = batch_size
        self.target_ratio = target_ratio
        self.n_target = max(1, int(round(batch_size * target_ratio)))
        self.n_source = batch_size - self.n_target

        self.source_idx = [i for i, s in enumerate(dataset.samples) if s["domain"] == "source"]
        self.target_idx = [i for i, s in enumerate(dataset.samples) if s["domain"] == "target"]

        if not self.source_idx:
            raise ValueError("No source samples found in dataset")
        if not self.target_idx:
            raise ValueError(
                "No target samples found in dataset; mixed training "
                "requires at least one target clip"
            )

        self._num_batches = len(self.source_idx) // self.n_source
        if self._num_batches == 0:
            raise ValueError(
                f"Not enough source samples ({len(self.source_idx)}) "
                f"for n_source={self.n_source} per batch"
            )

        self._rng = random.Random(seed)

        if verbose:
            print(
                f"[BalancedDomainSampler] batch_size={batch_size} "
                f"(source={self.n_source} + target={self.n_target}) | "
                f"pool: source={len(self.source_idx)}, target={len(self.target_idx)} | "
                f"batches/epoch={self._num_batches} | "
                f"target is oversampled with replacement"
            )

    def __iter__(self):
        source_pool = self.source_idx.copy()
        self._rng.shuffle(source_pool)

        for b in range(self._num_batches):
            s_start = b * self.n_source
            src = source_pool[s_start:s_start + self.n_source]
            tgt = self._rng.choices(self.target_idx, k=self.n_target)  # with replacement
            batch = src + tgt
            self._rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self._num_batches
