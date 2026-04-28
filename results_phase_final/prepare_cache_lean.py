"""
[LEAN ENTRYPOINT] Canonical script for the consolidated workflow.

Pre-compute mel-spectrogram caches for every split phase-2 is ALLOWED to
touch during training and apples-to-apples evaluation:

    1. dev   train  (sections 00-02, labelled-normal)           [train]
    2. eval  train  (sections 03-05, labelled-normal, NEW)      [train]
    3. dev   test   (sections 00-02, normal+anomaly, labelled)  [eval]

The 4th split — ``data/dcase_bearing_eval/bearing/test`` (sections 03-05,
unlabelled) — is INTENTIONALLY NOT CACHED HERE. It is the held-out
"prediction" split that must not be touched until a best domain-
generalization model has been selected from the three splits above.
A separate submission script will be added once that model is chosen.
"""

import argparse
import os

from dataset import BearingDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dev_train_dir", type=str,
        default=os.environ.get("BEARING_DEV_TRAIN_DIR",
                               "data/dcase_bearing_dev/bearing/train"),
    )
    p.add_argument(
        "--dev_test_dir", type=str,
        default=os.environ.get("BEARING_DEV_TEST_DIR",
                               "data/dcase_bearing_dev/bearing/test"),
    )
    p.add_argument(
        "--eval_train_dir", type=str,
        default=os.environ.get("BEARING_EVAL_TRAIN_DIR",
                               "data/dcase_bearing_eval/bearing/train"),
    )
    p.add_argument(
        "--dev_train_cache", type=str,
        default=os.environ.get("BEARING_DEV_TRAIN_CACHE", ""),
    )
    p.add_argument(
        "--dev_test_cache", type=str,
        default=os.environ.get("BEARING_DEV_TEST_CACHE", ""),
    )
    p.add_argument(
        "--eval_train_cache", type=str,
        default=os.environ.get("BEARING_EVAL_TRAIN_CACHE", ""),
    )
    p.add_argument("--force", action="store_true",
                   help="Recompute every .npy even if a valid cache exists.")
    args = p.parse_args()

    jobs = [
        ("dev train",  args.dev_train_dir,  args.dev_train_cache),
        ("eval train", args.eval_train_dir, args.eval_train_cache),
        ("dev test",   args.dev_test_dir,   args.dev_test_cache),
    ]

    for name, root, cache in jobs:
        if not os.path.isdir(root):
            print(f"[skip] {name}: {root} does not exist")
            continue
        print(f"\n=== {name}: {root} ===")
        ds = BearingDataset(root, cache_dir=(cache or None), return_label=True)
        ds.build_cache(force=args.force)

    print(
        "\nNOTE: eval-test (sections 03-05, unlabelled) was NOT cached.\n"
        "      That split is reserved for final prediction only and\n"
        "      must not be read during training or model selection."
    )


if __name__ == "__main__":
    main()
