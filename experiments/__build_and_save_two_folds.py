# experiments/build_two_folds_seed42_test20.py
import os
import sys
import warnings
import pickle as pkl
import random
import numpy as np

warnings.filterwarnings("ignore", message=".*weights_only=False.*")
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

# Project root = parent of this file (repo root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import *  # noqa: F401,F403

from omegaconf import OmegaConf
from src.dataset.datasets.mm_base import build_mm_datasets
from src.utils.marker_utils import load_marker_embeddings


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def main():
    SEED = 42
    TEST_FRACTION = 0.20  # "test20"
    OUT_DIR = os.path.join(project_root, "notebooks", "checkpoints")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"two_folds_seed{SEED}_test20.pkl")

    set_seed(SEED)

    print("[1/3] Building dataset...")

    # Keep "good" notebook style paths
    base_cfg = OmegaConf.load(os.path.join(project_root, "src/dataset/configs/base_config.yaml"))
    base_cfg.marker_embedding_dir = os.path.join(project_root, "src/dataset/esm2_t30_150M_UR50D")

    # This is important: ensures marker embeddings path exists and matches your notebook behavior
    _ = load_marker_embeddings(base_cfg.marker_embedding_dir)

    orion_subset_cfg = OmegaConf.load(os.path.join(project_root, "src/dataset/configs/orion_subset.yaml"))
    ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)
    ds = build_mm_datasets(ds_cfg)

    tissue_ids = ds[0].unimodal_datasets["cycif"].get_tissue_ids()
    tissue_ids = list(tissue_ids)

    # deterministic shuffle
    rng = np.random.RandomState(SEED)
    rng.shuffle(tissue_ids)

    n = len(tissue_ids)
    n_test = max(1, int(round(TEST_FRACTION * n)))
    n_train = n - n_test

    print(f"  Total tissues: {n}")
    print(f"  Per-fold: {n_train} train | {n_test} test")

    print("\n[2/3] Creating 2 folds...")

    # Fold1: first 20% is test
    fold1_test = tissue_ids[:n_test]
    fold1_train = tissue_ids[n_test:]

    # Fold2: next 20% is test (disjoint from fold1 test). If not enough, wrap but keep disjoint.
    start = n_test
    end = start + n_test
    fold2_test = tissue_ids[start:end]
    if len(fold2_test) < n_test:
        # not expected for your n=35, but keep safe
        needed = n_test - len(fold2_test)
        fold2_test = fold2_test + tissue_ids[end:end + needed]

    fold2_test = list(dict.fromkeys(fold2_test))  # preserve order, unique
    fold2_test = [t for t in fold2_test if t not in fold1_test]  # enforce disjoint tests

    # If disjoint enforcement shortened it, top-up from remaining tissues (still deterministic)
    if len(fold2_test) < n_test:
        remaining = [t for t in tissue_ids if (t not in fold1_test and t not in fold2_test)]
        fold2_test = fold2_test + remaining[: (n_test - len(fold2_test))]

    fold2_train = [t for t in tissue_ids if t not in fold2_test]

    print(f"  Fold1: {len(fold1_train)} train | {len(fold1_test)} test | overlap: {len(set(fold1_train) & set(fold1_test))}")
    print(f"  Fold2: {len(fold2_train)} train | {len(fold2_test)} test | overlap: {len(set(fold2_train) & set(fold2_test))}")
    print(f"  Fold1 test ∩ Fold2 test: {len(set(fold1_test) & set(fold2_test))}")

    folds = {
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "all_tissue_ids_shuffled": tissue_ids,
        "fold1": {"train_tids": fold1_train, "test_tids": fold1_test},
        "fold2": {"train_tids": fold2_train, "test_tids": fold2_test},
    }

    print("\n[3/3] Saving...")
    with open(out_path, "wb") as f:
        pkl.dump(folds, f)

    print("=" * 80)
    print(f"  Saved folds -> {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
