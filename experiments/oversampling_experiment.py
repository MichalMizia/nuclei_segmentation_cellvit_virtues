import sys
import os
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*")
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pickle as pkl
from omegaconf import OmegaConf
import random
import gc
from copy import deepcopy

# project imports
from src.dataset.datasets.mm_base import build_mm_datasets
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import (
    build_flex_dual_virtues_encoder,
)
from src.utils.marker_utils import load_marker_embeddings
from src.utils.utils import load_checkpoint_safetensors
from src.models.wrappers.virtues_wrapper import VirtuesWrapper
from src.dataset.datasets.embeddings_dataset import EmbeddingsDataset
from src.models.cellvit_decoder import CellViTDecoder
from src.utils.metrics import CombinedLoss
from src.models.utils.train_loop import train_loop


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

print("=" * 80)
print("CELL CLASS OVERSAMPLING EXPERIMENT")
print("=" * 80)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print("\n[1/5] Loading Dataset...")
base_cfg = OmegaConf.load("../src/dataset/configs/base_config.yaml")
base_cfg.marker_embedding_dir = "../src/dataset/esm2_t30_150M_UR50D"
marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)
orion_subset_cfg = OmegaConf.load("../src/dataset/configs/orion_subset.yaml")
ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)
ds = build_mm_datasets(ds_cfg)

tissue_ids = ds[0].unimodal_datasets["cycif"].get_tissue_ids()
print(f"  Total tissues: {len(tissue_ids)}")

# ============================================================================
# 2. Build Encoder
# ============================================================================
print("\n[2/5] Building Encoder...")
with open(VIRTUES_WEIGHTS_PATH + "/config.pkl", "rb") as f:
    virtues_cfg = pkl.load(f)

encoder = build_flex_dual_virtues_encoder(virtues_cfg, marker_embeddings)
encoder.cuda()
weights = load_checkpoint_safetensors(
    VIRTUES_WEIGHTS_PATH + "/checkpoints/checkpoint-94575/model.safetensors"
)
weights_encoder = {
    k[len("encoder.") :]: v for k, v in weights.items() if k.startswith("encoder.")
}
encoder.load_state_dict(weights_encoder, strict=False)
print("  Encoder loaded successfully")

# ============================================================================
# 3. Build Wrappers and Load Embeddings
# ============================================================================
print("\n[3/5] Loading Precomputed Embeddings...")

wrapper = VirtuesWrapper(encoder=encoder, device="cuda", autocast_dtype=torch.float16)
path_sp_he = os.path.join("/data", "embeddings", "virtues_sp_he")
wrapper.load_embeddings(path_sp_he)
print(f"  SP+HE embeddings: {len(wrapper.embeddings)} tissues")

# ============================================================================
# 4. Create Datasets
# ============================================================================
print("\n[4/5] Creating Datasets...")


def collate_batch(batch):
    _, he_img, mask, pss, intermediate_pss = zip(*batch)
    inter = [
        torch.stack([it[i] for it in intermediate_pss])
        for i in range(len(intermediate_pss[0]))
    ]
    return torch.stack(pss), torch.stack(mask), torch.stack(he_img), inter


all_items = [
    (tid, emb["pss"], emb["intermediate_pss"])
    for tid, emb in wrapper.embeddings.items()
]
n = len(all_items)

# Fold 1: First 50% of data (0-25% train, 25-50% test)
train_items1 = all_items[: int(0.25 * n)]
test_items1 = all_items[int(0.25 * n) : int(0.5 * n)]

# Fold 2: Last 50% of data (50-75% train, 75-100% test)
train_items2 = all_items[int(0.5 * n) : int(0.75 * n)]
test_items2 = all_items[int(0.75 * n) :]

print("Creating datasets...")
train_ds1 = EmbeddingsDataset(train_items1, ds=ds[0], batches_from_item=25)
test_ds1 = EmbeddingsDataset(test_items1, ds=ds[0], batches_from_item=25)
train_ds2 = EmbeddingsDataset(train_items2, ds=ds[0], batches_from_item=25)
test_ds2 = EmbeddingsDataset(test_items2, ds=ds[0], batches_from_item=25)

print(
    f"  Dataset sizes: train1={len(train_ds1)}, test1={len(test_ds1)}, train2={len(train_ds2)}, test2={len(test_ds2)}"
)

# ============================================================================
# 5. Build Decoder and Loss Function
# ============================================================================
print("\n[5/5] Building Decoder...")

decoder = CellViTDecoder(
    embed_dim=512,
    num_nuclei_classes=10,
    drop_rate=0.3,
    original_channels=19,
    patch_dropout_rate=0.0,
)
decoder.to("cuda")
initial_decoder_state = deepcopy(decoder.state_dict())

criterion = CombinedLoss(
    num_classes=10,
    ce_weight=0.4,
    dice_weight=0.4,
    ft_weight=0.2,
)

print("  Decoder and loss function ready.")


def compute_cell_oversampling_weights(dataset, gamma_s=0.5, num_classes=10):
    """
    Compute cell class oversampling weights (ignoring tissue types).

    Based on CellViT eq. (12):

    Args:
        dataset: EmbeddingsDataset instance
        gamma_s: Oversampling strength [0, 1]. 0=no oversampling, 1=max balancing
        num_classes: Number of nuclei classes INCLUDING background (so 10 total: 0-9)

    Returns:
        weights: Tensor of sampling weights for each sample
    """
    print(f"  Computing weights with gamma_s={gamma_s}...")

    N_train = len(dataset)

    # Create binary vectors for cell classes present in each sample
    # cell_presence[i, c] = 1 if class c is present in sample i
    cell_presence = torch.zeros((N_train, num_classes))

    for i in range(N_train):
        _, _, mask, _, _ = dataset[i]

        unique_classes = torch.unique(mask)
        for c in unique_classes:
            c_int = int(c.item())
            if 0 <= c_int < num_classes:  # Excluding background (class 0)
                cell_presence[i, c_int] = 1

    # total number of class presences across all samples
    # EXCLUDING background class (index 0)
    N_cell = cell_presence[:, 1:].sum().item()

    # Count how many samples contain each cell class
    cell_class_counts = cell_presence.sum(dim=0)  # Shape: (num_classes,)

    w_cell = torch.zeros(N_train)

    for i in range(N_train):
        weight_sum = 0.0

        for c in range(1, num_classes):  # Start from 1 to skip background
            if cell_presence[i, c] == 1:
                numerator = N_cell**gamma_s
                denominator = gamma_s * cell_class_counts[c] + (1 - gamma_s) * N_cell

                if denominator > 0:
                    weight_sum += numerator / denominator

        w_cell[i] = (1 - gamma_s) + gamma_s * weight_sum

    print(f"  Weight distribution:")
    percentiles = [0, 25, 50, 75, 100]
    for p in percentiles:
        val = torch.quantile(w_cell, p / 100.0).item()
        print(f"    {p}th percentile: {val:.4f}")

    return w_cell / w_cell.max()


def create_dataloaders(train_ds, test_ds, gamma_s, batch_size=128):
    """
    Create train and test dataloaders with optional oversampling.

    Args:
        train_ds: Training dataset
        test_ds: Test dataset
        gamma_s: Oversampling strength [0, 1]
        batch_size: Batch size

    Returns:
        train_loader, test_loader
    """
    if gamma_s == 0.0:
        # No oversampling - standard random sampling
        print("  Using standard random sampling (no oversampling)")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_batch,
            pin_memory=True,
        )
    else:
        # Compute oversampling weights for cell classes
        weights = compute_cell_oversampling_weights(
            train_ds, gamma_s=gamma_s, num_classes=10
        )
        sampler = WeightedRandomSampler(
            weights=weights,  # type: ignore
            num_samples=len(train_ds),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_batch,
            pin_memory=True,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    return train_loader, test_loader


all_results = []


def run_experiment(description, train_ds, test_ds, gamma_s, fold):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"  Gamma_s: {gamma_s} | Fold: {fold}")
    print("=" * 80)

    print("\nCreating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        train_ds, test_ds, gamma_s, batch_size=128
    )

    decoder.load_state_dict(initial_decoder_state)

    optimizer = optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, eta_min=1e-6
    )

    print("\nStarting training...")
    _, train_losses, val_losses, val_dices = train_loop(
        train_loader,
        test_loader,
        decoder,
        criterion,
        optimizer,
        scheduler,
        num_epochs=30,
        early_stopping_patience=30,
        num_classes=10,
        device="cuda",
        save_path=None,
        include_skip_connections=True,
    )

    best_dice = max(val_dices)
    best_epoch = val_dices.index(best_dice) + 1

    all_results.append(
        {
            "description": description,
            "gamma_s": gamma_s,
            "fold": fold,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_dices": val_dices,
            "best_dice": best_dice,
            "best_epoch": best_epoch,
        }
    )

    print(f"\n  ✓ Best Dice: {best_dice:.4f} at epoch {best_epoch}")

    del train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()


GAMMA_S_VALUES = [0.0, 0.45, 0.85]

print("\n" + "=" * 80)
print("STARTING EXPERIMENTS")
print(f"Gamma_s values: {GAMMA_S_VALUES}")
print("Folds per gamma_s: 2")
print(f"Total experiments: {len(GAMMA_S_VALUES) * 2}")
print("=" * 80)

for gamma_s in GAMMA_S_VALUES:
    run_experiment(
        f"Cell Oversampling gamma_s={gamma_s}",
        train_ds1,
        test_ds1,
        gamma_s,
        fold=1,
    )

    run_experiment(
        f"Cell Oversampling gamma_s={gamma_s}",
        train_ds2,
        test_ds2,
        gamma_s,
        fold=2,
    )

# ============================================================================
# Save and Display Results
# ============================================================================
output_file = "cell_oversampling_results.pkl"
with open(output_file, "wb") as f:
    pkl.dump(all_results, f)

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED")
print("=" * 80)
print(f"\nResults saved to: {output_file}")

# Detailed results
print("\n" + "=" * 80)
print("DETAILED RESULTS")
print("=" * 80)
print(
    f"{'Configuration':<40} {'Gamma_s':<10} {'Fold':<6} {'Best Dice':<12} {'Epoch':<6}"
)
print("-" * 80)
for result in all_results:
    print(
        f"{result['description']:<40} "
        f"{result['gamma_s']:<10.2f} "
        f"{result['fold']:<6} "
        f"{result['best_dice']:<12.4f} "
        f"{result['best_epoch']:<6}"
    )

# Average results per gamma_s
print("\n" + "=" * 80)
print("AVERAGE PERFORMANCE ACROSS FOLDS")
print("=" * 80)
print(f"{'Gamma_s':<10} {'Avg Dice':<20} {'Std Dice':<20}")
print("-" * 80)

for gamma_s in GAMMA_S_VALUES:
    matching = [r for r in all_results if r["gamma_s"] == gamma_s]
    if matching:
        avg_dice = np.mean([r["best_dice"] for r in matching])
        std_dice = np.std([r["best_dice"] for r in matching])
        print(f"{gamma_s:<10.2f} {avg_dice:.4f} ± {std_dice:.4f}")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Find best gamma_s
best_gamma_results = {}
for gamma_s in GAMMA_S_VALUES:
    matching = [r for r in all_results if r["gamma_s"] == gamma_s]
    avg_dice = np.mean([r["best_dice"] for r in matching])
    best_gamma_results[gamma_s] = avg_dice

best_gamma = max(best_gamma_results, key=best_gamma_results.get)  # type: ignore
print(f"Best gamma_s: {best_gamma} (Avg Dice: {best_gamma_results[best_gamma]:.4f})")
print(f"No oversampling (gamma_s=0.0): {best_gamma_results[0.0]:.4f}")
print(f"Improvement: {best_gamma_results[best_gamma] - best_gamma_results[0.0]:+.4f}")

print("\n" + "=" * 80)
