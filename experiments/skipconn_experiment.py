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
from torch.utils.data import DataLoader
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
from src.models.utils.class_weights import compute_class_weights
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
print("SKIP CONNECTIONS & INPUT MODALITY EXPERIMENT")
print("=" * 80)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print("\n[1/6] Loading Dataset...")
base_cfg = OmegaConf.load("../src/dataset/configs/base_config.yaml")
base_cfg.marker_embedding_dir = "../src/dataset/esm2_t30_150M_UR50D"
marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)
orion_subset_cfg = OmegaConf.load("../src/dataset/configs/orion_subset.yaml")
ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)
ds = build_mm_datasets(ds_cfg)

tissue_ids = ds[0].unimodal_datasets["cycif"].get_tissue_ids()
train_len = int(0.8 * len(tissue_ids))
train_tids = tissue_ids[:train_len]

print(f"  Total tissues: {len(tissue_ids)}")
print(f"  Training tissues: {len(train_tids)}")

# ============================================================================
# 3. Build Encoder
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
# 4. Build Wrappers and Load Embeddings
# ============================================================================
print("\n[3/5] Loading Precomputed Embeddings...")

# Wrapper 1: SP + HE
wrapper_sp_he = VirtuesWrapper(
    encoder=encoder, device="cuda", autocast_dtype=torch.float16
)
path_sp_he = os.path.join("/data", "embeddings", "virtues_sp_he")
wrapper_sp_he.load_embeddings(path_sp_he)
print(f"  SP+HE embeddings: {len(wrapper_sp_he.embeddings)} tissues")

# Wrapper 2: SP only
wrapper_sp_only = VirtuesWrapper(
    encoder=encoder, device="cuda", autocast_dtype=torch.float16
)
path_sp_only = os.path.join("/data", "embeddings", "virtues_sp_only")
wrapper_sp_only.load_embeddings(path_sp_only)
print(f"  SP-only embeddings: {len(wrapper_sp_only.embeddings)} tissues")

# ============================================================================
# 5. Build Dataloaders
# ============================================================================
print("\n[4/5] Building Dataloaders...")


def collate_batch(batch):
    _, he_img, mask, pss, intermediate_pss = zip(*batch)
    inter = [
        torch.stack([it[i] for it in intermediate_pss])
        for i in range(len(intermediate_pss[0]))
    ]
    return torch.stack(pss), torch.stack(mask), torch.stack(he_img), inter


def create_dataloaders(wrapper, batch_size=128):
    """Create train/test loaders for 2 folds"""
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

    train_ds1 = EmbeddingsDataset(train_items1, ds=ds[0], batches_from_item=25)
    test_ds1 = EmbeddingsDataset(test_items1, ds=ds[0], batches_from_item=25)
    train_ds2 = EmbeddingsDataset(train_items2, ds=ds[0], batches_from_item=25)
    test_ds2 = EmbeddingsDataset(test_items2, ds=ds[0], batches_from_item=25)

    loaders = {
        "fold1": {
            "train": DataLoader(
                train_ds1,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_batch,
            ),
            "test": DataLoader(
                test_ds1,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_batch,
            ),
        },
        "fold2": {
            "train": DataLoader(
                train_ds2,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_batch,
            ),
            "test": DataLoader(
                test_ds2,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_batch,
            ),
        },
    }

    return loaders


loaders_sp_he = create_dataloaders(wrapper_sp_he)
loaders_sp_only = create_dataloaders(wrapper_sp_only)
print("  Dataloaders created for both modalities")

# ============================================================================
# 6. Build Decoders and Loss Functions
# ============================================================================
print("\n[5/5] Building Decoders...")

# Decoder 1: With skip connections (original_channels=19 for SP+HE)
decoder_with_skip = CellViTDecoder(
    embed_dim=512,
    num_nuclei_classes=10,
    drop_rate=0.3,
    original_channels=19,  # HE (3) + SP (16) = 19
    patch_dropout_rate=0.0,
)
decoder_with_skip.to("cuda")
initial_state_with_skip = deepcopy(decoder_with_skip.state_dict())

# Decoder 2: Without skip connections (original_channels=None, defaults to embed_dim)
decoder_no_skip = CellViTDecoder(
    embed_dim=512,
    num_nuclei_classes=10,
    drop_rate=0.3,
    original_channels=None,  # No skip connections
    patch_dropout_rate=0.0,
)
decoder_no_skip.to("cuda")
initial_state_no_skip = deepcopy(decoder_no_skip.state_dict())

print("  Decoder with skip connections: original_channels=19")
print("  Decoder without skip connections: original_channels=None")

criterion = CombinedLoss(
    num_classes=10,
    ce_weight=0.4,
    dice_weight=0.4,
    ft_weight=0.2,
)

# ============================================================================
# 7. Experiment Runner
# ============================================================================
all_results = []


def run_experiment(
    description,
    decoder,
    initial_state,
    train_loader,
    test_loader,
    modality,
    use_skip,
    fold,
):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"  Modality: {modality} | Skip Connections: {use_skip} | Fold: {fold}")
    print("=" * 80)

    # Reset decoder to initial weights
    decoder.load_state_dict(initial_state)

    # Create fresh optimizer and scheduler
    optimizer = optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, eta_min=1e-6
    )

    # Train
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
        include_skip_connections=use_skip,
    )

    best_dice = max(val_dices)
    best_epoch = val_dices.index(best_dice) + 1

    all_results.append(
        {
            "description": description,
            "modality": modality,
            "use_skip": use_skip,
            "fold": fold,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_dices": val_dices,
            "best_dice": best_dice,
            "best_epoch": best_epoch,
        }
    )

    print(f"\n  ✓ Best Dice: {best_dice:.4f} at epoch {best_epoch}")

    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# 8. Run All Experiments (4 configs × 2 folds = 8 experiments)
# ============================================================================
print("\n" + "=" * 80)
print("STARTING EXPERIMENTS")
print("Configurations: 4 (SP+HE Skip, SP+HE No Skip, SP Skip, SP No Skip)")
print("Folds per config: 2")
print("Total experiments: 8")
print("=" * 80)

experiments = [
    # (description, decoder, initial_state, loaders, modality, use_skip)
    (
        "SP+HE with Skip Connections",
        decoder_with_skip,
        initial_state_with_skip,
        loaders_sp_he,
        "SP+HE",
        True,
    ),
    (
        "SP+HE without Skip Connections",
        decoder_no_skip,
        initial_state_no_skip,
        loaders_sp_he,
        "SP+HE",
        False,
    ),
    (
        "SP-only with Skip Connections",
        decoder_with_skip,
        initial_state_with_skip,
        loaders_sp_only,
        "SP-only",
        True,
    ),
    (
        "SP-only without Skip Connections",
        decoder_no_skip,
        initial_state_no_skip,
        loaders_sp_only,
        "SP-only",
        False,
    ),
]

for desc, decoder, init_state, loaders, modality, use_skip in experiments:
    # Run fold 1
    run_experiment(
        desc,
        decoder,
        init_state,
        loaders["fold1"]["train"],
        loaders["fold1"]["test"],
        modality,
        use_skip,
        fold=1,
    )

    # Run fold 2
    run_experiment(
        desc,
        decoder,
        init_state,
        loaders["fold2"]["train"],
        loaders["fold2"]["test"],
        modality,
        use_skip,
        fold=2,
    )

# ============================================================================
# 9. Save and Display Results
# ============================================================================
output_file = "skipconn_experiment_results.pkl"
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
    f"{'Configuration':<40} {'Modality':<10} {'Skip':<6} {'Fold':<6} {'Best Dice':<12} {'Epoch':<6}"
)
print("-" * 80)
for result in all_results:
    print(
        f"{result['description']:<40} "
        f"{result['modality']:<10} "
        f"{'Yes' if result['use_skip'] else 'No':<6} "
        f"{result['fold']:<6} "
        f"{result['best_dice']:<12.4f} "
        f"{result['best_epoch']:<6}"
    )

# Average results per configuration
print("\n" + "=" * 80)
print("AVERAGE PERFORMANCE ACROSS FOLDS")
print("=" * 80)
print(f"{'Configuration':<40} {'Modality':<10} {'Skip':<6} {'Avg Dice':<20}")
print("-" * 80)

configs = [
    ("SP+HE with Skip Connections", "SP+HE", True),
    ("SP+HE without Skip Connections", "SP+HE", False),
    ("SP-only with Skip Connections", "SP-only", True),
    ("SP-only without Skip Connections", "SP-only", False),
]

for desc, modality, use_skip in configs:
    matching = [r for r in all_results if r["description"] == desc]
    if matching:
        avg_dice = np.mean([r["best_dice"] for r in matching])
        std_dice = np.std([r["best_dice"] for r in matching])
        print(
            f"{desc:<40} "
            f"{modality:<10} "
            f"{'Yes' if use_skip else 'No':<6} "
            f"{avg_dice:.4f} ± {std_dice:.4f}"
        )

# Summary comparison
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Compare skip vs no skip
skip_results = [r for r in all_results if r["use_skip"]]
no_skip_results = [r for r in all_results if not r["use_skip"]]
avg_skip = np.mean([r["best_dice"] for r in skip_results])
avg_no_skip = np.mean([r["best_dice"] for r in no_skip_results])
print(f"With Skip Connections:    {avg_skip:.4f}")
print(f"Without Skip Connections: {avg_no_skip:.4f}")
print(f"Difference:               {avg_skip - avg_no_skip:+.4f}")

# Compare SP+HE vs SP-only
sp_he_results = [r for r in all_results if r["modality"] == "SP+HE"]
sp_only_results = [r for r in all_results if r["modality"] == "SP-only"]
avg_sp_he = np.mean([r["best_dice"] for r in sp_he_results])
avg_sp_only = np.mean([r["best_dice"] for r in sp_only_results])
print(f"\nSP+HE:    {avg_sp_he:.4f}")
print(f"SP-only:  {avg_sp_only:.4f}")
print(f"Difference: {avg_sp_he - avg_sp_only:+.4f}")

print("\n" + "=" * 80)
