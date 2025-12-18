import sys
import os
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

# Fix shared memory issues with multiprocessing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

print("=" * 80)
print("BOUNDARY ATTENTION EXPERIMENT")
print("=" * 80)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print("\n[1/6] Loading Dataset...")
base_cfg = OmegaConf.load(os.path.join(project_root, "src/dataset/configs/base_config.yaml"))
base_cfg.marker_embedding_dir = os.path.join(project_root, "src/dataset/esm2_t30_150M_UR50D")
marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)
orion_subset_cfg = OmegaConf.load(os.path.join(project_root, "src/dataset/configs/orion_subset.yaml"))
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
with open(os.path.join(VIRTUES_WEIGHTS_PATH, "config.pkl"), "rb") as f:
    virtues_cfg = pkl.load(f)

encoder = build_flex_dual_virtues_encoder(virtues_cfg, marker_embeddings)
encoder.cuda()
weights = load_checkpoint_safetensors(
    os.path.join(VIRTUES_WEIGHTS_PATH, "checkpoints/checkpoint-94575/model.safetensors")
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

    # Fold 1
    train_items1 = all_items[: int(0.25 * n)]
    test_items1 = all_items[int(0.25 * n) : int(0.5 * n)]

    # Fold 2
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
                pin_memory=False,
            ),
            "test": DataLoader(
                test_ds1,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_batch,
                pin_memory=False,
            ),
        },
        "fold2": {
            "train": DataLoader(
                train_ds2,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_batch,
                pin_memory=False,
            ),
            "test": DataLoader(
                test_ds2,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_batch,
                pin_memory=False,
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
print("\n[5/5] Building Decoder...")

# Decoder: Skip connections WITH boundary attention
decoder_with_boundary = CellViTDecoder(
    embed_dim=512,
    num_nuclei_classes=10,
    drop_rate=0.3,
    original_channels=19,  # HE (3) + SP (16) = 19
    patch_dropout_rate=0.0,
    boundary_attention=True
)
decoder_with_boundary.to("cuda")
initial_state_with_boundary = deepcopy(decoder_with_boundary.state_dict())

print("  Decoder WITH boundary attention: original_channels=19, boundary_attention=True")

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
    boundary_attention
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
        boundary_attention=boundary_attention
        
    )

    best_dice = max(val_dices)
    best_epoch = val_dices.index(best_dice) + 1

    all_results.append(
        {
            "description": description,
            "modality": modality,
            "use_skip": use_skip,
            "boundary_attention": boundary_attention,
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
# 8. Run All Experiments (1 config × 2 folds = 2 experiments)
# ============================================================================
print("\n" + "=" * 80)
print("STARTING EXPERIMENTS")
print("Configurations: 1 (With Boundary Attention)")
print("Folds per config: 2")
print("Total experiments: 2")
print("=" * 80)

experiments = [
    # (description, decoder, initial_state, loaders, modality, use_skip, boundary_attention)
    (
        "SP+HE WITH Boundary Attention",
        decoder_with_boundary,
        initial_state_with_boundary,
        loaders_sp_he,
        "SP+HE",
        True,
        True
    )
]

for desc, decoder, init_state, loaders, modality, use_skip, boundary_attention in experiments:
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
        boundary_attention=boundary_attention
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
        boundary_attention=boundary_attention
    )

# ============================================================================
# 9. Save and Display Results
# ============================================================================
output_file = "boundary_attention_experiment_results.pkl"
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
    f"{'Configuration':<40} {'Boundary Attn':<15} {'Fold':<6} {'Best Dice':<12} {'Epoch':<6}"
)
print("-" * 80)
for result in all_results:
    print(
        f"{result['description']:<40} "
        f"{'Yes' if result['boundary_attention'] else 'No':<15} "
        f"{result['fold']:<6} "
        f"{result['best_dice']:<12.4f} "
        f"{result['best_epoch']:<6}"
    )

# Average results per configuration
print("\n" + "=" * 80)
print("AVERAGE PERFORMANCE ACROSS FOLDS")
print("=" * 80)
print(f"{'Configuration':<40} {'Boundary Attn':<15} {'Avg Dice':<20}")
print("-" * 80)

configs = [
    ("SP+HE WITH Boundary Attention", True),
]

for desc, boundary_attention in configs:
    matching = [r for r in all_results if r["description"] == desc]
    if matching:
        avg_dice = np.mean([r["best_dice"] for r in matching])
        std_dice = np.std([r["best_dice"] for r in matching])
        print(
            f"{desc:<40} "
            f"{'Yes' if boundary_attention else 'No':<15} "
            f"{avg_dice:.4f} ± {std_dice:.4f}"
        )

# Summary
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Overall performance with boundary attention
with_boundary_results = [r for r in all_results if r["boundary_attention"]]
avg_with_boundary = np.mean([r["best_dice"] for r in with_boundary_results])
std_with_boundary = np.std([r["best_dice"] for r in with_boundary_results])
print(f"SP+HE WITH Boundary Attention: {avg_with_boundary:.4f} ± {std_with_boundary:.4f}")
print(f"Best fold performance: {max([r['best_dice'] for r in with_boundary_results]):.4f}")
print(f"Worst fold performance: {min([r['best_dice'] for r in with_boundary_results]):.4f}")

print("\n" + "=" * 80)
