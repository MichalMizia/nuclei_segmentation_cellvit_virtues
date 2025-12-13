import sys
import os
import warnings

warnings.filterwarnings(
    "ignore", message=".*weights_only=False.*"
)  # ignore warning from torch for loading models
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import *

print(project_root)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from omegaconf import OmegaConf
import random
import gc
from copy import deepcopy

# project imports
from src.dataset.datasets.mm_base import build_mm_datasets, MultimodalDataset
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import (
    build_flex_dual_virtues_encoder,
)
from src.utils.marker_utils import load_marker_embeddings
from src.utils.utils import load_checkpoint_safetensors
from src.models.utils.class_weights import compute_class_weights
from src.models.wrappers.virtues_wrapper import VirtuesWrapper
from src.dataset.datasets.embeddings_dataset import EmbeddingsDataset


from src.models.cellvit_decoder import CellViTDecoder
from src.utils.metrics import CombinedLoss, calculate_dice_score, calculate_f1_score
from src.models.utils.train_loop import train_loop


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 1. Load Dataset
print("Loading Dataset...")
base_cfg = OmegaConf.load("../src/dataset/configs/base_config.yaml")
base_cfg.marker_embedding_dir = "../src/dataset/esm2_t30_150M_UR50D"
marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)
orion_subset_cfg = OmegaConf.load("../src/dataset/configs/orion_subset.yaml")
ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)
ds = build_mm_datasets(ds_cfg)

uniprot_to_name = (
    ds[0].unimodal_datasets["cycif"].get_marker_embedding_index_to_name_dict()
)
tissue_ids = ds[0].unimodal_datasets["cycif"].get_tissue_ids()
tid = tissue_ids[0]
train_len = int(0.8 * len(tissue_ids))
train_tids = tissue_ids[:train_len]
test_tids = tissue_ids[train_len:]
channels = ds[0].unimodal_datasets["cycif"].get_marker_embedding_indices(tid).numpy()
tissue = ds[0].unimodal_datasets["cycif"].get_tissue(tid).numpy()

print(f"Total tissues: {len(tissue_ids)}")
print(f"Tissue shape: {tissue.shape}")
print(f"Channels: {channels}")
print("Markers:", [uniprot_to_name[c] for c in channels])

# 2. Compute Class Weights
print("Computing Class Weights...")
train_class_weights = compute_class_weights(train_tids, ds[0], 10)
print("Train Class Weights:", train_class_weights.numpy())

# 3. Build Encoder
print("Building Encoder...")
with open(VIRTUES_WEIGHTS_PATH + "/config.pkl", "rb") as f:
    virtues_cfg = pkl.load(f)

encoder = build_flex_dual_virtues_encoder(virtues_cfg, marker_embeddings)
encoder.cuda()
weights = load_checkpoint_safetensors(
    VIRTUES_WEIGHTS_PATH + "/checkpoints/checkpoint-94575/model.safetensors"
)
# rename weights
weights_encoder = {}
for k, v in weights.items():
    if k.startswith("encoder."):
        weights_encoder[k[len("encoder.") :]] = v

encoder.load_state_dict(weights_encoder, strict=False)

# 4. Build the wrapper for precomputed embeddings
# and load the precomputed embeddings
print("Building Virtues Wrapper and Loading Embeddings...")
wrapper = VirtuesWrapper(encoder=encoder, device="cuda", autocast_dtype=torch.float16)
# path = os.path.join(DATA_DIR, "embeddings", "virtues_sp_only")
path = os.path.join("/data", "embeddings", "virtues_sp_he")
wrapper.load_embeddings(path)

for tid in wrapper.embeddings:
    print(
        f"Tissue ID: {tid} | PSS Shape: {wrapper.embeddings[tid]['pss'].shape} | Intermediate PSS Shapes: {[ip.shape for ip in wrapper.embeddings[tid]['intermediate_pss']]}"
    )
    break

# 5. Build Dataloaders
print("Building Dataloaders...")
all_items = [
    (tid, emb["pss"], emb["intermediate_pss"])
    for tid, emb in wrapper.embeddings.items()
]
n = len(all_items)
train_items1 = all_items[: int(0.25 * n)]  # only 25% for fast experimenting
test_items1 = all_items[int(0.25 * n) : int(0.5 * n)]
train_items2 = all_items[int(0.5 * n) : int(0.75 * n)]
test_items2 = all_items[int(0.75 * n) :]

train_ds1 = EmbeddingsDataset(train_items1, ds=ds[0], batches_from_item=25)
test_ds1 = EmbeddingsDataset(test_items1, ds=ds[0], batches_from_item=25)
train_ds2 = EmbeddingsDataset(train_items2, ds=ds[0], batches_from_item=25)
test_ds2 = EmbeddingsDataset(test_items2, ds=ds[0], batches_from_item=25)


def collate_batch(batch):
    _, he_img, mask, pss, intermediate_pss = zip(*batch)
    inter = [
        torch.stack([it[i] for it in intermediate_pss])
        for i in range(len(intermediate_pss[0]))
    ]
    return (
        torch.stack(pss),
        torch.stack(mask),
        torch.stack(he_img),
        inter,
    )


kwargs = {
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "collate_fn": collate_batch,
}
train_loader1 = DataLoader(train_ds1, shuffle=True, **kwargs)
test_loader1 = DataLoader(test_ds1, shuffle=False, **kwargs)
train_loader2 = DataLoader(train_ds2, shuffle=True, **kwargs)
test_loader2 = DataLoader(test_ds2, shuffle=False, **kwargs)

# 6. Build Decoder Model
print("Building Decoder Model...")
decoder = CellViTDecoder(
    embed_dim=512,
    num_nuclei_classes=10,
    drop_rate=0.3,
    original_channels=19,  # he + sp image
    patch_dropout_rate=0.0,
)
decoder.to("cuda")
initial_state_dict = deepcopy(decoder.state_dict())

# fmt: off
criterion_no_ft_no_classweights = CombinedLoss(num_classes=10, ce_weight=0.4, dice_weight=0.4, ft_weight=0)
criterion_ft_no_classweights = CombinedLoss(num_classes=10, ce_weight=0.4, dice_weight=0.4, ft_weight=0.2)
criterion_no_ft_classweights = CombinedLoss(num_classes=10, ce_weight=0.4, dice_weight=0.4, ft_weight=0, class_weights=train_class_weights.cuda())
criterion_ft_classweights = CombinedLoss(num_classes=10, ce_weight=0.4, dice_weight=0.4, ft_weight=0.2, class_weights=train_class_weights.cuda())
# fmt: on

# 4 experiments
# a) Focal Tversky set to 0, no class weights
# b) Focal Tversky set to 0, with class weights
# c) Focal Tversky set to 0.2, no class weights
# d) Focal Tversky set to 0.2, with class weights

all_results = []


def run_experiment(description, criterion, train_loader, test_loader, fold=1):
    print(f"Running: {description}")

    decoder.load_state_dict(initial_state_dict)

    optimizer = optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, eta_min=1e-6
    )

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
        save_path=None,  # type: ignore
        include_skip_connections=True,
    )
    best_dice = max(val_dices)
    best_epoch = val_dices.index(best_dice) + 1
    all_results.append(
        {
            "description": description,
            "fold": fold,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_dices": val_dices,
            "best_dice": best_dice,
            "best_epoch": best_epoch,
        }
    )

    print(f"Best Dice: {best_dice:.4f}")

    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()


print("\n" + "=" * 70)
print("Starting Experiments (4 loss configs x 2 folds = 8 total)")
print("=" * 70)
# fmt: off
run_experiment("FT=0.0, No Class Weights", criterion_no_ft_no_classweights, train_loader1, test_loader1, fold=1)
run_experiment("FT=0.0, No Class Weights", criterion_no_ft_no_classweights, train_loader2, test_loader2, fold=2)

run_experiment("FT=0.0, With Class Weights", criterion_no_ft_classweights, train_loader1, test_loader1, fold=1)
run_experiment("FT=0.0, With Class Weights", criterion_no_ft_classweights, train_loader2, test_loader2, fold=2) 

run_experiment("FT=0.2, No Class Weights", criterion_ft_no_classweights, train_loader1, test_loader1, fold=1)
run_experiment("FT=0.2, No Class Weights", criterion_ft_no_classweights, train_loader2, test_loader2, fold=2)

run_experiment("FT=0.2, With Class Weights", criterion_ft_classweights, train_loader1, test_loader1, fold=1)
run_experiment("FT=0.2, With Class Weights", criterion_ft_classweights, train_loader2, test_loader2, fold=2)
# fmt: on

output_file = "criterion_experiment_results.pkl"
with open(output_file, "wb") as f:
    pkl.dump(all_results, f)

print(f"Results saved to: {output_file}")

print("\nSummary:")
print(f"{'Loss Configuration':<35} {'Fold':<8} {'Best Dice':<12} {'Epoch':<6}")
print("-" * 70)
for result in all_results:
    print(
        f"{result['description']:<35} {result['fold']:<8} {result['best_dice']:<12.4f} {result['best_epoch']:<6}"
    )

descriptions = [
    "FT=0.0, No Class Weights",
    "FT=0.0, With Class Weights",
    "FT=0.2, No Class Weights",
    "FT=0.2, With Class Weights",
]

for config in descriptions:
    matching = [r for r in all_results if r["description"] == config]
    avg_dice = np.mean([r["best_dice"] for r in matching])
    std_dice = np.std([r["best_dice"] for r in matching])
    print(f"{config:<35} Avg Dice: {avg_dice:.4f} ± {std_dice:.4f}")
