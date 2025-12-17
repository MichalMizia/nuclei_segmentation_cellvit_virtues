# experiments/boundary_masked_fullcontext.py

import sys
import os
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# ============================================================
# Project root (script is run from repo root)
# ============================================================
project_root = os.getcwd()
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

# =========================
# Project imports
# =========================
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


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

print("=" * 80)
print("FULL-CONTEXT BOUNDARY vs MASKED ATTENTION EXPERIMENT")
print("=" * 80)

# =========================
# 1. Load Dataset
# =========================
print("\n[1/5] Loading Dataset...")

base_cfg = OmegaConf.load("src/dataset/configs/base_config.yaml")
base_cfg.marker_embedding_dir = "src/dataset/esm2_t30_150M_UR50D"
marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)

orion_subset_cfg = OmegaConf.load("src/dataset/configs/orion_subset.yaml")
ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)
ds = build_mm_datasets(ds_cfg)

# =========================
# 2. Build Encoder
# =========================
print("\n[2/5] Building Encoder...")

with open(os.path.join(VIRTUES_WEIGHTS_PATH, "config.pkl"), "rb") as f:
    virtues_cfg = pkl.load(f)

encoder = build_flex_dual_virtues_encoder(
    virtues_cfg, marker_embeddings
).cuda()

weights = load_checkpoint_safetensors(
    os.path.join(
        VIRTUES_WEIGHTS_PATH,
        "checkpoints",
        "checkpoint-94575",
        "model.safetensors",
    )
)

encoder.load_state_dict(
    {k[len("encoder.") :]: v for k, v in weights.items() if k.startswith("encoder.")},
    strict=False,
)

print("  Encoder loaded")

# =========================
# 3. Load Precomputed Embeddings (FULL DATA)
# =========================
print("\n[3/5] Loading Embeddings...")

from src.models.wrappers.virtues_wrapper import VirtuesWrapper
wrapper = VirtuesWrapper(encoder=encoder, device="cuda", autocast_dtype=torch.float16)

from_scratch=False
emb_path = os.path.join("/data", "embeddings", "virtues_sp_only")
if from_scratch:
    # SP Only, include_he_data=False by default
    wrapper.process_dataset(ds[0], return_intermediates=True, intermediate_layers=[4,8,12])
    wrapper.save_embeddings(emb_path)
else:
    wrapper.load_embeddings(emb_path)


print(f"  Loaded {len(wrapper.embeddings)} tissues")

# =========================
# 4. Build Dataloaders (FULL DATA)
# =========================
print("\n[4/5] Building Dataloaders...")

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
train_items = all_items[: int(0.8 * n)]
test_items  = all_items[int(0.8 * n) :]

train_ds = EmbeddingsDataset(train_items, ds=ds[0], batches_from_item=25)
test_ds  = EmbeddingsDataset(test_items,  ds=ds[0], batches_from_item=25)

train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True, collate_fn=collate_batch
)
test_loader = DataLoader(
    test_ds, batch_size=128, shuffle=False, collate_fn=collate_batch
)

# =========================
# 5. Loss
# =========================
criterion = CombinedLoss(
    num_classes=10,
    ce_weight=0.4,
    dice_weight=0.4,
    ft_weight=0.2,
)

# =========================
# 6. Experiment Runner
# =========================
results = []

def run_experiment(name, boundary_att, masked_att):
    print("\n" + "=" * 80)
    print(f"Running: {name}")
    print(f"  Boundary Attention: {boundary_att}")
    print(f"  Masked Attention:   {masked_att}")
    print("=" * 80)

    decoder = CellViTDecoder(
        embed_dim=512,
        num_nuclei_classes=10,
        drop_rate=0.3,
        original_channels=19,
        patch_dropout_rate=0.0,
        boundary_attention=boundary_att,
        use_feature_gating=False,
        use_masked_attention=masked_att,
    ).cuda()

    optimizer = optim.AdamW(
        decoder.parameters(), lr=5e-4, weight_decay=1e-2
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, eta_min=1e-6
    )

    _, _, _, val_dices = train_loop(
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
        boundary_attention=boundary_att,
        use_feature_gating=False,
        use_masked_attention=masked_att,
    )

    best_dice = max(val_dices)
    best_epoch = int(np.argmax(val_dices) + 1)

    results.append(
        {
            "name": name,
            "boundary_attention": boundary_att,
            "masked_attention": masked_att,
            "best_dice": best_dice,
            "best_epoch": best_epoch,
            "val_dices": val_dices,
        }
    )

    print(f"✓ Best Dice: {best_dice:.4f} at epoch {best_epoch}")

    torch.cuda.empty_cache()
    gc.collect()


# =========================
# 7. Run 4 Experiments
# =========================
run_experiment("Baseline (No Boundary, No Masked)", False, False)
run_experiment("Boundary Attention Only",           True,  False)
run_experiment("Masked Attention Only",             False, True)
run_experiment("Boundary + Masked Attention",       True,  True)

# =========================
# 8. Save Results
# =========================
out_path = "boundary_masked_fullcontext_results.pkl"
with open(out_path, "wb") as f:
    pkl.dump(results, f)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for r in results:
    print(
        f"{r['name']:<35} | "
        f"Best Dice: {r['best_dice']:.4f} | "
        f"Epoch: {r['best_epoch']}"
    )

print(f"\nResults saved to: {out_path}")
print("=" * 80)
