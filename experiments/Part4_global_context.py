import sys
import os
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl
from omegaconf import OmegaConf
import importlib
print("\n" + "="*60)
print("TRAINING - Global COntext")
print("\n" + "="*60)

# --- Environment & Path Setup ---
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

# Set project root (assuming script is run from project root)
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import *
from src.dataset.datasets.mm_base import build_mm_datasets
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import build_flex_dual_virtues_encoder
from src.utils.marker_utils import load_marker_embeddings
from src.utils.utils import load_checkpoint_safetensors
from src.models.wrappers.virtues_wrapper import VirtuesWrapper
from src.dataset.datasets.embeddings_dataset import EmbeddingsDataset
from src.models.cellvit_decoder import CellViTDecoder
from src.utils.metrics import CombinedLoss
from src.models.utils.train_loop import train_loop
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These two lines are critical for absolute reproducibility 
    # but can slightly slow down training.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call it immediately after imports
set_seed(42)

# --- Configuration ---
num_classes = 10
checkpoint_dir = os.path.join(project_root, "notebooks", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
device = "cuda"

# Path Fixes
base_cfg = OmegaConf.load("src/dataset/configs/base_config.yaml")
base_cfg.marker_embedding_dir = "src/dataset/esm2_t30_150M_UR50D"
marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)
orion_subset_cfg = OmegaConf.load("src/dataset/configs/orion_subset.yaml")
ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)

# --- Dataset & Sequential Split ---
ds = build_mm_datasets(ds_cfg)
tissue_ids = ds[0].unimodal_datasets["cycif"].get_tissue_ids()

train_len = int(0.8 * len(tissue_ids))
train_tids = tissue_ids[:train_len]
test_tids = tissue_ids[train_len:]

# --- Load VirTues Encoder & Weights ---
with open(os.path.join(VIRTUES_WEIGHTS_PATH, "config.pkl"), "rb") as f:
    virtues_cfg = pkl.load(f)

encoder = build_flex_dual_virtues_encoder(virtues_cfg, marker_embeddings)
encoder.cuda()

weights_path = os.path.join(VIRTUES_WEIGHTS_PATH, "checkpoints", "checkpoint-94575", "model.safetensors")
weights = load_checkpoint_safetensors(weights_path)
weights_encoder = {k[len("encoder."):]: v for k, v in weights.items() if k.startswith("encoder.")}
encoder.load_state_dict(weights_encoder, strict=False)

wrapper = VirtuesWrapper(encoder=encoder, device=device, autocast_dtype=torch.float16)

# --- Load Precomputed Embeddings (SP ONLY) ---
path = os.path.join("/data", "embeddings", "virtues_sp_only")
wrapper.load_embeddings(path)

# --- Split Loaded Items ---
all_items = [(tid, emb["pss"], emb["intermediate_pss"]) for tid, emb in wrapper.embeddings.items()]
n = len(all_items)
train_items = all_items[: int(0.8 * n)]
test_items  = all_items[int(0.8 * n):]

train_ds = EmbeddingsDataset(train_items, ds=ds[0], batches_from_item=25)
test_ds  = EmbeddingsDataset(test_items,  ds=ds[0], batches_from_item=25)

def collate_batch(batch):
    _, he_img, mask, pss, intermediate_pss = zip(*batch)
    inter = [torch.stack([it[i] for it in intermediate_pss]) for i in range(len(intermediate_pss[0]))]
    return torch.stack(pss), torch.stack(mask), torch.stack(he_img), inter

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_batch, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_batch, pin_memory=True)

# --- Model / Optimizer / Criterion ---
decoder = CellViTDecoder(
    embed_dim=512,  
    num_nuclei_classes=10,
    drop_rate=0.3,
    original_channels=19, 
    patch_dropout_rate=0.0,
    boundary_attention=True,
    use_global_context=True
)
decoder.to(device)

optimizer = optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-2)
criterion = CombinedLoss(num_classes=num_classes)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

# --- Training Loop ---
print("="*60)
print(f"REPLICATION RUN: Sequential Split | No Class Weights")
print(f"Train tissues: {len(train_tids)} | Test tissues: {len(test_tids)}")
print("="*60)

decoder, train_losses, val_losses, val_dices = train_loop(
    train_loader,
    test_loader,
    decoder,
    criterion,
    optimizer,
    scheduler,
    num_epochs=86,
    early_stopping_patience=30,
    num_classes=10,
    device=device,
    save_path=os.path.join(checkpoint_dir, "_best_global_context.pth"),
    include_skip_connections=True,
    boundary_attention=True,
    use_masked_attention=False
)

# --- Final Metrics and Save ---
avg_dice = np.mean(val_dices)
max_dice = np.max(val_dices)

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY - Global Context")
print(f"Max Validation Dice: {max_dice:.4f}")
print(f"Avg Validation Dice: {avg_dice:.4f}")
print(f"Metrics saved to: {os.path.join(checkpoint_dir, '_training_global_context.npz')}")
print("="*60)

np.savez(
    os.path.join(checkpoint_dir, "_training_global_context.npz"),
    train_losses=np.array(train_losses),
    val_losses=np.array(val_losses),
    val_dices=np.array(val_dices),
)