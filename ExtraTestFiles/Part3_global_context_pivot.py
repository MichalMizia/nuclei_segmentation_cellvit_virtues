

# === Cell 2: Imports / path setup ===
import sys
import os
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*")  # ignore warning from torch for loading models
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

# Resolve project root robustly (works when running from repo root).
# This file lives in `notebooks/`, so its parent is the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
	sys.path.insert(0, project_root)

from config import *  # noqa: F401,F403

print(project_root)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle as pkl
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
import importlib


# === Cell 3: Project imports ===
from src.dataset.datasets.mm_base import build_mm_datasets, MultimodalDataset
from src.dataset.datasets.embeddings_dataset import EmbeddingsDataset  # precomputed embeddings ds
from src.utils.plot_utils import visualize_multichannel_image
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import build_flex_dual_virtues_encoder
from src.utils.marker_utils import load_marker_embeddings
from src.utils.utils import load_checkpoint_safetensors


# === Cell 4: Decoder/loss imports + globals ===
from src.models.cellvit_decoder import CellViTDecoder
import src.utils.metrics as m

importlib.reload(m)
from src.utils.metrics import (
	CombinedLoss,
	calculate_dice_score,
	calculate_f1_score,
	calculate_panoptic_quality,
	calculate_iou_score,
)
from src.models.utils.train_loop import train_loop

num_classes = 10
checkpoint_dir = os.path.join(project_root, "notebooks", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
device = "cuda"


# === Cell 5: Dataset config ===
base_cfg = OmegaConf.load(os.path.join(project_root, "src/dataset/configs/base_config.yaml"))
base_cfg.marker_embedding_dir = os.path.join(project_root, "src/dataset/esm2_t30_150M_UR50D")
marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)
orion_subset_cfg = OmegaConf.load(os.path.join(project_root, "src/dataset/configs/orion_subset.yaml"))
ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)


# === Cell 6: Build dataset ===
ds = build_mm_datasets(ds_cfg)


# === Cell 8: Basic dataset sanity checks (visualization omitted) ===
uniprot_to_name = ds[0].unimodal_datasets["cycif"].get_marker_embedding_index_to_name_dict()

tissue_ids = ds[0].unimodal_datasets["cycif"].get_tissue_ids()
train_len = int(0.8 * len(tissue_ids))
train_tids = tissue_ids[:train_len]
test_tids = tissue_ids[train_len:]

channels = np.array(ds[0].unimodal_datasets["cycif"].get_marker_embedding_indices(tissue_ids[0]))
tissue = np.array(ds[0].unimodal_datasets["cycif"].get_tissue(tissue_ids[0]))

print(f"Tissue shape: {tissue.shape}")
print(f"SP: {list(map(lambda x: uniprot_to_name[x], channels))}")


# === Cell 9: Class weights ===
from src.models.utils.class_weights import compute_class_weights

train_class_weights = compute_class_weights(train_tids, ds[0], 10)
print(train_class_weights)


# === Cell 10: Load VirTues encoder + weights ===
with open(os.path.join(VIRTUES_WEIGHTS_PATH, "config.pkl"), "rb") as f:
	virtues_cfg = pkl.load(f)

encoder = build_flex_dual_virtues_encoder(virtues_cfg, marker_embeddings)
encoder.cuda()

weights = load_checkpoint_safetensors(
	os.path.join(VIRTUES_WEIGHTS_PATH, "checkpoints/checkpoint-94575/model.safetensors")
)

# rename weights
weights_encoder = {}
for k, v in weights.items():
	if k.startswith("encoder."):
		weights_encoder[k[len("encoder.") :]] = v

encoder.load_state_dict(weights_encoder, strict=False)


# === Cell 11: Wrapper ===
from src.models.wrappers.virtues_wrapper import VirtuesWrapper

wrapper = VirtuesWrapper(encoder=encoder, device="cuda", autocast_dtype=torch.float16)


# === Cell 14: Load precomputed embeddings (or compute) ===
from_scratch = False
path = os.path.join("/data", "embeddings", "virtues_sp_only")
if from_scratch:
	# SP Only, include_he_data=False by default
	wrapper.process_dataset(ds[0], return_intermediates=True, intermediate_layers=[4, 8, 12])
	wrapper.save_embeddings(path)
else:
	wrapper.load_embeddings(path)


# === Cell 15: Sanity marker ===
print("OK")


# === Cell 16: Print one embedding shape ===
for key, emb in wrapper.embeddings.items():
	print(key, emb["pss"].shape, emb["intermediate_pss"][0].shape, len(emb["intermediate_pss"]))
	break


# === Cell 17: Train/test loaders from embeddings ===
all_items = [(tid, emb["pss"], emb["intermediate_pss"]) for tid, emb in wrapper.embeddings.items()]
n = len(all_items)
train_items = all_items[: int(0.8 * n)]
test_items = all_items[int(0.8 * n) :]

# 25x25 batches from 375x375 => each is 15x15 patches => 120x120 px (120=15xpatch_size)
train_ds = EmbeddingsDataset(train_items, ds=ds[0], batches_from_item=25)
test_ds = EmbeddingsDataset(test_items, ds=ds[0], batches_from_item=25)


def collate_batch(batch):
	_, he_img, mask, pss, intermediate_pss = zip(*batch)
	inter = [torch.stack([it[i] for it in intermediate_pss]) for i in range(len(intermediate_pss[0]))]
	return torch.stack(pss), torch.stack(mask), torch.stack(he_img), inter  # pss: (B, h, w, D), mask: (B, H, W)


train_loader = DataLoader(
	train_ds,
	batch_size=128,
	shuffle=True,
	num_workers=0,
	collate_fn=collate_batch,
	pin_memory=True,
)
test_loader = DataLoader(
	test_ds,
	batch_size=128,
	shuffle=False,
	num_workers=0,
	collate_fn=collate_batch,
	pin_memory=True,
)


# === Cell 18: (Duplicate imports in notebook; kept for fidelity) ===
from src.models.cellvit_decoder import CellViTDecoder
from src.models.utils.train_loop import train_loop


# === Cell 19: Model/optimizer/criterion ===
decoder = CellViTDecoder(
	embed_dim=512,
	num_nuclei_classes=10,
	drop_rate=0.3,
	original_channels=19,  # he + sp image
	patch_dropout_rate=0.0,
	boundary_attention=False,
	use_masked_attention=False,
	use_global_context=True
)
decoder.to(device)

optimizer = optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-2)
criterion = CombinedLoss(num_classes=num_classes)  # CE + Dice + Focal-Tversky
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)


# === Cell 20: Train ===
decoder, train_losses, val_losses, val_dices = train_loop(  # printing clears every 10 epochs to not freeze jupyter
	train_loader,
	test_loader,
	decoder,
	criterion,
	optimizer,
	scheduler,
	num_epochs=100,
	early_stopping_patience=30,
	num_classes=10,
	device="cuda",
	save_path=os.path.join(checkpoint_dir, "best_cellvit_global_context.pth"),
	include_skip_connections=True,
	boundary_attention=False,
	use_masked_attention=False
)


# === Cell 21: Save metrics ===
np.savez(
	os.path.join(checkpoint_dir, "training_metrics_global_context.npz"),
	train_losses=np.array(train_losses),
	val_losses=np.array(val_losses),
	val_dices=np.array(val_dices),
)


# (Cells 22+25 were visualization; intentionally omitted.)

