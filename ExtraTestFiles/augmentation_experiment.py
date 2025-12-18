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
import kornia.augmentation as K
from copy import deepcopy

# project imports
from src.dataset.datasets.mm_base import build_mm_datasets
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import (
    build_flex_dual_virtues_encoder,
)
from src.utils.marker_utils import load_marker_embeddings
from src.utils.utils import load_checkpoint_safetensors
from src.models.wrappers.virtues_wrapper import VirtuesWrapper
from src.dataset.datasets.embeddings_dataset import ImageDataset
from src.models.cellvit_decoder import CellViTDecoder
from src.utils.metrics import CombinedLoss
from src.models.utils.train_loop import train_loop
from src.utils.metrics import calculate_dice_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

print("=" * 80)
print("CELL CLASS AUGMENTATION EXPERIMENT")
print("=" * 80)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print("\n[1/4] Loading Dataset...")
base_cfg = OmegaConf.load("../src/dataset/configs/base_config.yaml")
base_cfg.marker_embedding_dir = "../src/dataset/esm2_t30_150M_UR50D"
marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)
orion_subset_cfg = OmegaConf.load("../src/dataset/configs/orion_subset.yaml")
ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)
ds = build_mm_datasets(ds_cfg)

tissue_ids = ds[0].unimodal_datasets["cycif"].get_tissue_ids()
channels = (
    ds[0].unimodal_datasets["cycif"].get_marker_embedding_indices(tissue_ids[0]).cuda()
)
print(f"  Total tissues: {len(tissue_ids)}")

# 2. Build Encoder
print("\n[2/4] Building Encoder...")
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

# 3. Create 3-Fold Datasets
print("\n[3/4] Creating Datasets...")

n = len(tissue_ids)
tissue_ids = list(tissue_ids)
fold_size = n // 3

fold1_tids = tissue_ids[:fold_size]  # 0-33%
fold2_tids = tissue_ids[fold_size : 2 * fold_size]  # 33-66%
fold3_tids = tissue_ids[2 * fold_size :]  # 66-100%

cv_splits = [
    {
        "fold": 1,
        "train_tids": fold2_tids + fold3_tids,
        "val_tids": fold1_tids,
    },
    {
        "fold": 2,
        "train_tids": fold1_tids + fold3_tids,
        "val_tids": fold2_tids,
    },
    {
        "fold": 3,
        "train_tids": fold1_tids + fold2_tids,
        "val_tids": fold3_tids,
    },
]

print(
    f"  Fold 1: Train={len(cv_splits[0]['train_tids'])}, Val={len(cv_splits[0]['val_tids'])}"
)
print(
    f"  Fold 2: Train={len(cv_splits[1]['train_tids'])}, Val={len(cv_splits[1]['val_tids'])}"
)
print(
    f"  Fold 3: Train={len(cv_splits[2]['train_tids'])}, Val={len(cv_splits[2]['val_tids'])}"
)


def collate_batch(batch):
    _, he_img, cycif_img, mask = zip(*batch)
    return torch.stack(he_img), torch.stack(cycif_img), torch.stack(mask)


kwargs = {
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "collate_fn": collate_batch,
}
channels_list = [channels.clone().detach() for _ in range(kwargs["batch_size"])]
# 4. Build Decoder and Loss Function
print("\n[4/4] Building Decoder...")

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


geo_aug = K.AugmentationSequential(
    # K.RandomResizedCrop((120, 120), scale=(0.8, 1.0), ratio=(1, 1), p=0.5),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation90(times=(1, 3), p=0.5),
    data_keys=["input", "input", "mask"],
).cuda()

# image_aug = K.AugmentationSequential(
#     K.RandomGaussianBlur((3, 3), (0.3, 0.3), p=0.3),
#     data_keys=["input", "input"],
# ).cuda()

he_color = K.AugmentationSequential(
    K.ColorJitter(0.2, 0.15, 0.1, 0.03, p=0.5),
    # K.RandomGaussianNoise(0.0, 0.01, p=0.25),
    data_keys=["input"],
).cuda()


def apply_augmentations(
    he_img, cycif_img, mask, cycif_channel_dropout_p=0.1, device="cuda"
):
    """
    Apply geometric + appearance augmentations for HE and CyCIF using Kornia.

    he_img:      (B, H, W, 3)
    cycif_img:   (B, H, W, C)
    mask:        (B, H, W)
    """

    B, H, W, cycif_C = cycif_img.shape

    he_img = he_img.permute(0, 3, 1, 2).float().cuda()  # (B,3,H,W)
    cycif_img = cycif_img.permute(0, 3, 1, 2).float().cuda()  # (B,C,H,W)
    mask = mask.unsqueeze(1).long().cuda()  # (B,1,H,W)

    he_img, cycif_img, mask = geo_aug(he_img, cycif_img, mask)

    he_img = he_color(he_img)

    if cycif_channel_dropout_p > 0:
        dropout_mask = torch.rand((B, cycif_C), device=device) > cycif_channel_dropout_p
        dropout_mask = dropout_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        cycif_img = cycif_img * dropout_mask

    # (B,3,H,W) | (B,C,H,W) | (B,H,W)
    return he_img, cycif_img, mask.squeeze(1)


def no_augmentation(he_img, cycif_img, mask):
    """
    No augmentation - just convert format.

    Args:
        he_img:      (B, H, W, 3)
        cycif_img:   (B, H, W, C)
        mask:        (B, H, W)

    Returns:
        he_img:      (B, 3, H, W)
        cycif_img:   (B, C, H, W)
        mask:        (B, H, W)
    """
    he_img = he_img.permute(0, 3, 1, 2).float().cuda()
    cycif_img = cycif_img.permute(0, 3, 1, 2).float().cuda()
    mask = mask.long().cuda()
    return he_img, cycif_img, mask


all_results = []

early_stopping_patience = 30
num_epochs = 50
num_classes = 10


def run_experiment(
    description, train_loader, test_loader, use_augmentation=False, fold=1
):
    """
    Run a single training experiment.

    Args:
        description: Experiment description
        train_loader: Training dataloader
        test_loader: Test dataloader
        use_augmentation: Whether to apply augmentations
        fold: Fold number (1 or 2)
    """
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"  Use Augmentation: {use_augmentation} | Fold: {fold}")
    print("=" * 80)

    train_losses = []
    val_losses = []
    val_dices = []
    best_val_dice = 0.0
    early_stop_epochs = 0

    decoder.load_state_dict(initial_decoder_state)

    optimizer = optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=num_epochs // 2, eta_min=1e-6
    )

    for epoch in range(num_epochs):
        # --- Training Phase ---
        decoder.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            he_img, cycif_img, mask = batch  # (B, H, W, 3), (B, H, W, C), (B, H, W)

            if use_augmentation:
                he_img, cycif_img, mask = apply_augmentations(
                    he_img, cycif_img, mask, cycif_channel_dropout_p=0.1, device="cuda"
                )
            else:
                he_img, cycif_img, mask = no_augmentation(he_img, cycif_img, mask)

            he_list = [he_img[i] for i in range(he_img.shape[0])]
            cycif_list = [cycif_img[i] for i in range(cycif_img.shape[0])]

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, pss, intermediates = encoder.forward_list(
                        multiplex=cycif_list,
                        he=he_list,
                        channel_ids=channels_list[: len(cycif_list)],
                        return_intermediates=True,
                    )

            pss = torch.stack(pss).squeeze((1, 2)).permute(0, 3, 1, 2)
            intermediate_pss = [intermediates[layer] for layer in [4, 8, 12]]
            intermediates = [
                torch.stack(interm).squeeze((1, 2)).permute(0, 3, 1, 2)
                for interm in intermediate_pss
            ]
            img = torch.cat([he_img, cycif_img], dim=1)

            input = [img.cuda()] + [ip.cuda() for ip in intermediates] + [pss.cuda()]

            optimizer.zero_grad()
            outputs = decoder(input)
            pred_logits = outputs["nuclei_type_map"]

            loss = criterion(pred_logits, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            del pss, intermediates, he_img, cycif_img, mask, pred_logits, loss
            del img, input, outputs
            gc.collect()
            torch.cuda.empty_cache()

        avg_train_loss = running_loss / steps if steps > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        decoder.eval()
        val_running_loss = 0.0
        val_running_dice = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in test_loader:
                he_img, cycif_img, mask = batch  # (B, H, W, 3), (B, H, W, C), (B, H, W)

                he_img, cycif_img, mask = no_augmentation(he_img, cycif_img, mask)

                he_list = [he_img[i] for i in range(he_img.shape[0])]
                cycif_list = [cycif_img[i] for i in range(cycif_img.shape[0])]

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, pss, intermediates = encoder.forward_list(
                        multiplex=cycif_list,
                        he=he_list,
                        channel_ids=channels_list[: len(cycif_list)],
                        return_intermediates=True,
                    )

                pss = torch.stack(pss).squeeze((1, 2)).permute(0, 3, 1, 2)
                intermediate_pss = [intermediates[layer] for layer in [4, 8, 12]]
                intermediates = [
                    torch.stack(interm).squeeze((1, 2)).permute(0, 3, 1, 2)
                    for interm in intermediate_pss
                ]
                img = torch.cat([he_img, cycif_img], dim=1)

                input = (
                    [img.cuda()] + [ip.cuda() for ip in intermediates] + [pss.cuda()]
                )

                outputs = decoder(input)
                pred_logits = outputs["nuclei_type_map"]

                loss = criterion(pred_logits, mask)
                val_running_loss += loss.item()

                pred_mask = torch.argmax(pred_logits, dim=1)
                dice = calculate_dice_score(pred_mask, mask, num_classes)
                val_running_dice += dice

                val_steps += 1
                del mask, he_img, cycif_img, pss, intermediates
                del pred_logits, loss, input, outputs, img, pred_mask
                gc.collect()
                torch.cuda.empty_cache()

        avg_val_loss = val_running_loss / val_steps if val_steps > 0 else 0
        avg_val_dice = val_running_dice / val_steps if val_steps > 0 else 0

        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.1e} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            print(f"  --> New Best Dice: {best_val_dice:.4f}")
            early_stop_epochs = 0
        else:
            early_stop_epochs += 1
            if early_stop_epochs >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {early_stop_epochs} epochs without improvement."
                )
                break

    # Store results
    best_epoch = val_dices.index(best_val_dice) + 1
    all_results.append(
        {
            "description": description,
            "use_augmentation": use_augmentation,
            "fold": fold,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_dices": val_dices,
            "best_dice": best_val_dice,
            "best_epoch": best_epoch,
        }
    )

    print(f"\n  ✓ Best Dice: {best_val_dice:.4f} at epoch {best_epoch}")

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()


# 5. Run All Experiments
print("\n" + "=" * 80)
print("STARTING EXPERIMENTS")
print("Configurations: [With Aug, No Aug] x 2 folds = 4 experiments")
print("=" * 80)

for split in cv_splits:
    fold_num = split["fold"]

    train_ds = ImageDataset(split["train_tids"], ds=ds[0], batches_from_item=25)
    val_ds = ImageDataset(split["val_tids"], ds=ds[0], batches_from_item=25)

    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)

    print(f"\n{'='*80}")
    print(f"FOLD {fold_num}")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print("=" * 80)

    run_experiment(
        "With Augmentation",
        train_loader,
        val_loader,
        use_augmentation=True,
        fold=fold_num,
    )

    run_experiment(
        "No Augmentation",
        train_loader,
        val_loader,
        use_augmentation=False,
        fold=fold_num,
    )

# 6. Save and Display Results
output_file = "augmentation_experiment_results.pkl"
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
    f"{'Configuration':<30} {'Augmentation':<15} {'Fold':<6} {'Best Dice':<12} {'Epoch':<6}"
)
print("-" * 80)

for result in all_results:
    aug_status = "Yes" if result["use_augmentation"] else "No"
    print(
        f"{result['description']:<30} "
        f"{aug_status:<15} "
        f"{result['fold']:<6} "
        f"{result['best_dice']:<12.4f} "
        f"{result['best_epoch']:<6}"
    )

# Average results per configuration
print("\n" + "=" * 80)
print("AVERAGE PERFORMANCE ACROSS FOLDS")
print("=" * 80)
print(f"{'Configuration':<30} {'Avg Dice':<20} {'Std Dice':<20}")
print("-" * 80)

for use_aug in [False, True]:
    matching = [r for r in all_results if r["use_augmentation"] == use_aug]
    if matching:
        avg_dice = np.mean([r["best_dice"] for r in matching])
        std_dice = np.std([r["best_dice"] for r in matching])
        config_name = "With Augmentation" if use_aug else "No Augmentation"
        print(f"{config_name:<30} {avg_dice:.4f} ± {std_dice:.4f}")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

no_aug_results = [r for r in all_results if not r["use_augmentation"]]
aug_results = [r for r in all_results if r["use_augmentation"]]

if no_aug_results and aug_results:
    no_aug_avg = np.mean([r["best_dice"] for r in no_aug_results])
    aug_avg = np.mean([r["best_dice"] for r in aug_results])

    print(f"No Augmentation avg: {no_aug_avg:.4f}")
    print(f"With Augmentation avg: {aug_avg:.4f}")
    print(
        f"Improvement: {aug_avg - no_aug_avg:+.4f} ({100*(aug_avg - no_aug_avg)/no_aug_avg:+.2f}%)"
    )

print("\n" + "=" * 80)
