import os
import sys
import warnings
import random
import gc
import pickle as pkl
import numpy as np

warnings.filterwarnings("ignore", message=".*weights_only=False.*")
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import *  # noqa: F401,F403

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.dataset.datasets.mm_base import build_mm_datasets
from src.dataset.datasets.embeddings_dataset import EmbeddingsDataset
from src.utils.marker_utils import load_marker_embeddings
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import build_flex_dual_virtues_encoder
from src.utils.utils import load_checkpoint_safetensors
from src.models.wrappers.virtues_wrapper import VirtuesWrapper
from src.models.cellvit_decoder import CellViTDecoder
from src.utils.metrics import CombinedLoss
from src.models.utils.train_loop import train_loop
from src.models.utils.class_weights import compute_class_weights


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_nchw_image(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      - (H,W) -> (1,H,W)
      - (H,W,C) -> (C,H,W)
      - (C,H,W) -> unchanged
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.dim() == 2:
        return x.unsqueeze(0)

    if x.dim() == 3:
        # If last dim looks like channels (<=32), assume HWC
        if x.shape[-1] <= 32 and x.shape[0] > 32:
            return x.permute(2, 0, 1).contiguous()
        # Otherwise assume CHW
        return x

    raise ValueError(f"Unexpected image dim={x.dim()} shape={tuple(x.shape)}")


def _to_chw_feat(x: torch.Tensor) -> torch.Tensor:
    """
    For features like pss / intermediate:
      - (H,W,D) -> (D,H,W)
      - (D,H,W) -> unchanged
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.dim() != 3:
        raise ValueError(f"Unexpected feat dim={x.dim()} shape={tuple(x.shape)}")

    # If last dim looks like embedding dim (>=64), assume HWD
    if x.shape[-1] >= 64 and x.shape[0] < 64:
        return x.permute(2, 0, 1).contiguous()
    return x


def collate_batch(batch):
    # EmbeddingsDataset returns: (tid, he_img, mask, pss, intermediate_pss)
    _, he_img, mask, pss, intermediate_pss = zip(*batch)

    he_img = torch.stack([_to_nchw_image(x) for x in he_img])  # (B,C,H,W)
    mask = torch.stack([torch.as_tensor(m).long() for m in mask])  # (B,H,W)

    pss = torch.stack([_to_chw_feat(x) for x in pss])  # (B,512,h,w)

    # intermediate_pss: list per sample, each is a list of layers
    inter = []
    num_layers = len(intermediate_pss[0])
    for i in range(num_layers):
        layer_feats = [_to_chw_feat(sample[i]) for sample in intermediate_pss]
        inter.append(torch.stack(layer_feats))  # (B,512,h,w)

    return pss, mask, he_img, inter


def build_ds_and_wrapper():
    # Dataset config (same style as notebook)
    base_cfg = OmegaConf.load(os.path.join(project_root, "src/dataset/configs/base_config.yaml"))
    base_cfg.marker_embedding_dir = os.path.join(project_root, "src/dataset/esm2_t30_150M_UR50D")
    marker_embeddings = load_marker_embeddings(base_cfg.marker_embedding_dir)

    orion_subset_cfg = OmegaConf.load(os.path.join(project_root, "src/dataset/configs/orion_subset.yaml"))
    ds_cfg = OmegaConf.merge(base_cfg, orion_subset_cfg)
    ds = build_mm_datasets(ds_cfg)

    # VirTues encoder
    with open(os.path.join(VIRTUES_WEIGHTS_PATH, "config.pkl"), "rb") as f:
        virtues_cfg = pkl.load(f)

    encoder = build_flex_dual_virtues_encoder(virtues_cfg, marker_embeddings).cuda()

    weights = load_checkpoint_safetensors(
        os.path.join(VIRTUES_WEIGHTS_PATH, "checkpoints", "checkpoint-94575", "model.safetensors")
    )
    weights_encoder = {k[len("encoder.") :]: v for k, v in weights.items() if k.startswith("encoder.")}
    encoder.load_state_dict(weights_encoder, strict=False)

    wrapper = VirtuesWrapper(encoder=encoder, device="cuda", autocast_dtype=torch.float16)
    return ds, wrapper


def make_loaders_for_fold(ds, wrapper, train_tids, test_tids, batch_size=128):
    # select items by tissue id, not dict order
    all_items = [(tid, emb["pss"], emb["intermediate_pss"]) for tid, emb in wrapper.embeddings.items()]
    by_tid = {tid: (pss, inter) for (tid, pss, inter) in all_items}

    train_items = [(tid, by_tid[tid][0], by_tid[tid][1]) for tid in train_tids if tid in by_tid]
    test_items  = [(tid, by_tid[tid][0], by_tid[tid][1]) for tid in test_tids  if tid in by_tid]

    train_ds = EmbeddingsDataset(train_items, ds=ds[0], batches_from_item=25)
    test_ds  = EmbeddingsDataset(test_items,  ds=ds[0], batches_from_item=25)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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

    # derive original_channels AFTER collate makes it NCHW
    _, _, img, _ = next(iter(train_loader))
    original_channels = int(img.shape[1])
    return train_loader, test_loader, original_channels


def run_one_experiment(
    name, fold_name, train_loader, test_loader, original_channels,
    boundary_attention, use_masked_attention, num_epochs, checkpoint_dir, class_weights=None
):
    print("\n" + "=" * 90)
    print(f"[{fold_name}] {name}")
    print(f"  original_channels={original_channels} | boundary={boundary_attention} | masked={use_masked_attention} | epochs={num_epochs}")
    print("=" * 90)

    decoder = CellViTDecoder(
        embed_dim=512,
        num_nuclei_classes=10,
        drop_rate=0.3,
        original_channels=original_channels,
        patch_dropout_rate=0.0,
        boundary_attention=boundary_attention,
        use_masked_attention=use_masked_attention,
    ).to("cuda")

    optimizer = optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    # Loss: keep same, optionally inject class weights if your CombinedLoss supports it
    try:
        criterion = CombinedLoss(num_classes=10, class_weights=class_weights).to("cuda")
    except TypeError:
        criterion = CombinedLoss(num_classes=10).to("cuda")

    decoder, train_losses, val_losses, val_dices = train_loop(
        train_loader,
        test_loader,
        decoder,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        early_stopping_patience=30,
        num_classes=10,
        device="cuda",
        save_path=os.path.join(checkpoint_dir, f"best_{fold_name}_{name.replace(' ', '_')}.pth"),
        include_skip_connections=True,
        boundary_attention=boundary_attention,
        use_masked_attention=use_masked_attention,
    )

    best_dice = float(np.max(val_dices))
    best_epoch = int(np.argmax(val_dices) + 1)

    np.savez(
        os.path.join(checkpoint_dir, f"metrics_{fold_name}_{name.replace(' ', '_')}.npz"),
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        val_dices=np.array(val_dices),
    )

    torch.cuda.empty_cache()
    gc.collect()

    print(f"  ✓ Best Dice: {best_dice:.4f} @ epoch {best_epoch}")
    return {
        "fold": fold_name,
        "name": name,
        "boundary_attention": boundary_attention,
        "masked_attention": use_masked_attention,
        "original_channels": original_channels,
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "val_dices": list(map(float, val_dices)),
    }


def main():
    set_seed(42)

    num_epochs = 100  # 2 folds × 2 experiments × 100 epochs = 400 epoch-runs
    checkpoint_dir = os.path.join(project_root, "notebooks", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    folds_path = os.path.join(checkpoint_dir, "two_folds_seed42_test20.pkl")
    with open(folds_path, "rb") as f:
        folds = pkl.load(f)

    ds, wrapper = build_ds_and_wrapper()

    # SP ONLY embeddings
    emb_path = os.path.join("/data", "embeddings", "virtues_sp_only")
    wrapper.load_embeddings(emb_path)
    print(f"Loaded embeddings: {emb_path} ({len(wrapper.embeddings)} tissues)")

    results = []

    experiments = [
        ("SP only + boundary", True,  False)
    ]

    for fold_name in ["fold1", "fold2"]:
        train_tids = folds[fold_name]["train_tids"]
        test_tids  = folds[fold_name]["test_tids"]

        # class weights computed from TRAIN tids (good practice)
        try:
            class_weights = compute_class_weights(train_tids, ds[0], 10)
            print(f"[{fold_name}] class_weights = {class_weights}")

            #  FIX: move class weights to CUDA (CrossEntropy expects same device as logits)
            if isinstance(class_weights, np.ndarray):
                class_weights = torch.from_numpy(class_weights)
            class_weights = class_weights.to(device="cuda", dtype=torch.float32)
        except Exception as e:
            print(f"[{fold_name}] WARNING: class weights compute failed: {e}")
            class_weights = None

        train_loader, test_loader, original_channels = make_loaders_for_fold(
            ds, wrapper, train_tids, test_tids
        )

        for name, boundary_attention, masked_attention in experiments:
            res = run_one_experiment(
                name=name,
                fold_name=fold_name,
                train_loader=train_loader,
                test_loader=test_loader,
                original_channels=original_channels,
                boundary_attention=boundary_attention,
                use_masked_attention=masked_attention,
                num_epochs=num_epochs,
                checkpoint_dir=checkpoint_dir,
                class_weights=class_weights,
            )
            results.append(res)

    out_path = os.path.join(checkpoint_dir, "fileA_results_sp_only_vs_boundary.pkl")
    with open(out_path, "wb") as f:
        pkl.dump(results, f)

    print("\n" + "=" * 90)
    print("FILE A SUMMARY")
    for r in results:
        print(f"{r['fold']:<6} | {r['name']:<22} | best={r['best_dice']:.4f} @ {r['best_epoch']}")
    print(f"\nSaved -> {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
