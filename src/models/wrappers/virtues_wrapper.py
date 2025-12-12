import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import (
    FlexDualVirTuesEncoder,
)
from src.dataset.datasets.mm_base import MultimodalDataset


class VirtuesWrapper:
    """
    Wraps an encoder and dataset utilities to process batched tissue tiles.

    Args:
        encoder: model with a forward_list(multiplex, he, channel_ids) or forward(...)
        device: torch device string, e.g. "cuda" or "cpu"
        patch_size: size of square patch (P)
        autocast_dtype: optional torch.dtype for autocast on CUDA (e.g., torch.float16)
    """

    def __init__(
        self,
        encoder: FlexDualVirTuesEncoder,
        device: str = "cuda",
        autocast_dtype: torch.dtype = torch.float16,
    ):
        self.encoder = encoder
        self.device = device
        self.autocast_dtype = autocast_dtype

        self.encoder.to(self.device)
        self.encoder.eval()

        self.embeddings = {}
        self.masks = {}

    @torch.no_grad()
    def process_dataset(
        self,
        ds: MultimodalDataset,
        unimodal_ds: str = "cycif",
        include_he_data: bool = False,
    ):
        """"""
        patch_size = 8  # virtues patch size
        tissue_ids = ds.unimodal_datasets[unimodal_ds].get_tissue_ids()
        tid = tissue_ids[0]
        C, H, W = ds.unimodal_datasets[unimodal_ds].get_tissue(tid).shape
        batch_size = (H // patch_size) * (W // patch_size)  # number of patches
        channels = ds.unimodal_datasets[unimodal_ds].get_marker_embedding_indices(tid)
        channels = channels.to(self.device)
        channels_list = [channels.clone().detach() for _ in range(batch_size)]

        print("=" * 40)
        print(f"Dataset '{unimodal_ds}' with {len(tissue_ids)} tissues".center(40))
        print(f"No. of patches {batch_size}".center(40))
        print(f"Channels per patch {C}".center(40))
        print(f"Final PSS shape ({batch_size},{512})".center(40))
        print("=" * 40)

        for tid in tissue_ids:
            tissue = ds.unimodal_datasets["cycif"].get_tissue(tid)  # (C, H, W)
            tissue_list = self._image_to_patches(
                tissue, patch_size=patch_size
            )  # list of (C, P, P)
            tissue_list = [t.to(self.device) for t in tissue_list]
            if include_he_data:
                he = ds.unimodal_datasets["he"].get_tissue(tid)  # (3, H, W)
                he_list = self._image_to_patches(
                    he, patch_size=patch_size
                )  # list of (3, P, P)
                he_list = [h.to(self.device) for h in he_list]
            else:
                he_list = [None] * batch_size

            pss = []
            max_len = 625
            for i in tqdm(range(0, batch_size, max_len)):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # we cant just pass the whole list at once because cuda complains
                    _, pss_temp = self.encoder.forward_list(
                        multiplex=tissue_list[i : min(i + max_len, batch_size)],
                        he=he_list[i : min(i + max_len, len(he_list))],
                        channel_ids=channels_list[i : min(i + max_len, batch_size)],
                    )
                pss.extend(pss_temp)

            pss = torch.stack(pss).squeeze((1, 2))  # (B, D)
            self.embeddings[tid] = {
                "pss": pss.cpu(),
            }

    def init_masks(
        self,
        ds: MultimodalDataset,
        unimodal_ds: str = "cycif",
        task="broad_cell_type",
        resize=True,
    ):
        """Initialize masks for all tissues in the dataset."""
        tissue_ids = ds.unimodal_datasets[unimodal_ds].get_tissue_ids()
        for tid in tissue_ids:
            self.masks[tid] = ds.get_cell_mask(tid, task=task, resize=resize)

    @torch.no_grad()
    def process_batch(
        self,
        batch: torch.Tensor,
        channel_ids: list[torch.Tensor] | torch.Tensor,
        he_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Process a batch of tissue tiles using encoder.forward_list by converting the batch
        into per-sample lists.

        Args:
            batch: (B, C, P, P)
            channel_ids: one of:
                - Tensor of shape (C,) -> broadcast to B
                - List of length B, each Tensor of shape (C,)
            he_batch: optional (B, 3, P, P)
        Returns:
            pss: (B, P // 8, P // 8, D)
        """
        assert batch.ndim == 4, "batch must be (B, C, P, P)"
        B, C, P, _ = batch.shape
        if he_batch is not None:
            assert he_batch.shape[:2] == (B, 3), "he_batch must be (B, 3, P, P)"
            assert he_batch.shape[-2:] == (P, P), "he_batch patch size mismatch"

        batch = batch.to(self.device, non_blocking=True)

        multiplex_list = [batch[i] for i in range(B)]  # each (C, P, P)

        if he_batch is not None:
            he_batch = he_batch.to(self.device, non_blocking=True)
            he_list = [he_batch[i] for i in range(B)]  # each (3, P, P)
        else:
            he_list = [None] * B

        # channel_ids has to be a list of length B of (C,) tensors
        if isinstance(channel_ids, torch.Tensor):
            assert (
                channel_ids.ndim == 1 and channel_ids.numel() == C
            ), "channel_ids tensor must be shape (C,)"
            base = channel_ids.to(self.device)
            channel_ids = [base.clone() for _ in range(B)]

        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            # channel_tokens, pss, channels tokens of final shape (B, C, D)
            _, pss_list = self.encoder.forward_list(
                multiplex=multiplex_list,
                he=he_list,
                channel_ids=channel_ids,
            )

        pss = torch.stack(pss_list)
        return pss

    def save_embeddings(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for tid, data in self.embeddings.items():
            torch.save(data, os.path.join(save_dir, f"{tid}.pth"))

    def load_embeddings(self, load_dir: str):
        self.embeddings = {}
        for fname in os.listdir(load_dir):
            if not fname.endswith(".pth"):
                continue
            tid = fname[:-4]
            self.embeddings[tid] = torch.load(
                os.path.join(load_dir, fname), map_location="cpu"
            )

    def _image_to_patches(
        self, img: torch.Tensor, patch_size: int
    ) -> list[torch.Tensor]:
        """img: (C, H, W) -> list[(C, P, P)]"""
        C, H, W = img.shape
        P = patch_size
        if H % P != 0 or W % P != 0:
            raise ValueError(f"Image size {(H, W)} not divisible by patch_size={P}")

        cols = F.unfold(img.unsqueeze(0), kernel_size=P, stride=P)  # (1, C*P*P, L)
        patches = cols.transpose(1, 2).reshape(-1, C, P, P)  # (L, C, P, P)
        return [p for p in patches]

    def _patches_to_image(
        self, patches: list[torch.Tensor], hw: tuple[int, int] | torch.Size
    ) -> torch.Tensor:
        """patches: list[(C, P, P)] -> img: (C, H, W)"""
        C, P, _ = patches[0].shape
        H, W = hw
        L = (H // P) * (W // P)
        if H % P != 0 or W % P != 0:
            raise ValueError(f"Target size {(H, W)} not divisible by patch_size={P}")
        assert len(patches) == L, f"Expected {L} patches, got {len(patches)}"

        stacked = torch.stack(patches)  # (L,C,P,P)
        cols = stacked.reshape(L, C * P * P).transpose(0, 1).unsqueeze(0)  # (1,C*P*P,L)
        img = F.fold(cols, output_size=(H, W), kernel_size=P, stride=P)  # (1,C,H,W)
        return img.squeeze(0)  # (C,H,W)
