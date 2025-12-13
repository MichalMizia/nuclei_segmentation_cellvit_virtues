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
        include_cycif_data: bool = True,
        return_intermediates: bool = False,
        intermediate_layers: list[int] = [4, 8, 12],
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

        if not include_cycif_data and not include_he_data:
            raise ValueError(
                "At least one of include_cycif_data or include_he_data must be True"
            )

        print("=" * 40)
        print(f"Dataset '{unimodal_ds}' with {len(tissue_ids)} tissues".center(40))
        print(f"No. of patches {batch_size}".center(40))
        print(f"Channels per patch {C}".center(40))
        print(f"Final PSS shape ({batch_size},{512})".center(40))
        print("=" * 40)

        for tid in tissue_ids:
            if include_cycif_data:
                cycif = ds.unimodal_datasets["cycif"].get_tissue(tid)  # (C, H, W)
                cycif_list = self._image_to_patches(
                    cycif, patch_size=patch_size
                )  # list of (C, P, P)
                cycif_list = [t.to(self.device) for t in cycif_list]
            else:
                cycif_list = [None] * batch_size

            if include_he_data:
                he = ds.unimodal_datasets["he"].get_tissue(tid)  # (3, H, W)
                he_list = self._image_to_patches(
                    he, patch_size=patch_size
                )  # list of (3, P, P)
                he_list = [h.to(self.device) for h in he_list]
            else:
                he_list = [None] * batch_size

            pss = []
            intermediate_pss = [[] for _ in range(len(intermediate_layers))]
            max_len = 625
            for i in tqdm(range(0, batch_size, max_len)):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # we cant just pass the whole list at once because cuda complains
                    data = self.encoder.forward_list(
                        multiplex=cycif_list[i : min(i + max_len, batch_size)],
                        he=he_list[i : min(i + max_len, len(he_list))],
                        channel_ids=channels_list[i : min(i + max_len, batch_size)],
                        return_intermediates=return_intermediates,
                    )
                pss_temp = data[1]
                pss.extend(pss_temp)
                if return_intermediates:
                    interm_temp = data[2]
                    for layer_idx, layer in enumerate(intermediate_layers):
                        intermediate_pss[layer_idx].extend(interm_temp[layer])

            pss = torch.stack(pss).squeeze((1, 2))  # (B, D)
            if return_intermediates:
                intermediate_pss = [
                    torch.stack(interm).squeeze((1, 2)).cpu()
                    for interm in intermediate_pss
                ]  # list of (B, D)
                self.embeddings[tid] = {
                    "pss": pss.cpu(),
                    "intermediate_pss": intermediate_pss,
                }
            else:
                self.embeddings[tid] = {"pss": pss.cpu()}

    def save_embeddings(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for tid, data in self.embeddings.items():
            torch.save(data, os.path.join(save_dir, f"{tid}.pth"))

    def load_embeddings(self, load_dir: str):
        self.embeddings = {}
        for fname in tqdm(sorted(os.listdir(load_dir))):
            if not fname.endswith(".pth"):
                continue
            tid = fname[:-4]
            self.embeddings[tid] = torch.load(
                os.path.join(load_dir, fname), map_location="cpu", weights_only=True
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
