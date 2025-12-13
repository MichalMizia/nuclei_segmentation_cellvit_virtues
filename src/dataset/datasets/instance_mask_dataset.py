import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from src.dataset.datasets.mm_base import MultimodalDataset
from tqdm import tqdm
import torchvision.transforms as T
from src.utils.instance_map_utils import gen_instance_hv_map

# fmt: off
class InstanceMaskDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[str, torch.Tensor, list[torch.Tensor]]],
        ds: MultimodalDataset,
        batches_from_item: int = 1,
        include_cycif=True,
        include_he=True,
    ):
        """items: list of (tissue_id, pss, intermediate_pss) where pss: (N,D), intermediate_pss: list of (N,D)
        ds: MultimodalDataset, to get cell masks and he images
        batches_from_item: int, number of batches to split each item into"""
        dim = int(np.sqrt(items[0][1].shape[0])) # sqrt(N), assume square images
        B = batches_from_item
        self.items = []

        if not include_cycif and not include_he:
            raise ValueError("At least one of include_cycif or include_he must be True")

        masks = [torch.from_numpy(ds.get_cell_mask(tid, task="broad_cell_type", resize=True)) for tid, _, _ in tqdm(items, desc="Preloading masks")]
        instance_masks = [torch.from_numpy(ds.get_cell_mask(tid, task="segmentation", resize=True)) for tid, _, _ in tqdm(items, desc="Preloading instance masks")]
        hv_maps = [torch.from_numpy(gen_instance_hv_map(im.numpy())) for im in tqdm(instance_masks, desc="Computing HV maps")]
        
        if include_he:
            he_imgs = [ds.unimodal_datasets["he"].get_tissue(tid) for tid, _, _ in tqdm(items, desc="Preloading H&E images")]
        else:
            he_imgs = [torch.zeros((3, masks[0].shape[0], masks[0].shape[1])) for _ in items]
        if include_cycif:
            cycif_imgs = [ds.unimodal_datasets["cycif"].get_tissue(tid) for tid, _, _ in tqdm(items, desc="Preloading CyCIF images")]
        else:
            cycif_imgs = [torch.zeros((1, masks[0].shape[0], masks[0].shape[1])) for _ in items]

        mask_shape = masks[0].shape
        N = mask_shape[0] // B
        assert mask_shape[0] % B == 0 and mask_shape[1] % B == 0, f"mask {mask_shape[0]}x{mask_shape[1]} not divisible by B={B}"
        assert he_imgs[0].shape[1] == mask_shape[0] and he_imgs[0].shape[2] == mask_shape[1], f"he image {he_imgs[0].shape} not matching mask {mask_shape}"
        assert cycif_imgs[0].shape[1] == mask_shape[0] and cycif_imgs[0].shape[2] == mask_shape[1], f"cycif image {cycif_imgs[0].shape} not matching mask {mask_shape}"

        masks_tiles_list = rearrange(torch.stack(masks), "n (b1 h) (b2 w) -> n (b1 b2) h w", b1=B, b2=B, h=N, w=N)
        instance_masks_tiles_list = rearrange(torch.stack(instance_masks), "n (b1 h) (b2 w) -> n (b1 b2) h w", b1=B, b2=B, h=N, w=N)
        hv_maps_tiles_list = rearrange(torch.stack(hv_maps), "n (b1 h) (b2 w) c -> n (b1 b2) h w c", b1=B, b2=B, h=N, w=N)
        he_imgs_tiles_list = rearrange(torch.stack(he_imgs), "n c (b1 h) (b2 w) -> n (b1 b2) h w c", b1=B, b2=B, h=N, w=N)
        cycif_imgs_tiles_list = rearrange(torch.stack(cycif_imgs), "n c (b1 h) (b2 w) -> n (b1 b2) h w c", b1=B, b2=B, h=N, w=N)

        for ind, (tid, pss, intermediate_pss) in tqdm(enumerate(items), desc="Processing items"):
            mask_tiles = masks_tiles_list[ind]  # (B*B, h, w)
            he_img_tiles = he_imgs_tiles_list[ind]  # (B*B, h, w, 3)
            cycif_img_tiles = cycif_imgs_tiles_list[ind]  # (B*B, h, w, C)
            instance_mask_tiles = instance_masks_tiles_list[ind]  # (B*B, h, w)
            hv_map_tiles = hv_maps_tiles_list[ind]  # (B*B, h

            pss_tiles = rearrange(pss, "(b1 h b2 w) d -> (b1 b2) h w d", b1=B, b2=B, h=dim // B, w=dim // B)
            intermediate_pss_tiles = [rearrange(t, "(b1 h b2 w) d -> (b1 b2) h w d", b1=B, b2=B, h=dim // B, w=dim // B) for t in intermediate_pss]

            for i in range(B * B):
                # from first dim, take all, from second dim take index
                intermediate_pss_tile = [t[i] for t in intermediate_pss_tiles]
                img_tiles = []
                if include_he:
                    img_tiles.append(he_img_tiles[i])
                if include_cycif:
                    img_tiles.append(cycif_img_tiles[i])
                    
                img_tiles = torch.cat(img_tiles, dim=-1)  # (h,w,3+C) or (h,w,3) or (h,w,C)
                item = (
                    tid,
                    img_tiles,
                    mask_tiles[i],
                    pss_tiles[i],
                    intermediate_pss_tile,
                    instance_mask_tiles[i],
                    hv_map_tiles[i]
                )
                self.items.append(item) # pss_tiles[i]: (h,w,D), mask_tiles[i]: (H,W)

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        """Returns: tid: str, he_img: (H,W,3), mask: (H,W), pss: (h,w,D), intermediate_pss: list of (h,w,D)"""
        return self.items[idx]
