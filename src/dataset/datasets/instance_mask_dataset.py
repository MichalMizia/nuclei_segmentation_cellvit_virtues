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
        hv_path: str | None = None,
        load_hv_maps: bool = False,
        save_hv_maps: bool = False,
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

        if load_hv_maps:
            hv_maps = []
            for tid, _, _ in tqdm(items, desc="Loading HV maps"):
                hv_map_path = f"{hv_path}/{tid}_hv_map.npy"
                hv_map = np.load(hv_map_path)  # (H,W,2)
                hv_maps.append(torch.from_numpy(hv_map))
        else:
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
        hv_maps_tiles_list = rearrange(torch.stack(hv_maps), "n c (b1 h) (b2 w) -> n (b1 b2) c h w", b1=B, b2=B, h=N, w=N)
        he_imgs_tiles_list = rearrange(torch.stack(he_imgs), "n c (b1 h) (b2 w) -> n (b1 b2) h w c", b1=B, b2=B, h=N, w=N)
        cycif_imgs_tiles_list = rearrange(torch.stack(cycif_imgs), "n c (b1 h) (b2 w) -> n (b1 b2) h w c", b1=B, b2=B, h=N, w=N)

        if save_hv_maps and hv_path is not None:
            import os
            os.makedirs(hv_path, exist_ok=True)
            for ind, (tid, _, _) in enumerate(items):
                hv_map = hv_maps[ind].numpy()  # (H,W,2)
                hv_map_path = os.path.join(hv_path, f"{tid}_hv_map.npy")
                np.save(hv_map_path, hv_map)

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

class ImageInstanceMaskDataset(Dataset):
    def __init__(
            self,
            items: list[str],
            ds: MultimodalDataset,
            batches_from_item: int = 1,
            resize=None,
            include_cycif=True,
            normalize_he=True,
            load_hv_maps: bool = False,
            hv_path: str | None = None, 
        ):
        """items: list of (tissue_id) where pss: (N,D), intermediate_pss: list of (N,D)
        ds: MultimodalDataset, to get cell masks and he images
        batches_from_item: int, number of batches to split each item into
        resize: tuple (H,W) to resize masks and images to"""
        B = batches_from_item
        self.items = []

        if load_hv_maps:
            hv_maps = {}
            for tid in tqdm(items, desc="Loading HV maps"):
                hv_map_path = f"{hv_path}/{tid}_hv_map.npy"
                hv_map = np.load(hv_map_path)
                hv_maps[tid] = torch.from_numpy(hv_map)


        for tid in tqdm(items):
            mask = torch.from_numpy(ds.get_cell_mask(tid, task="broad_cell_type", resize=True))
            instance_mask = torch.from_numpy(ds.get_cell_mask(tid, task="segmentation", resize=True))
            if load_hv_maps:
                hv_map = hv_maps[tid]
            else:
                hv_map = torch.zeros((mask.shape[0], mask.shape[1], 2))
            
            if normalize_he:
                he_img = ds.unimodal_datasets["he"].get_tissue(tid) # (3, H, W)
            else:
                he_img = torch.from_numpy(ds.unimodal_datasets["he"]._get_tissue_all_channels(tid)).float() / 255.0
                norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                he_img = norm(he_img)

            if include_cycif:
                cycif_img = ds.unimodal_datasets["cycif"].get_tissue(tid) # (C, H, W)
            else: # zeros for compatibility
                cycif_img = torch.zeros((1, he_img.shape[1], he_img.shape[2]))

            if resize is not None:
                transform = T.Resize(resize, interpolation=T.InterpolationMode.NEAREST)
                mask = transform(mask.unsqueeze(0)).squeeze(0)
                instance_mask = transform(instance_mask.unsqueeze(0)).squeeze(0)
                transform = T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR)
                he_img = transform(he_img)
                cycif_img = transform(cycif_img)
                hv_map = transform(hv_map) # (2, H, W)
            mask_h, mask_w = mask.shape # mask is tensor of shape (H,W)

            assert mask_h % B == 0 and mask_w % B == 0, f"mask {mask_h}x{mask_w} not divisible by B={B}"
            assert he_img.shape[1] == mask_h and he_img.shape[2] == mask_w, f"he image {he_img.shape} not matching mask {mask.shape}"
            assert cycif_img.shape[1] == mask_h and cycif_img.shape[2] == mask_w, f"cycif image {cycif_img.shape} not matching mask {mask.shape}"
            
            mask_tiles = rearrange(mask, "(b1 h) (b2 w) -> (b1 b2) h w", b1=B, b2=B, h=mask_h // B, w=mask_w // B)
            instance_mask_tiles = rearrange(instance_mask, "(b1 h) (b2 w) -> (b1 b2) h w", b1=B, b2=B, h=mask_h // B, w=mask_w // B)
            he_img_tiles = rearrange(he_img, "c (b1 h) (b2 w) -> (b1 b2) h w c", b1=B, b2=B, h=mask_h // B, w=mask_w // B)
            cycif_img_tiles = rearrange(cycif_img, "c (b1 h) (b2 w) -> (b1 b2) h w c", b1=B, b2=B, h=mask_h // B, w=mask_w // B)
            hv_maps_tiles_list = rearrange(hv_map, "c (b1 h) (b2 w) -> (b1 b2) c h w", b1=B, b2=B, h=mask_h // B, w=mask_w // B)

            for i in range(B * B):
                self.items.append((tid, he_img_tiles[i], cycif_img_tiles[i], mask_tiles[i], instance_mask_tiles[i], hv_maps_tiles_list[i]))

    def __len__(self): 
        return len(self.items)
    
    def __getitem__(self, idx):
        """Returns: tid: str, he_img: (H,W,3), cycif_img: (H,W,C), mask: (H,W)"""
        tid, he_img, cycif_img, mask, instance_mask, hv_map = self.items[idx]
        return tid, he_img, cycif_img, mask, instance_mask, hv_map
