import sys
import os
import gc
import torch
import numpy as np
import pickle as pkl
from omegaconf import OmegaConf
from loguru import logger
from config import *
import random
from src.dataset.datasets.mm_base import build_mm_datasets
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import build_flex_dual_virtues_encoder
from src.utils.marker_utils import load_marker_embeddings
from src.utils.utils import load_checkpoint_safetensors

class OrionImageFeeder:
    def __init__(self, config_path="../src/dataset/configs/base_config.yaml", 
                 subset_config_path="../src/dataset/configs/orion_subset.yaml",
                 embedding_dir="../src/dataset/esm2_t30_150M_UR50D"):
        
        print("Initializing OrionImageFeeder...")
        
        # 1. Load Configuration
        self.base_cfg = OmegaConf.load(config_path)
        self.base_cfg.marker_embedding_dir = embedding_dir
        self.marker_embeddings = load_marker_embeddings(self.base_cfg.marker_embedding_dir)
        
        self.orion_subset_cfg = OmegaConf.load(subset_config_path)
        self.ds_cfg = OmegaConf.merge(self.base_cfg, self.orion_subset_cfg)
        
        # 2. Build Dataset
        print("Building MM Datasets...")
        self.ds = build_mm_datasets(self.ds_cfg)
        self.cycif_ds = self.ds[0].unimodal_datasets["cycif"]
        
        # 3. Load Label Encoder
        self.label_encoder = None
        label_encoder_path = os.path.join(self.ds[0].cell_masks_dir, "labelencoder_broad_cell_type.npy")
        if os.path.exists(label_encoder_path):
            self.label_encoder = np.load(label_encoder_path, allow_pickle=True)
            print(f"Label Encoder loaded: {self.label_encoder}")
        else:
            print(f"Warning: Label Encoder not found at {label_encoder_path}")
            
        # 4. Initialize VirTues Encoder
        print("Initializing VirTues Encoder...")
        with open(VIRTUES_WEIGHTS_PATH + "/config.pkl", "rb") as f:
            self.virtues_cfg = pkl.load(f)

        self.virtues = build_flex_dual_virtues_encoder(self.virtues_cfg, self.marker_embeddings)
        self.virtues.cuda()
        self.virtues.eval() # Set to eval mode

        weights = load_checkpoint_safetensors(VIRTUES_WEIGHTS_PATH + "/checkpoints/checkpoint-94575/model.safetensors")
        weights_encoder = {}
        for k, v in weights.items():
            if k.startswith("encoder."):
                weights_encoder[k[len("encoder."):]] = v

        self.virtues.load_state_dict(weights_encoder, strict=False)
        print("VirTues Encoder initialized and weights loaded.")
        
        self.uniprot_to_name = self.cycif_ds.get_marker_embedding_index_to_name_dict()

    def get_all_tids(self):
        return self.cycif_ds.get_tissue_ids()

    def iterate_image_orion(self, tids, crop_size=None):
        """
        Generator that yields processed data for each tissue ID.
        """
        for i, tid in enumerate(tids):
            #print(f"Processing tissue {i+1}/{len(tids)}: {tid}")
            
            try:
                # 1. Load Data
                tissue = self.cycif_ds.get_tissue(tid) # (C, H, W)
                channels = np.array(self.cycif_ds.get_marker_embedding_indices(tid))
                mask = self.ds[0].get_cell_mask(tid, task="broad_cell_type", resize=True)
                
                if mask is None:
                    print(f"Warning: No mask found for {tid}, creating zero mask.")
                    H, W = tissue.shape[1], tissue.shape[2]
                    mask = np.zeros((H, W), dtype=np.int32)

                # 2. Prepare Tensors (Move to GPU)
                # Note: We process one image at a time to save memory
                mx_image = torch.tensor(tissue).cuda(non_blocking=True).unsqueeze(0) # (1, C, H, W)
                mx_channel_ids = torch.tensor(channels).cuda(non_blocking=True).unsqueeze(0) # (1, C)
                mx_mask = torch.tensor(mask).cuda(non_blocking=True).unsqueeze(0) # (1, H, W)

                # 3. Crop (if needed) - currently hardcoded to top-left crop as per previous notebook
                # Ideally, this should be sliding window or random crops for training
                if crop_size is not None:
                    mx_image = mx_image[:, :, :crop_size, :crop_size]
                    mx_mask = mx_mask[:, :crop_size, :crop_size]
                
                # 4. Run VirTues Encoder
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    # forward_list expects lists of tensors
                    # mx_images list of (C, H, W)
                    # mx_channel_ids list of (C)
                    # But here we have batches (1, ...), so we need to unbind or adjust forward_list usage
                    # The previous notebook used: 
                    # mx_images = [tissue_tensor]
                    # mx_channel_ids = [channel_tensor]
                    # virtues.forward_list(mx_images, [None], mx_channel_ids)
                    
                    # Let's match that structure exactly
                    input_images = [mx_image.squeeze(0)]
                    input_channels = [mx_channel_ids.squeeze(0)]
                    
                    channel_tokens, pss = self.virtues.forward_list(input_images, [None], input_channels)
                    # pss is a list of (H_patch, W_patch, D), stack it
                    pss = torch.stack(pss) # (1, H_patch, W_patch, D)

                # 5. Yield Results
                yield {
                    'tid': tid,
                    'pss': pss,
                    'channel_tokens': channel_tokens,
                    'mask': mx_mask,
                    'original_image': tissue, # Keep on CPU for visualization if needed
                    'original_mask': mask,    # Keep on CPU
                    'channels': channels
                }

            except Exception as e:
                print(f"Error processing {tid}: {e}")
                continue
            
            finally:
                # 6. Cleanup
                # Explicitly delete large tensors to free GPU memory
                if 'mx_image' in locals(): del mx_image
                if 'mx_channel_ids' in locals(): del mx_channel_ids
                if 'mx_mask' in locals(): del mx_mask
                if 'pss' in locals(): del pss
                if 'channel_tokens' in locals(): del channel_tokens
                
                torch.cuda.empty_cache()
                gc.collect()
    
    def iterate_image_orion_all(self, tids, crop_size=256):
        """
        New Method: Splits large images into a grid of 'crop_size', shuffles the grid,
        and yields every single patch. Guarantees full coverage of the tissue.
        """
        # Shuffle tissues so we don't learn in the same order every epoch
        tids_list = list(tids)
        random.shuffle(tids_list)

        for i, tid in enumerate(tids_list):
            try:
                # 1. Load Full Image & Mask
                tissue = self.cycif_ds.get_tissue(tid) # (C, H, W)
                channels = np.array(self.cycif_ds.get_marker_embedding_indices(tid))
                mask = self.ds[0].get_cell_mask(tid, task="broad_cell_type", resize=True)
                
                if mask is None:
                    H_full, W_full = tissue.shape[1], tissue.shape[2]
                    mask = np.zeros((H_full, W_full), dtype=np.int32)

                H_full, W_full = tissue.shape[1], tissue.shape[2]
                
                # 2. Generate Grid Coordinates
                coords = []
                
                if crop_size is None:
                    coords.append((0, 0))
                    use_full = True
                else:
                    use_full = False
                    # Generate top-left (y, x) coordinates
                    y_range = list(range(0, H_full, crop_size))
                    x_range = list(range(0, W_full, crop_size))
                    
                    # Adjust edges: if the last step goes out of bounds, 
                    # shift it back so it takes the last 'crop_size' pixels exactly.
                    # This avoids padding.
                    if y_range[-1] + crop_size > H_full:
                        y_range[-1] = max(0, H_full - crop_size)
                    if x_range[-1] + crop_size > W_full:
                        x_range[-1] = max(0, W_full - crop_size)
                        
                    # Remove duplicates if image was smaller than crop_size
                    y_range = sorted(list(set(y_range)))
                    x_range = sorted(list(set(x_range)))

                    for y in y_range:
                        for x in x_range:
                            coords.append((y, x))
                                
                    # 3. Shuffle the Grid
                    # This is crucial so the model doesn't drift on one specific region
                    random.shuffle(coords)

                # 4. Iterate over tiles
                for (y_start, x_start) in coords:
                    
                    # Crop logic
                    if use_full:
                        img_crop = tissue
                        mask_crop = mask
                    else:
                        y_end = y_start + crop_size
                        x_end = x_start + crop_size
                        img_crop = tissue[:, y_start:y_end, x_start:x_end]
                        mask_crop = mask[y_start:y_end, x_start:x_end]

                    # Prepare Tensors
                    mx_image = torch.tensor(img_crop).cuda(non_blocking=True).unsqueeze(0)
                    mx_channel_ids = torch.tensor(channels).cuda(non_blocking=True).unsqueeze(0)
                    mx_mask = torch.tensor(mask_crop).cuda(non_blocking=True).unsqueeze(0)

                    # Run VirTues Encoder on this specific Crop
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        input_images = [mx_image.squeeze(0)]
                        input_channels = [mx_channel_ids.squeeze(0)]
                        
                        channel_tokens, pss = self.virtues.forward_list(input_images, [None], input_channels)
                        pss = torch.stack(pss) # (1, H_patch, W_patch, D)

                    yield {
                        'tid': tid,
                        'pss': pss,
                        'channel_tokens': channel_tokens,
                        'mask': mx_mask,
                        'channels': channels,
                        'coords': (y_start, x_start)
                    }
                    
                    # Cleanup tensors per tile
                    del mx_image, mx_mask, pss, channel_tokens

            except Exception as e:
                print(f"Error processing {tid} at tile {y_start},{x_start}: {e}")
                continue
            
            finally:
                # Cleanup full image arrays when done with this TID
                if 'tissue' in locals(): del tissue
                if 'mask' in locals(): del mask
                torch.cuda.empty_cache()
                gc.collect()