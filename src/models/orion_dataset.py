import sys
import os
import gc
import torch
import numpy as np
import pickle as pkl
from omegaconf import OmegaConf
from loguru import logger
from config import *
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

    def iterate_image_orion(self, tids, crop_size=128):
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