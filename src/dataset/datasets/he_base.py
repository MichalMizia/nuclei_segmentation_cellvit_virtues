import os
import torch
import numpy as np
from src.dataset.datasets.imaging_dataset import ImagingDataset
from einops import rearrange
from PIL import Image
from einops import rearrange
from src.utils.normalize_utils import get_normalize_metadata
from loguru import logger


Image.MAX_IMAGE_PIXELS = 50000 * 50000 # Increase PIL limit to 50k x 50k pixels

class HEDataset(ImagingDataset):

    def __init__(self, name, path, rnd_crop_size, crop_strategy='random_crops', **kwargs):
        """
        name: Name of the dataset
        path: Path to the dataset
        rnd_crop_size: Size of the random crops
        crop_stragegy: Strategy to crop the images (either 'random_crops' or 'grid_crops')
        """
        super().__init__(name, path, rnd_crop_size, 'he', crop_strategy)

        self.normalization_metadata = get_normalize_metadata('he', rnd_crop_size, crop_strategy)
        logger.info(f'H&E Normalization metadata: {self.normalization_metadata}')

        self.rnd_crop_folder = os.path.join(self.root_dir, self.modality, self.normalization_metadata.rnd_crop_folder_name)
        os.makedirs(self.rnd_crop_folder, exist_ok=True)

        self.img_folder = os.path.join(self.root_dir, self.modality, "images_resized") # TODO change this

        # ImageNet values, note that mmvirtues uses the Hibou normalizations
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        # HIBOU mean/std values
        # self.mean_he =  torch.tensor([0.7068, 0.5755, 0.722])[:, None, None]
        # self.std_he = torch.tensor([0.195, 0.2316, 0.1816])[:, None, None]

    def _get_tissue_all_channels(self, tissue_id):
        path = os.path.join(self.img_folder, tissue_id + ".png")
        img = np.array(Image.open(path))
        img = rearrange(img, 'h w c -> c h w')
        return img

    def get_tissue(self, tissue_id):
        path = os.path.join(self.img_folder, tissue_id + ".png")
        img = np.array(Image.open(path))
        img = rearrange(img, 'h w c -> c h w')
        img = self._preprocess(img)
        return img

    def get_tissue_mask(self, tissue_id):
        mask = np.load(os.path.join(self.tissue_masks_dir, f"{tissue_id}.npy"))
        return mask
    
    def get_tissue_size(self, tissue_id):
        path = os.path.join(self.img_folder, tissue_id + ".png")
        im = Image.open(path, mode='r')
        return im.size[1], im.size[0]

    def get_marker_embedding_indices(self, tissue_id):
        raise NotImplementedError("HE does not have markers")
    
    def get_crop(self, tissue_id, crop_id, **kwargs):
        path = os.path.join(self.rnd_crop_folder, f"{tissue_id}_{crop_id}.png")
        crop = np.array(Image.open(path))
        crop = rearrange(crop, 'h w c -> c h w')
        crop = self._preprocess(crop)
        return crop
    
    def _save_crop(self, crop, tissue_id, idx):
        crop = rearrange(crop, 'c h w -> h w c')
        crop = Image.fromarray(crop)
        crop.save(os.path.join(self.rnd_crop_folder, f"{tissue_id}_{idx}.png"))
    
    def _preprocess(self, crop):
        crop = torch.from_numpy(crop / 255).float()
        crop = (crop - self.mean) / self.std
        return crop
