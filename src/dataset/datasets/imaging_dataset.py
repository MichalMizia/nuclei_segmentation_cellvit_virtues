import os
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from skimage.transform import resize
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter

import numpy as np
from loguru import logger
import pandas as pd

class ImagingDataset(ABC):

    def __init__(self, name, path, rnd_crop_size, modality, crop_strategy):
        """
        name: Name of the dataset
        path: Path to the dataset
        rnd_crop_size: Size of the random crops
        normalization: Normalization method to apply to the images
        """
        self.name = name
        self.modality = modality
        self.rnd_crop_size = rnd_crop_size
        self.crop_strategy = crop_strategy

        self.root_dir = path
        self.tissue_masks_dir = os.path.join(self.root_dir, "tissue_masks")

        self.tissue_annotations = pd.read_csv(os.path.join(self.root_dir, "tissue_annotations.csv"))
        self.tissue_annotations.set_index("tissue_id", inplace=True)
        self.tissue_annotations["tissue_id"] = self.tissue_annotations.index
        self.tissue_annotations = self.tissue_annotations[self.tissue_annotations[self.modality]==1] #TODO overwrite this

        self.rnd_crops_per_image = {}

    def get_tissue_ids(self):
        """
        Gets all tissue ids available
        """
        return self.tissue_annotations["tissue_id"].values
    
    @abstractmethod
    def _get_tissue_all_channels(self, tissue_id):
        """
        Gets the full tissue image without filtering channels for a given tissue id. 
        """
        pass

    @abstractmethod
    def get_tissue(self, tissue_id):
        """
        Gets the tissue image for a given tissue id. 
        """
        pass

    def get_tissue_mask(self, tissue_id):
        """
        Gets the tissue segmentation mask for a given tissue id
        """
        mask = np.load(os.path.join(self.tissue_masks_dir, f"{tissue_id}.npy"))
        return mask

    @abstractmethod
    def get_tissue_size(self, tissue_id):
        """
        Gets the size the of the tissue for a given tissue id
        """
        pass

    @abstractmethod
    def get_marker_embedding_indices(self, tissue_id):
        """
        Gets the indices of the measured markers w.r.t to the marker embedding 
        """
        pass

    @abstractmethod
    def get_crop(self, tissue_id, crop_id):
        """
        Gets a specific crop based on its tissue id and crop id
        """
        pass

    def get_rnd_crop(self, tissue_id):
        """
        Gets a random crop for a given tissue
        """
        if not tissue_id in self.rnd_crops_per_image:
            crop_files = os.listdir(self.rnd_crop_folder)
            crop_files = [f for f in os.listdir(self.rnd_crop_folder) if f.startswith(tissue_id + "_")]
            self.rnd_crops_per_image[tissue_id] = len(crop_files)
        i = np.random.randint(0, self.rnd_crops_per_image[tissue_id])
        return self.get_crop(tissue_id, i)

    @abstractmethod
    def _save_crop(self, crop, tissue_id, crop_id):
        """
        Saves a crop. Implement this using modality specific techniques.
        """
        pass

    def _create_crops(self, tissue_id: str, row_coords: list[int], col_coords: list[int], tissue_mask: np.ndarray):
        """ Creates crops for an image. Only creates crops where the mean tissue mask is > 0.8
        Args:
            tissue_id (str): tissue_id of full tissue image
            row_coords (list[int]): all row coordinates to create crops
            col_coords (list[int]): all column coordinates to create crops
            tissue_mask (np.ndarray): tissue mask of the full tissue image. If the tissue mask should not be used, set it to np.ones((H, W))

        Returns:
            list[int]: the row and column coordinates of the valid crops (where the tissue mask was > 0.8)
        """
        assert len(row_coords) == len(col_coords), "row_coords and col_coords must have the same length"

        logger.info(f"Loading image {tissue_id}")
        image  = self._get_tissue_all_channels(tissue_id)

        logger.info(f"Loading image {tissue_id}")
        image  = self._get_tissue_all_channels(tissue_id) 
        H, W = image.shape[1], image.shape[2]

        assert max(row_coords) < H, f"row_coords must be smaller than the image height {H}"
        assert max(col_coords) < W, f"col_coords must be smaller than the image width {W}"
        assert np.abs(np.array(tissue_mask.shape) - (H, W)).sum() < 10, \
            f"Tissue mask does not roughly match image shape for {tissue_id}. This should be taken care of in the mmDatset class."
        
        logger.info(f"Creating a maximum of {len(row_coords)} crops for {tissue_id} with image size {H}x{W}")
        valid_row_coords, valid_col_coords = [], []
        cid = 0
        for row, col in zip(row_coords, col_coords):
            mean = np.mean(tissue_mask[row : row + self.rnd_crop_size, col : col + self.rnd_crop_size].astype(np.float32))
            if mean < 0.3:
                logger.debug(f"Skipping crop {cid} for {tissue_id} with mean tissue mask value {mean} and coordinate ({row}, {col})")
                continue    # Skip if the crop is mostly background
            crop = image[:, row:row+self.rnd_crop_size, col:col+self.rnd_crop_size]
            self._save_crop(crop, tissue_id, cid)
            valid_row_coords.append(row)
            valid_col_coords.append(col)
            cid += 1
        logger.info(f"Created {len(valid_row_coords)} crops for {tissue_id}")

        if len(valid_row_coords) == 0:
            logger.warning(f"No valid crops could be created for image {tissue_id}. This is probably due to the tissue mask mostly indicating 'no tissue' areas... Creating a single center crop instead.")
            row = H // 2 - self.rnd_crop_size // 2
            col = W // 2 - self.rnd_crop_size // 2
            crop = image[:, row:row+self.rnd_crop_size, col:col+self.rnd_crop_size]
            self._save_crop(crop, tissue_id, 0)
            valid_row_coords.append(row)
            valid_col_coords.append(col)
        return valid_row_coords, valid_col_coords

    def _count_random_crops(self):
        """
        Counts the available crops per tissue
        """
        _crop_files = os.listdir(self.rnd_crop_folder)
        df = pd.DataFrame({"files": _crop_files})
        df["tid"] = df["files"].str.rsplit("_", n=1, expand=True)[0]
        self.rnd_crops_per_image = df.groupby("tid").size().to_dict()
        self.rnd_crops_per_image = {k: v for k, v in self.rnd_crops_per_image.items() if k in self.get_tissue_ids()}
        
        for tissue_id in self.get_tissue_ids():
            if tissue_id not in self.rnd_crops_per_image:
                self.rnd_crops_per_image[tissue_id] = 0