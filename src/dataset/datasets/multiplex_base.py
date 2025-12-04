import os
from glob import glob
import numpy as np
import pandas as pd
from src.dataset.datasets.imaging_dataset import ImagingDataset
from src.utils.marker_utils import load_marker_embedding_dict
from src.utils.transform_utils import CustomGaussianBlur, custom_median_filter
from src.utils.normalize_utils import get_normalize_metadata
from loguru import logger
from src.utils.utils import is_rank0
import torch
import torch.nn.functional as F

class MultiplexDataset(ImagingDataset):
    @logger.catch
    def __init__(self, name, path, modality, rnd_crop_size, normalization, embedding_dir, crop_strategy='random_crops', removed_channel_names=None, custom_gene_dict=None, **kwargs):
        """
        name: Name of the dataset
        path: Path to the dataset
        rnd_crop_size: Size of the random crops
        normalization: Normalization method to apply to the images
        crop_strategy: Strategy to crop the images (either 'random_crops' or 'grid_crops')
        """
        super().__init__(name, path, rnd_crop_size, modality, crop_strategy)

        self.normalization = normalization
        self.normalization_metadata = get_normalize_metadata(normalization, rnd_crop_size, crop_strategy)
        
        if is_rank0():
            logger.info(f'Multiplex Normalization metadata: {self.normalization_metadata}')

        self.rnd_crop_folder = os.path.join(self.root_dir, self.modality, self.normalization_metadata.rnd_crop_folder_name)
        
        if not os.path.exists(self.rnd_crop_folder) or len(glob(os.path.join(self.rnd_crop_folder, "*.npy"))) == 0:
            if is_rank0():
                logger.warning(f"Crop folder {self.rnd_crop_folder} does not exist or is empty")    
            raise FileNotFoundError(f"Crop folder {self.rnd_crop_folder} does not exist or is empty")
        else:
            if is_rank0():
                logger.info(f"Crop folder {self.rnd_crop_folder} exists")


        self.modality_dir = os.path.join(self.root_dir, self.modality)
        self.metadata_dir = os.path.join(self.modality_dir, "metadata")
        self.img_folder = os.path.join(self.modality_dir, "images_resized") # @yc: change from images_prepared to images
        self.img_folder_log = os.path.join(self.modality_dir, "images_prepared")

        self.channels = pd.read_csv(os.path.join(self.modality_dir, self.normalization_metadata.channel_file_name + ".csv"))

        if self.normalization_metadata.normalizer_name == "global_std" or self.normalization_metadata.normalizer_name == "raw":
            self.mean_value_for_normalization = self.channels[self.normalization_metadata.mean_name].values
            self.std_value_for_normalization = self.channels[self.normalization_metadata.std_name].values

        if os.path.exists(os.path.join(self.root_dir, self.modality, "channels_per_image.csv")):
            self.channels_per_image = pd.read_csv(os.path.join(self.root_dir, self.modality, "channels_per_image.csv"))
            self.channels_per_image.set_index("tissue_id", inplace=True)
        else:
            self.channels_per_image = pd.DataFrame(index=self.tissue_annotations.index, columns=self.channels["channel_name"].values)
            self.channels_per_image.fillna(1, inplace=True)
        self.channels_per_image = self.channels_per_image.replace(0, False)
        self.channels_per_image = self.channels_per_image.replace(1, True)
        
        if removed_channel_names is not None:
            logger.info(f"Removing channels {removed_channel_names} from channels_per_image")
            logger.info(f"Getting indices per image")

            channel_names = list(self.channels_per_image.columns)
            self.image_channels_dict = {}
            for image_name, row in self.channels_per_image.iterrows():
                present_channels = [ch for ch in channel_names if row[ch]]
                remaining_channels = [
                    ch for ch in present_channels if ch not in removed_channel_names
                ]
                channel_indices = [present_channels.index(ch) for ch in remaining_channels]
                self.image_channels_dict[image_name] = channel_indices

            for channel_name in removed_channel_names:
                if channel_name in self.channels_per_image.columns:
                    self.channels_per_image[channel_name] = False
                else:
                    if is_rank0():
                        logger.warning(f"Channel {channel_name} not found in channels_per_image")
                        
        if self.normalization_metadata.normalizer_name == "q_99" or self.normalization_metadata.normalizer_name == "global_std":
            try:
                self.quantile = pd.read_csv(os.path.join(self.root_dir, self.modality, self.normalization_metadata.quantile_path),
                                            index_col=0)
            except Exception as e:
                if is_rank0():
                    logger.error(f'Error loading quantile file: {e}')
                raise e
            if is_rank0():
                logger.info(f'Quantile: {self.quantile.head()}')
            self.quantile_mask = self.quantile.reindex(index=self.channels_per_image.index,
                                                        columns=self.channels_per_image.columns)
            
            self.quantile_mask = self.quantile_mask.fillna(0)
            self.quantile_mask = self.quantile_mask > 0

        elif self.normalization_metadata.normalizer_name in {"tm_local_q_global_std", "tm_local_q_rescale_global_std", "q99_mean_std", "log_compress_fg_0.99", "global_log_compress_fg_0.99"}:
            try:
                self.quantile = pd.read_csv(os.path.join(self.root_dir, self.modality, self.normalization_metadata.quantile_path),
                                            index_col=0)
                self.means_file = pd.read_csv(os.path.join(self.root_dir, self.modality, self.normalization_metadata.mean_name),
                                            index_col=0)
                self.stds_file = pd.read_csv(os.path.join(self.root_dir, self.modality, self.normalization_metadata.std_name),
                                            index_col=0)
            except Exception as e:
                if is_rank0():
                    logger.error(f'Error loading quantile file: {e}')
                raise e
            if is_rank0():
                # logger.info(f'Quantile: {self.quantile.head()}')
                # logger.info(f'Means: {self.means_file.head()}')
                # logger.info(f'Stds: {self.stds_file.head()}')

                # find duplicate indices and columns
                duplicate_indices = self.quantile.index[self.quantile.index.duplicated(keep=False)]
                duplicate_columns = self.quantile.columns[self.quantile.columns.duplicated(keep=False)]
                logger.debug(f"Duplicate indices: {duplicate_indices}")
                logger.debug(f"Duplicate columns: {duplicate_columns}")
            
            self.quantile_mask = self.quantile.reindex(index=self.channels_per_image.index,
                                                        columns=self.channels_per_image.columns)
            self.std_mask = self.stds_file.reindex(index=self.channels_per_image.index,
                                                        columns=self.channels_per_image.columns)
            # non-nan quantile values are considered
            self.quantile_mask = self.quantile_mask.notna() & self.quantile_mask > 0 # remove images where 99%ile quantile is still 0 meaning 99% of the image has 0 value
            self.std_mask = self.std_mask.notna() & self.std_mask > 0 # remove static images
            self.quantile_mask = self.quantile_mask & self.std_mask # consider channels which are not-nan and have non-zero std
            # logger.debug(f"Num unique index: {self.quantile_mask.index.nunique()}, length: {len(self.quantile_mask.index)}")
            # logger.debug(f"Num unique columns: {self.quantile_mask.columns.nunique()}, length: {len(self.quantile_mask.columns)}")
            # __quantiles = self.quantile[self.quantile_mask].stack()
            # print("self.quantile shape:", self.quantile.shape)
            # print("self.quantile_mask shape:", self.quantile_mask.shape)
            # print("self.channels_per_image shape:", self.channels_per_image.shape)
            # print("Number of True in mask:", self.quantile_mask.sum().sum())
            # print("Sample of self.quantile where mask is True:", self.quantile[self.quantile_mask].stack().head())
            # print("Any NaN in __quantiles?", __quantiles.isna().any())
            # print(__quantiles)
            # assert __quantiles.notna().all().all(), f'Quantile values are nan for {self.name}'
            assert self.quantile_mask.sum().sum() > 0, f'No channels found with quantile > 0 and std > 0 for {self.name}'

        # TODO REWORK THIS, DATASET ONLY RESPONSIBLE FOR UNIPROT IDS
        self.uniprot_to_index = load_marker_embedding_dict(embedding_dir=embedding_dir)
        if custom_gene_dict is not None:
            self.uniprot_df = pd.read_csv(os.path.join(self.modality_dir, custom_gene_dict))
        else:
            self.uniprot_df = pd.read_csv(os.path.join(self.modality_dir, f"gene_dict_{self.name}.csv"))

        marker_indices = [] # contains the row index of the marker in the embedding matrix
        channel_indices_with_embedding = [] # contains the channel indices w.r.t list of all channels for which a embedding has been found
        mask_channels_with_embeddings = []
        for i, row in self.uniprot_df.iterrows():
            rowname = row["name"]
            uniprot_id = row["protein_id"]
            if uniprot_id in self.uniprot_to_index:
                marker_indices.append(self.uniprot_to_index[uniprot_id])
                channel_indices_with_embedding.append(i)
                mask_channels_with_embeddings.append(True)
            else:
                mask_channels_with_embeddings.append(False)
                if is_rank0():
                    logger.warning(f'Could not find embedding for {rowname} with {uniprot_id}')
        self.marker_indices = np.array(marker_indices) # length = channels with embedding
        self.channel_indices_with_embedding = np.array(channel_indices_with_embedding) # length = channels with embedding
        self.mask_channels_with_embeddings = np.array(mask_channels_with_embeddings) # length = all channels, true if channel has embedding
        if is_rank0():
            logger.info(f"Found embeddings for {len(self.marker_indices)} markers. Embeddings for {len(self.uniprot_df) - len(self.marker_indices)} markers not found.")

        self.gaussian_blur = CustomGaussianBlur(kernel_size=3, sigma=1.0)
        self.kwargs = kwargs
        self.apply_median = self.kwargs.get("apply_median", False)
        if is_rank0():
            logger.info(f"Apply median filter: {self.apply_median}")
        self.apply_gaussian = self.kwargs.get("apply_gaussian", False)
        if is_rank0():
            logger.info(f"Apply gaussian filter: {self.apply_gaussian}")
            
        if self.kwargs.get("disable_quantile_mask", False) and hasattr(self, 'quantile_mask'):
            if is_rank0():
                logger.warning(f"Disabling quantile mask for {self.name} dataset")
            delattr(self, 'quantile_mask')
        
    def _get_tissue_all_channels(self, tissue_id):
        path = os.path.join(self.img_folder, tissue_id + ".npy")
        img = np.load(path)
        return img.astype(np.float32)

    def get_tissue(self, tissue_id, process=True, remove_channels=False, log=False):
        if log:
            path = os.path.join(self.img_folder_log, tissue_id + ".npy")
        else:
            path = os.path.join(self.img_folder, tissue_id + ".npy")
        img = np.load(path)

        if not process:
            return img.astype(np.float32)
        
        # @yc remove here
        # only keep the channels in channels_per_image
        if remove_channels:
            img = img[self.image_channels_dict[tissue_id]]
            # img = img[self.channels_per_image.loc[tissue_id].values]
        
        # restrict img to channels with marker embeddings
        bmask_measured_channels = self.channels_per_image.loc[tissue_id].values
        bmask_channels_with_embeddings = self.mask_channels_with_embeddings[bmask_measured_channels]

        if hasattr(self, 'quantile_mask'):
            bmask_channels_with_embeddings = bmask_channels_with_embeddings & \
                        self.quantile_mask.loc[tissue_id].values[bmask_measured_channels] # (C,)
            
        img = img[bmask_channels_with_embeddings]

        img = self._preprocess(img, tissue_id, bmask_channels_with_embeddings, bmask_measured_channels)
        return img
    
    def get_tissue_mask(self, tissue_id):
        mask = np.load(os.path.join(self.tissue_masks_dir, f"{tissue_id}.npy"))
        return mask
    
    def get_tissue_size(self, tissue_id):
        path = os.path.join(self.img_folder, tissue_id + ".npy")
        img = np.load(path, mmap_mode='r')
        return img.shape[1], img.shape[2]
    
    def get_marker_embedding_indices(self, tissue_id):
        # Ugly, but works: 
        bmask_channels_per_image = self.channels_per_image.loc[tissue_id].values # total channesl
        bmask_channels_with_embedding_per_image = bmask_channels_per_image[self.channel_indices_with_embedding] # length = channels with embedding
        
        # if self.normalization == 'q_99' or self.normalization == 'q99_mean_std':
        if hasattr(self, 'quantile_mask'):
            quantile_per_image = self.quantile_mask.loc[tissue_id].values[self.channel_indices_with_embedding]
            bmask_channels_with_embedding_per_image = bmask_channels_with_embedding_per_image & quantile_per_image
        return torch.from_numpy(self.marker_indices[bmask_channels_with_embedding_per_image])
    
    def get_marker_names(self, tissue_id):
        # Doesn't work yet, but not called 
        bmask_channels_per_image = self.channels_per_image.loc[tissue_id].values # total channels
        bmask_channels_with_embedding_per_image = bmask_channels_per_image[self.channel_indices_with_embedding] # length = channels with embedding
        channel_indices_with_embedding_per_image = self.channel_indices_with_embedding[bmask_channels_with_embedding_per_image]
        return self.uniprot_df.loc[channel_indices_with_embedding_per_image]["name"].values
    
    def get_marker_uniprot_ids(self, tissue_id):
        # Doesn't work yet, but not called 
        bmask_channels_per_image = self.channels_per_image.loc[tissue_id].values # total channels
        bmask_channels_with_embedding_per_image = bmask_channels_per_image[self.channel_indices_with_embedding] # length = channels with embedding
        channel_indices_with_embedding_per_image = self.channel_indices_with_embedding[bmask_channels_with_embedding_per_image]
        return self.uniprot_df.loc[channel_indices_with_embedding_per_image]["protein_id"].values
    
    def get_marker_embedding_index_to_name_dict(self):
        mapping = {}
        for marker_index, channel_index in zip(self.marker_indices, self.channel_indices_with_embedding):
            mapping[marker_index] = self.uniprot_df.iloc[channel_index]["name"]
        return mapping
    
    def get_marker_embedding_index_to_uniprot_dict(self):
        mapping = {}
        for marker_index, channel_index in zip(self.marker_indices, self.channel_indices_with_embedding):
            mapping[marker_index] = self.uniprot_df.iloc[channel_index]["protein_id"]
        return mapping
        
    def get_crop(self, tissue_id, crop_id, process=True, remove_channels=False):
        path = os.path.join(self.rnd_crop_folder, f"{tissue_id}_{crop_id}.npy")
        crop = np.load(path)
        
        # @yc remove here
        # only keep the channels in channels_per_image
        if remove_channels:
            crop = crop[self.image_channels_dict[tissue_id]]
            # crop = crop[self.channels_per_image.loc[tissue_id].values]
        # crop has shape: C_max x H x W
        # channels_per_image has shape: C_true True values where C_true <= C_max
        # Quantile_mask has shape C_max
        # self.mask_channels_with_embeddings has shape C_max

        bmask_measured_channels = self.channels_per_image.loc[tissue_id].values # (C,)
        crop_mask = self.mask_channels_with_embeddings[bmask_measured_channels] # (C,)

        # logger.debug(f'TID: {tissue_id}, CropID: {crop_id}, Crop shape: {crop.shape}, Crop mask shape: {crop_mask.shape} bm: {bmask_measured_channels.shape}, qm: \
        #                 {self.quantile_mask.loc[tissue_id].values.shape}, mask: {self.mask_channels_with_embeddings.shape}')

        # if self.normalization == 'q_99' or self.normalization == 'q99_mean_std':
        if hasattr(self, 'quantile_mask'):
            crop_mask = crop_mask & \
                        self.quantile_mask.loc[tissue_id].values[bmask_measured_channels] # (C,)
        

        crop = crop[crop_mask]
        if not process:
            return crop

        crop = self._preprocess(crop, tissue_id, crop_mask, bmask_measured_channels)
        # restrict crop to channels with marker embeddings
        # bmask_measured_channels = self.channels_per_image.loc[tissue_id].values # (C,)
        # bmask_quantile = self.quantile_mask.loc[tissue_id].values
        # bmask_measured_channels = bmask_measured_channels & bmask_quantile
        # bmask_channels_with_embeddings = self.mask_channels_with_embeddings[bmask_measured_channels]
        # crop = crop[bmask_channels_with_embeddings]
        return crop
    
    def _save_crop(self, crop, tissue_id, crop_id):
        np.save(os.path.join(self.rnd_crop_folder, f"{tissue_id}_{crop_id}.npy"), crop)

    def _preprocess(self, crop, tissue_id, crop_mask=None, bmask_measured_channels=None):
        """
        Applies normalization and gaussian blur to the crop
        Args:
            crop: C x H x W crop
            tissue_id: tissue_id of the crop
            crop_mask: mask of the measued channels with embeddings loaded and quantile valid, shape: C_measured
            bmask_measured_channels: mask of the measured channels in the tissue. shape: C_total with sum=C_measured
        Returns:
            crop: np.ndarray of type float32
        """
        if self.normalization == 'tm_local_q_global_std':
            clip_values = self.quantile.loc[tissue_id].values[bmask_measured_channels]
            clip_values = clip_values[crop_mask]
            clip_values = torch.from_numpy(clip_values).float()[:, None, None]
            min_ = torch.zeros_like(clip_values)
            crop = torch.from_numpy(crop).float()
            crop = torch.clamp(crop, min=min_, max=clip_values)
            crop = torch.log1p(crop)
            crop = self.gaussian_blur(crop)
            log_mean = self.means_file.loc[tissue_id].values[bmask_measured_channels]
            log_mean = log_mean[crop_mask]
            log_mean = torch.from_numpy(log_mean).float()[:, None, None]
            log_std = self.stds_file.loc[tissue_id].values[bmask_measured_channels]
            log_std = log_std[crop_mask]
            log_std = torch.from_numpy(log_std).float()[:, None, None]
            crop = (crop - log_mean) / (log_std + 1e-9) # avoid division by zero
            return crop

        elif self.normalization == 'tm_local_q_rescale_global_std':
            clip_values = self.quantile.loc[tissue_id].values[bmask_measured_channels]
            clip_values = clip_values[crop_mask]
            clip_values = torch.from_numpy(clip_values).float()[:, None, None]
            min_ = torch.zeros_like(clip_values)
            crop = torch.from_numpy(crop).float()
            crop = 255 * torch.clamp(crop, min=min_, max=clip_values) / clip_values 
            crop = torch.log1p(crop)
            crop = self.gaussian_blur(crop)
            log_mean = self.means_file.loc[tissue_id].values[bmask_measured_channels]
            log_mean = log_mean[crop_mask]
            log_mean = torch.from_numpy(log_mean).float()[:, None, None]
            log_std = self.stds_file.loc[tissue_id].values[bmask_measured_channels]
            log_std = log_std[crop_mask]
            log_std = torch.from_numpy(log_std).float()[:, None, None]
            crop = (crop - log_mean) / (log_std + 1e-9) # avoid division by zero
            return crop
        
        elif self.normalization == 'raw':
            mean = self.mean_value_for_normalization
            std = self.std_value_for_normalization
            bmask = self.channels_per_image.loc[tissue_id]
            mean = mean[bmask] # C
            std = std[bmask] # C
            mean = mean[crop_mask] # C_measured
            std = std[crop_mask]
            mean = mean[:, None, None]
            std = std[:, None, None]
            crop = (crop - mean) / std
            return torch.from_numpy(crop).float()

        elif self.normalization == 'global_std':
            # We do: clip q99 -> log1p -> gaussian_blur -> standardization. 
            clip_values = self.quantile.loc[tissue_id].values[bmask_measured_channels]
            clip_values = clip_values[crop_mask]
            clip_values = torch.from_numpy(clip_values).float()[:, None, None]
            min_ = torch.zeros_like(clip_values)
            crop = torch.from_numpy(crop).float()
            crop = torch.clamp(crop, min=min_, max=clip_values) # / clip_values

            crop = torch.log1p(crop)
            crop = self.gaussian_blur(crop)

            log_mean = self.mean_value_for_normalization
            log_std = self.std_value_for_normalization

            bmask = self.channels_per_image.loc[tissue_id]
            log_mean = log_mean[bmask]
            log_std = log_std[bmask]

            log_mean = log_mean[crop_mask][:, None, None] # C_measured
            log_std = log_std[crop_mask][:, None, None]
            log_mean = torch.from_numpy(log_mean).float()
            log_std = torch.from_numpy(log_std).float()

            crop = (crop - log_mean) / log_std
            return crop
        
        elif self.normalization == 'q_99':
            clip_values = self.quantile.loc[tissue_id].values[bmask_measured_channels]
            clip_values = clip_values[crop_mask] 
            clip_values = torch.from_numpy(clip_values).float()[:, None, None] 
            min_ = torch.zeros_like(clip_values)
            crop = torch.from_numpy(crop).float()
            
            crop = torch.clamp(crop, min=min_, max=clip_values) / clip_values # much faster than numpy clip


            crop = self.gaussian_blur(crop)
            return crop
        
        elif self.normalization == 'log_compress_fg_0.99' or self.normalization == 'log_compress_fg_0.99_focused' \
            or self.normalization == 'global_log_compress_fg_0.99_focused':
            clip_values = self.quantile.loc[tissue_id].values[bmask_measured_channels]
            clip_values = clip_values[crop_mask]
            clip_values = torch.from_numpy(clip_values).float()[:, None, None]
            mean_values = self.means_file.loc[tissue_id].values[bmask_measured_channels]
            mean_values = mean_values[crop_mask]
            mean_values = torch.from_numpy(mean_values).float()[:, None, None]
            std_values = self.stds_file.loc[tissue_id].values[bmask_measured_channels]
            std_values = std_values[crop_mask]
            std_values = torch.from_numpy(std_values).float()[:, None, None]
            crop = torch.from_numpy(crop).float()
            crop = crop / clip_values
            crop[crop > 1] = torch.log(
                torch.e + crop[crop > 1] - 1
            )
            crop = (crop - mean_values) / std_values
            if self.apply_gaussian:
                crop = self.gaussian_blur(crop)
            if self.apply_median:
                crop = custom_median_filter(crop.unsqueeze(0), kernel_size=3, padding='reflect').squeeze(0)
            return crop

        
        elif self.normalization == 'q99_mean_std':
            # try:
            clip_values = self.quantile.loc[tissue_id].values[bmask_measured_channels]
            # except Exception as e:
            #     if is_rank0():
            #         logger.error(f'Error loading quantile file: {e}')
            #         logger.error(tissue_id)
            #     raise e
            clip_values = clip_values[crop_mask]
            clip_values = torch.from_numpy(clip_values).float()[:, None, None]
            min_ = torch.zeros_like(clip_values)
            crop = torch.from_numpy(crop).float()
            crop = torch.clamp(crop, min=min_, max=clip_values) 
            mean_values = self.means_file.loc[tissue_id].values[bmask_measured_channels]
            mean_values = mean_values[crop_mask]
            mean_values = torch.from_numpy(mean_values).float()[:, None, None]
            std_values = self.stds_file.loc[tissue_id].values[bmask_measured_channels]
            std_values = std_values[crop_mask]
            std_values = torch.from_numpy(std_values).float()[:, None, None]
            crop = (crop - mean_values) / std_values
            crop = self.gaussian_blur(crop)
            return crop




        else:
            raise NotImplementedError(f"Normalization method {self.normalization} not implemented for multiplex datasets")