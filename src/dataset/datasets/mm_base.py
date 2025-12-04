import os
from src.utils.utils import is_rank0
import numpy as np
import pandas as pd
from src.dataset.datasets.he_base import HEDataset
from src.dataset.datasets.multiplex_base import MultiplexDataset
from multiprocessing import Pool
from tqdm import tqdm
from loguru import logger
from skimage.transform import resize
from sklearn.model_selection import GroupShuffleSplit

def build_mm_datasets(conf, **kwargs):
    datasets = []
    for ds in conf.datasets.keys():
        ds_conf = conf.datasets[ds]
        mm_dataset = build_mm_dataset(conf, ds_conf, **kwargs)
        datasets.append(mm_dataset)
    
    return datasets

def build_mm_dataset(conf, ds_conf, **kwargs):
    custom_gene_dict = None
    if hasattr(ds_conf, 'custom_gene_dict'):
        custom_gene_dict = ds_conf.custom_gene_dict
    load_mask = getattr(ds_conf, 'load_mask', False)
    mask_type = getattr(ds_conf, 'mask_type', 'broad_cell_type')
    dataset = MultimodalDataset(name=ds_conf.name, path=ds_conf.path, modalities=ds_conf.modalities, rnd_crop_size=conf.image_info.rnd_crop_size, normalization=conf.image_info.normalization,
                                embedding_dir=conf.marker_embedding_dir, crop_strategy=conf.image_info.crop_strategy, custom_gene_dict=custom_gene_dict, load_mask=load_mask, mask_type=mask_type, **kwargs)
    return dataset

class MultimodalDataset():

    def __init__(self, name, path, modalities, rnd_crop_size, normalization, embedding_dir, crop_strategy='random_crops', removed_channel_names=None, custom_gene_dict=None, load_mask=False, mask_type='broad_cell_type',**kwargs):
        """
        name: Name of the dataset
        path: Path to the parent directory of the dataset
        modalities: List of modalities in the dataset
        rnd_crop_size: Size of the random crops
        normalization: Normalization to apply to the images
        """
        self.name = name
        if is_rank0():
            logger.debug(f"Loading dataset {name} from {path}")
        self.root_dir = path

        self.modalities = modalities
        self.rnd_crop_size = rnd_crop_size

        self.unimodal_datasets = dict()
        for modality in modalities:
            if modality == 'he':
                self.unimodal_datasets[modality] = HEDataset(name, path, rnd_crop_size, crop_strategy, **kwargs)
            elif modality in ['codex', 'imc', 'cycif']:
                self.unimodal_datasets[modality] = MultiplexDataset(name, path, modality, rnd_crop_size, normalization, embedding_dir, crop_strategy, removed_channel_names=removed_channel_names, custom_gene_dict=custom_gene_dict, **kwargs)
                if self.unimodal_datasets[modality].rnd_crop_size != rnd_crop_size:
                    logger.warning(f"Found different random crop size: changing from {rnd_crop_size} to {self.unimodal_datasets[modality].rnd_crop_size}")
                    self.rnd_crop_size = self.unimodal_datasets[modality].rnd_crop_size

            else:
                raise NotImplementedError(f"Modality {modality} not implemented")

        self.tissue_masks_dir = os.path.join(self.root_dir, "tissue_masks")
        self.cell_masks_dir = os.path.join(self.root_dir, "cell_masks")

        self.tissue_annotations = pd.read_csv(os.path.join(self.root_dir, "tissue_annotations.csv"))
        self.tissue_annotations.set_index("tissue_id", inplace=True)
        self.tissue_annotations["tissue_id"] = self.tissue_annotations.index
        self.tissue_annotations["alignment"] = self.tissue_annotations["alignment"].fillna("single modality")
        filter = self.tissue_annotations[modalities].sum(axis=1) > 0
        self.tissue_annotations = self.tissue_annotations[filter]
        self.load_mask = load_mask
        self.mask_type = mask_type

        # NEED TO FIX
        if 'split' not in self.tissue_annotations.columns:
            if is_rank0():
                logger.warning("No split information found in tissue annotations. Creating a new split.")

            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_idx, test_idx in gss.split(self.tissue_annotations, groups=self.tissue_annotations["patient_id"]):
                self.tissue_annotations.loc[self.tissue_annotations.index[train_idx], "split"] = "train"
                self.tissue_annotations.loc[self.tissue_annotations.index[test_idx], "split"] = "test"

            

        # Set crop strategy (either random_crops or grid_crops)
        crop_strategies = [self.unimodal_datasets[mod].crop_strategy for mod in self.modalities]
        self.crop_strategy = crop_strategies[0]
        assert all([crop_strategies[i] == crop_strategies[0] for i in range(len(crop_strategies))]), "All modalities should have the same crop strategy"
        # assert self.crop_strategy in ['random_crops', 'grid_crops', 'focused_crops'], "Crop strategy should be either 'random_crops' or 'grid_crops'"

        self.rnd_crops_per_image = {}
        if os.path.exists(os.path.join(self.root_dir,  f"index_{self.crop_strategy}_{self.rnd_crop_size}.csv")):
            self._count_random_crops()

            self.crop_annotations = pd.read_csv(os.path.join(self.root_dir,  f"index_{self.crop_strategy}_{self.rnd_crop_size}.csv"))
            self.crop_annotations = self.crop_annotations[self.crop_annotations.tissue_id.isin(self.get_tissue_ids())]
            self.unique_crops = self.crop_annotations.drop_duplicates(subset=['tissue_id', 'crop_id']).drop('modality', axis=1) # double entries can occur if multiple modalities are present
        else:
            logger.error(f"Crop index file {os.path.join(self.root_dir,  f'index_{self.crop_strategy}_{self.rnd_crop_size}.csv')} does not exist. Please create crops first.")
            
        

    def __len__(self):
        return len(self.unique_crops)

    def __getitem__(self, idx):
        row = self.unique_crops.iloc[idx]
        tissue_id = row["tissue_id"]
        crop_id = row["crop_id"]
        modalities = self.get_modalities_of_tissue(tissue_id)
        sample = {'tissue_id': tissue_id, 'crop_id': crop_id}
        for modality in modalities:
            crop = self.get_crop(tissue_id, crop_id, modality)
            sample[modality] = crop
            
        # --- ADDED: Load Mask ---
        if self.load_mask:
            # Assuming get_cell_mask_crop fetches the correct mask type or uses a default
            # You might need to pass mask_type to get_cell_mask_crop if it supports it
            # The provided code for get_cell_mask_crop uses task="segmentation", check if that aligns with "broad_cell_type"
            # or if you need to implement fetching specific mask types.
            # For now, let's try using get_cell_mask_crop.
            mask = self.get_cell_mask_crop(tissue_id, crop_id) # You might need to adjust task/type here
            if mask is not None:
                sample[self.mask_type] = mask
        # ------------------------

        return sample
    

    def get_tissue_ids(self, split=None):
        """
        Get all tissue ids in the dataset
        """
        if split is None:
            return self.tissue_annotations["tissue_id"].values
        else:
            return self.tissue_annotations[self.tissue_annotations["split"]==split]["tissue_id"].values
    
    def get_tissue(self, tissue_id, modality, process=True, remove_channels=False):
        """
        Gets the tissue image for a given tissue id and modality
        """
        return self.unimodal_datasets[modality].get_tissue(tissue_id, process=process, remove_channels=remove_channels)
    
    def get_tissue_mask(self, tissue_id):
        """
        Gets the tissue segmentation mask for a given tissue id
        """
        mask = np.load(os.path.join(self.tissue_masks_dir, f"{tissue_id}.npy")).astype(bool)
        return mask
    
    def get_cell_mask(self, tissue_id, task="segmentation", resize=True):
        """
        Gets the cell mask for a given tissue id
        Args:
            tissue_id: id of the tissue
            task: name of the task. Defaults to segementation for cell instance mask.
        """
        if not os.path.exists(os.path.join(self.cell_masks_dir, task)):
            raise ValueError(f"Cell mask directory for task {task} does not exist.")
        if not os.path.exists(os.path.join(self.cell_masks_dir, task, f"{tissue_id}.npy")):
            logger.info(f"Cell mask for tissue {tissue_id} and task {task} does not exist. ")
            return None
            raise ValueError(f"Cell mask for tissue {tissue_id} and task {task} does not exist.")
        mask = np.load(os.path.join(self.cell_masks_dir, task, f"{tissue_id}.npy"))
        if resize:
            modalities = self.get_modalities_of_tissue(tissue_id)
            tissue_size = self.unimodal_datasets[modalities[0]].get_tissue_size(tissue_id)
            if mask.shape != tissue_size:
                print('resizeing!')
                mask = resize(mask, tissue_size, order=0)
        return mask
    
    def get_cell_mask_crop(self, tissue_id, crop_id, task="segmentation"):
        mask = self.get_cell_mask(tissue_id, task=task, resize=True)
        if mask is None:
            return None
        df = self.crop_annotations[(self.crop_annotations.tissue_id == tissue_id) & (self.crop_annotations.crop_id == crop_id)]
        row = df['row'].iloc[0]
        col = df['col'].iloc[0]
        mask = mask[row:row+self.rnd_crop_size, col:col+self.rnd_crop_size]
        return mask

    def get_marker_embedding_indices(self, tissue_id, modality):
        """
        Gets the indices of the measured markers w.r.t to the marker embedding 
        """
        return self.unimodal_datasets[modality].get_marker_embedding_indices(tissue_id)
    
    def get_crop(self, tissue_id, crop_id, modality, remove_channels=False):
        """
        Gets a specific crop of a tissue for a given modality
        """
        return self.unimodal_datasets[modality].get_crop(tissue_id, crop_id, remove_channels=remove_channels)
    
    def get_rnd_crop(self, tissue_id, modality):
        """
        Gets a random crop of a tissue for a given modality
        """
        return self.unimodal_datasets[modality].get_rnd_crop(tissue_id)
    
    def get_modalities_of_tissue(self, tissue_id):
        """
        Gets the modalties available for a given tissue
        """
        annotation = self.tissue_annotations.loc[tissue_id]
        return [modality for modality in self.modalities if annotation[modality]==1]

    def create_crops_all(self):
        """
        Creates random crops for all tissues in the dataset
        """

        for mod in self.modalities:
            # check if random crop folder is not empty
            crop_files = os.listdir(self.unimodal_datasets[mod].rnd_crop_folder)
            assert len(crop_files) == 0, f"Crop folder {self.unimodal_datasets[mod].rnd_crop_folder} is not empty. It is recommended to delete old crop folders before creating new crops."
        crop_index = []
        for tissue_id in tqdm(self.get_tissue_ids()):
            subindex = self.create_crops_tissue_id(tissue_id)
            crop_index += subindex
        crop_index = pd.DataFrame(crop_index)

        if self.crop_strategy == 'random_crops':
            crop_index.to_csv(os.path.join(self.root_dir, f"index_random_crops_{self.rnd_crop_size}.csv"), index=False)
        elif self.crop_strategy == 'grid_crops':
            crop_index.to_csv(os.path.join(self.root_dir, f"index_grid_crops_{self.rnd_crop_size}.csv"), index=False)

    def create_crops_tissue_id(self, tissue_id):
        """
        Creates random crops for a specific tissue
        Args:
            tissue_id: Id of the tissue to create crops for
            random: If True, creates random crops. If False, creates crops in a grid with stride of rnd_crop_size//2.
        """
        crop_index = []

        modalities = self.get_modalities_of_tissue(tissue_id)

        # Ugly workaround to match width and height of all modalities. This should not be necessary, but due to the way the images are created some images have one pixel difference in width or height.
        h_ws = [self.unimodal_datasets[modalities[i]].get_tissue_size(tissue_id) for i in range(len(modalities))]
        H = min([h_w[0] for h_w in h_ws])
        W = min([h_w[1] for h_w in h_ws])
        logger.debug(f"Minimum Height and width is: {H}, {W} of {h_ws}")
        
        # H, W = self.unimodal_datasets[modalities[0]].get_tissue_size(tissue_id)
        
        if H <= self.rnd_crop_size or W <= self.rnd_crop_size:
            logger.error(f"Image {tissue_id} is too small for random crops. Skipping.")
            return crop_index
        
        match self.crop_strategy:
            case "random_crops":
                max_crops = (H / self.rnd_crop_size) * (W / self.rnd_crop_size)
                max_crops = int(max_crops) * 4
                row_coords = np.random.randint(0, H - self.rnd_crop_size, max_crops) 
                col_coords = np.random.randint(0, W - self.rnd_crop_size, max_crops) 
            case "grid_crops":
                rows = np.arange(0, H - self.rnd_crop_size, self.rnd_crop_size // 2)
                columns = np.arange(0, W - self.rnd_crop_size, self.rnd_crop_size // 2)
                row_coords, col_coords = np.meshgrid(rows, columns)
                row_coords = row_coords.flatten()
                col_coords = col_coords.flatten()
            case _:
                raise NotImplementedError(f"Crop strategy {self.crop_strategy} not implemented")

        tissue_mask = self.get_tissue_mask(tissue_id)  
        tissue_mask = resize(tissue_mask, (H, W))

        num_cops_modality = []
        for modality in modalities:
            dataset = self.unimodal_datasets[modality]
            row_coords, col_coords = dataset._create_crops(tissue_id, row_coords, col_coords, tissue_mask) # careful, row and column coords are overwritten based on first modality
            if len(row_coords) == 0:
                raise ValueError(f"No valid crops found for image {tissue_id} in modality {dataset.modality}. This is highly problematic and should be investigated.")
            for cid, (row, col) in enumerate(zip(row_coords, col_coords)):
                crop_index.append({'tissue_id': tissue_id,
                                   'crop_id': cid,
                                   'modality': modality,
                                   'row': row,
                                   'col': col})
            num_cops_modality.append(len(row_coords))
        
        assert all([num_cops_modality[i] == num_cops_modality[0] for i in range(len(num_cops_modality))]), f"Number of crops for {tissue_id} is not the same across modalities. Its advisable to delete all crops of this tissue and re-run the script"
        return crop_index
    
    def _count_random_crops(self):
        """
        Counts the random crops available.
        """
        for dataset in self.unimodal_datasets.values():
            dataset._count_random_crops()
            for tissue_id in dataset.get_tissue_ids():
                if tissue_id in self.rnd_crops_per_image:
                    assert self.rnd_crops_per_image[tissue_id] == dataset.rnd_crops_per_image[tissue_id], f"Number of crops for {tissue_id} is not the same across modalities. Its advisable to delete all crops of this tissue and re-run the script \
                        . Found {self.rnd_crops_per_image[tissue_id]} for {self.name} and {dataset.rnd_crops_per_image[tissue_id]} for {dataset.name}"
                else:
                    self.rnd_crops_per_image[tissue_id] = dataset.rnd_crops_per_image[tissue_id]