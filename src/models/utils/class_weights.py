import torch
from src.dataset.datasets.mm_base import MultimodalDataset


def compute_class_weights(
    train_tids, ds: MultimodalDataset, num_classes: int, clamp_val=20.0
) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies in the dataset.

    Args:
        dataset (NucleiSegmentationDataset): The dataset containing segmentation masks.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the computed class weights.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float32)

    for tid in train_tids:
        mask = ds.get_cell_mask(tid, task="broad_cell_type", resize=False) # (H, W)
        mask = torch.from_numpy(mask).cuda()
        for class_idx in range(num_classes):
            class_counts[class_idx] += torch.sum(mask == class_idx).item()

    total_pixels = class_counts.sum().item()
    class_frequencies = class_counts / total_pixels
    eps = 1e-6  # avoid zero division
    class_weights = 1.0 / (class_frequencies + eps)
    class_weights = class_weights / class_weights[0]  # background weight = 1.0

    class_weights = torch.clip(class_weights, max=clamp_val)

    return class_weights
