import torch


def compute_cell_oversampling_weights(dataset, gamma_s=0.5, num_classes=10):
    """
    Compute cell class oversampling weights (ignoring tissue types).

    Based on CellViT eq. (12):

    Args:
        dataset: EmbeddingsDataset instance
        gamma_s: Oversampling strength [0, 1]. 0=no oversampling, 1=max balancing
        num_classes: Number of nuclei classes INCLUDING background (so 10 total: 0-9)

    Returns:
        weights: Tensor of sampling weights for each sample
    """
    N_train = len(dataset)

    # Create binary vectors for cell classes present in each sample
    # cell_presence[i, c] = 1 if class c is present in sample i
    cell_presence = torch.zeros((N_train, num_classes))

    for i in range(N_train):
        di = dataset[i]
        if len(di) == 4:
            mask = di[3]  # ImageDataset returns (tid, he_img, cycif_img, mask)
        else:
            mask = di[2]  # EmbeddingsDataset

        unique_classes = torch.unique(mask)
        for c in unique_classes:
            c_int = int(c.item())
            if 0 <= c_int < num_classes:  # Excluding background (class 0)
                cell_presence[i, c_int] = 1

    # total number of class presences across all samples
    # EXCLUDING background class (index 0)
    N_cell = cell_presence[:, 1:].sum().item()

    # Count how many samples contain each cell class
    cell_class_counts = cell_presence.sum(dim=0)  # Shape: (num_classes,)

    w_cell = torch.zeros(N_train)

    for i in range(N_train):
        weight_sum = 0.0

        for c in range(1, num_classes):  # Start from 1 to skip background
            if cell_presence[i, c] == 1:
                numerator = N_cell**gamma_s
                denominator = gamma_s * cell_class_counts[c] + (1 - gamma_s) * N_cell

                if denominator > 0:
                    weight_sum += numerator / denominator

        w_cell[i] = (1 - gamma_s) + gamma_s * weight_sum

    return w_cell / w_cell.max()
