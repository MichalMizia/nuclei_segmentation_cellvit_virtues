from scipy.ndimage import find_objects
import numpy as np


def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """
    Generate horizontal and vertical distance maps. Vectorized version of CellVit function found at
    https://github.com/TIO-IKIM/CellViT/blob/main/cell_segmentation/utils/tools.py#L24

    Args:
        inst_map: Instance map (H, W) with unique integer per instance

    Returns:
        hv_map: (2, H, W) - horizontal and vertical gradients [-1, 1]
    """
    orig_inst_map = inst_map.astype(np.int32)

    x_map = np.zeros(orig_inst_map.shape, dtype=np.float32)
    y_map = np.zeros(orig_inst_map.shape, dtype=np.float32)

    inst_list = np.unique(orig_inst_map)
    inst_list = inst_list[inst_list != 0]

    slices = find_objects(orig_inst_map)

    for inst_id in inst_list:
        if inst_id == 0 or inst_id - 1 >= len(slices):
            continue

        slice_obj = slices[inst_id - 1]
        if slice_obj is None:
            continue

        y_slice, x_slice = slice_obj
        y_start, y_end = y_slice.start, y_slice.stop
        x_start, x_end = x_slice.start, x_slice.stop

        y_start = max(0, y_start - 2)
        x_start = max(0, x_start - 2)
        y_end = min(orig_inst_map.shape[0], y_end + 2)
        x_end = min(orig_inst_map.shape[1], x_end + 2)

        inst_map_crop = (orig_inst_map[y_start:y_end, x_start:x_end] == inst_id).astype(
            np.uint8
        )

        if inst_map_crop.shape[0] < 2 or inst_map_crop.shape[1] < 2:
            continue

        y_coords, x_coords = np.nonzero(inst_map_crop)
        if len(y_coords) == 0:
            continue

        inst_com_y = int(np.mean(y_coords) + 0.5)
        inst_com_x = int(np.mean(x_coords) + 0.5)

        h, w = inst_map_crop.shape
        inst_y_range = np.arange(1, h + 1) - inst_com_y
        inst_x_range = np.arange(1, w + 1) - inst_com_x

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        inst_x = inst_x.astype(np.float32)
        inst_y = inst_y.astype(np.float32)
        inst_x[inst_map_crop == 0] = 0
        inst_y[inst_map_crop == 0] = 0

        x_neg = inst_x < 0
        x_pos = inst_x > 0
        y_neg = inst_y < 0
        y_pos = inst_y > 0

        if x_neg.any():
            inst_x[x_neg] /= -inst_x[x_neg].min()
        if x_pos.any():
            inst_x[x_pos] /= inst_x[x_pos].max()
        if y_neg.any():
            inst_y[y_neg] /= -inst_y[y_neg].min()
        if y_pos.any():
            inst_y[y_pos] /= inst_y[y_pos].max()

        mask_region = inst_map_crop > 0
        x_map[y_start:y_end, x_start:x_end][mask_region] = inst_x[mask_region]
        y_map[y_start:y_end, x_start:x_end][mask_region] = inst_y[mask_region]

    hv_map = np.stack([x_map, y_map])
    return hv_map
