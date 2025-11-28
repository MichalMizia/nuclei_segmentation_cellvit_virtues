import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.ndimage import gaussian_filter
import numpy as np
import torch

def plot_rgb_image(image, mask=None, ax=None, title=None):
    """
    Plots an RGB image with an optional mask overlay.
    Args:
    image: (3, h, w) or (h, w, 3) tensor. Channel dimension is detected automatically.
    mask: (H, W) boolean tensor. If provided, will overlay the mask on the image.
    """
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    h, w = image.shape[0], image.shape[1]

    if ax == None:
        _, ax = plt.subplots()
    ax.imshow(image)

    if mask is not None:
        H, W = mask.shape[0], mask.shape[1]
        patch_size_h = h // H
        patch_size_w = w // W
        for r in range(H):
            for c in range(W):
                if mask[r, c] == True:
                    ax.add_patch(plt.Rectangle((c*patch_size_w, r*patch_size_h), patch_size_w, patch_size_h, color="white"))

    if title is not None:
        ax.set_title(title)
    
    return ax

def plot_intensity_image(image, mask=None, ax=None, vmin=None, vmax=None, cmap='inferno', title=None,
                         return_mappable=False):
    """
    Plots an intensity image with an optional mask overlay.
    Args:
    image: (h, w) tensor.
    mask: (H, W) boolean tensor. If provided, will overlay the mask on the image.
    """
    h, w = image.shape[0], image.shape[1]

    if ax == None:
        _, ax = plt.subplots()
    g = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

    if mask is not None:
        H, W = mask.shape[0], mask.shape[1]
        patch_size_h = h // H
        patch_size_w = w // W
        for r in range(H):
            for c in range(W):
                if mask[r, c] == True:
                    ax.add_patch(plt.Rectangle((c*patch_size_w, r*patch_size_h), patch_size_w, patch_size_h, color="white"))

    if title is not None:
        ax.set_title(title)
    if return_mappable:
        return ax, g
    return ax

def visualize_multichannel_image(_img, cmap_name='hsv', clip=True, apply_gaussian=False, normalize=False, title=None, ax=None, return_rgb=False):
    """
    Create a composite RGB visualization of a multi-channel image. t
    
    Args:
        img (np.ndarray): Input image of shape (C, H, W)
        cmap_name (str): Name of matplotlib colormap (e.g., 'hsv', 'tab20')
        
    Returns:
        np.ndarray: RGB image of shape (H, W, 3), values in [0, 1]
    """
    img = _img.copy()
    assert img.ndim == 3, "Input image must have shape (C, H, W)"
    C, H, W = img.shape
    if clip:
        q99 = np.quantile(img, 0.99, axis=(1,2))
        img = np.clip(img, 0, q99[:, None, None])
        if normalize:
            img = img / q99[:, None, None]
    if apply_gaussian:
        img = gaussian_filter(img, sigma=1, axes=(1, 2))

    cmap = get_cmap(cmap_name, C)
    
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Normalize each channel and blend
    for i in range(C):
        ch = img[i]
        ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        color = np.array(cmap(i)[:3])
        rgb += ch_norm[..., None] * color

    # Normalize composite to [0, 1]
    rgb /= rgb.max() + 1e-8
    if return_rgb:
        return rgb
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    g = ax.imshow(rgb)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    # fig.colorbar(g, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    # plt.show()
    return img

def plot_mask(mask, id_to_name, name_to_color, ax=None, legend=False, ticks=False, return_rgb=False):
    from matplotlib import patches as mpatches

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return np.array(list(int(hex_color[i:i+2], 16) for i in (0, 2, 4)), dtype=np.uint8)

    def transform_mask_to_RGB(mask, id_to_rgba):
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for id, color in id_to_rgba.items():
            rgb[mask == id] = color
        return rgb

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    id_to_rgb = {id: hex_to_rgb(name_to_color[name]) for id, name in id_to_name.items()}
    mask_rgb = transform_mask_to_RGB(mask, id_to_rgb)

    values = sorted(id_to_name.keys())
    labels = [id_to_name[v] for v in values]
    colors = [name_to_color[id_to_name[v]] for v in values]
    # cmap = ListedColormap(colors)
    # if ax is None:
    #     plt.imshow(mask, cmap=cmap, vmin=values[0], vmax=values[-1])
    # else:
    #     ax.imshow(mask, cmap=cmap, vmin=values[0], vmax=values[-1])
    if return_rgb:
        return mask_rgb
    
    if ax is None:
        plt.imshow(mask_rgb)
    else:
        ax.imshow(mask_rgb)

    if not ticks:
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

    if legend:
        # pathces with black stroke
        patches = [mpatches.Patch(facecolor=colors[i], label=labels[i], edgecolor='k') for i in range(len(values))]
        plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1, 1))