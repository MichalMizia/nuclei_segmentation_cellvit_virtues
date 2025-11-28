import numpy as np
import random
import torch
from torchvision.transforms import v2
import math
from einops import rearrange
from loguru import logger
import torch.nn.functional as F
from typing import Dict, Tuple, List

class DropChannels(object):

    def __init__(self, p=0.5, fraction_range=[0.5,0.5]):
        self.p = p
        self.fraction_range = fraction_range

    @logger.catch
    def __call__(self, *args):
        if np.random.rand() < self.p or self.p == 1.0:
            lengths = set([len(arg) for arg in args])
            assert len(lengths) == 1, 'All inputs must have the same number of channels'
            num_channels = len(args[0])
            fraction = random.uniform(*self.fraction_range)
            num_channels_to_keep = np.ceil(num_channels * fraction).astype(int)
            indices = np.random.choice(num_channels, num_channels_to_keep, replace=False)
            return [arg[indices] for arg in args]
        else:
            return args
        
        
class DropChannelsFixedNumber(object):

    def __init__(self, p=0.5, num_keep=1):
        self.p = p
        self.num_keep = num_keep

    def __call__(self, *args):

        if np.random.rand() < self.p or self.p == 1.0:
            num_channels_to_keep = min(self.num_keep, len(args[0]))
            indices = np.random.choice(len(args[0]), num_channels_to_keep, replace=False)
            return [arg[indices] for arg in args]
        

class DropChannelsFixedNumberRange(object):

    def __init__(self, p=0.5, num_keep_min=1, num_keep_max=1):
        self.p = p
        self.num_keep_min = num_keep_min
        self.num_keep_max = num_keep_max

    def __call__(self, *args):

        if np.random.rand() < self.p or self.p == 1.0:
            # num_channels_to_keep = min(self.num_keep, len(args[0]))
            num_channels_to_keep = np.random.randint(self.num_keep_min, self.num_keep_max + 1)
            num_channels_to_keep = min(num_channels_to_keep, len(args[0]))
            indices = np.random.choice(len(args[0]), num_channels_to_keep, replace=False)
            return [arg[indices] for arg in args]

class DropChannelsNuclearKnown(object):
    def __init__(self, num_choose=2):
        """
        Args:
            num_choose: Number of channels (except nuclear) to choose.
        """
        self.num_choose = num_choose

    def choice_with_fixed(self, num_channels, fixed_index):
        # always the first element should be fixed_index
        # then sample num_choose indices from the rest
        indices = [fixed_index]
        if self.num_choose == -1:
            remaining_indices = list(set(range(num_channels)) - {fixed_index})
            indices += remaining_indices
            return indices
        remaining_indices = list(set(range(num_channels)) - {fixed_index})
        if len(remaining_indices) < self.num_choose:
            indices += remaining_indices
        else:
            indices += np.random.choice(remaining_indices, self.num_choose, replace=False).tolist()
        return indices

    def __call__(self, *args, fixed_index=None):
        indices = self.choice_with_fixed(len(args[0]), fixed_index=fixed_index)
        if len(indices) > 0:
            return [arg[indices] for arg in args]
        else:
            return args

class HierchicalChannelSampling(object):

    def __init__(self, min_channels=1):
        self.min_channels = min_channels

    @logger.catch
    def __call__(self, *args):
        # if np.random.rand() < self.p or self.p == 1.0:
        lengths = set([len(arg) for arg in args])
        assert len(lengths) == 1, 'All inputs must have the same number of channels'
        num_channels = len(args[0])
        num_channels_to_keep = np.random.randint(self.min_channels, num_channels + 1)
        indices = np.random.choice(num_channels, num_channels_to_keep, replace=False)
        return [arg[indices] for arg in args]


class CustomGaussianBlur(object):

    def __init__(self, kernel_size, sigma):
        self.transform = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            return self.transform(img).numpy()
        else:
            return self.transform(img)
        

class CustomGaussianBlur(object):

    def __init__(self, kernel_size, sigma):
        self.transform = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            return self.transform(img).numpy()
        else:
            return self.transform(img)

class RandomRotation(object):

    def __init__(self):
        pass

    def __call__(self, img):
        """
        img: ... x H x W
        """
        flip = torch.flip if isinstance(img, torch.Tensor) else np.flip
        r = random.randint(0, 3)
        if r == 0:
            return img
        elif r == 1:
            return flip(img, (-1,)).transpose(-1, -2)
        elif r == 2:
            return flip(img, (-1, -2))
        elif r == 3:
            return flip(img, (-2,)).transpose(-1, -2)
        
class MultiImageRandomRotation(object):

    def __init__(self):
        pass
    
    def __call__(self, *imgs):
        """
        imgs: list of ... x H x W
        """
        flip = torch.flip if isinstance(imgs[0], torch.Tensor) else np.flip
        permute = torch.permute if isinstance(imgs[0], torch.Tensor) else np.transpose

        r = random.randint(0, 3)
        if r == 0:
            return [img for img in imgs]
        elif r == 1:
            # rotate by 90 degrees
            return [permute(flip(img, (-1,)), (0, -1, -2)) for img in imgs]
        elif r == 2:
            # rotate by 180 degrees
            return [flip(img, (-1, -2)) for img in imgs]
        elif r == 3:
            return [permute(flip(img, (-2,)), (0, -1, -2)) for img in imgs]

class RandomSymmety(object):

    def __init__(self):
        self.random_rotation = RandomRotation()
        pass

    def __call__(self, img):
        """
        img: ... x H x W
        """
        img = self.random_rotation(img)
        if np.random.rand() < 0.5:
            img = img.flip(-1)
        return img
    
class MultiImageRandomSymmetry(object):

    def __init__(self):
        self.random_rotation = MultiImageRandomRotation()
        pass

    def __call__(self, *imgs):
        imgs = self.random_rotation(*imgs)
        if np.random.rand() < 0.5:
            imgs = [img.flip(-1) for img in imgs]
        return imgs
    
class CropToPatchSize(object):

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        num_patches_x  = img.shape[-2] // self.patch_size
        num_patches_y = img.shape[-1] // self.patch_size
        return img[..., :num_patches_x * self.patch_size, :num_patches_y * self.patch_size]

class GridReshape(object):

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        assert img.shape[-2] % self.patch_size == 0 and img.shape[-1] % self.patch_size == 0, 'Image dimensions must be divisible by patch size'
        return rearrange(img, 'c (h p1) (w p2) -> c h w (p1 p2)', p1=self.patch_size, p2=self.patch_size)
    
class RandomRescaleChannel(object):

    def __init__(self, p=0.5, scale=(4/5, 5/4)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if np.random.rand() < self.p:
            scales = torch.rand(size=(img.shape[0],)) * (self.scale[1] - self.scale[0]) + self.scale[0]
            return img * scales[:, None, None]
        return img

class PerChannelRescale(object):

    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            m = img.max(axis=(-1, -2), keepdims=True)
            m = np.where(m == 0, np.ones_like(m), m)
            img = img / m
            return img
        elif isinstance(img, torch.Tensor):
            m = img.max(dim=-1)[0].max(dim=-1)[0][:,None,None]
            m = torch.where(m == 0, torch.ones_like(m), m)
            img = img / m
            return img
        else:
            raise ValueError('Unknown input type')

class PerChannelSelfStandardization(object):
    # implements self-standarization as recommended in https://arxiv.org/pdf/2301.05768
    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            mean = img.mean(axis=(-1, -2), keepdims=True)
            std = img.std(axis=(-1, -2), keepdims=True)
            std = np.where(std == 0, np.ones_like(std), std)
            return (img - mean) / std
        elif isinstance(img, torch.Tensor):
            mean = img.mean(axis=(-1, -2), keepdims=True)
            std = img.std(axis=(-1, -2), keepdims=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            return (img - mean) / std
        else:
            raise ValueError('Unknown input type')
    
class PerChannelSelfStandardizationNoCentering(object):

    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            std = img.std(axis=(-1, -2), keepdims=True)
            std = np.where(std == 0, np.ones_like(std), std)
            return img / std
        elif isinstance(img, torch.Tensor):        
            std = img.std(axis=(-1, -2), keepdims=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            return img / std
        else:
            raise ValueError('Unknown input type')
        
class PerChannelUnitSecondMomentum(object):

    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            sec_momentum = np.sqrt((img**2).mean(axis=(-1, -2), keepdims=True))
            sec_momentum = np.where(sec_momentum == 0, np.ones_like(sec_momentum), sec_momentum)
            return img / sec_momentum
        elif isinstance(img, torch.Tensor):
            sec_momentum = img.pow(2).mean(axis=(-1, -2), keepdims=True).sqrt()
            sec_momentum = torch.where(sec_momentum == 0, torch.ones_like(sec_momentum), sec_momentum)
            return img / sec_momentum
        else:
            raise ValueError('Unknown input type')

class GlobalNormalize(object):

    def __init__(self, mean, std):
        if isinstance(mean, np.ndarray):
            self.mean = mean
            self.std = std
        elif isinstance(mean, torch.Tensor):
            self.mean = mean.numpy()
            self.std = std.numpy()
        else:
            raise ValueError('Unknown input type')
        self.mean = self.mean[:, None, None]
        self.std = self.std[:, None, None]

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return (img - self.mean) / self.std
        elif isinstance(img, torch.Tensor):
            return (img - torch.from_numpy(self.mean)) / torch.from_numpy(self.std)
        else:
            raise ValueError('Unknown input type')

def get_normalization_transform(name, global_mean, global_std):
    if name == 'unit_rescale':
        return PerChannelRescale()
    elif name == 'global_std':
        return GlobalNormalize(global_mean, global_std)
    elif name == 'self_std':
        return PerChannelSelfStandardization()
    elif name == 'self_std_no_center':
        return PerChannelSelfStandardizationNoCentering()
    elif name == 'unit_second_momentum':
        return PerChannelUnitSecondMomentum()
    elif name == "none":
        return lambda x: x
    else:
        raise ValueError(f'Unknown normalization transform {name}')

def get_normalization_transform_rgb():
    return v2.Compose([
        v2.ToTensor(),
        # v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_inverse_normalization_transform_rgb():
    return v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

class MultiImageRandomCrop(object):
    """
    Crops the same view out of a list of images of equal size.
    """
    def __init__(self, size):
        self.size = size
        
    def __call__(self, *images):
        target_h = self.size[0]
        target_w = self.size[1]
        output = []
        """        
        for i in range(len(images)):
            img = images[i]
            h, w = img.shape[-2], img.shape[-1]

            if h < target_h or w < target_w:
                top = (target_h - h) // 2
                bottom = target_h - h - top
                left = (target_w - w) // 2
                right = target_w - w - left
                if img.dim() == 3:
                    img = v2.functional.pad(img, [left, top, right, bottom], padding_mode='constant')
                else:
                    img = img[None, :, :]
                    img = v2.functional.pad(img, [left, top, right, bottom], padding_mode='edge')
                    img = img.squeeze(0)
            output.append(img)
        """
        h, w = images[0].shape[-2], images[0].shape[-1]

        r = random.randint(0, h - target_h)
        c = random.randint(0, w - target_w)
        for i in range(len(images)):
            if images[i] is None:
                output.append(None)
                continue
            img = images[i]
            img = img[..., r:r+target_h, c:c+target_w]
            output.append(img)

        return output

 
def sample_mask_area(H, W, mask_ratio):
    ratio = mask_ratio['mask_ratio']
    mask_ratio_upper = ratio[1]
    mask_ratio_lower = ratio[0]
    rnd_mask_ratio = random.uniform(mask_ratio_lower, mask_ratio_upper)
    mask_volume = math.ceil(H * W * rnd_mask_ratio)
    return mask_volume

def dirichlet_v1(C, H, W, mask_ratio, min_visible_channels=3, debug=False):
    target_mean = (mask_ratio['mask_ratio'][0] + mask_ratio['mask_ratio'][1]) / 2
    alpha = mask_ratio['alpha']

    # 1. Sample Dirichlet and scale to target mean
    raw_ratios = np.random.dirichlet([alpha] * C)
    scaled_ratios = raw_ratios * (target_mean * C) / np.sum(raw_ratios)

    # 2. Force some channels to be visible (≤ 0.5)
    visible_idx = np.argsort(scaled_ratios)[:min_visible_channels]
    scaled_ratios[visible_idx] = np.minimum(scaled_ratios[visible_idx], 0.5)

    # 3. Renormalize masked channels to preserve target mean
    all_idx = np.arange(C)
    masked_idx = np.setdiff1d(all_idx, visible_idx)
    visible_sum = np.sum(scaled_ratios[visible_idx])
    target_total = target_mean * C
    remaining_budget = target_total - visible_sum
    current_sum = np.sum(scaled_ratios[masked_idx])
    if current_sum > 0:
        scaled_ratios[masked_idx] *= remaining_budget / current_sum

    # 4. Clip to [0, 1]
    scaled_ratios = np.clip(scaled_ratios, 0.0, 1.0)

    if debug:
        print(f"scaled_ratios:\n{scaled_ratios}")
        print(f"mean: {scaled_ratios.mean():.3f}, fully masked: {(scaled_ratios == 1.0).sum()}, visible: {(scaled_ratios <= 0.5).sum()}")

    # 5. Create patch-level binary masks
    masks = []
    for ratio in scaled_ratios:
        n_mask = int(H * W * ratio)
        mask = torch.zeros(H * W, dtype=torch.bool)
        mask[:n_mask] = 1
        mask = mask[torch.randperm(H * W)].reshape(H, W)
        masks.append(mask)

    return torch.stack(masks, dim=0)

def dirichlet_v2(
    C: int,
    H: int,
    W: int,
    mask_ratio_config: Dict[str, List[float]],
) -> Tuple[np.ndarray, np.ndarray, float]: # Added target_avg_mask_ratio to return
    """
    Generates masks for multi-channel images using Dirichlet sampling.

    Args:
        C (int): Number of channels.
        H (int): Height of the image/feature map per channel.
        W (int): Width of the image/feature map per channel.
        mask_ratio_config (Dict[str, List[float]]): Dictionary containing the key
            'mask_ratio' with a list [lower_ratio, higher_ratio] defining the
            range for the target *average* mask ratio across channels.
        alpha (float): Concentration parameter for the Dirichlet distribution.
                       Lower values (<1) increase variance, higher values (>1)
                       decrease variance around the mean. alpha=1 is uniform.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - masks (np.ndarray): Boolean mask array of shape (C, H, W).
                                  `True` indicates a masked token.
            - channel_ratios (np.ndarray): The actual mask ratio applied per
                                           channel, shape (C,).
            - target_avg_mask_ratio (float): The average ratio sampled for this run.
    """
    # if not (0.0 <= mask_ratio_config['mask_ratio'][0] <= 1.0 and \
    #         0.0 <= mask_ratio_config['mask_ratio'][1] <= 1.0 and \
    #         mask_ratio_config['mask_ratio'][0] <= mask_ratio_config['mask_ratio'][1]):
    #     raise ValueError("Invalid mask_ratio range. Must be [low, high] between 0.0 and 1.0.")

    # if alpha <= 0:
    #     raise ValueError("Dirichlet alpha parameter must be > 0.")

    # 1. Sample the target *average* mask ratio for this specific image
    lower_ratio, higher_ratio = mask_ratio_config['mask_ratio']
    alpha = mask_ratio_config['alpha']
    target_avg_mask_ratio = np.random.uniform(lower_ratio, higher_ratio)

    # 2. Sample proportions from the Dirichlet distribution
    dirichlet_params = np.full(C, alpha)
    proportions = np.random.dirichlet(dirichlet_params)

    # 3. Scale proportions to get target per-channel mask ratios
    channel_ratios_prelim = proportions * C * target_avg_mask_ratio

    # 4. Clip ratios to be within [0.0, 1.0]
    channel_ratios = np.clip(channel_ratios_prelim, 0.0, 1.0)

    # 5. Generate masks for each channel based on its specific ratio
    masks = np.zeros((C, H, W), dtype=bool)
    num_tokens_per_channel = H * W

    for c in range(C):
        channel_mask_ratio = channel_ratios[c]
        num_tokens_to_mask = int(np.round(num_tokens_per_channel * channel_mask_ratio))

        if num_tokens_to_mask == 0:
            # channel_ratios[c] = 0.0 # Ensure actual ratio is exactly 0
            continue
        if num_tokens_to_mask >= num_tokens_per_channel:
            masks[c, :, :] = True
            # channel_ratios[c] = 1.0 # Ensure actual ratio is exactly 1
            continue

        # Generate random indices to mask for this channel
        # Ensure we don't re-use indices if somehow num_tokens_to_mask > num_tokens_per_channel
        # Although clipping should prevent this unless H*W is very small.
        indices = np.random.choice(num_tokens_per_channel, num_tokens_to_mask, replace=False)

        # Convert flat indices to 2D coordinates
        row_indices, col_indices = np.unravel_index(indices, (H, W))

        # Apply the mask
        masks[c, row_indices, col_indices] = True

        # Update actual ratio based on rounding
        # channel_ratios[c] = num_tokens_to_mask / num_tokens_per_channel


    # return masks #, channel_ratios, target_avg_mask_ratio
    return torch.from_numpy(masks)


def generate_mask(C, H, W, mask_ratio, mask_strategy):
    if mask_strategy == 'uniform_independent':
        masks = []
        for _ in range(C):
            mask_area = sample_mask_area(H, W, mask_ratio)
            mask = torch.zeros(H*W, dtype=bool)
            mask[:mask_area] = True
            mask = mask[torch.randperm(H*W)].reshape(H, W)
            masks.append(mask)
        mask = torch.stack(masks, dim=0)
    elif mask_strategy == 'dirichlet_v1':
        mask = dirichlet_v1(C, H, W, mask_ratio)
    elif mask_strategy == 'dirichlet_v2':
        mask = dirichlet_v2(C, H, W, mask_ratio)

    elif mask_strategy == "uniform_coupled":
        mask_area = sample_mask_area(H, W, mask_ratio)
        total_area = H * W
        mask =  torch.zeros(H*W, dtype=bool)
        mask[:mask_area] = True
        mask = mask[torch.randperm(total_area)].reshape(H, W)
        mask = mask.unsqueeze(0).expand(C, H, W).clone()

    elif mask_strategy == "block_independent":
        masks = [generate_block_mask(H, W, mask_ratio) for _ in range(C)]
        mask = torch.stack(masks, dim=0)

    elif mask_strategy == "block_coupled":
        mask = generate_block_mask(H, W, mask_ratio)
        mask = mask.unsqueeze(0).expand(C, H, W)

    elif mask_strategy == "checkerboard1":
        x = torch.arange(W).unsqueeze(0).expand(H, W)
        y = torch.arange(H).unsqueeze(-1).expand(H, W)
        mask = (x + y) % 2 == 0
        mask = mask.unsqueeze(0).expand(C, H, W)
    elif mask_strategy == "checkerboard2":
        x = torch.arange(W).unsqueeze(0).expand(H, W)
        y = torch.arange(H).unsqueeze(-1).expand(H, W)
        mask = ((x//2) + (y//2)) % 2 == 0
        mask = mask.unsqueeze(0).expand(C, H, W)
    elif mask_strategy == "checkerboard3":
        x = torch.arange(W).unsqueeze(0).expand(H, W)
        y = torch.arange(H).unsqueeze(-1).expand(H, W)
        mask = ((x//3) + (y//3)) % 2 == 0
        mask = mask.unsqueeze(0).expand(C, H, W)
    elif mask_strategy == "checkerboard4":
        x = torch.arange(W).unsqueeze(0).expand(H, W)
        y = torch.arange(H).unsqueeze(-1).expand(H, W)
        mask = ((x//4) + (y//4)) % 2 == 0
        mask = mask.unsqueeze(0).expand(C, H, W)
    else:
        raise ValueError(f"Unknown mask strategy {mask_strategy}")
    return mask   

        
def generate_block_mask(H, W, mask_ratio):
    area_to_mask = sample_mask_area(H, W, mask_ratio)
    mask = torch.zeros(H, W, dtype=bool)
    num_masks_placed = 0
    while num_masks_placed < area_to_mask:
        sucess  = 0
        for _ in range(10):
            upper_bound  = area_to_mask - num_masks_placed
            lower_bound =  (min(H,W) // 3)**2
            target_area  = random.uniform(lower_bound, upper_bound)
            aspect_ratio = math.exp(random.uniform(math.log(0.3), math.log(1/0.3)))
            block_height = int(round(math.sqrt(target_area *aspect_ratio)))
            block_weight = int(round(math.sqrt(target_area / aspect_ratio)))
            if block_height < H and block_weight < W:
                row = random.randint(0, H - block_height)
                col = random.randint(0, W - block_weight)
                num_already_masked = mask[row:row + block_height, col:col + block_weight].sum()
                if 0 < block_height*block_weight - num_already_masked <= upper_bound:
                    placed  = block_height*block_weight - num_already_masked
                    mask[row:row + block_height, col:col + block_weight] = 1
                    num_masks_placed += placed
                    sucess += 1
            if sucess > 0:
                break
        if sucess == 0:
            break
    return mask

def custom_median_filter(input_tensor: torch.Tensor, kernel_size: int = 3, padding: str = 'reflect') -> torch.Tensor:
    """
    Applies a median filter to a 4D input tensor (batch, channels, height, width).
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
        kernel_size (int): Size of the kernel (must be odd, e.g., 3, 5, 7)
        padding (str): Padding mode ('reflect', 'replicate', or 'constant')
    
    Returns:
        torch.Tensor: Filtered tensor of the same shape as input
    """
    # Ensure kernel_size is odd
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    
    # Calculate padding
    pad = kernel_size // 2
    
    # Pad the input tensor
    padded = F.pad(input_tensor, (pad, pad, pad, pad), mode=padding)
    
    # Unfold the tensor to get all patches of size kernel_size x kernel_size
    # Shape after unfold: (B, C, H * W, kernel_size * kernel_size)
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    B, C, H, W, _, _ = patches.shape
    patches = patches.reshape(B, C, H, W, kernel_size * kernel_size)
    
    # Compute median along the last dimension (across the kernel)
    # Shape after median: (B, C, H, W)
    filtered = torch.median(patches, dim=-1).values
    
    return filtered