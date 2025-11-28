from dataclasses import dataclass
from torchvision.transforms.v2 import Compose
from src.utils.utils import is_rank0
from loguru import logger

class MyStr(str):
    def __eq__(self, other):
        return self.__contains__(other)

@dataclass(kw_only=True)
class BaseMultiplexNormalizeMetadata:
    normalizer_name: str
    rnd_crop_folder_name: str 
    channel_file_name: str
    mean_name: str
    std_name: str 

@dataclass(kw_only=True)
class BaseHENormalizeMetadata:
    rnd_crop_folder_name: str


@dataclass(kw_only=True)
class QuantileMultiplexNormalizeMetadata(BaseMultiplexNormalizeMetadata):
    quantile_path: str



def get_normalize_metadata(normalization, rnd_crop_size, crop_strategy):
    # normalization = MyStr(normalization)

    if 'no_overlap_' in crop_strategy:
        if is_rank0():
            logger.warning(f"Using no overlap crop strategy with size {rnd_crop_size}.")
        crop_strategy = crop_strategy.replace("no_overlap_", "")

    match normalization:
        case "global_std":
            normalize_metadata = BaseMultiplexNormalizeMetadata(
                normalizer_name="global_std",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}",
                channel_file_name="channels",
                mean_name="mean",
                std_name="std"
            )
        case "approx_global_std":
            normalize_metadata = QuantileMultiplexNormalizeMetadata(
                normalizer_name="approx_global_std",
                # rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}_no_log",
                channel_file_name="channels",
                mean_name="approximate_mean",
                std_name="approximate_std",
                quantile_path="quantiles/0.99/quantiles.csv"
            )
        case "tm_local_q_global_std":
            normalize_metadata = QuantileMultiplexNormalizeMetadata(
                normalizer_name="tm_local_q_global_std",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}_no_log",
                channel_file_name="channels",
                mean_name="quantiles/tm_clip99/global_level_means.csv",
                std_name="quantiles/tm_clip99/global_level_stds.csv",
                quantile_path="quantiles/tm_clip99/image_level_quantiles.csv"
            )
        case "tm_local_q_rescale_global_std":
            normalize_metadata = QuantileMultiplexNormalizeMetadata(
                normalizer_name="tm_local_q_rescale_global_std",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}_no_log",
                channel_file_name="channels",
                mean_name="quantiles/tm_rescale255_clip99/global_level_means.csv",
                std_name="quantiles/tm_rescale255_clip99/global_level_stds.csv",
                quantile_path="quantiles/tm_rescale255_clip99/image_level_quantiles.csv"
            )
        case "raw":
            normalize_metadata = BaseMultiplexNormalizeMetadata(
                normalizer_name="raw",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}_no_log",
                channel_file_name="channels_global_vals_no_log",
                mean_name="mean",
                std_name="std"
            )
        case "he":
            normalize_metadata = BaseHENormalizeMetadata(
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}"
            )
        
        case "q99_mean_std":
            normalize_metadata = QuantileMultiplexNormalizeMetadata(
                normalizer_name="q99_mean_std",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}_no_log",
                channel_file_name="channels",
                mean_name="quantiles/0.99/means.csv",
                std_name="quantiles/0.99/stds.csv",
                quantile_path="quantiles/0.99/quantiles.csv"
            )

        case "q_99":
            normalize_metadata = QuantileMultiplexNormalizeMetadata(
                normalizer_name="q_99",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}_no_log",
                channel_file_name="channels",
                mean_name="mean",
                std_name="std",
                quantile_path="quantiles/q99.csv"
            )

        case "global_log_compress_fg_0.99_focused":
            normalize_metadata = QuantileMultiplexNormalizeMetadata(
                normalizer_name="global_log_compress_fg_0.99",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}",
                channel_file_name="channels",
                mean_name="quantiles/log_compress_fg_0.99/global_means.csv",
                std_name="quantiles/log_compress_fg_0.99/global_stds.csv",
                quantile_path="quantiles/log_compress_fg_0.99/global_quantiles.csv"
            )
        
        case "log_compress_fg_0.99_focused":
            normalize_metadata = QuantileMultiplexNormalizeMetadata(
                normalizer_name="log_compress_fg_0.99",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}",
                channel_file_name="channels",
                mean_name="quantiles/log_compress_fg_0.99/means.csv",
                std_name="quantiles/log_compress_fg_0.99/stds.csv",
                quantile_path="quantiles/log_compress_fg_0.99/quantiles.csv"
            )
        
        case "log_compress_fg_0.99":
            normalize_metadata = QuantileMultiplexNormalizeMetadata(
                normalizer_name="log_compress_fg_0.99",
                rnd_crop_folder_name=f"{crop_strategy}_{rnd_crop_size}_no_log",
                channel_file_name="channels",
                mean_name="quantiles/log_compress_fg_0.99/means.csv",
                std_name="quantiles/log_compress_fg_0.99/stds.csv",
                quantile_path="quantiles/log_compress_fg_0.99/quantiles.csv"
            )
        
        case _:
            raise NotImplementedError(f"Normalization {normalization} not implemented")

    return normalize_metadata