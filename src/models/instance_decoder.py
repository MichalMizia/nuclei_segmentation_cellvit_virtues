import torch
import torch.nn as nn
from src.models.cellvit_decoder import CellViTDecoder
import numpy as np
from src.utils.cell_post_processor import DetectionCellPostProcessor
from typing import Literal, OrderedDict


class InstanceDecoder(CellViTDecoder):
    """
    Extended CellViT Decoder for instance segmentation that produces:
    - Nuclei type map (cell classification)
    - Nuclei binary map (cell vs background)
    - HV map (horizontal/vertical gradients for instance separation)

    Args:
        num_nuclei_classes (int): Number of nuclei classes for segmentation.
        embed_dim (int, optional): Embedding dimension from encoder. Defaults to 512.
        drop_rate (float, optional): Dropout rate. Defaults to 0.
        original_channels (int | None, optional): Number of channels in the original input image.
        upsample_bottleneck (bool): Whether to upsample the embedding on the first stage.
        patch_dropout_rate (float, optional): Patch dropout rate during training. Defaults to 0.0.
    """

    def __init__(
        self,
        num_nuclei_classes: int,
        embed_dim: int = 512,
        drop_rate: float = 0,
        original_channels: int | None = None,
        upsample_bottleneck: bool = False,
        patch_dropout_rate: float = 0.0,
    ):
        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            embed_dim=embed_dim,
            drop_rate=drop_rate,
            original_channels=original_channels,
            upsample_bottleneck=upsample_bottleneck,
            patch_dropout_rate=patch_dropout_rate,
        )

        self.nuclei_binary_map_decoder = self.create_upsampling_branch(
            num_classes=2,
            upsample_bottleneck=upsample_bottleneck,
        )

        self.hv_map_decoder = self.create_upsampling_branch(
            num_classes=2,  # 2 channels: horizontal and vertical gradients
            upsample_bottleneck=upsample_bottleneck,
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> dict:
        """
        Forward pass that generates all three outputs.

        Args:
            x (torch.Tensor): PSS tokens from encoder. Shape (B, H_patch, W_patch, D)
                or list of tensors for skip connections [he_image, z1, z2, z3, pss]

        Returns:
            dict: Dictionary containing:
                - 'nuclei_type_map': (B, num_classes, H, W) - Cell type classification logits
                - 'nuclei_binary_map': (B, 2, H, W) - Binary segmentation logits
                - 'hv_map': (B, 2, H, W) - Horizontal/vertical gradients [-1, 1]
        """
        if isinstance(x, torch.Tensor) and x.shape[-1] == self.embed_dim:
            x = self._apply_patch_dropout(x.permute(0, 3, 1, 2))
        elif isinstance(x, list) and all(
            t.shape[-1] in (self.embed_dim, self.original_channels) for t in x
        ):
            x = [self._apply_patch_dropout(t.permute(0, 3, 1, 2)) for t in x]

        if isinstance(x, list):
            z0, z1, z2, z3, z4 = x
        else:
            # Simulate skip connections
            z0 = z1 = z2 = z3 = z4 = x

        out_dict = {}

        # Binary Map (Cell vs Background)
        out_dict["nuclei_binary_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
        )

        # HV Map (Horizontal/Vertical gradients for instance separation)
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )

        return out_dict

    # def calculate_instance_map(
    #     self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    # ) -> tuple[torch.Tensor, list[dict]]:
    #     """Calculate Instance Map from network predictions (after Softmax output)

    #     Args:
    #         predictions (dict): Dictionary with the following required keys:
    #             * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
    #             * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
    #         magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

    #     Returns:
    #         Tuple[torch.Tensor, List[dict]]:
    #             * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
    #             * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
    #                 For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
    #     """
    #     # reshape to B, H, W, C
    #     predictions_ = predictions.copy()
    #     predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
    #         0, 2, 3, 1
    #     )
    #     predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

    #     cell_post_processor = DetectionCellPostProcessor(
    #         nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
    #     )
    #     instance_preds = []
    #     type_preds = []

    #     for i in range(predictions_["nuclei_binary_map"].shape[0]):
    #         pred_map = np.concatenate(
    #             [
    #                 torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
    #                 .detach()
    #                 .cpu()[..., None],
    #                 torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
    #                 .detach()
    #                 .cpu()[..., None],
    #                 predictions_["hv_map"][i].detach().cpu(),
    #             ],
    #             axis=-1,
    #         )
    #         instance_pred = cell_post_processor.post_process_single_image(pred_map)
    #         instance_preds.append(instance_pred[0])
    #         type_preds.append(instance_pred[1])

    #     return torch.Tensor(np.stack(instance_preds)), type_preds

    # def generate_instance_nuclei_map(
    #     self, instance_maps: torch.Tensor, type_preds: list[dict]
    # ) -> torch.Tensor:
    #     """Convert instance map (binary) to nuclei type instance map

    #     Args:
    #         instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
    #         type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

    #     Returns:
    #         torch.Tensor: Nuclei type instance map. Shape: (B, self.num_nuclei_classes, H, W)
    #     """
    #     batch_size, h, w = instance_maps.shape
    #     instance_type_nuclei_maps = torch.zeros(
    #         (batch_size, h, w, self.num_nuclei_classes)
    #     )
    #     for i in range(batch_size):
    #         instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
    #         instance_map = instance_maps[i]
    #         type_pred = type_preds[i]
    #         for nuclei, spec in type_pred.items():
    #             nuclei_type = spec["type"]
    #             instance_type_nuclei_map[:, :, nuclei_type][
    #                 instance_map == nuclei
    #             ] = nuclei

    #         instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

    #     instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
    #     return torch.Tensor(instance_type_nuclei_maps)
