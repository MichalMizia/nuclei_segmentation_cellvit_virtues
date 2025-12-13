import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from src.models.utils.blocks import Conv2DBlock, Deconv2DBlock


class CellViTDecoder(nn.Module):
    """
    CellViT Decoder that takes VirTues encoder output (PSS tokens) and produces segmentation masks.
    Since VirTues is frozen and only provides the bottleneck features (PSS), we simulate skip connections
    by reusing the bottleneck features and upsampling them as needed.

    Args:
        num_nuclei_classes (int): Number of nuclei classes for segmentation.
        embed_dim (int, optional): Embedding dimension from VirTues encoder. Defaults to 512.
        original_channels (int | None, optional): Number of channels in the original input image.
            Used if we pass the original image as a skip connection. If None, defaults to embed_dim, i.e
            we use the embedding features for all skip connections.
        upsample_bottleneck (bool): Whether to upsample the embedding on the first stage of cellvit decoder.
            If true, the total upsampling factor will be 16x, else 8x.

    Virtues patch_size=8, CellViT expected input patch_size=16, so we have to downsample somewhere.
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
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.num_nuclei_classes = num_nuclei_classes
        self.patch_dropout_rate = patch_dropout_rate
        if original_channels is not None:
            self.original_channels = original_channels
        else:
            self.original_channels = embed_dim

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # Decoder blocks (Skip connection processing)
        # Since we reuse the bottleneck features for all skips, the input dim is always embed_dim
        self.decoder0 = nn.Sequential(
            Conv2DBlock(
                self.original_channels,
                32,
                3,
                dropout=self.drop_rate,
            ),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        if upsample_bottleneck:
            self.decoder1 = nn.Sequential(
                Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
                Deconv2DBlock(
                    self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate
                ),
                Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
            )
            self.decoder2 = nn.Sequential(
                Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
                Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
            )
            self.decoder3 = nn.Sequential(
                Deconv2DBlock(  # upsampling block
                    self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate
                ),
            )
        else:
            self.decoder1 = nn.Sequential(
                Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
                Deconv2DBlock(
                    self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate
                ),
                Conv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
            )
            self.decoder2 = nn.Sequential(
                Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
                Conv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
            )
            self.decoder3 = nn.Sequential(
                Conv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
            )

        # Upsampling branches
        # self.nuclei_binary_map_decoder = self.create_upsampling_branch(2)
        # self.hv_map_decoder = self.create_upsampling_branch(2)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes, upsample_bottleneck=upsample_bottleneck
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> dict:
        """
        Args:
            x (torch.Tensor): PSS tokens from VirTues. Shape (B, H_patch, W_patch, D)
            or list of tensors for skip connections [he_image, z1, z2, z3, pss]

        Returns:
            dict: Segmentation maps
        """
        # Permute to (B, D, H, W)
        if isinstance(x, torch.Tensor) and x.shape[-1] == self.embed_dim:
            x = self._apply_patch_dropout(x.permute(0, 3, 1, 2))
        elif isinstance(x, list) and all(
            t.shape[-1] in (self.embed_dim, self.original_channels) for t in x
        ):
            x = [(self._apply_patch_dropout(t.permute(0, 3, 1, 2))) for t in x]

        if isinstance(x, list):
            z0, z1, z2, z3, z4 = x
        else:
            # Simulate skip connections
            z0 = z1 = z2 = z3 = z4 = x
        out_dict = {}

        # We only use the nuclei_type_map for the broad_cell_type task usually,
        # but we'll generate all for completeness/compatibility

        # Binary Map (Cell vs Background)
        # out_dict["nuclei_binary_map"] = self._forward_upsample(
        #     z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
        # )

        # # HV Map (Horizontal/Vertical gradients for instance separation)
        # out_dict["hv_map"] = self._forward_upsample(
        #     z0, z1, z2, z3, z4, self.hv_map_decoder
        # )

        # Type Map (Cell Classification)
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        return out_dict

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder,
    ) -> torch.Tensor:

        # Bottleneck
        b4 = branch_decoder.bottleneck_upsampler(z4)  # z4=pss, z0=he_image

        # Decoder 3
        b3 = self.decoder3(z3)
        if b3.shape[-2:] != b4.shape[-2:]:
            b3 = F.interpolate(
                b3, size=b4.shape[-2:], mode="bilinear", align_corners=False
            )
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))

        # Decoder 2
        b2 = self.decoder2(z2)
        if b2.shape[-2:] != b3.shape[-2:]:
            b2 = F.interpolate(
                b2, size=b3.shape[-2:], mode="bilinear", align_corners=False
            )
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))

        # Decoder 1
        b1 = self.decoder1(z1)
        if b1.shape[-2:] != b2.shape[-2:]:
            b1 = F.interpolate(
                b1, size=b2.shape[-2:], mode="bilinear", align_corners=False
            )
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))

        # Decoder 0
        b0 = self.decoder0(z0)
        if b0.shape[-2:] != b1.shape[-2:]:
            b0 = F.interpolate(
                b0, size=b1.shape[-2:], mode="bilinear", align_corners=False
            )

        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(
        self, num_classes: int, upsample_bottleneck: bool = False
    ) -> nn.Sequential:
        """
        Creates an upsampling branch for segmentation map generation.
        Args:
            num_classes (int): Number of output classes for the segmentation map.
            upsample_factor (int): Factor to upsample the input feature map. Options are 8/16, default is 8
        """
        if upsample_bottleneck:
            bottleneck_upsampler = nn.ConvTranspose2d(
                in_channels=self.embed_dim,
                out_channels=self.bottleneck_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            )
        else:
            bottleneck_upsampler = nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.bottleneck_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def _apply_patch_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly zero out tokens (patches) with probability `self.patch_dropout_rate`.
        Args:
            x: Tensor of shape (B, D, H, W)
        Returns:
            Tensor of same shape with some patches zeroed out.
        """
        if not self.training or self.patch_dropout_rate == 0.0:
            return x

        B, D, H, W = x.shape
        mask = torch.rand(B, 1, H, W, device=x.device) > self.patch_dropout_rate
        return x * mask
