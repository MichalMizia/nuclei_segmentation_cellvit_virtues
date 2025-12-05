import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from src.cellvit.models.utils.blocks import Conv2DBlock, Deconv2DBlock

class CellViTDecoder(nn.Module):
    
    def __init__(
        self,
        num_nuclei_classes: int,
        embed_dim: int = 512,
        drop_rate: float = 0,
    ):
        print("Reloaded Decoder")
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.num_nuclei_classes = num_nuclei_classes

        # Bridge maintains resolution
        self.bridge = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1)

        # Pooling layers to restore feature hierarchy
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # Decoder blocks
        self.decoder0 = nn.Sequential(
            Conv2DBlock(self.embed_dim, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        # Upsampling branches
        self.nuclei_binary_map_decoder = self.create_upsampling_branch(2)
        self.hv_map_decoder = self.create_upsampling_branch(2)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(self.num_nuclei_classes)

    def forward(self, x: torch.Tensor) -> dict:
        if x.shape[-1] == self.embed_dim:
            x = x.permute(0, 3, 1, 2)
        
        x = self.bridge(x)
        
        # Artificial feature pyramid
        z0 = x               # High Res (1x)
        z1 = x               # High Res (1x)
        z2 = self.pool2(x)   # Mid Res (2x downsample)
        z3 = self.pool4(x)   # Low Res (4x downsample)
        z4 = self.pool4(x)   # Bottleneck (4x downsample)

        out_dict = {}
        
        def get_map(decoder_branch):
            return self._forward_upsample(z0, z1, z2, z3, z4, decoder_branch)

        out_dict["nuclei_binary_map"] = get_map(self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = get_map(self.hv_map_decoder)
        out_dict["nuclei_type_map"] = get_map(self.nuclei_type_maps_decoder)

        return out_dict

    def _forward_upsample(self, z0, z1, z2, z3, z4, branch_decoder):
        b4 = branch_decoder.bottleneck_upsampler(z4)
        
        # Decoder 3
        b3 = self.decoder3(z3)
        if b3.shape[-2:] != b4.shape[-2:]:
            b3 = F.interpolate(b3, size=b4.shape[-2:], mode='bilinear', align_corners=False)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        
        # Decoder 2
        b2 = self.decoder2(z2)
        if b2.shape[-2:] != b3.shape[-2:]:
            b2 = F.interpolate(b2, size=b3.shape[-2:], mode='bilinear', align_corners=False)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        
        # Decoder 1
        b1 = self.decoder1(z1)
        if b1.shape[-2:] != b2.shape[-2:]:
            b1 = F.interpolate(b1, size=b2.shape[-2:], mode='bilinear', align_corners=False)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        
        # Decoder 0
        b0 = self.decoder0(z0)

        # --- THIS IS THE ONLY CHANGE ---
        # OLD: if b0.shape != b1.shape: b0 = interpolate(b0, size=b1) (This caused the shrinking)
        # NEW: We force b1 (context) to match b0 (high-res input)
        if b1.shape[-2:] != b0.shape[-2:]:
            b1 = F.interpolate(b1, size=b0.shape[-2:], mode='bilinear', align_corners=False)
        # -------------------------------
        
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        bottleneck_upsampler = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=2, stride=2, padding=0),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0),
        )

        decoder = nn.Sequential(
            OrderedDict([
                ("bottleneck_upsampler", bottleneck_upsampler),
                ("decoder3_upsampler", decoder3_upsampler),
                ("decoder2_upsampler", decoder2_upsampler),
                ("decoder1_upsampler", decoder1_upsampler),
                ("decoder0_header", decoder0_header),
            ])
        )
        return decoder