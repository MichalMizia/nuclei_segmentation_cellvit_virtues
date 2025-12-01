import torch
import torch.nn as nn
from collections import OrderedDict
from src.cellvit.models.utils.blocks import Conv2DBlock, Deconv2DBlock

class VirTuesCellViT(nn.Module):
    def __init__(self, virtues_encoder, num_nuclei_classes=6, virtues_embed_dim=512, drop_rate=0):
        super().__init__()
        self.encoder = virtues_encoder
        self.embed_dim = virtues_embed_dim
        self.drop_rate = drop_rate
        
        # ... (Decoder config same as before) ...
        if self.embed_dim < 512:
            self.skip_dim_11 = 256; 
            self.skip_dim_12 = 128; 
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512; 
            self.skip_dim_12 = 256; 
            self.bottleneck_dim = 512

        # ... (Decoder Blocks same as before) ...
        self.decoder0 = nn.Sequential(Conv2DBlock(self.embed_dim, 32, 3, dropout=self.drop_rate), Conv2DBlock(32, 64, 3, dropout=self.drop_rate))
        self.decoder1 = nn.Sequential(Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate), Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate), Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate))
        self.decoder2 = nn.Sequential(Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate), Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate))
        self.decoder3 = nn.Sequential(Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate))

        self.nuclei_binary_map_decoder = self.create_upsampling_branch(2) 
        self.hv_map_decoder = self.create_upsampling_branch(2)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(self.num_nuclei_classes)

    # --- UPDATED FORWARD METHOD ---
    def forward(self, multiplex_images, he_images, channel_ids):
        """
        Args:
            multiplex_images: [B, C, H, W] (CyCIF)
            he_images: [B, 3, H, W] (H&E) - NEW ARGUMENT
            channel_ids: List of channel indices
        """
        # We verify if he_images is a tensor or list (VirTues expects list of tensors)
        if isinstance(he_images, torch.Tensor):
            he_list = [he_images[i] for i in range(he_images.shape[0])]
        else:
            he_list = he_images

        if isinstance(multiplex_images, torch.Tensor):
            mx_list = [multiplex_images[i] for i in range(multiplex_images.shape[0])]
        else:
            mx_list = multiplex_images

        # Pass BOTH to VirTues
        _, pss_list = self.encoder.forward_list(mx_list, he_list, channel_ids)
        
        # Stack PSS
        x = torch.stack(pss_list) 
        z_final = x.permute(0, 3, 1, 2) # [B, 512, 16, 16]

        # Feature Expansion
        z4 = z_final; z3 = z_final; z2 = z_final; z1 = z_final; z0 = z_final 

        # Decode
        out_dict = {}
        out_dict["nuclei_binary_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
        # ... (other heads) ...

        return out_dict

    # ... (Helpers _forward_upsample and create_upsampling_branch remain same) ...
    # (Paste them from previous cells)
    def _forward_upsample(self, z0, z1, z2, z3, z4, branch_decoder):
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        if b3.shape[-2:] != b4.shape[-2:]: b3 = nn.functional.interpolate(b3, size=b4.shape[-2:])
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        if b2.shape[-2:] != b3.shape[-2:]: b2 = nn.functional.interpolate(b2, size=b3.shape[-2:])
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        if b1.shape[-2:] != b2.shape[-2:]: b1 = nn.functional.interpolate(b1, size=b2.shape[-2:])
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        if b0.shape[-2:] != b1.shape[-2:]: b0 = nn.functional.interpolate(b0, size=b1.shape[-2:])
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
        return branch_output

    def create_upsampling_branch(self, num_classes):
        bottleneck_upsampler = nn.ConvTranspose2d(self.embed_dim, self.bottleneck_dim, 2, 2, 0, 0)
        decoder3_upsampler = nn.Sequential(Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate), Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate), Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate), nn.ConvTranspose2d(self.bottleneck_dim, 256, 2, 2, 0, 0))
        decoder2_upsampler = nn.Sequential(Conv2DBlock(256 * 2, 256, dropout=self.drop_rate), Conv2DBlock(256, 256, dropout=self.drop_rate), nn.ConvTranspose2d(256, 128, 2, 2, 0, 0))
        decoder1_upsampler = nn.Sequential(Conv2DBlock(128 * 2, 128, dropout=self.drop_rate), Conv2DBlock(128, 128, dropout=self.drop_rate), nn.ConvTranspose2d(128, 64, 2, 2, 0, 0))
        decoder0_header = nn.Sequential(Conv2DBlock(64 * 2, 64, dropout=self.drop_rate), Conv2DBlock(64, 64, dropout=self.drop_rate), nn.Conv2d(64, num_classes, 1, 1, 0))
        return nn.Sequential(OrderedDict([("bottleneck_upsampler", bottleneck_upsampler), ("decoder3_upsampler", decoder3_upsampler), ("decoder2_upsampler", decoder2_upsampler), ("decoder1_upsampler", decoder1_upsampler), ("decoder0_header", decoder0_header)]))