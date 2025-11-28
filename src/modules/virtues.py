import torch
from torch import nn
try:
    import xformers
    from src.modules.flex_dual_virtues.layers.transformer_xformers import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock
except ImportError:
    from src.modules.flex_dual_virtues.layers.transformers_flashattention import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock 
from src.modules.flex_dual_virtues.layers.positional_embeddings import LearnablePositionalEmbedding2D, PositionalEmbedding2D
from einops import rearrange

def build_virtues_mae(conf, protein_emb):

    return VirTuesMAE(protein_emb=protein_emb,
                          patch_size=conf.image_info.patch_size,
                          model_dim=conf.model.model_dim,
                          feedforward_dim=conf.model.feedforward_dim,
                          encoder_pattern=conf.model.encoder_pattern,
                          num_encoder_heads=conf.model.num_encoder_heads,
                          mae_decoder_pattern=conf.model.decoder_pattern,
                          mae_num_decoder_heads=conf.model.num_decoder_heads,
                          mae_num_hidden_layers_head=conf.model.num_hidden_layers_head,
                          dropout=conf.model.dropout,
                          pos_emb=conf.model.pos_emb,
                    )

class VirTuesEncoder(nn.Module):

    def __init__(self,
                protein_emb,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                encoder_pattern="hwhw",
                num_encoder_heads=8,
                dropout=0.0,
                pos_emb="rope",
                ):
        super().__init__()
        self.patch_size = patch_size

        self.register_buffer("protein_emb", protein_emb, persistent=False)
        self.patch_summary_token = nn.Parameter(torch.randn(model_dim)/model_dim**2)
        self.masked_token = nn.Parameter(torch.randn(model_dim)/model_dim**2)

        self.patch_encoder = nn.Linear(patch_size**2, model_dim)
        self.protein_encoder = nn.Linear(protein_emb.shape[1], model_dim)     
        
        if pos_emb == "learnable":
            self.positional_embedding = LearnablePositionalEmbedding2D(model_dim, max_pos=100)
        elif pos_emb == "absolute_beginning":
            self.positional_embedding = PositionalEmbedding2D(model_dim=model_dim, max_width_or_height=100)
        else:
            self.positional_embedding = None

        enc_layers = []
        for pattern in encoder_pattern:
            if pattern == "|" or pattern == "v":
                enc_layers.append(MarkerAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            elif pattern == "-" or pattern == "h":
                enc_layers.append(ChannelAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            elif pattern == "f":
                enc_layers.append(FullAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            else:
                raise ValueError("encoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention)")
        
        self.encoder = nn.ModuleList(enc_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, channel_ids, mask=None):
        """
        x: B x C x h x w
        channel_ids: B x C
        Returns:
        x: B x C x H x W x D
        ps: B x H x W x D
        """
        h, w = x.shape[-2], x.shape[-1]
        B = x.shape[0]
        C = x.shape[1]
        H = h // self.patch_size
        W = w // self.patch_size

        x = rearrange(x, "B C (H p) (W q) -> B C H W (p q)", p=self.patch_size, q=self.patch_size)

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(B, C+1, H, W, 2)
        pos = rearrange(pos, "b c h w d -> b c (h w) d")

        x = self.patch_encoder(x)

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.masked_token.expand(x.shape), x)
            mask = rearrange(mask, "b c h w -> b c (h w)")
        x = rearrange(x, "b c h w d -> b c (h w) d")

        proteins = self.protein_emb[channel_ids] # B x C x P
        proteins = self.protein_encoder(proteins) # B x C x D
        proteins = proteins.unsqueeze(2).expand(*x.shape)
        x = x + proteins

        x = torch.concat([self.patch_summary_token.expand(B, 1, H*W, x.shape[-1]), x], dim=1)
        if mask is not None:
            mask = torch.concat([torch.zeros(B, 1, H*W, dtype=torch.bool, device=mask.device), mask], dim=1)

        if self.positional_embedding is not None:
            x = self.positional_embedding(x, pos)

        for layer in self.encoder:
            if mask is None:
                x = layer.forward(x, pos)
            else:
                x = layer.forward_masked(x, pos, mask)

        x = rearrange(x, "b c (h w) d -> b c h w d", h=H, w=W)
        patch_summ = x[:, 0]
        x = x[:, 1:]
        return x, patch_summ

    def forward_list(self, x, channel_ids, mask=None):
        """
        x: list of tensors C_i x h x w
        channel_ids: list of tensors C_i
        mask: list of tensors C_i x H x W or None
        Returns:
        x: list of tensors C_i x H x W x D
        ps: list of tensors H x W x D
        """
        h, w = x[0].shape[-2], x[0].shape[-1]
        B = len(x)
        channels_per_sample = [len(channels) for channels in channel_ids]
        H = h // self.patch_size
        W = w // self.patch_size

        x = [rearrange(x_i, 'C (H p) (W q) -> C H W (p q)', p=self.patch_size, q=self.patch_size) for x_i in x]

        x = torch.concat(x, dim=0) # sum(C_i) x H x W x D
        print('0 dtype', self.patch_encoder(x).dtype)
        print('1 dtype', x.dtype)
        channel_ids = torch.concat(channel_ids, dim=0) # sum(C_i)
        if mask is not None:
            mask = torch.concat(mask, dim=0) # sum(C_i) x H x W

        sum_C = x.shape[0]
        sum_C = sum_C + len(channels_per_sample)

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(sum_C, H, W, 2)
        pos = rearrange(pos, "c h w d -> c (h w) d")

        x = self.patch_encoder(x)
        print('2 dtype', x.dtype)

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.masked_token.expand(x.shape), x)
            mask = rearrange(mask, "c h w -> c (h w)")

        x = rearrange(x, "c h w d -> c (h w) d")

        proteins = self.protein_emb[channel_ids] # sum_C x P
        proteins = self.protein_encoder(proteins) # sum_C x D
        proteins = proteins.unsqueeze(1).expand(*x.shape)
        x = x + proteins

        x = torch.split(x, channels_per_sample, dim=0)
        x = [torch.concat([self.patch_summary_token.expand(1, H*W, x_i.shape[-1]), x_i], dim=0) for x_i in x]
        x = torch.concat(x, dim=0)
        if mask is not None:
            mask = torch.split(mask, channels_per_sample, dim=0)
            mask = [torch.concat([torch.zeros(1, H*W, dtype=torch.bool, device=mask_i.device), mask_i], dim=0) for mask_i in mask]
            mask = torch.concat(mask, dim=0)

        channels_per_sample = [c + 1 for c in channels_per_sample]

        if self.positional_embedding is not None:
            x = self.positional_embedding(x, pos)

        print('3 dtype', x.dtype)
        for layer in self.encoder:
            if mask is None:
                x = layer.forward_cc(x, pos, channels_per_sample)
            else:
                x = layer.forward_cc_masked(x, pos, mask, channels_per_sample)
        
        x = rearrange(x, "c (h w) d -> c h w d", h=H, w=W)
        x = torch.split(x, channels_per_sample, dim=0) # list of tensors C_i x H x W x D
        
        ps = [x_i[0] for x_i in x]
        x = [x_i[1:] for x_i in x]

        return x, ps

class VirTuesDecoder(nn.Module):

    def __init__(self,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                pattern="hwhw",
                num_heads=8,
                num_hidden_layers_head=0,
                dropout=0.0,
                pos_emb="rope",
                ):
        super().__init__()
        
        decoder_layers = []
        if num_hidden_layers_head > 0:
            for _ in range(num_hidden_layers_head -1):
                decoder_layers.append(nn.Linear(model_dim, model_dim))
                decoder_layers.append(nn.GELU())
        decoder_layers.append(nn.Linear(model_dim, patch_size**2))
        self.decoder_mlp = nn.Sequential(*decoder_layers)
        
        dec_layers = []
        for pattern in pattern:
            if pattern == "|" or pattern == "v":
                dec_layers.append(MarkerAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            elif pattern == "-" or pattern == "h":
                dec_layers.append(ChannelAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            elif pattern == "f":
                dec_layers.append(FullAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            else:
                raise ValueError("decoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention)")
        self.decoder = nn.ModuleList(dec_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, ps):
        """
        x: B x C x H x W x D
        ps: B x H x W x D
        Returns : B x C x H x W x D
        """
        B, C, H, W, D = x.shape
        ps = ps[:, None, None, :, :, :].expand(B, C, 1, H, W, D)
        x = x[:,:, None, :, :, :]
        x = torch.concat([ps, x], dim=2) # B x C x 2 x H x W x D
        x = rearrange(x, "b c e h w d -> (b c) e (h w) d")

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(B*C, 2, H, W, 2)
        pos = rearrange(pos, "b e h w d -> b e (h w) d")

        for layer in self.decoder:
            x = layer.forward(x, pos)

        x = x[:, 1] # (B C) S D
        x = self.decoder_mlp(x)
        x = rearrange(x, "(b c) (h w) d -> b c h w d", b=B, c=C, h=H, w=W)
        return x 

    def forward_list(self, x, ps):
        """
        x: list of tensors C_i x H x W x D
        ps: list of tensors H x W x D
        Returns: 
        x: sum(C_i) x H x W x D
        """
        H, W, D = x[0].shape[1], x[0].shape[2], x[0].shape[3]
        x = torch.concat([
            torch.concat([
                ps_i[None, None, :, :, :].expand(x_i.shape[0], 1, H, W, D),
                x_i[:, None , :, :, :],
            ], dim=1)            
            for x_i, ps_i in zip(x, ps)
        ], dim=0) # sum(C_i) x 2 x H x W x D
        x = rearrange(x, "c e h w d -> (c e) (h w) d")

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(x.shape[0], H, W, 2)
        pos = rearrange(pos, "c h w d -> c (h w) d")

        channels = x.shape[0] // 2
        channels_per_sample = [2]*channels

        for layer in self.decoder:
            x = layer.forward_cc(x, pos, channels_per_sample) # x: sum(C_i)*2 x S x D

        x = x[1::2] # sum(C_i) x S x D
        x = self.decoder_mlp(x)
        x = rearrange(x, "c (h w) d -> c h w d", h=H, w=W)
        return x

class VirTuesMAE(nn.Module):

    def __init__(self,
                protein_emb,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                encoder_pattern="hwhw",
                num_encoder_heads=8,
                mae_decoder_pattern="hwhw",
                mae_num_decoder_heads=8,
                mae_num_hidden_layers_head=0,
                dropout=0.0,
                pos_emb="rope",
                ):
        super().__init__()
        self.encoder = VirTuesEncoder(protein_emb, patch_size, model_dim, feedforward_dim, encoder_pattern, num_encoder_heads, dropout, pos_emb)
        self.mae_decoder = VirTuesDecoder(patch_size, model_dim, feedforward_dim, mae_decoder_pattern, mae_num_decoder_heads, mae_num_hidden_layers_head, dropout, pos_emb)
        
    def embed(self, x, channel_ids, mask=None, return_dict=False, place_on_cpu=False):
        x, ps = self.encoder.forward(x, channel_ids, mask=mask) # B x C x H x W x D, B x H x W x D
        cls_token = ps.mean(dim=(1,2)) # B x D
        if return_dict:
            results = [{
                "cls_token": cls_token,
                "patch_summary_token": ps,
            }]
            if place_on_cpu:
                for res in results:
                    for key in res.keys():
                        res[key] = res[key].cpu()
            return results
        else:
            return cls_token, ps
       
    def forward_single(self, x, channel_ids, mask=None):
        """
        x: list of tensors B x C x H x W x D
        channel_ids: list of tensors B x C
        mask: list of tensors B x C x H x W or None
        Returns:
        cls_tokens: B x D
        x: B x C x H x W x D
        """
        x, ps = self.encoder.forward(x, channel_ids, mask=mask)
        recon = self.mae_decoder.forward(x, ps) # B x C x H x W x D
        return recon

    def forward(self, x, channel_ids, mask=None):
        """
        x: list of tensors C_i x H x W x D
        channel_ids: list of tensors C_i
        mask: list of tensors C_i x H x W or None
        Returns:
        cls_tokens: B x D
        x: sum(C_i) x H x W x D        
        """
        # NOTE: Modified here to fit the trainer feature
        x, ps = self.encoder.forward_list(x, channel_ids, mask=mask) # list C_i x H x W x D,  list of H x W x D
        recon = self.mae_decoder.forward_list(x, ps) # sum(C_i) x H x W x D
        return recon
    
    def reconstruct(self, x, channel_ids, mask=None, inject_mask_token_before_decoder=None):
        x, ps = self.encoder.forward(x, channel_ids, mask=mask)
        if inject_mask_token_before_decoder is not None:
            x = inject_mask_token_before_decoder.expand(x.shape)
        recon = self.mae_decoder.forward(x, ps) # B x C x H x W x D
        return recon
