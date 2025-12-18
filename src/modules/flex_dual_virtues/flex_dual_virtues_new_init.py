import torch
from torch import nn
from loguru import logger
from src.utils.utils import is_rank0
import math 

try:
    import xformers
    raise ImportError("xFormers disabled for debugging")
    from src.modules.flex_dual_virtues.layers.transformer_xformers import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock, PatchAttentionBlock
    from src.modules.flex_dual_virtues.layers.mask_utils_xformers import build_selfattention_bias_channel_concat, build_selfattention_bias, SA_BIAS_CACHE
    USE_XFORMERS = True
    USE_FLASHATTENTION = False
    if is_rank0():
        logger.info(f'Using xformers for FlexDualVirTues')
    
except ImportError:
    from src.modules.flex_dual_virtues.layers.transformers_flashattention import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock, PatchAttentionBlock
    from src.modules.flex_dual_virtues.layers.mask_utils_flashattention import build_selfattention_bias_channel_concat, build_selfattention_bias, SA_BIAS_CACHE
    USE_XFORMERS = False
    USE_FLASHATTENTION = True
    if is_rank0():
        logger.info(f'Using flash attention for FlexDualVirTues')
# try:
#     from modules.flex_dual_virtues.layers.transformers_flashattention import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock, CrossAttentionBlock
# except ImportError:
#     raise ImportError("Please install flash attention: pip install flash-attn --no-build-isolation")
from src.modules.flex_dual_virtues.layers.positional_embeddings import LearnablePositionalEmbedding2D, PositionalEmbedding2D, RotaryPositionalEmbedding2D
from einops import rearrange
from itertools import groupby


def sinusoidal_positional_embeddings(seq_len, dim):
    """
    Create sinusoidal positional embeddings (1D).
    
    Args:
        seq_len (int): Length of the sequence.
        dim (int): Dimension of the embedding.
        device (str): Device to place the tensor.
    
    Returns:
        torch.Tensor: [seq_len, dim] positional embeddings.
    """
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    embeddings = torch.zeros((seq_len, dim))
    embeddings[:, 0::2] = torch.sin(position * div_term)  # even indices
    embeddings[:, 1::2] = torch.cos(position * div_term)  # odd indices
    return embeddings


def build_flex_dual_virtues(conf, protein_emb, **kwargs):
    if is_rank0():
        logger.info(f'Building FlexDualVirTues with sqrt init')
    return FlexDualVirTuesMAE(protein_emb=protein_emb,
                          protein_emb_type=conf.model.protein_emb_type if hasattr(conf.model, 'protein_emb_type') else 'esm',
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
                          joint_patch_encoder=conf.model.joint_patch_encoder if hasattr(conf.model, 'joint_patch_encoder') else False,
                          separate_encoders=conf.model.separate_encoders if hasattr(conf.model, 'separate_encoders') else False,
                          separate_encoder_pattern_he=conf.model.separate_encoder_pattern_he if hasattr(conf.model, 'separate_encoder_pattern_he') else "hhhh",
                          separate_encoder_pattern_multiplex=conf.model.separate_encoder_pattern_multiplex if hasattr(conf.model, 'separate_encoder_pattern_multiplex') else "hvhv",
                          separate_decoders=conf.model.separate_decoders if hasattr(conf.model, 'separate_decoders') else False,
                          only_ps_tokens_decoder=conf.model.only_ps_tokens_decoder if hasattr(conf.model, 'only_ps_tokens_decoder') else False,
                          group_layers=conf.model.group_layers if hasattr(conf.model, 'group_layers') else False,
                          norm_after_encoder_decoder=conf.model.norm_after_encoder_decoder if hasattr(conf.model, 'norm_after_encoder_decoder') else False,
                          **kwargs
                    )  

def build_flex_dual_virtues_encoder(conf, protein_emb, pos_emb=None, **kwargs):
    
    return FlexDualVirTuesEncoder(protein_emb=protein_emb,
                            patch_size=conf.image_info.patch_size,
                            protein_emb_type='esm',
                            model_dim=conf.model.model_dim,
                            feedforward_dim=conf.model.feedforward_dim,
                            encoder_pattern=conf.model.encoder_pattern,
                            num_encoder_heads=conf.model.num_encoder_heads,
                            dropout=conf.model.dropout,
                            pos_emb=pos_emb,
                            joint_patch_encoder=conf.model.joint_patch_encoder if hasattr(conf.model, 'joint_patch_encoder') else False,
                            separate_encoders=conf.model.separate_encoders if hasattr(conf.model, 'separate_encoders') else False,
                            group_layers=conf.model.group_layers if hasattr(conf.model, 'group_layers') else False,
                            norm_after_encoder_decoder=conf.model.norm_after_encoder_decoder if hasattr(conf.model, 'norm_after_encoder_decoder') else False,
                        )


class FlexDualVirTuesEncoder(nn.Module):

    def __init__(self,
                protein_emb,
                protein_emb_type,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                encoder_pattern="hvhv",
                num_encoder_heads=8,
                dropout=0.0,
                pos_emb="rope",
                joint_patch_encoder=False,
                separate_encoders=False,
                separate_encoder_pattern_he="hhhh",
                separate_encoder_pattern_multiplex="hvhv",
                group_layers=False,
                norm_after_encoder_decoder=False,
                **kwargs,
                ):
        super().__init__()

        self.patch_size = patch_size
        self.joint_patch_encoder = joint_patch_encoder
        self.separate_encoders = separate_encoders
        self.norm_after_encoder_decoder = norm_after_encoder_decoder

        self.use_protein_emb = protein_emb_type != "empty"

        if protein_emb_type == 'esm' or protein_emb_type == 'one_hot':
            if is_rank0():
                logger.info(f'Using protein embedding: {protein_emb_type} with shape {protein_emb.shape}')
            self.register_buffer("protein_emb", protein_emb, persistent=False)
            self.protein_encoder = nn.Linear(protein_emb.shape[1], model_dim)     
        elif protein_emb_type == 'empty':
            logger.info(f'Not using protein embedding')
        elif protein_emb_type == 'esm_learnable':
            self.protein_emb = nn.Parameter(protein_emb)
            self.protein_encoder = nn.Linear(protein_emb.shape[1], model_dim)
            logger.info(f'Using protein embedding: {protein_emb_type} with shape {protein_emb.shape}')
        elif protein_emb_type == 'sinusoidal':
            pos1d = sinusoidal_positional_embeddings(protein_emb.shape[0], model_dim)
            self.register_buffer("protein_emb", pos1d, persistent=False)
            self.protein_encoder = nn.Identity()
            logger.info(f'Using protein embedding: {protein_emb_type} with shape {pos1d.shape}')
        else:
            raise ValueError("Invalid protein_emb argument. Should be 'esm', 'one_hot' or 'empty'.")
        
        self.protein_fusion_type = kwargs.get("protein_fusion_type", "add")
        if is_rank0():
            logger.info(f'Using protein fusion type: {self.protein_fusion_type}')

        if self.protein_fusion_type == 'cross_attention':
            self.protein_cross_attn = CrossAttentionBlock(
                model_dim=model_dim,
                num_heads=num_encoder_heads,
                dropout=dropout,
                pos_type=pos_emb
            )


        self.use_legacy_init = kwargs.get("use_legacy_init", False)
        if self.use_legacy_init:
            if is_rank0():
                logger.info(f'Using legacy initialization with power 2')
            power = 2
        else:
            power = 0.5
        self.patch_summary_token = nn.Parameter(torch.randn(model_dim)/model_dim**power)
        self.num_registers = kwargs.get("num_registers", 0)
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(self.num_registers, model_dim)/model_dim**power)
            if is_rank0():
                logger.info(f'Using {self.num_registers} registers')

        self.masked_token = nn.Parameter(torch.randn(model_dim)/model_dim**power)

        self.he_marker = nn.Parameter(torch.randn(model_dim)/model_dim**power)

        if self.joint_patch_encoder:
            self.rgb_to_intensity = nn.Linear(3, 1)
            self.patch_encoder = nn.Linear(patch_size**2, model_dim)
        else:
            self.he_patch_encoder = nn.Linear(3*patch_size**2, model_dim)
            self.multiplex_patch_encoder = nn.Linear(patch_size**2, model_dim)
        
        if pos_emb == "learnable":
            self.positional_embedding = LearnablePositionalEmbedding2D(model_dim, max_pos=100)
        elif pos_emb == "absolute_beginning":
            self.positional_embedding = PositionalEmbedding2D(model_dim=model_dim, max_width_or_height=100)
        else:
            self.positional_embedding = None

        enc_layers = []
        if group_layers:
            encoder_pattern = [(label, sum(1 for _ in group)) for label, group in groupby(encoder_pattern)]
            separate_encoder_pattern_he = [(label, sum(1 for _ in group)) for label, group in groupby(separate_encoder_pattern_he)]
            separate_encoder_pattern_multiplex = [(label, sum(1 for _ in group)) for label, group in groupby(separate_encoder_pattern_multiplex)]
        else:
            encoder_pattern = [(label, 1) for label in encoder_pattern]
            separate_encoder_pattern_he = [(label, 1) for label in separate_encoder_pattern_he]
            separate_encoder_pattern_multiplex = [(label, 1) for label in separate_encoder_pattern_multiplex]

        for pattern, depth in encoder_pattern:
            if pattern == "|" or pattern == "v":
                enc_layers.append(MarkerAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=None, num_layers=depth))
            elif pattern == "-" or pattern == "h":
                enc_layers.append(ChannelAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
            elif pattern == "f":
                enc_layers.append(FullAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
            elif pattern == "p":
                enc_layers.append(PatchAttentionBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                    
            else:
                raise ValueError("encoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention) or 'p' (PatchAttention)")
        
        if norm_after_encoder_decoder:
            self.layer_norm = nn.LayerNorm(model_dim)

        self.encoder = nn.ModuleList(enc_layers)

        if self.separate_encoders:
            enc_layers = []
            for pattern, depth in separate_encoder_pattern_he:
                if pattern == "|" or pattern == "v":
                    enc_layers.append(MarkerAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=None, num_layers=depth))
                elif pattern == "-" or pattern == "h":
                    enc_layers.append(ChannelAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "f":
                    enc_layers.append(FullAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "p":
                    enc_layers.append(PatchAttentionBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                else:
                    raise ValueError("encoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention) or 'p' (PatchAttention)")
            self.he_encoder = nn.ModuleList(enc_layers)

            enc_layers = []
            for pattern, depth in separate_encoder_pattern_multiplex:
                if pattern == "|" or pattern == "v":
                    enc_layers.append(MarkerAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=None, num_layers=depth))
                elif pattern == "-" or pattern == "h":
                    enc_layers.append(ChannelAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "f":
                    enc_layers.append(FullAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "p":
                    enc_layers.append(PatchAttentionBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                else:
                    raise ValueError("encoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention) or 'p' (PatchAttention)")
            self.multiplex_encoder = nn.ModuleList(enc_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def empty_tensor(self, shape, dtype, device):
        return torch.zeros(0, *shape, dtype=dtype, device=device)

    def forward_list(self, multiplex, he, channel_ids, multiplex_mask=None, he_mask=None, get_router_logits=False, return_intermediates=False):
        """
        he: list of tensors 3 x h x w or None of length B
        multiplex: list of tensors C_i x h x w or None of length B
        channel_ids: list of tensors C_i or None of length B
        he_mask: list of tensors H x W or None
        multiplex_mask: list of tensors C_i x H x W or None

        Returns:
        x: list of tensors C_i x H x W x D
        ps: list of tensors H x W x D
        """
        he_none_mask = [h is None for h in he]
        multiplex_none_mask = [m is None for m in multiplex]

        B_he_not_none = len(he) - sum(he_none_mask)
        B_multiplex_not_none = len(multiplex) - sum(multiplex_none_mask)
        B = len(he)

        if False in he_none_mask:
            id = he_none_mask.index(False)
            h, w = he[id].shape[1], he[id].shape[2]
            dtype = he[id].dtype
            device = he[id].device
        else:
            id = multiplex_none_mask.index(False)
            h, w = multiplex[id].shape[1], multiplex[id].shape[2]
            dtype = multiplex[id].dtype
            device = multiplex[id].device
        
        H, W = h // self.patch_size, w // self.patch_size

        multiplex_channels_per_sample = [(len(channels) if channels is not None else 0) for channels in channel_ids]
        he_channels_per_sample = [1 if he_i is not None else 0 for he_i in he]

        multiplex = [(rearrange(multiplex_i, 'C (H p) (W q) -> C H W (p q)', p=self.patch_size, q=self.patch_size) if multiplex_i is not None else torch.zeros(0, H, W, self.patch_size**2, dtype=dtype, device=device)) for multiplex_i in multiplex]
        if self.joint_patch_encoder:
            he = [(rearrange(he_i, 'C h w -> h w C').unsqueeze(0) if he_i is not None else torch.zeros(0, h, w, 3, dtype=dtype, device=device)) for he_i in he] # B_he_not_none x H x W x 3
            he = torch.concat(he, dim=0) # B_he_not_none x H x W x 3
            he = self.rgb_to_intensity(he) # B_he_not_none x H x W x 1
            he = rearrange(he, "b (H p) (W q) c -> b H W (p q c)", p=self.patch_size, q=self.patch_size)
        else:
            he = [(rearrange(he_i, 'C (H p) (W q) -> H W (C p q)', p=self.patch_size, q=self.patch_size).unsqueeze(0) if he_i is not None else torch.zeros(0, H, W, 3*self.patch_size**2, dtype=dtype, device=device)) for he_i in he]
            he = torch.concat(he, dim=0) # B x H x W x 3*patch_size**2 or B x H x W x patch_size**2

        multiplex = torch.concat(multiplex, dim=0) # sum(C_i) x H x W x D
        channel_ids = torch.concat([cids if cids is not None else torch.zeros(0, dtype=torch.long, device=device) for cids in channel_ids], dim=0) # sum(C_i)

        if multiplex_mask is not None:
            multiplex_mask = torch.concat([(m if m is not None else torch.zeros(0, H, W, dtype=torch.bool, device=device)) for m in multiplex_mask], dim=0) # sum(C_i) x H x W

        if he_mask is not None:
            he_mask = torch.concat([(m.unsqueeze(dim=0) if m is not None else torch.zeros(0, H, W, dtype=torch.bool, device=device)) for m in he_mask], dim=0)

        num_multiplex_channels = multiplex.shape[0]

        sum_C = multiplex.shape[0] + he.shape[0] + B * (1+self.num_registers)

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=multiplex.device), torch.arange(W, device=multiplex.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(sum_C, H, W, 2)
        pos = rearrange(pos, "c h w d -> c (h w) d")

        pos_multiplex = torch.stack(torch.meshgrid(torch.arange(H, device=multiplex.device), torch.arange(W, device=multiplex.device), indexing="ij"), dim=-1) # H x W x 2
        pos_multiplex = pos_multiplex.expand(multiplex.shape[0], H, W, 2)
        pos_multiplex = rearrange(pos_multiplex, "c h w d -> c (h w) d")

        pos_he = torch.stack(torch.meshgrid(torch.arange(H, device=he.device), torch.arange(W, device=he.device), indexing="ij"), dim=-1) # H x W x 2
        pos_he = pos_he.expand(he.shape[0], H, W, 2)
        pos_he = rearrange(pos_he, "b h w d -> b (h w) d")

        if self.joint_patch_encoder:
            multiplex = self.patch_encoder(multiplex)
            he = self.patch_encoder(he)
        else:
            multiplex = self.multiplex_patch_encoder(multiplex)
            he = self.he_patch_encoder(he)


        if multiplex_mask is not None:  
            multiplex = torch.where(multiplex_mask.unsqueeze(-1), self.masked_token.expand(multiplex.shape), multiplex)
            multiplex_mask = rearrange(multiplex_mask, "c h w -> c (h w)")
        if he_mask is not None:
            he = torch.where(he_mask.unsqueeze(-1), self.masked_token.expand(he.shape), he)
            he_mask = rearrange(he_mask, "b h w -> b (h w)")

        multiplex = rearrange(multiplex, "c h w d -> c (h w) d")
        he = rearrange(he, "b h w d -> b (h w) d")


        if self.use_protein_emb:
            
            proteins = self.protein_emb[channel_ids] # sum_C x P
            proteins = self.protein_encoder(proteins) # sum_C x D
            proteins = proteins.unsqueeze(1).expand(*multiplex.shape)
            if self.protein_fusion_type == 'add':
                multiplex = multiplex + proteins

            elif self.protein_fusion_type == 'cross_attention':
                multiplex = self.protein_cross_attn(multiplex, proteins, pos_multiplex, multiplex_channels_per_sample)


        he = he + self.he_marker.unsqueeze(0).unsqueeze(0)

        if get_router_logits:
            router_logits = []

        if self.separate_encoders:
            for layer in self.he_encoder:
                if he.shape[0] > 0:
                    if he_mask is None:
                        he = layer.forward_cc(he, pos_he, [1]*he.shape[0], get_router_logits=get_router_logits)
                        if get_router_logits:
                            he, rl = he
                            if rl is not None: router_logits.append(rl)
                    elif (~he_mask).sum() > 0:
                        he = layer.forward_cc_masked(he, pos_he, he_mask, [1]*he.shape[0], get_router_logits=get_router_logits)
                        if get_router_logits:
                            he, rl = he
                            if rl is not None: router_logits.append(rl)
            SA_BIAS_CACHE.clear()

            for layer in self.multiplex_encoder:
                if multiplex.shape[0] > 0:
                    if multiplex_mask is None:
                        multiplex = layer.forward_cc(multiplex, pos_multiplex, multiplex_channels_per_sample, get_router_logits=get_router_logits)
                        if get_router_logits:
                            multiplex, rl = multiplex
                            if rl is not None: router_logits.append(rl)
                    elif (~multiplex_mask).sum() > 0:
                        multiplex = layer.forward_cc_masked(multiplex, pos_multiplex, multiplex_mask, multiplex_channels_per_sample, get_router_logits=get_router_logits)
                        if get_router_logits:
                            multiplex, rl = multiplex
                            if rl is not None: router_logits.append(rl)
            SA_BIAS_CACHE.clear()
            

        multiplex = torch.split(multiplex, multiplex_channels_per_sample, dim=0)
        he = torch.split(he, he_channels_per_sample, dim=0)

        if self.num_registers > 0:
            x = [torch.concat([
                self.patch_summary_token.expand(1, H*W, self.patch_summary_token.shape[-1]),
                self.register_tokens.unsqueeze(1).expand(self.num_registers, H*W, self.register_tokens.shape[-1]),
                multiplex_i,
                he_i
                ], dim=0) for multiplex_i, he_i in zip(multiplex, he)]
        else:
            x = [torch.concat([
                    self.patch_summary_token.expand(1, H*W, multiplex_i.shape[-1]),
                    multiplex_i,
                    he_i
                    ], dim=0) for multiplex_i, he_i in zip(multiplex, he)]

        
        x = torch.concat(x, dim=0)
        x_channels_per_sample = [mc + hec + 1 + self.num_registers for mc, hec in zip(multiplex_channels_per_sample, he_channels_per_sample)]

        mask = None
        if multiplex_mask is not None or he_mask is not None:
            he_mask = he_mask if he_mask is not None else torch.zeros(B_he_not_none, H*W, dtype=torch.bool, device=multiplex[0].device) # B x H*W
            multiplex_mask = multiplex_mask if multiplex_mask is not None else torch.zeros(num_multiplex_channels, H*W, dtype=torch.bool, device=multiplex[0].device)

            he_mask = torch.split(he_mask, he_channels_per_sample, dim=0)
            multiplex_mask = torch.split(multiplex_mask, multiplex_channels_per_sample, dim=0)
            
            mask = [torch.concat([
                torch.zeros(1+self.num_registers, H*W, dtype=torch.bool, device=multiplex_mask_i.device),
                multiplex_mask_i,
                he_mask_i
                ], dim=0) for multiplex_mask_i, he_mask_i in zip(multiplex_mask, he_mask)]
            mask = torch.concat(mask, dim=0)

        if self.positional_embedding is not None:
            x = self.positional_embedding(x, pos)

        if return_intermediates:
            intermediates = []

        for layer in self.encoder:
            if mask is None:
                x = layer.forward_cc(x, pos, x_channels_per_sample, get_router_logits=get_router_logits)
                if get_router_logits:
                    x, rl = x
                    if rl is not None: router_logits.append(rl)
            else:
                x = layer.forward_cc_masked(x, pos, mask, x_channels_per_sample, get_router_logits=get_router_logits)
                if get_router_logits:
                    x, rl = x
                    if rl is not None: 
                        router_logits.append(rl)

            if return_intermediates:
                intermediates.append(x.clone()) 

        if self.norm_after_encoder_decoder:
            x = self.layer_norm(x)

        # reset attn_bias cache
        SA_BIAS_CACHE.clear()
        x = rearrange(x, "c (h w) d -> c h w d", h=H, w=W)
        x = torch.split(x, x_channels_per_sample, dim=0) # list of tensors C_i x H x W x D
        
        ps = [x_i[0] for x_i in x]
        x = [x_i[1 + self.num_registers:] for x_i in x]

        if return_intermediates:
            intermediates_reshaped = []
            for inter in intermediates:
                inter = rearrange(inter, "c (h w) d -> c h w d", h=H, w=W)
                inter = torch.split(inter, x_channels_per_sample, dim=0)
                ps_inter = [inter_i[0] for inter_i in inter]
                intermediates_reshaped.append(ps_inter)
            
            if get_router_logits:
                return x, ps, router_logits, intermediates_reshaped
            return x, ps, intermediates_reshaped

        if get_router_logits:
            return x, ps, router_logits

        return x, ps

class FlexDualVirTuesDecoder(nn.Module):

    def __init__(self,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                pattern="hvhv",
                num_heads=8,
                num_hidden_layers_head=0,
                dropout=0.0,
                pos_emb="rope",
                separate_decoders=False,
                group_layers=False,
                norm_after_encoder_decoder=False,
                **kwargs
                ):
        super().__init__()

        self.separate_decoders = separate_decoders
        self.norm_after_encoder_decoder = norm_after_encoder_decoder
        self.patch_size = patch_size
        
        multiplex_decoder_layers = []
        if num_hidden_layers_head > 0:
            for _ in range(num_hidden_layers_head -1):
                multiplex_decoder_layers.append(nn.Linear(model_dim, model_dim))
                multiplex_decoder_layers.append(nn.GELU())
        multiplex_decoder_layers.append(nn.Linear(model_dim, patch_size**2))

        if kwargs.get("add_sigmoid_to_multiplex_decoder", False):
            multiplex_decoder_layers.append(nn.Sigmoid())

        self.multiplex_decoder_mlp = nn.Sequential(*multiplex_decoder_layers)
        
        he_decoder_layers = []
        if num_hidden_layers_head > 0:
            for _ in range(num_hidden_layers_head -1):
                he_decoder_layers.append(nn.Linear(model_dim, model_dim))
                he_decoder_layers.append(nn.GELU())
        he_decoder_layers.append(nn.Linear(model_dim, 3*patch_size**2))
        self.he_decoder_mlp = nn.Sequential(*he_decoder_layers)

        if group_layers:
            groups = groupby(pattern)
            pattern = [(label, sum(1 for _ in group)) for label, group in groups]
        else:
            pattern = [(label, 1) for label in pattern]

        if self.separate_decoders:
            dec_layers = []
            for pattern, depth in pattern:
                if pattern == "|" or pattern == "v":
                    dec_layers.append(MarkerAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "-" or pattern == "h":
                    dec_layers.append(ChannelAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "f":
                    dec_layers.append(FullAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                else:
                    raise ValueError("decoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention)")
            self.he_decoder = nn.ModuleList(dec_layers)

            dec_layers = []
            for pattern, depth in pattern:
                if pattern == "|" or pattern == "v":
                    dec_layers.append(MarkerAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "-" or pattern == "h":
                    dec_layers.append(ChannelAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "f":
                    dec_layers.append(FullAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                else:
                    raise ValueError("decoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention)")
            self.multiplex_decoder = nn.ModuleList(dec_layers)

        else:
            dec_layers = []
            for pattern, depth in pattern:
                if pattern == "|" or pattern == "v":
                    dec_layers.append(MarkerAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "-" or pattern == "h":
                    dec_layers.append(ChannelAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                elif pattern == "f":
                    dec_layers.append(FullAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth))
                else:
                    raise ValueError("decoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention)")
            self.decoder = nn.ModuleList(dec_layers)
    
        if norm_after_encoder_decoder:
            self.layer_norm = nn.LayerNorm(model_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_list(self, x, ps, multiplex_channels_per_sample, he_channels_per_sample, get_router_logits=False, multiplex_mask=None, he_mask=None):
        """
        x: list of tensors C_i+1 x H x W x D
        ps: list of tensors H x W x D
        Returns: 
        x: sum(C_i) x h x w
        he: B x 3 x h x w
        """
        H, W, D = x[0].shape[1], x[0].shape[2], x[0].shape[3]
        x_channels_per_sample = [a + b for a, b in zip(multiplex_channels_per_sample, he_channels_per_sample)] # includes channel for H&E

        # x = torch.concat([
        #     torch.concat([
        #         ps_i[None, None, :, :, :].expand(x_i.shape[0], 1, H, W, D),
        #         x_i[:, None , :, :, :],
        #     ], dim=1)            
        #     for x_i, ps_i in zip(x, ps)
        # ], dim=0) # sum(C_i+1) x 2 x H x W x D
        x_copy = torch.empty(sum(x_channels_per_sample), 2, H, W, D, dtype=x[0].dtype, device=x[0].device) # sum(C_i+1) x 2 x H x W x D
        current_idx = 0
        for i, (x_i, ps_i) in enumerate(zip(x, ps)):
            num_channels = x_i.shape[0]
            x_copy[current_idx:current_idx+num_channels, 0] = ps_i.expand(num_channels, H, W, D)
            x_copy[current_idx:current_idx+num_channels, 1] = x_i
            current_idx += num_channels
        x = x_copy

        # mask 
        if multiplex_mask is not None:
            multiplex_mask = torch.concat([(m if m is not None else torch.zeros(0, H, W, dtype=torch.bool, device=x.device)) for m in multiplex_mask], dim=0) # sum(C_i) x H x W

        if he_mask is not None:
            he_mask = torch.concat([(m.unsqueeze(dim=0) if m is not None else torch.zeros(0, H, W, dtype=torch.bool, device=x.device)) for m in he_mask], dim=0)

        mask = None
        if multiplex_mask is not None or he_mask is not None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device) # sum(C_i+1) x 2 x H x W

            he_mask = he_mask if he_mask is not None else torch.ones(sum(he_channels_per_sample), H, W, dtype=torch.bool, device=x[0].device) # B x H*W, torch.ones because of negation afterwards
            multiplex_mask = multiplex_mask if multiplex_mask is not None else torch.ones(sum(multiplex_channels_per_sample), H, W, dtype=torch.bool, device=x[0].device)

            he_mask = torch.split(he_mask, he_channels_per_sample, dim=0)
            multiplex_mask = torch.split(multiplex_mask, multiplex_channels_per_sample, dim=0)
            
            mask_tokens = [torch.concat([
                multiplex_mask_i,
                he_mask_i
                ], dim=0) for multiplex_mask_i, he_mask_i in zip(multiplex_mask, he_mask)]
            mask_tokens = torch.concat(mask_tokens, dim=0)
            mask[:, 0] = ~mask_tokens

        if get_router_logits:
            router_logits = []

        if self.separate_decoders: # EJ: no MoE implemented yet
            x = torch.split(x, x_channels_per_sample, dim=0)
            multiplex = [x_i[:c] for x_i, c in zip(x, multiplex_channels_per_sample)]
            multiplex = torch.concat(multiplex, dim=0) # sum(C_i) x 2 x H x W x D
            multiplex = rearrange(multiplex, 'c e h w d -> (c e) (h w) d')
            multiplex_examples = multiplex.shape[0] // 2
            he = [x_i[c:] for x_i, c in zip(x, multiplex_channels_per_sample)]
            he = torch.concat(he, dim=0)
            he = rearrange(he, 'c e h w d -> (c e) (h w) d')
            he_examples = he.shape[0] // 2

            he_pos = torch.stack(torch.meshgrid(torch.arange(H, device=he.device), torch.arange(W, device=he.device), indexing="ij"), dim=-1) # H x W x 2
            he_pos = he_pos.expand(he.shape[0], H, W, 2)
            he_pos = rearrange(he_pos, "c h w d -> c (h w) d")

            multiplex_pos = torch.stack(torch.meshgrid(torch.arange(H, device=multiplex.device), torch.arange(W, device=multiplex.device), indexing="ij"), dim=-1) # H x W x 2
            multiplex_pos = multiplex_pos.expand(multiplex.shape[0], H, W, 2)
            multiplex_pos = rearrange(multiplex_pos, "c h w d -> c (h w) d")

            if multiplex_examples > 0:
                for layer in self.multiplex_decoder:
                    multiplex = layer.forward_cc(multiplex, multiplex_pos, [2]*multiplex_examples)
            SA_BIAS_CACHE.clear()

            if he_examples > 0:
                for layer in self.he_decoder:
                    he = layer.forward_cc(he, he_pos, [2]*he_examples)
            SA_BIAS_CACHE.clear()

            multiplex = multiplex[1::2] # sum(C_i) x S x D
            he = he[1::2] # B x S x D

        else:
            x = rearrange(x, "c e h w d -> (c e) (h w) d")
            pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
            pos = pos.expand(x.shape[0], H, W, 2)
            pos = rearrange(pos, "c h w d -> c (h w) d")
            examples = x.shape[0] // 2

            if mask is not None:
                mask = rearrange(mask, "c e h w -> (c e) (h w)")

            for layer in self.decoder:
                if mask is not None:
                    x = layer.forward_cc_masked(x, pos, mask, [2]*examples, get_router_logits=get_router_logits) # x: sum(C_i+1)*2 x S x D
                else:
                    x = layer.forward_cc(x, pos, [2]*examples, get_router_logits=get_router_logits) # x: sum(C_i+1)*2 x S x D, num_of_channels = 2 -> only attends between ch_i and ps
                
                if get_router_logits:
                    x, rl = x
                    if rl is not None: router_logits.append(rl)

            if self.norm_after_encoder_decoder:
                x = self.layer_norm(x)

            x = x[1::2] # sum(C_i+1) x S x D
            x = torch.split(x, x_channels_per_sample, dim=0)
            multiplex = [x_i[:c] for x_i, c in zip(x, multiplex_channels_per_sample)]
            he = [x_i[c:] for x_i, c in zip(x, multiplex_channels_per_sample)]

            multiplex = torch.concat(multiplex, dim=0) # sum(C_i) x S x D
            he = torch.concat(he, dim=0) # B x S x D

        # Reset attn_bias cache
        SA_BIAS_CACHE.clear()

        multiplex = self.multiplex_decoder_mlp(multiplex)
        he = self.he_decoder_mlp(he)

        multiplex = rearrange(multiplex, "c (h w) (p q) -> c (h p) (w q)", h=H, w=W, p=self.patch_size, q=self.patch_size)
        multiplex = torch.split(multiplex, multiplex_channels_per_sample, dim=0)
        multiplex = [multiplex_i if multiplex_i.shape[0] > 0 else None for multiplex_i in multiplex]

        he = rearrange(he, 'b (h w) (c p q) -> b c (h p) (w q)', h=H, w=W, c=3, p=self.patch_size, q=self.patch_size)
        he = torch.split(he, he_channels_per_sample, dim=0)
        he = [he_i.squeeze(0) if he_i.shape[0] > 0 else None for he_i in he]

        if get_router_logits:
            return multiplex, he, router_logits
        return multiplex, he

class FlexDualVirTuesMAE(nn.Module):

    def __init__(self,
                protein_emb,
                protein_emb_type,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                encoder_pattern="hvhv",
                num_encoder_heads=8,
                mae_decoder_pattern="hvhv",
                mae_num_decoder_heads=8,
                mae_num_hidden_layers_head=0,
                dropout=0.0,
                pos_emb="rope",
                joint_patch_encoder=False,
                separate_encoders=False,
                separate_encoder_pattern_he="hhhh",
                separate_encoder_pattern_multiplex="hvhv",
                separate_decoders=False,
                group_layers=False,
                norm_after_encoder_decoder=False,
                **kwargs
                ):
        super().__init__()

        self.encoder = FlexDualVirTuesEncoder(protein_emb, protein_emb_type, patch_size, model_dim, feedforward_dim, encoder_pattern, num_encoder_heads, dropout, pos_emb, joint_patch_encoder, separate_encoders, separate_encoder_pattern_he, separate_encoder_pattern_multiplex,
                                              group_layers, norm_after_encoder_decoder, **kwargs)
        self.mae_decoder = FlexDualVirTuesDecoder(patch_size, model_dim, feedforward_dim, mae_decoder_pattern, mae_num_decoder_heads, mae_num_hidden_layers_head, dropout, pos_emb, separate_decoders,
                                                  group_layers, norm_after_encoder_decoder, **kwargs)
        self.use_koleo = kwargs.get("use_koleo", False)
        self.get_router_logits = kwargs.get("get_router_logits", False)
        self.only_ps_tokens_decoder = kwargs.get("only_ps_tokens_decoder", False)

    def forward(self, multiplex, he, channel_ids, multiplex_mask=None, he_mask=None, return_tokens=False, **kwargs):
        """
        x: list of tensors C_i x H x W x D
        channel_ids: list of tensors C_i
        mask: list of tensors C_i x H x W or None
        Returns:
        cls_tokens: B x D
        x: sum(C_i) x H x W x D        
        """
        multiplex_channels_per_sample = [m.shape[0] if m is not None else 0 for m in multiplex]
        he_channels_per_sample = [1 if h is not None else 0 for h in he] # we counts RGB as one channels
        if not self.get_router_logits:
            x, ps = self.encoder.forward_list(multiplex, he, channel_ids, multiplex_mask=multiplex_mask, he_mask=he_mask, get_router_logits=self.get_router_logits) # list C_i x H x W x D,  list of H x W x D
            if self.only_ps_tokens_decoder:
                multiplex, he = self.mae_decoder.forward_list(x, ps, multiplex_channels_per_sample, he_channels_per_sample, multiplex_mask=multiplex_mask, he_mask=he_mask, get_router_logits=self.get_router_logits) # sum(C_i) x H x W x D
            else:
                multiplex, he = self.mae_decoder.forward_list(x, ps, multiplex_channels_per_sample, he_channels_per_sample, get_router_logits=self.get_router_logits)
            if self.use_koleo:
                self.ps = torch.stack(ps, dim=0)
                self.ps = rearrange(self.ps, "b h w d -> b (h w) d")
            if not return_tokens:
                return multiplex, he
            else:
                return multiplex, he, x, ps
        else:
            x, ps, enc_router_logits = self.encoder.forward_list(multiplex, he, channel_ids, multiplex_mask=multiplex_mask, he_mask=he_mask, get_router_logits=self.get_router_logits) # list C_i x H x W x D,  list of H x W x D
            if self.only_ps_tokens_decoder:
                multiplex, he, dec_router_logits = self.mae_decoder.forward_list(x, ps, multiplex_channels_per_sample, he_channels_per_sample, multiplex_mask=multiplex_mask, he_mask=he_mask, get_router_logits=self.get_router_logits)
            else:
                multiplex, he, dec_router_logits = self.mae_decoder.forward_list(x, ps, multiplex_channels_per_sample, he_channels_per_sample, get_router_logits=self.get_router_logits) # sum(C_i) x H x W x D

            if self.use_koleo:
                self.ps = torch.stack(ps, dim=0)
                self.ps = rearrange(self.ps, "b h w d -> b (h w) d")
            if not return_tokens:
                return multiplex, he, enc_router_logits, dec_router_logits
            else:
                return multiplex, he, x, ps, enc_router_logits, dec_router_logits
    
    def reconstruct(self, multiplex, he, channel_ids, multiplex_mask=None, he_mask=None, inject_mask_token_before_decoder=False):
        raise NotImplementedError("Reconstruction is not implemented for FlexDualVirTuesMAE. Use forward method instead.")
        x, ps = self.encoder.forward(multiplex, he, channel_ids, multiplex_mask=multiplex_mask, he_mask=he_mask)
        if inject_mask_token_before_decoder:
            x = self.encoder.masked_token.expand(x.shape)
        multiplex, he = self.mae_decoder.forward(x, ps) # B x C x H x W x D
        return multiplex, he

    def reconstruct_mpx(self, multiplex, channel_ids, multiplex_mask=None):
        """
        multiplex: list of tensors C_i x H x W x D
        channel_ids: list of tensors C_i
        multiplex_mask: list of tensors C_i x H x W or None
        """
        he = [None] * len(multiplex)
        he_mask = [None] * len(multiplex)
        multiplex, he = self.forward(multiplex=multiplex, he=he, channel_ids=channel_ids, multiplex_mask=multiplex_mask, he_mask=he_mask, return_tokens=False)
        return multiplex
    
    def embed(self, multiplex, he, channel_ids, multiplex_mask=None, he_mask=None, return_dict=False, place_on_cpu=False, **kwargs):
        x, ps = self.encoder.forward_list(multiplex, he, channel_ids, multiplex_mask=multiplex_mask, he_mask=he_mask)
        cls_tokens = [ps_i.mean(dim=(0, 1)) for ps_i in ps]
        if return_dict:
            results = [{
                "cls_token": cls_tokens,
                "patch_summary_token": ps,
            }]
            if place_on_cpu:
                for res in results:
                    for key in res.keys():
                        # res[key] = res[key].cpu()
                        res[key] = [t.cpu() for t in res[key]]
            return results
        else:
            return cls_tokens, ps

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """
        Override the state_dict method to get rid of the _orig_mod. prefix of compiled modules.
        """
        state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict = {k.replace("_orig_mod.", "."): v for k, v in state_dict.items()}
        return state_dict
    
    def compile_rope(self):
        for module in self.modules():
            if isinstance(module, (MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock)):
                for submodule in module.modules():
                    # print("compiling submodule", submodule)
                    if isinstance(submodule, (RotaryPositionalEmbedding2D)):
                        submodule.compile()