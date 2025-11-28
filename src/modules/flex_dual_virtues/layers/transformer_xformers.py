import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from src.modules.flex_dual_virtues.layers.attention_xformers import MHAwithPosEmb
from src.modules.flex_dual_virtues.layers.basics import build_feedforward
from src.modules.flex_dual_virtues.layers.mask_utils_xformers import (
    build_selfattention_bias, build_selfattention_bias_channel_concat,
    cached_build_block_diagonal_mask, cached_build_selfattention_bias,
    cached_build_selfattention_bias_channel_concat, merge_batch, split_batch)
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, activation="gelu", bias=True, inbuilt_pos_emb="absolute", num_layers=1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([TransformerEncoderBlock(d_model, num_heads, dim_feedforward, dropout=dropout, activation=activation, bias=bias, inbuilt_pos_emb=inbuilt_pos_emb) for _ in range(num_layers)])

    def forward(self, src, src_pos=None, src_key_padding_mask=None, cu_seq_len=None, max_seq_len=None, get_router_logits=False):
        """
        Args:
            src: (B, S, d_model) source sequence
            attn_mask: (B, S, S) boolean mask or float mask
            key_padding_mask: (B, S) boolean mask or float mask
        """   
        if get_router_logits:
            for layer in self.layers:
                src, router_logits = layer(src, src_pos=src_pos, src_key_padding_mask=src_key_padding_mask, cu_seq_len=cu_seq_len, max_seq_len=max_seq_len, get_router_logits=get_router_logits)
            return src, router_logits
        else:
            for layer in self.layers:
                src = layer(src, src_pos=src_pos, src_key_padding_mask=src_key_padding_mask, cu_seq_len=cu_seq_len, max_seq_len=max_seq_len)
        return src

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation = "gelu", bias=True, inbuilt_pos_emb="absolute"):
        super(TransformerEncoderBlock, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.activation = activation

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.multi_head_attention = MHAwithPosEmb(embed_dim=d_model, num_heads=nhead, dropout=dropout, bias=bias, inbuilt_pos_emb=inbuilt_pos_emb)
        self.feedforward = build_feedforward(d_model, dim_feedforward, activation, dropout)

        self.layernorm1 = nn.LayerNorm(d_model, bias=bias)
        self.layernorm2 = nn.LayerNorm(d_model, bias=bias)
        
        
    def forward(self, src, src_pos=None, mask=None, src_key_padding_mask=None, return_attention=False, before_softmax=False):
        """
        Args:
            src: (B, S, d_model) source sequence
            attn_mask: (B, S, S) boolean mask or float mask
            key_padding_mask: (B, S) boolean mask or float mask
        """
        
        """ Post LN Attention 
        src = self.layernorm1(src + self.multi_head_attention(query=src, key=src, value=src, query_pos=src_pos, key_pos=src_pos, mask=mask, key_padding_mask=src_key_padding_mask))
        src = self.layernorm2(src + self.feedforward(src))
        """
        # Pre-LN MHA
        lsrc = self.layernorm1(src)
        if return_attention:
            mha, attention = self.multi_head_attention(query=lsrc, key=lsrc, value=lsrc, query_pos=src_pos, key_pos=src_pos, mask=mask, key_padding_mask=src_key_padding_mask, return_attention=return_attention, before_softmax=before_softmax)
            src = src + mha
        else:
            src = src + self.multi_head_attention(query=lsrc, key=lsrc, value=lsrc, query_pos=src_pos, key_pos=src_pos, mask=mask, key_padding_mask=src_key_padding_mask)
        lsrc = self.layernorm2(src)
        src = src + self.feedforward(lsrc)

        if return_attention:
            return src, attention
        return src
 
class ChannelAttentionEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()
        self.encoder_layer = TransformerEncoder(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers)

    def forward(self, x, pos, return_attention=False, before_softmax=False, **kwargs):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B C) S D")
        pos = rearrange(pos, "B C S D -> (B C) S D")
        if return_attention:
            x, attention = self.encoder_layer(src=x, src_pos=pos, return_attention=return_attention, before_softmax=before_softmax)
            x = rearrange(x, "(B C) S D -> B C S D", B=B)
            return x, attention
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "(B C) S D -> B C S D", B=B)
        return x
    
    def forward_masked(self, x, pos, mask, **kwargs):
        """
        x: BxCxSxD
        mask: BxCxS True indicating a masked token.
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B C) S D")
        pos = rearrange(pos, "B C S D -> (B C) S D")
        mask = rearrange(mask, "B C S -> (B C) S")

        x_false, pos_false = x[~mask].unsqueeze(0), pos[~mask].unsqueeze(0)
        attn_bias = cached_build_selfattention_bias("ChannelAttention_masked", mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x[~mask] = x_false[0]
        x = rearrange(x, "(B C) S D -> B C S D", B=B)
        return x
    
    def forward_cc(self, x, pos, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        x = self.encoder_layer(src=x, src_pos=pos)
        return x

    def forward_cc_masked(self, x, pos, mask, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        mask: CxS True indicating a masked token.
        """
        x_false, pos_false = x[~mask].unsqueeze(0), pos[~mask].unsqueeze(0)
        attn_bias = cached_build_selfattention_bias("ChannelAttention_cc_masked", mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x[~mask] = x_false[0]
        return x

class MarkerAttentionEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()
        self.encoder_layer = TransformerEncoder(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers)

    def forward(self, x, pos, return_attention=False, before_softmax=False, **kwargs):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B S) C D")
        pos = rearrange(pos, "B C S D -> (B S) C D")
        if return_attention:
            x, attention = self.encoder_layer(src=x, src_pos=pos, return_attention=return_attention, before_softmax=before_softmax)
            x = rearrange(x, "(B S) C D -> B C S D", B=B)
            return x, attention
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "(B S) C D -> B C S D", B=B)
        return x

    def forward_masked(self, x, pos, mask, **kwargs):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B S) C D")
        pos = rearrange(pos, "B C S D -> (B S) C D")
        mask = rearrange(mask, "B C S -> (B S) C")

        x_false, pos_false = x[~mask].unsqueeze(0), pos[~mask].unsqueeze(0)
        attn_bias = cached_build_selfattention_bias("MarkerAttention", mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x[~mask] = x_false[0]

        x = rearrange(x, "(B S) C D -> B C S D", B=B)
        return x
    
    def forward_cc(self, x, pos, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        S = x.shape[1]
        q_seq_lens = channels_per_sample * S
        x = rearrange(x, "C S D -> (S C) D").unsqueeze(0)
        pos = rearrange(pos, "C S D -> (S C) D").unsqueeze(0)
        attn_bias = cached_build_block_diagonal_mask("MarkerAttention_cc", q_seq_len=q_seq_lens)
        x = self.encoder_layer(src=x, src_pos=pos, mask=attn_bias)
        x = x.squeeze(0)
        x = rearrange(x, "(S C) D -> C S D", S=S)
        return x
    
    def forward_cc_masked(self, x, pos, mask, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        x = rearrange(x, "C S D -> S C D")
        pos = rearrange(pos, "C S D -> S C D")
        mask = rearrange(mask, "C S -> S C")

        x_false, pos_false = x[~mask].unsqueeze(0), pos[~mask].unsqueeze(0)
        attn_bias = cached_build_selfattention_bias_channel_concat("MarkerAttention_cc_masked", mask, channels_per_sample, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x[~mask] = x_false[0]

        x = rearrange(x, "S C D -> C S D")
        return x

class FullAttentionEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()
        self.encoder_layer = TransformerEncoder(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers)

    def forward(self, x, pos, **kwargs):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> B (C S) D")
        pos = rearrange(pos, "B C S D -> B (C S) D")
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "B (C S) D -> B C S D", C=C)
        return x
    
    def forward_masked(self, x, pos, mask, **kwargs):
        """
        x: BxCxSxD
        pos: BxCxSx2
        mask: BxCxS
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> B (C S) D")
        pos = rearrange(pos, "B C S D -> B (C S) D")
        mask = rearrange(mask, "B C S -> B (C S)")

        x_false, pos_false = x[~mask].unsqueeze(0), pos[~mask].unsqueeze(0)
        attn_bias = cached_build_selfattention_bias("FullAttention_masked", mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x[~mask] = x_false[0]

        x = rearrange(x, "B (C S) D -> B C S D", C=C)
        return x

    def forward_cc(self, x, pos, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        S = x.shape[1]
        q_seq_lens = [c * S for c in channels_per_sample]
        x = rearrange(x, "C S D -> (C S) D").unsqueeze(0)
        pos = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)
        attn_bias = cached_build_block_diagonal_mask("FullAttention_cc", q_seq_len=q_seq_lens)
        x = self.encoder_layer(src=x, src_pos=pos, mask=attn_bias)
        x = x.squeeze(0)
        x = rearrange(x, "(C S) D -> C S D", S=S)

        return x
    
    def forward_cc_masked(self, x, pos, mask, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        mask: CxS
        """
        x = rearrange(x, "C S D -> (C S) D")
        pos = rearrange(pos, "C S D -> (C S) D")
        mask = rearrange(mask, "C S -> (C S)")

        if not attn_bias:
            S = x.shape[1]    
            tokens_per_sample = [c * S for c in channels_per_sample]
            attn_bias = cached_build_selfattention_bias_channel_concat("FullAttention_cc_masked", mask, tokens_per_sample, use_true_as_query=False)
            
        x_false, pos_false = x[~mask].unsqueeze(0), pos[~mask].unsqueeze(0)
        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x[~mask] = x_false[0]

        x = rearrange(x, "(C S) D -> C S D", S=S)
        return x

class PatchAttentionBlock(nn.Module):
    """
    Implements attention between patch summary tokens.
    Could be used e.g., after "vvv" blocks.
    """
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()
        self.encoder_layer = TransformerEncoder(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers)

    def forward(self, x, pos, **kwargs):
        """
        Implements attention between patch summary tokens.
        x: BxCxSxD
        pos: BxCxSx2
        """

        B, C, S, D = x.shape
        # x = rearrange(x, "B C S D -> (B S) C D")
        # pos = rearrange(pos, "B C S D -> (B S) C D")
        
        # Mask everything except patch summary tokens
        # mask = torch.ones(B, C, S, dtype=torch.bool, device=x.device)
        # mask[:, 0, :] = False
        # mask = rearrange(mask, "B C S -> (B S) C")

        # x_false, pos_false = x[~mask].unsqueeze(0), pos[~mask].unsqueeze(0)
        # attn_bias = cached_build_selfattention_bias("PatchAttention", mask, use_true_as_query=False)
        # x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
    
        # x[~mask] = x_false[0]

        # Only take patch summary tokens (first token of each patch)
        ps = x[:, 0]  # BxSxD
        pos = pos[:, 0]  # BxSx2
        ps = self.encoder_layer(src=ps, src_pos=pos)
        x[:, 0] = ps  # BxSxD
        
        # x = rearrange(x, "(B S) C D -> B C S D", B=B)
        return x
    
    # Patch summary tokens are never masked. Other masking is not intended.
    def forward_masked(self, x, pos, mask, **kwargs):
        return self.forward(x, pos, **kwargs)

    def forward_cc(self, x, pos, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        # C, S, D = x.shape
        # x = rearrange(x, "C S D -> S C D")
        # pos = rearrange(pos, "C S D -> S C D")
        # mask = torch.ones(C, S, dtype=torch.bool, device=x.device)
        # mask[:, 0] = False
        # mask = rearrange(mask, "C S -> S C")

        # x_false, pos_false = x[~mask].unsqueeze(0), pos[~mask].unsqueeze(0)
        # # attn_bias = build_selfattention_bias(mask, use_true_as_query=False)
        # attn_bias = cached_build_selfattention_bias("PatchAttention_cc", mask, use_true_as_query=False)

        # x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)

        # x[~mask] = x_false[0]
        # x = rearrange(x, "S C D -> C S D")

        # Only take patch summary tokens (first token of each channel)
        ps_position = np.cumsum(channels_per_sample)
        ps_position -= ps_position[0]
        ps = x[ps_position]                  # BxSxD
        ps = self.encoder_layer(src=ps, src_pos=pos[ps_position])
        x[ps_position] = ps
        return x
    
    def forward_cc_masked(self, x, pos, mask, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        return self.forward_cc(x, pos, channels_per_sample, **kwargs)