import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from src.modules.flex_dual_virtues.layers.attention_flashattention import \
    MHAwithPosEmb
from src.modules.flex_dual_virtues.layers.basics import build_feedforward
from src.modules.flex_dual_virtues.layers.mask_utils_flashattention import (
    build_selfattention_bias, build_selfattention_bias_channel_concat,
    cached_build_selfattention_bias,
    cached_build_selfattention_bias_channel_concat,
    cached_get_non_zero_indices)


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
        
        
    def forward(self, src, src_pos=None, src_key_padding_mask=None,
                cu_seq_len=None, max_seq_len=None, get_router_logits=False):
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
        src = src + self.multi_head_attention(query=lsrc, key=lsrc, value=lsrc, query_pos=src_pos, key_pos=src_pos, key_padding_mask=src_key_padding_mask,
                                                cu_seq_len=cu_seq_len, max_seq_len=max_seq_len)
        lsrc = self.layernorm2(src)

        feedfwd_output = self.feedforward(lsrc)
        if get_router_logits:
            # only if tuple
            if isinstance(feedfwd_output, tuple):
                feedfwd_output, router_logits = feedfwd_output
                # logger.warning(f"feedfwd shape: {feedfwd_output.shape}, router_logits shape: {router_logits.shape}")
            else:
                router_logits = None
        src = src + feedfwd_output

        if get_router_logits:
            return src, router_logits
        else:
            return src

class ChannelAttentionEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()

        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers)
        else:   # For backward compatibility
            self.encoder_layer = TransformerEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb)

    def forward(self, x, pos):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        raise NotImplementedError('Not yet implemented for FlashAttn')
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B C) S D")
        pos = rearrange(pos, "B C S D -> (B C) S D")
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "(B C) S D -> B C S D", B=B)
        return x
    
    def forward_masked(self, x, pos, mask):
        """
        x: BxCxSxD
        mask: BxCxS True indicating a masked token.
        pos: BxCxSx2
        """
        raise NotImplementedError('Not yet implemented for FlashAttn')
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B C) S D")
        pos = rearrange(pos, "B C S D -> (B C) S D")
        mask = rearrange(mask, "B C S -> (B C) S")

        x_false, pos_false = x[mask_indices].unsqueeze(0), pos[mask_indices].unsqueeze(0)
        attn_bias = build_selfattention_bias(mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x[mask_indices] = x_false[0]

        x = rearrange(x, "(B C) S D -> B C S D", B=B)
        return x
    
    def forward_cc(self, x, pos, channels_per_sample, get_router_logits=False):
        """
        x: CxSxD
        pos: CxSx2
        """
        # raise NotImplementedError('Not yet implemented for FlashAttn')
        # x = self.encoder_layer(src=x, src_pos=pos)
        # return x
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        return self.forward_cc_masked(x, pos, mask, channels_per_sample, get_router_logits=get_router_logits)

    def forward_cc_masked(self, x, pos, mask, channels_per_sample, get_router_logits=False):
        """
        x: CxSxD
        pos: CxSx2
        mask: CxS True indicating a masked token.
        """
        mask_indices = cached_get_non_zero_indices("ChannelAttention_cc_Masked_Mask_indices", ~mask)
        x_false, pos_false = x[mask_indices].unsqueeze(0), pos[mask_indices].unsqueeze(0)
        # attn_bias = build_selfattention_bias(mask, use_true_as_query=False)

        seq_lens, max_seq_len = cached_build_selfattention_bias("ChannelAttention_cc_masked", mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false,
                                      cu_seq_len=seq_lens, max_seq_len=max_seq_len, get_router_logits=get_router_logits)
        if get_router_logits:
            x_false, router_logits = x_false
        x[mask_indices] = x_false[0]
        if get_router_logits:
            return x, router_logits
        else:
            return x

class MarkerAttentionEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()

        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers)
        else:   # For backward compatibility
            self.encoder_layer = TransformerEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb)

    def forward(self, x, pos):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        raise NotImplementedError('Not yet implemented for FlashAttn')
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B S) C D")
        pos = rearrange(pos, "B C S D -> (B S) C D")
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "(B S) C D -> B C S D", B=B)
        return x

    def forward_masked(self, x, pos, mask, get_router_logits=False):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B S) C D")
        pos = rearrange(pos, "B C S D -> (B S) C D")
        mask = rearrange(mask, "B C S -> (B S) C")

        mask_indices = cached_get_non_zero_indices("MarkerAttention_masked_Mask_indices", ~mask)
        x_false, pos_false = x[mask_indices].unsqueeze(0), pos[mask_indices].unsqueeze(0)
        # attn_bias = build_selfattention_bias(mask, use_true_as_query=False)
        q_seq_lens, max_seq_len = cached_build_selfattention_bias("MarkerAttention_masked", mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, cu_seq_len=q_seq_lens, max_seq_len=max_seq_len,
                                        get_router_logits=get_router_logits)
        if get_router_logits:
            x_false, router_logits = x_false
    
        x[mask_indices] = x_false[0]

        x = rearrange(x, "(B S) C D -> B C S D", B=B)
        return x
    
    def forward_cc(self, x, pos, channels_per_sample, get_router_logits=False):
        """
        x: CxSxD
        pos: CxSx2
        """
        S = x.shape[1]
        q_seq_lens = channels_per_sample * S
        x = rearrange(x, "C S D -> (S C) D").unsqueeze(0)
        pos = rearrange(pos, "C S D -> (S C) D").unsqueeze(0)
        # attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seq_lens)
        q_seq_lens = [0] + q_seq_lens # since FlashAttn requires the first element to be 0
        max_seq_len = max(q_seq_lens)
        q_seq_lens = torch.tensor(q_seq_lens, device=x.device).cumsum(dim=0, dtype=torch.int32)
        x = self.encoder_layer(src=x, src_pos=pos, cu_seq_len=q_seq_lens, max_seq_len=max_seq_len, 
                                get_router_logits=get_router_logits).squeeze(0)
        if get_router_logits:
            x, router_logits = x
        x = rearrange(x, "(S C) D -> C S D", S=S)
        if get_router_logits:
            return x, router_logits
        else:
            return x
    
    def forward_cc_masked(self, x, pos, mask, channels_per_sample, get_router_logits=False):
        """
        x: CxSxD
        pos: CxSx2
        """
        x = rearrange(x, "C S D -> S C D")
        pos = rearrange(pos, "C S D -> S C D")
        mask = rearrange(mask, "C S -> S C")

        mask_indices = cached_get_non_zero_indices("MarkerAttention_cc_Masked_Mask_indices", ~mask)
        x_false, pos_false = x[mask_indices].unsqueeze(0), pos[mask_indices].unsqueeze(0)
        # attn_bias = build_selfattention_bias_channel_concat(mask, channels_per_sample, use_true_as_query=False)

        seq_lens, max_seq_len = cached_build_selfattention_bias_channel_concat("MarkerAttention_cc_masked", mask, tuple(channels_per_sample), use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false,
                                     cu_seq_len=seq_lens, max_seq_len=max_seq_len,
                                        get_router_logits=get_router_logits)
        if get_router_logits:
            x_false, router_logits = x_false
        x[mask_indices] = x_false[0]
        x = rearrange(x, "S C D -> C S D")
        if get_router_logits:
            return x, router_logits
        else:
            return x

class FullAttentionEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()

        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers)
        else:   # For backward compatibility
            self.encoder_layer = TransformerEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb)
                   
    def forward(self, x, pos):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        raise NotImplementedError('Not yet implemented for FlashAttn')
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> B (C S) D")
        pos = rearrange(pos, "B C S D -> B (C S) D")
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "B (C S) D -> B C S D", C=C)
        return x
    
    def forward_masked(self, x, pos, mask):
        """
        x: BxCxSxD
        pos: BxCxSx2
        mask: BxCxS
        """
        raise NotImplementedError('Not yet implemented for FlashAttn')
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> B (C S) D")
        pos = rearrange(pos, "B C S D -> B (C S) D")
        mask = rearrange(mask, "B C S -> B (C S)")

        x_false, pos_false = x[mask_indices].unsqueeze(0), pos[mask_indices].unsqueeze(0)
        attn_bias = build_selfattention_bias(mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x[mask_indices] = x_false[0]

        x = rearrange(x, "B (C S) D -> B C S D", C=C)
        return x

    def forward_cc(self, x, pos, channels_per_sample, get_router_logits=False):
        """
        x: CxSxD
        pos: CxSx2
        """
        S = x.shape[1]
        q_seq_lens = [c * S for c in channels_per_sample]
        x = rearrange(x, "C S D -> (C S) D").unsqueeze(0)
        pos = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)
        # attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seq_lens)

        q_seq_lens = [0] + q_seq_lens # since FlashAttn requires the first element to be 0
        max_seq_len = max(q_seq_lens)
        q_seq_lens = torch.tensor(q_seq_lens, device=x.device).cumsum(dim=0, dtype=torch.int32)

        x = self.encoder_layer(src=x, src_pos=pos, cu_seq_len=q_seq_lens, max_seq_len=max_seq_len,
                                get_router_logits=get_router_logits)
        if get_router_logits:
            x, router_logits = x
        x = x.squeeze(0)
        x = rearrange(x, "(C S) D -> C S D", S=S)

        if get_router_logits:
            return x, router_logits

        return x
    
    def forward_cc_masked(self, x, pos, mask, channels_per_sample, get_router_logits=False):
        """
        x: CxSxD
        pos: CxSx2
        mask: CxS
        """
        S = x.shape[1]    
        x = rearrange(x, "C S D -> (C S) D")
        pos = rearrange(pos, "C S D -> (C S) D")
        mask = rearrange(mask, "C S -> (C S)")

        tokens_per_sample = [c * S for c in channels_per_sample]
        mask_indices = cached_get_non_zero_indices("FullAttention_cc_Masekd_Mask_indices", ~mask)
        x_false, pos_false = x[mask_indices].unsqueeze(0), pos[mask_indices].unsqueeze(0)
        # attn_bias = build_selfattention_bias_channel_concat(mask, tokens_per_sample, use_true_as_query=False)
        
        # mask = ~mask
        # seq_lens = torch.empty(len(channels_per_sample) + 1, dtype=torch.int32, device=x.device)
        # current_index = 0
        # for i, t in enumerate(channels_per_sample):
        #     seq_lens[i + 1] = mask[current_index:current_index + t].sum()
        #     current_index += t
        # seq_lens[0] = 0
        # seq_lens = seq_lens[seq_lens != 0] # Filter-out zero length sequences for self-attention
        # seq_lens = seq_lens.tolist()
        # max_seq_len = max(seq_lens)
        # seq_lens = torch.tensor(seq_lens, device=x.device).cumsum(dim=0, dtype=torch.int32)
        raise Exception("Needs to be revised!!")
        seq_lens, max_seq_len = cached_build_selfattention_bias_channel_concat("FullAttention_cc_masked", mask, tuple(tokens_per_sample), use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, cu_seq_len=seq_lens, max_seq_len=max_seq_len,
                                        get_router_logits=get_router_logits)
        if get_router_logits:
            x_false, router_logits = x_false
        x[mask_indices] = x_false[0]

        x = rearrange(x, "(C S) D -> C S D", S=S)
        if get_router_logits:
            return x, router_logits
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.0, pos_type="learnable"):
        super().__init__()
        self.attention_module = MHAwithPosEmb(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            inbuilt_pos_emb=pos_type
        )
        
    def forward(self, x_query, x_keyval, pos, multiplex_channels_per_sample):
        """
        x: [C_total, S, D]
        protein_emb: [C_total, S, D]
        """
        S = x_query.shape[1]
        q_seq_lens = [c * S for c in multiplex_channels_per_sample]
        _x_attn = rearrange(x_query, "C S D -> (C S) D").unsqueeze(0)
        _pos = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)
        _prot = rearrange(x_keyval, "C S D -> (C S) D").unsqueeze(0)
        q_seq_lens = [0] + q_seq_lens
        max_seq_lens = max(q_seq_lens)
        q_seq_lens = torch.tensor(q_seq_lens, device=x_query.device).cumsum(dim=0, dtype=torch.int32)
        ca = self.attention_module(
            query=_x_attn,
            key=_prot,
            value=_prot,
            query_pos=_pos,
            key_pos=_pos,
            key_padding_mask=None,
            cu_seq_len=q_seq_lens,
            max_seq_len=max_seq_lens
        )
        ca = rearrange(ca.squeeze(0), "(C S) D -> C S D", S=S)
        return ca


class PatchAttentionBlock(nn.Module):
    """
    Implements attention between patch summary tokens.
    Could be used e.g., after "vvv" blocks.
    """
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()

        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers)
        else:   # For backward compatibility
            self.encoder_layer = TransformerEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb)

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
        ps = rearrange(ps, "B S D -> (B S) D")
        pos = rearrange(pos, "B S D -> (B S) D")
        q_seq_lens = [0] + [S] * B
        q_seq_lens = torch.tensor(q_seq_lens, device=x.device).cumsum(dim=0, dtype=torch.int32)
        ps = self.encoder_layer(src=ps, src_pos=pos, max_seq_len=S, cu_seq_len=torch.tensor(q_seq_lens))
        ps = rearrange(ps, "(B S) D -> B S D", S=S)
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
        C, S, D = x.shape
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
        ps = rearrange(ps, "B S D -> (B S) D") 
        pos = pos[ps_position]               # BxSx2
        pos = rearrange(pos, "B S D -> (B S) D")

        q_seq_lens = [0] + [S] * len(ps_position)
        q_seq_lens = torch.tensor(q_seq_lens, device=x.device).cumsum(dim=0, dtype=torch.int32)
        ps = ps.unsqueeze(0)
        pos = pos.unsqueeze(0)
        ps = self.encoder_layer(src=ps, src_pos=pos, max_seq_len=S, cu_seq_len=q_seq_lens)
        ps = rearrange(ps.squeeze(0), "(B S) D -> B S D", S=S)
        x[ps_position] = ps
        
        return x
    
    def forward_cc_masked(self, x, pos, mask, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        return self.forward_cc(x, pos, channels_per_sample, **kwargs)