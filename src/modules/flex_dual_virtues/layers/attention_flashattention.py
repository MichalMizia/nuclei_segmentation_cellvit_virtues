
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from src.modules.flex_dual_virtues.layers.positional_embeddings import (
    LearnablePositionalEmbedding2D, PositionalEmbedding2D,
    RotaryPositionalEmbedding2D)
from torch.nn import functional as F


class MHAwithPosEmb(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, inbuilt_pos_emb="absolute",
                 keyval_embed_dim=None):
        
        super().__init__()
        if keyval_embed_dim is None:
            keyval_embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(keyval_embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(keyval_embed_dim, embed_dim, bias=bias)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        if inbuilt_pos_emb == "absolute":
            self.pos_emb = PositionalEmbedding2D(model_dim=self.embed_dim)
            self.pos_after_linear = False
            self.pos_before_linear = True
        elif inbuilt_pos_emb == "rope":
            self.pos_emb = RotaryPositionalEmbedding2D(model_dim=self.head_dim)
            self.pos_after_linear = True
            self.pos_before_linear = False
        elif inbuilt_pos_emb == "protein_learnable":
            self.pos_emb = LearnablePositionalEmbedding2D(model_dim=self.embed_dim)
            self.pos_after_linear = False
            self.pos_before_linear = False
        elif inbuilt_pos_emb == "learnable":
            self.pos_after_linear = False
            self.pos_before_linear = False
        elif inbuilt_pos_emb == "absolute_beginning" or inbuilt_pos_emb is None:
            self.pos_after_linear = False
            self.pos_before_linear = False
        else:
            raise ValueError("pos_embedding must be 'absolute' or 'rope' or 'learnable' or 'absolute_beginning' or None")
    
    def forward(self, query, key, value, query_pos=None, key_pos=None, key_padding_mask=None, return_attention=False,
                cu_seq_len=None, max_seq_len=None):
        """
        Args:
            q: (B, L, embed_dim)
            k: (B, S, embed_dim)
            v: (B, S, embed_dim)
            q_pos: (B, L, 2) 2D positions of query # different when using proteinlearnable
            k_pos: (B, S, 2) 2D positions of key
        """
        assert cu_seq_len is None or key_padding_mask is None, "cu_seq_len and key_padding_mask cannot be provided at the same time"

        bs = query.size(0)
        src_length = key.size(1)
        target_length = query.size(1)

        if self.pos_before_linear:
            if query_pos is not None and key_pos is not None:
                query = self.pos_emb(query, query_pos)
                key = self.pos_emb(key, key_pos)

        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bs, src_length), "key_padding_mask shape must be (B, S)"
            key_padding_mask = key_padding_mask.view(bs, 1, src_length)# (B, 1, S)
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = torch.zeros_like(key_padding_mask, dtype=torch.float32).masked_fill(key_padding_mask, float("-inf"))
                attn_mask = key_padding_mask #(B, 1, S)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, target_length, -1) # (B, num_heads, L, S)

            query = query.reshape(bs, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, L, head_dim)
            key = key.reshape(bs, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, S, head_dim)
            value = value.reshape(bs, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, S, head_dim)

            if self.pos_after_linear:
                # if query_pos is not None and key_pos is not None:
                #     query_pos = query_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # (B, num_heads, L, 2)
                #     key_pos = key_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # (B, num_heads, S, 2)

                #     query = self.pos_emb(query, query_pos)
                #     key = self.pos_emb(key, key_pos)
                if query_pos is not None:
                    query_pos = query_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # (B, num_heads, L, 2)
                    query = self.pos_emb(query, query_pos)
                if key_pos is not None:
                    key_pos = key_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # (B, num_heads, S, 2)
                    key = self.pos_emb(key, key_pos)

            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=self.dropout) # (B, num_heads, L, head_dim)
            attn_output = attn_output.transpose(1, 2).reshape(bs, -1, self.num_heads * self.head_dim) # (B, L, embed_dim)
            return self.W_o(attn_output) # (B, L, embed_dim)

        else:
            query = query.reshape(bs, -1, self.num_heads, self.head_dim) # (B, L, num_heads,head_dim)
            key = key.reshape(bs, -1, self.num_heads, self.head_dim) # (B, S, num_heads, head_dim)
            value = value.reshape(bs, -1, self.num_heads, self.head_dim) # (B, S, num_heads, head_dim)

            if self.pos_after_linear:
                if query_pos is not None:
                    query_pos = query_pos.unsqueeze(2).expand(-1, -1, self.num_heads, -1) # (B, L, num_heads, 2)
                    query = self.pos_emb(query, query_pos)
                if key_pos is not None:
                    key_pos = key_pos.unsqueeze(2).expand(-1, -1, self.num_heads, -1) # (B, S, num_heads, 2)
                    key = self.pos_emb(key, key_pos)

            if torch.is_autocast_enabled():
                query, key, value = query.half(), key.half(), value.half()


            # if return_attention:
            #     print(mask)
            #     # manually calculate attn_output:
            #     _scale = 1.0 / query.shape[-1] ** 0.5
            #     query = query * _scale # (B, L, num_heads, head_dim)
            #     query = query.transpose(1, 2) # (B, num_heads, L, head_dim)
            #     key = key.transpose(1, 2) # (B, num_heads, S, head_dim)
            #     value = value.transpose(1, 2) # (B, num_heads, S, head_dim)
            #     attn = query @ key.transpose(-2, -1) # (B, num_heads, L, S)
            #     if mask is not None:
            #         attn = attn + mask
            #     attn = F.dropout(attn.softmax(-1), self.dropout)
            #     attn_output = (attn @ value).transpose(1, 2) # (B, L, num_heads, head_dim)
            #     attn_output = attn_output.reshape(bs, -1, self.num_heads * self.head_dim)
            #     return self.W_o(attn_output)
            

            # attn_output = memory_efficient_attention(query, key, value, attn_bias=mask, p=self.dropout) # (B, L, num_heads, 2)

            attn_output = flash_attn_varlen_qkvpacked_func(
                qkv=torch.stack([query.squeeze(0), key.squeeze(0), value.squeeze(0)], dim=1),
                cu_seqlens=cu_seq_len,
                max_seqlen=max_seq_len,
                deterministic=True
            ).unsqueeze(0) # (B, L, num_heads, head_dim)

            attn_output = attn_output.reshape(bs, -1, self.num_heads * self.head_dim) # (B, L, embed_dim)
            return self.W_o(attn_output) # (B, L, embed_dim)
