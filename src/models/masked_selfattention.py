import torch
import torch.nn as nn
import math


class MaskedSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, fg_mask):
        """
        x: [B, C, H, W]
        fg_mask: [B, 1, H, W] (1 = allowed, 0 = blocked)
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial
        x_flat = x.flatten(2).transpose(1, 2)   # [B, N, C]

        qkv = self.qkv(x_flat)                   # [B, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(C)  # [B, N, N]

        # Build attention mask
        fg = fg_mask.flatten(2).squeeze(1)      # [B, N]
        attn_mask = fg[:, None, :]               # [B, 1, N]

        # Mask out background keys
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)

        out = attn @ v                           # [B, N, C]
        out = self.proj(out)

        # Restore spatial
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out
