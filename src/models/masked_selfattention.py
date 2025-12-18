import torch
import torch.nn as nn
import math


class MaskedSelfAttention(nn.Module):
    """
    Single-head masked self-attention over spatial locations.
    Mask blocks attention to background positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            fg_mask: [B, 1, H, W]  (1 = foreground, 0 = background)

        Returns:
            Tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dims
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]

        # QKV
        qkv = self.qkv(x_flat)                 # [B, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention logits
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(C)  # [B, N, N]

        # Build key mask (block background)
        fg = fg_mask.flatten(2).squeeze(1)    # [B, N]
        attn = attn.masked_fill(fg[:, None, :] == 0, float("-inf"))

        # Softmax
        attn = attn.softmax(dim=-1)

        # Attention output
        out = attn @ v                         # [B, N, C]
        out = self.proj(out)

        # Restore spatial shape
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out
