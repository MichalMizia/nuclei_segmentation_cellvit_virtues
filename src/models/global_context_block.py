import torch
import torch.nn as nn


class GlobalContextBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C, H, W) with global context mixed in.
        """
        B, C, H, W = x.shape
        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Apply Norm and Attention
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        # Residual connection
        x_out = x_flat + attn_out
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        return x_out