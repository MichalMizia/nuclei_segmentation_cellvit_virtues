import torch
import torch.nn as nn

from torch import LongTensor

class PositionalEmbedding2D(nn.Module):

    def __init__(self, model_dim : int, max_width_or_height : int = 1200, temperature : float = 10000.):
        super(PositionalEmbedding2D, self).__init__()

        assert model_dim % 4 == 0, 'Embedding dimension must be multiple of 4 for 2D positional embedding'

        dim_pe = model_dim // 2

        possible_positions = torch.arange(max_width_or_height, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_pe , 2, dtype=torch.float32) * - (torch.log(torch.tensor(temperature)) / dim_pe))
        pos = possible_positions * div_term
        sin = torch.sin(pos)
        cos = torch.cos(pos)

        self.register_buffer('positional_embeddings', torch.zeros(max_width_or_height, dim_pe))

        self.positional_embeddings[:, 0::2] = sin
        self.positional_embeddings[:, 1::2] = cos

    # positions = (batch_size, seq_len, 2)
    def forward(self, x, positions : LongTensor):
        """
        Computes positional embeddings corresponding to 2D input positions
        Args:
            x: (..., model_dim) tensor
            positions: (..., 2) tensor tensor of 2D positions
        Returns:
            (..., model_dim) tensor of positional embeddings
        """
        rows = positions.select(dim=-1, index=0)
        cols = positions.select(dim=-1, index=1)

        row_pos_emb = self.positional_embeddings[rows]
        col_pos_emb = self.positional_embeddings[cols]

        pos_emb = torch.cat([row_pos_emb, col_pos_emb], dim=-1)
        return x + pos_emb    


class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(self, model_dim : int, max_seq_length : int = 1200, temperature : float = 10000.):
        super(RotaryPositionalEmbedding1D, self).__init__()

        assert model_dim % 2 == 0, 'Embedding dimension must be multiple of 2 for 1D positional embedding'
        self.model_dim = model_dim

        possible_positions = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim , 2, dtype=torch.float32) * - (torch.log(torch.tensor(temperature)) / model_dim))
        pos = possible_positions * div_term
        sin = torch.sin(pos)
        sin = torch.concat([sin, sin], dim=-1)
        self.register_buffer('sin', sin)
        cos = torch.cos(pos)
        cos = torch.concat([cos, cos], dim=-1)
        self.register_buffer('cos', cos)

    def invert_negate(self, x):
        return torch.cat([-x[...,self.model_dim // 2:], x[...,:self.model_dim // 2]], dim=-1)

    def forward(self, x, pos):
        """
        Applies rotary positional encoding to input tensor
        Args:
            x: (..., model_dim) tensor
            pos: (..., ) tensor of positions
        """
        x = x * self.cos[pos] +  self.invert_negate(x) * self.sin[pos]
        return x

class RotaryPositionalEmbedding2D(nn.Module):

    def __init__(self, model_dim : int, max_pos : int = 1200, temperature : float = 10000.):
        super(RotaryPositionalEmbedding2D, self).__init__()

        assert model_dim % 4 == 0, 'Embedding dimension must be multiple of 4 for 2D positional embedding'
        self.model_dim = model_dim
        self.rope1d = RotaryPositionalEmbedding1D(model_dim // 2, max_pos, temperature)

    def forward(self, x, pos):
        """
        Applies 2D rotary positional encoding to input tensor
        Args:
            x: (..., model_dim) tensor
            pos: (..., 2) tensor of 2D positions
        """
        d = self.model_dim // 2

        x1 = x[..., :d]
        x2 = x[..., d:]

        x1 = self.rope1d(x1, pos.select(dim=-1, index=0))
        x2 = self.rope1d(x2, pos.select(dim=-1, index=1))

        return torch.cat([x1, x2], dim=-1)
    
class LearnablePositionalEmbedding2D(nn.Module):

    def __init__(self, model_dim, max_pos=100):
        super(LearnablePositionalEmbedding2D, self).__init__()
        self.pos_embeddings = nn.Parameter(torch.randn(max_pos, max_pos, model_dim) / model_dim**2)

    def forward(self, x, pos):
        """
        Applies learnable positional embedding to input tensor
        Args:
            x: (..., model_dim) tensor
            pos: (..., 2) tensor of 2D positions
        """
        to_add = self.pos_embeddings[pos[...,0], pos[...,1]]
        return x + to_add


# From: https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py
class VisionRotaryPositionalEmbeddings(nn.Module):
    """
    This class implements two-dimensional Rotary Positional Embeddings (RoPE) for images
    based on the axial frequency 2D RoPE described in https://arxiv.org/pdf/2403.13298.

    The position embedding is simply applied to the x-axis and y-axis separately, encoding
    the x and y position of each patch within every tile.. The embedding is applied to each
    tile identically.

    Note: This module assumes the CLS token embedding is appended at the end of the sequence.

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the full input image. In this case, the function will consider your image as a single tile.
        dim (int): Embedding dimension. Unlike :class:`~torchtune.modules.RotaryPositionalEmbeddings`, this is
            usually set to the dim of each head in the attention module divided by 2, computed as
            ``embed_dim // num_heads // 2``. The divide by 2 accounts for x and y positions.
        base (int): The base for the geometric progression used to compute
            the rotation angles
        append_cls_token (bool): Set to True if CLS token embedding is at the end of the sequence in the vision transformer,
            False if is in the beginning of the sequence. RoPE is zeroed out for the CLS token. Default is True.
    """

    def __init__(
        self,
        patch_size: int,
        tile_size: int,
        dim: int,
        base: int = 10_000,
        append_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.patch_grid_size = tile_size // patch_size
        self.seq_len = self.patch_grid_size**2 + 1
        self.dim = dim
        self.base = base
        self.append_cls_token = append_cls_token
        self.rope_init()

    def rope_init(self):
        dim = self.dim // 2
        theta = 1.0 / (
            self.base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self) -> None:
        # Create position indices for each patch in the tile
        patches_per_tile = self.patch_grid_size**2
        patch_idx = torch.arange(
            patches_per_tile, dtype=self.theta.dtype, device=self.theta.device
        )
        # # Add a placeholder index for CLS token - will not be used in RoPE
        # if self.append_cls_token:
        #     patch_idx = torch.cat(
        #         [
        #             patch_idx,
        #             -1 * torch.ones(1, dtype=patch_idx.dtype, device=patch_idx.device),
        #         ]
        #     )
        # else:
        #     patch_idx = torch.cat(
        #         [
        #             -1 * torch.ones(1, dtype=patch_idx.dtype, device=patch_idx.device),
        #             patch_idx,
        #         ]
        #     )
        # Encode x and y positions of each patch in the tile
        patch_x_pos = patch_idx % self.patch_grid_size
        patch_y_pos = patch_idx // self.patch_grid_size

        # Outer product of theta and position index; output tensor has
        # a shape of [patches_per_tile + 1, dim // 4]
        x_theta = torch.einsum("i, j -> ij", patch_x_pos + 1, self.theta).float()
        y_theta = torch.einsum("i, j -> ij", patch_y_pos + 1, self.theta).float()

        # Shape: [patches_per_tile + 1, dim]
        freqs = torch.cat([x_theta, y_theta], dim=-1)
        # Zero out CLS token position frequencies
        freqs = freqs.masked_fill(patch_idx.unsqueeze(-1) < 0, 0)

        # cache includes both the cos and sin components and so the output shape is
        # [patches_per_tile + 1, dim, 2]
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``[b, s, n_h, h_d]``
            **kwargs (Any): additional keyword arguments. This is kept to match the forward signature of
                :class:`~torchtune.modules.RotaryPositionalEmbeddings`.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        bsz, _, n_h, h_d = x.shape

        # reshape input; the last dimension is used for computing the output.
        # Split tile dimension from the sequence dimension
        # Cast to float to match the reference implementation
        # tensor has shape [b, max_num_tiles, s // max_num_tiles, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(bsz, -1, self.seq_len, n_h, h_d // 2, 2)

        # reshape the cache for broadcasting
        rope_cache = self.cache.view(1, 1, self.seq_len, 1, h_d // 2, 2)

        # tensor has shape [b, max_num_tiles, s // max_num_tiles, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # Squash tile dimension back into sequence dimension - tensor has shape [b, s, n_h, h_d]
        x_out = x_out.reshape(bsz, -1, n_h, h_d)
        return x_out.type_as(x)