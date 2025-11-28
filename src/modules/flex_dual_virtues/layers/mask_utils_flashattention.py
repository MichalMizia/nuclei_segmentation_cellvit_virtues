import torch
import functools
from collections import OrderedDict

class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val
    
SA_BIAS_CACHE = CacheDict(cache_len=20)


def cached_build_selfattention_bias(cache_key:str, split_mask:torch.Tensor, use_true_as_query=True, return_seq_lens=False, device="cuda"):
    """
    split_mask: ... x S tensor of bools
    If cross_attention, False will be used as queries and True as keys and values.
    """
    if cache_key in SA_BIAS_CACHE:
        return SA_BIAS_CACHE[cache_key]

    seq_lens = build_selfattention_bias(split_mask, use_true_as_query=use_true_as_query, return_seq_lens=return_seq_lens)
    seq_lens = [0] + seq_lens # since FlashAttn requires the first element to be 0
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, device=device).cumsum(dim=0, dtype=torch.int32)

    SA_BIAS_CACHE[cache_key] = (seq_lens, max_seq_len)
    return seq_lens, max_seq_len


def cached_build_selfattention_bias_channel_concat(cache_key:str, split_mask:torch.Tensor, tokens_per_sequence, use_true_as_query=True, device="cuda"):
    """
    This builds a BlockDiagonalMask when in the final dimension of the input tensor, the channels
    of different samples are concatenated and should not attend to each other.
    split_mask: ... x S tensor of bools
    tokens_per_sequence: number of channels per sample summing up to S i.e. S_i
    """
    if cache_key in SA_BIAS_CACHE:
        return SA_BIAS_CACHE[cache_key]

    seq_lens = build_selfattention_bias_channel_concat(split_mask, tokens_per_sequence, use_true_as_query=use_true_as_query)
    seq_lens = [0] + seq_lens # since FlashAttn requires the first element to be 0
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, device=device).cumsum(dim=0, dtype=torch.int32)
    SA_BIAS_CACHE[cache_key] = (seq_lens, max_seq_len)
    return seq_lens, max_seq_len

@torch.no_grad()
def build_selfattention_bias(split_mask:torch.Tensor, use_true_as_query=True, return_seq_lens=False):
    """
    split_mask: ... x S tensor of bools
    If cross_attention, False will be used as queries and True as keys and values.
    """

    if use_true_as_query:
        seq_lens = split_mask.sum(-1)
    else:
        seq_lens = (~split_mask).sum(-1)
    seq_lens = seq_lens[seq_lens != 0] # Filter-out zero length sequences for self-attention
    seq_lens = seq_lens.tolist()
    # if return_seq_lens:
    return seq_lens

# @torch.compile
def build_selfattention_bias_channel_concat(split_mask:torch.Tensor, tokens_per_sequence, use_true_as_query=True):
    """
    This builds a BlockDiagonalMask when in the final dimension of the input tensor, the channels
    of different samples are concatenated and should not attend to each other.
    split_mask: ... x S tensor of bools
    tokens_per_sequence: number of channels per sample summing up to S i.e. S_i
    """
    split_masks = split_mask.split(tokens_per_sequence, dim=-1) # list of ... x S_i tensors
    seq_lens = torch.stack([calc_seq_lens_sums(split_mask, use_true_as_query=use_true_as_query) for split_mask in split_masks], dim=-1)
    seq_lens = seq_lens[seq_lens != 0] # Filter-out zero length sequences for self-attention
    seq_lens = seq_lens.tolist()
    # if return_seq_lens:
    return seq_lens

@torch.no_grad()
def calc_seq_lens_sums(split_mask:torch.Tensor, use_true_as_query=True):
    """
    split_mask: ... x S tensor of bools
    If cross_attention, False will be used as queries and True as keys and values.
    """
    if use_true_as_query:
        seq_lens = split_mask.sum(-1)
    else:
        seq_lens = (~split_mask).sum(-1)
    return seq_lens

@torch.no_grad()
def cached_get_non_zero_indices(cache_key: str, x:torch.Tensor):
    """
    x: ... x S tensor of bools
    Returns a list of indices of non-zero elements in the last dimension
    """
    if cache_key in SA_BIAS_CACHE:
        return SA_BIAS_CACHE[cache_key]
    
    indices = get_non_zero_indices(x)
    SA_BIAS_CACHE[cache_key] = indices
    return indices

@torch.no_grad()
def get_non_zero_indices(x: torch.Tensor):
    """
    x: ... x S tensor of bools
    Returns a list of indices of non-zero elements in the last dimension
    """
    return x.nonzero(as_tuple=True)