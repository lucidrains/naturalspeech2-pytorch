from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
class AlignerNet(torch.nn.Module):
    """alignment model https://arxiv.org/pdf/2108.10447.pdf """
    def __init__(
        self,
        dim_in=80,
        dim_hidden=512,
        attn_channels=80,
        temperature=0.0005,
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)

        self.key_layers = nn.ModuleList([
            nn.Conv1d(
                dim_hidden,
                dim_hidden * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_hidden * 2, attn_channels, kernel_size=1, padding=0, bias=True)
        ])

        self.query_layers = nn.ModuleList([
            nn.Conv1d(
                dim_in,
                dim_in * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_in * 2, dim_in, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_in, attn_channels, kernel_size=1, padding=0, bias=True)
        ])

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor = None):
        key_out = keys
        for layer in self.key_layers:
            key_out = layer(key_out)

        query_out = queries
        for layer in self.query_layers:
            query_out = layer(query_out)

        key_out = rearrange(key_out, 'b c t -> b c 1 t')
        query_out = rearrange(query_out, 'b c t -> b c t 1')

        attn_logp = torch.cdist(query_out, key_out)

        if mask is not None:
            attn_logp.data.masked_fill_(~rearrange(mask, 'b t -> b t ()'), -torch.finfo(attn_logp.dtype).max)

        attn = self.softmax(attn_logp)
        return attn, attn_logp

def maximum_path(value, mask, const=None):
    if const is None:
        const = -np.inf
    value = value * mask

    b, t_x, t_y = value.shape
    dir_matrix = np.zeros_like(value, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)

    value = rearrange(value, 'b c t -> b t c')
    mask = rearrange(mask, 'b c t -> b t c')

    for j in range(t_y):
        v_prev = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=const)[:, :-1]
        v_max = np.maximum(v_prev, v)
        max_mask = v_max == v
        dir_matrix[:, :, j] = max_mask

        v = np.where(x_range <= j, v_max + value[:, j:j+1, :], const)

    dir_matrix = np.where(mask, dir_matrix, 1)

    path = np.zeros_like(value, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1

    for j in reversed(range(t_y)):
        path[index, np.arange(b), j] = 1
        index = index + dir_matrix[:, :, j][index, np.arange(b)] - 1

    path = path * mask.astype(np.float32)
    return torch.from_numpy(rearrange(path, 'b t c -> b c t'))

class ForwardSumLoss():
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_logprob.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_logprob = rearrange(attn_logprob, 'b c t -> c b t')

        # Add blank label
        attn_logprob = F.pad(attn_logprob, (1, 0, 0, 0, 0, 0), self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        device = attn_logprob.device
        attn_logprob.masked_fill_(torch.arange(max_key_len + 1, device=device, dtype=torch.long).view(1, 1, -1) > key_lens.view(1, -1, 1), -1e15)

        attn_logprob = self.log_softmax(attn_logprob)

        # Target sequences
        target_seqs = torch.arange(1, max_key_len + 1, device=device, dtype=torch.long).unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.ctc_loss(attn_logprob, target_seqs, query_lens, key_lens)

        return cost



class BinLoss():
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
