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

        key_out = rearrange(key_out, 'b c t -> b t c')
        query_out = rearrange(query_out, 'b c t -> b t c')
        print(query_out.shape, key_out.shape)
        attn_logp = torch.cdist(query_out, key_out).unsqueeze(1)
        # attn_logp = rearrange(key_out, 'b c t -> b 1 c t')
        # attn_factor = (query_out[:, :, :, None] - key_out[:, :, None]) ** 2
        # attn_logp = -self.temperature * attn_factor.sum(1, keepdim=True)
        print("attn_logp: ", attn_logp.shape)
        if mask is not None:
            print(mask.shape)
            attn_logp.data.masked_fill_(~mask.bool().unsqueeze(2), -float("inf"))

        attn = self.softmax(attn_logp)
        return attn, attn_logp

def maximum_path(value, mask, const=None):
    if const is None:
        const = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=const)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], const)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path

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

class Aligner(nn.Module):
    def __init__(self, dim_in, dim_hidden, attn_channels=80 ,temperature=0.0005):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.attn_channels = attn_channels
        self.temperature = temperature
        self.aligner = AlignerNet(dim_in = self.dim_in, 
                                  dim_hidden = self.dim_hidden,
                                  attn_channels = self.attn_channels,
                                  temperature = self.temperature)
    def forward(self, x, x_mask, y, y_mask):
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        alignment_soft, alignment_logprob = self.aligner(y.transpose(1, 2), x, x_mask)
        print(rearrange(alignment_soft, 'b 1 c t -> b t c').shape, rearrange(attn_mask, 'b 1 c t -> b c t').shape)
        alignment_soft = rearrange(alignment_soft, 'b 1 c t -> b t c')
        alignment_mas = maximum_path(
            alignment_soft.contiguous(),
            rearrange(attn_mask, 'b 1 c t -> b c t').contiguous()
        )
        alignment_hard = torch.sum(alignment_mas, -1).int()
        return alignment_hard, alignment_soft, alignment_logprob, alignment_mas
    
if __name__ == '__main__':
    batch_size = 10
    seq_len_y = 200  # length of sequence y
    seq_len_x = 35
    feature_dim = 80  # feature dimension

    x = torch.randn(batch_size, 512, seq_len_x)
    y = torch.randn(batch_size, seq_len_y, feature_dim)
    
    # Create masks
    x_mask = torch.ones(batch_size, 1, seq_len_x)
    y_mask = torch.ones(batch_size, 1, seq_len_y)

    align = Aligner(dim_in = 80, dim_hidden=512, attn_channels=80)
    print(x.shape, x_mask.shape, y.shape, y_mask.shape)
    alignment_hard, alignment_soft, alignment_logprob, alignment_mas = align(x, x_mask, y, y_mask)

    print(alignment_hard.shape)
    print(alignment_soft.shape)
    print(alignment_logprob.shape)
    print(alignment_mas.shape)