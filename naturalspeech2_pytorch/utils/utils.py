import torch
from einops import repeat, rearrange

def average_over_durations(values, durs):
    """
        - in:
            - values: B, 1, T_de
            - durs: B, T_en
        - out:
            - avg: B, 1, T_en
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = repeat(durs_cums_starts, 'bs l -> bs n l', n=n_formants)
    dce = repeat(durs_cums_ends, 'bs l -> bs n l', n=n_formants)

    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).to(values.dtype)
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).to(values.dtype)

    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems).to(values.dtype)
    return avg

def create_mask(sequence_length, max_len):
    dtype, device = sequence_length.dtype, sequence_length.device
    seq_range = torch.arange(max_len, dtype=dtype, device=device)
    sequence_length = rearrange(sequence_length, 'b -> b 1')
    seq_range = rearrange(seq_range, 't -> 1 t')
    return seq_range < sequence_length
