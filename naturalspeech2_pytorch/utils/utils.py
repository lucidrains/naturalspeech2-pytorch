import torch
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
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).to(values.dtype)
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).to(values.dtype)

    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems).to(values.dtype)
    return avg