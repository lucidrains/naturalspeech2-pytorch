import math
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import torchaudio

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from audiolm_pytorch import SoundStream, EncodecWrapper
from audiolm_pytorch.data import SoundDataset, get_dataloader

from beartype import beartype
from beartype.typing import Tuple, Union, Optional, List
from beartype.door import is_bearable

from naturalspeech2_pytorch.attend import Attend
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer, ESpeak

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

# constants

mlist = nn.ModuleList

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# tensor helpers

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# sinusoidal positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# peripheral models

# phoneme - pitch - speech prompt - duration predictors

class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        *,
        tokenizer: Optional[Tokenizer] = None,
        num_tokens = None,
        dim = 512,
        dim_hidden = 1024,
        kernel_size = 9,
        depth = 6,
        dim_head = 64,
        heads = 8,
        conv_dropout = 0.2,
        attn_dropout = 0.,
        use_flash = False
    ):
        super().__init__()

        self.tokenizer = tokenizer
        num_tokens = default(num_tokens, tokenizer.vocab_size if exists(tokenizer) else None)

        self.token_emb = nn.Embedding(num_tokens + 1, dim) if exists(num_tokens) else nn.Identity()
        self.pad_id = num_tokens

        same_padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            Rearrange('b n c -> b c n'),
            CausalConv1d(dim, dim_hidden, kernel_size),
            nn.SiLU(),
            nn.Dropout(conv_dropout),
            Rearrange('b c n -> b n c'),
        )

        self.transformer = Transformer(
            dim = dim_hidden,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            dropout = attn_dropout,
            use_flash = use_flash
        )

    @beartype
    def forward(
        self,
        x: Union[Tensor, List[str]],
        mask = None
    ):
        if is_bearable(x, List[str]):
            assert exists(self.tokenizer)
            x = self.tokenizer.texts_to_tensor_ids(x)

        is_padding = x < 0
        x = x.masked_fill(is_padding, self.pad_id)

        x = self.token_emb(x)
        x = self.conv(x)
        x = self.transformer(x, mask = mask)
        return x

class SpeechPromptEncoder(nn.Module):

    @beartype
    def __init__(
        self,
        dim_codebook,
        dims: Tuple[int] = (256, 2048, 2048, 2048, 2048, 512, 512, 512),
        *,
        depth,
        heads = 8,
        dim_head = 64,
        dropout = 0.2,
        kernel_size = 9,
        padding = 4,
        use_flash_attn = True

    ):
        super().__init__()

        dims = [dim_codebook, *dims]

        self.dim, self.dim_out = dims[0], dims[-1]

        dim_pairs = zip(dims[:-1], dims[1:])

        modules = []
        for dim_in, dim_out in dim_pairs:
            modules.extend([
                nn.Conv1d(dim_in, dim_out, kernel_size, padding = padding),
                nn.SiLU()
            ])

        self.conv = nn.Sequential(
            Rearrange('b n c -> b c n'),
            *modules,
            Rearrange('b c n -> b n c')
        )

        self.transformer = Transformer(
            dim = dims[-1],
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            dropout = dropout,
            use_flash = use_flash_attn
        )

    def forward(self, x):
        assert x.shape[-1] == self.dim

        x = self.conv(x)
        x = self.transformer(x)
        return x

# duration and pitch predictor seems to be the same

class DurationPitchPredictor(nn.Module):
    def __init__(
        self,
        dim=512,
        depth=30,
        kernel_size=9,
        heads=8,
        dim_head=64,
        dim_hidden=2048,
        dropout=0.1,
        attn_every=3,
        dropout_attn=0.5
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            is_attn_layer = _ % attn_every == 0
            self.layers.append(nn.ModuleList([
                nn.Sequential(
                    Rearrange('b n c -> b c n'),
                    nn.Conv1d(dim_hidden, dim_hidden, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.LayerNorm(dim_hidden),
                    nn.Dropout(dropout_attn if is_attn_layer else dropout),
                    Rearrange('b c n -> b n c'),
                ),
                RMSNorm(dim) if is_attn_layer else None,
                Attention(
                    dim_hidden,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    use_flash=True,
                    cross_attn_include_queries=True
                ) if is_attn_layer else None
            ]))

        self.to_pred = nn.Sequential(
            nn.Linear(dim_hidden, 2),
            nn.ReLU()
        )

    @beartype
    def forward(
        self,
        x: Union[Tensor, List[str]],
        encoded_prompts,
        duration=None,
        pitch=None
    ):
        for conv, norm, attn in self.layers:
            x = conv(x)
            if attn is not None:
                x = attn(norm(x), encoded_prompts) + x

        duration_pred, pitch_pred = self.to_pred(x).unbind(dim=-1)

        duration_return = F.l1_loss(duration, duration_pred) if duration is not None else duration_pred
        pitch_return = F.l1_loss(pitch, pitch_pred) if pitch is not None else pitch_pred

        return duration_return, pitch_return

# use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198
# in lieu of "q-k-v" attention with the m queries becoming key / values on which ddpm network is conditioned on

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_context = None,
        num_latents = 64, # m in the paper
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        use_flash_attn = True
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std = 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    use_flash = use_flash_attn,
                    cross_attn_include_queries = True
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x, mask = None):
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = repeat(self.latents, 'n d -> b n d', b = batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask = mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

# model, which is wavenet + transformer

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size, = self.kernel_size
        dilation, = self.dilation
        stride, = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value = 0.)
        return super().forward(causal_padded_x)

class WavenetResBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dilation,
        kernel_size = 3,
        skip_conv = False,
        dim_cond_mult = None
    ):
        super().__init__()

        self.cond = exists(dim_cond_mult)
        self.to_time_cond = None

        if self.cond:
            self.to_time_cond = nn.Linear(dim * dim_cond_mult, dim * 2)

        self.conv = CausalConv1d(dim, dim, kernel_size, dilation = dilation)
        self.res_conv = CausalConv1d(dim, dim, 1)
        self.skip_conv = CausalConv1d(dim, dim, 1) if skip_conv else None

    def forward(self, x, t = None):

        if self.cond:
            assert exists(t)
            t = self.to_time_cond(t)
            t = rearrange(t, 'b c -> b c 1')
            t_gamma, t_beta = t.chunk(2, dim = -2)

        res = self.res_conv(x)

        x = self.conv(x)

        if self.cond:
            x = x * t_gamma + t_beta

        x = x.tanh() * x.sigmoid()

        x = x + res

        skip = None
        if exists(self.skip_conv):
            skip = self.skip_conv(x)

        return x, skip


class WavenetStack(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers,
        kernel_size = 3,
        has_skip = False,
        dim_cond_mult = None
    ):
        super().__init__()
        dilations = 2 ** torch.arange(layers)

        self.has_skip = has_skip
        self.blocks = mlist([])

        for dilation in dilations.tolist():
            block = WavenetResBlock(
                dim = dim,
                kernel_size = kernel_size,
                dilation = dilation,
                skip_conv = has_skip,
                dim_cond_mult = dim_cond_mult
            )

            self.blocks.append(block)

    def forward(self, x, t):
        residuals = []
        skips = []

        if isinstance(x, Tensor):
            x = (x,) * len(self.blocks)

        for block_input, block in zip(x, self.blocks):
            residual, skip = block(block_input, t)

            residuals.append(residual)
            skips.append(skip)

        if self.has_skip:
            return torch.stack(skips)

        return residuals

class Wavenet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        stacks,
        layers,
        init_conv_kernel = 3,
        dim_cond_mult = None
    ):
        super().__init__()
        self.init_conv = CausalConv1d(dim, dim, init_conv_kernel)
        self.stacks = mlist([])

        for ind in range(stacks):
            is_last = ind == (stacks - 1)

            stack = WavenetStack(
                dim,
                layers = layers,
                dim_cond_mult = dim_cond_mult,
                has_skip = is_last
            )

            self.stacks.append(stack)

        self.final_conv = CausalConv1d(dim, dim, 1)

    def forward(self, x, t = None):

        x = self.init_conv(x)

        for stack in self.stacks:
            x = stack(x, t)

        return self.final_conv(x.sum(dim = 0))

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, dim_cond = None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond = None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim = -1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta

class ConditionableTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_causal_conv = False,
        dim_cond_mult = None,
        cross_attn = False,
        use_flash = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = mlist([])

        cond = exists(dim_cond_mult)

        maybe_adaptive_norm_kwargs = dict(scale = not cond, dim_cond = dim * dim_cond_mult) if cond else dict()
        rmsnorm = partial(RMSNorm, **maybe_adaptive_norm_kwargs)

        for _ in range(depth):
            self.layers.append(mlist([
                rmsnorm(dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash = use_flash),
                rmsnorm(dim) if cross_attn else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash = use_flash) if cross_attn else None,
                rmsnorm(dim),
                FeedForward(dim = dim, mult = ff_mult, causal_conv = ff_causal_conv)
            ]))

        self.to_pred = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim, bias = False)
        )

    def forward(
        self,
        x,
        times = None,
        context = None
    ):
        t = times

        for attn_norm, attn, cross_attn_norm, cross_attn, ff_norm, ff in self.layers:
            res = x
            x = attn_norm(x, cond = t)
            x = attn(x) + res

            if exists(cross_attn):
                assert exists(context)
                res = x
                x = cross_attn_norm(x, cond = t)
                x = cross_attn(x, context = context) + res

            res = x
            x = ff_norm(x, cond = t)
            x = ff(x) + res

        return self.to_pred(x)

class Model(nn.Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        wavenet_layers = 8,
        wavenet_stacks = 4,
        dim_cond_mult = 4,
        use_flash_attn = True,
        speech_prompt_encoder: Optional[SpeechPromptEncoder] = None,
        dim_prompt = None,
        num_latents_m = 32,   # number of latents to be perceiver resampled ('q-k-v' with 'm' queries in the paper)
        resampler_depth = 2,
        cond_drop_prob = 0.
    ):
        super().__init__()
        self.dim = dim

        # time condition

        dim_time = dim * dim_cond_mult

        self.to_time_cond = Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim + 1, dim_time),
            nn.SiLU()
        )

        # prompt condition

        condition_on_prompt = exists(speech_prompt_encoder)

        self.cond_drop_prob = cond_drop_prob # for classifier free guidance
        self.condition_on_prompt = condition_on_prompt
        self.to_prompt_cond = None

        if self.condition_on_prompt:
            dim_prompt = default(dim_prompt, speech_prompt_encoder.dim_out)

            self.speech_prompt_encoder = speech_prompt_encoder
            assert speech_prompt_encoder.dim_out == dim_prompt

            self.null_prompt_cond = nn.Parameter(torch.randn(dim_time))
            self.null_prompt_tokens = nn.Parameter(torch.randn(num_latents_m, dim))

            nn.init.normal_(self.null_prompt_cond, std = 0.02)
            nn.init.normal_(self.null_prompt_tokens, std = 0.02)

            self.to_prompt_cond = Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.Linear(dim_prompt, dim_time),
                nn.SiLU()
            )

            self.perceiver_resampler = PerceiverResampler(
                dim = dim,
                dim_context = dim_prompt,
                num_latents = num_latents_m,
                depth = resampler_depth,
                dim_head = dim_head,
                heads = heads,
                use_flash_attn = use_flash_attn
            )

        # conditioning includes time and optionally prompt

        dim_cond_mult = dim_cond_mult * (2 if condition_on_prompt else 1)

        # wavenet

        self.wavenet = Wavenet(
            dim = dim,
            stacks = wavenet_stacks,
            layers = wavenet_layers,
            dim_cond_mult = dim_cond_mult
        )

        # transformer

        self.transformer = ConditionableTransformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_causal_conv = True,
            dim_cond_mult = dim_cond_mult,
            use_flash = use_flash_attn,
            cross_attn = condition_on_prompt
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        times,
        prompt = None,
        prompt_mask = None,
        cond_drop_prob = None
    ):
        b = x.shape[0]
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)

        t = self.to_time_cond(times)
        c = None

        if exists(self.to_prompt_cond):
            assert exists(prompt)
            encoded_prompt = self.speech_prompt_encoder(prompt)

            prompt_cond = self.to_prompt_cond(encoded_prompt)

            prompt_cond = torch.where(
                rearrange(drop_mask, 'b -> b 1'),
                self.null_prompt_cond,
                prompt_cond,
            )

            t = torch.cat((t, prompt_cond), dim = -1)

            resampled_prompt_tokens = self.perceiver_resampler(encoded_prompt, mask = prompt_mask)

            c = torch.where(
                rearrange(drop_mask, 'b -> b 1 1'),
                self.null_prompt_tokens,
                resampled_prompt_tokens
            )

        x = rearrange(x, 'b n d -> b d n')
        x = self.wavenet(x, t)
        x = rearrange(x, 'b d n -> b n d')

        x = self.transformer(x, t, context = c)
        return x

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, causal_conv = False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange('b d n -> b n d'),
        )

    return Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        conv,
        nn.Linear(dim_inner, dim)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash = False,
        cross_attn_include_queries = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal = causal, dropout = dropout, use_flash = use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim = -2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        dim_head = 64,
        heads = 8,
        use_flash = False,
        dropout = 0.,
        ff_mult = 4,
        final_norm = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim),
                Attention(
                    dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    dropout = dropout,
                    use_flash = use_flash
                ),
                RMSNorm(dim),
                FeedForward(
                    dim,
                    mult = ff_mult
                )
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x, mask = None):
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), mask = mask) + x
            x = ff(ff_norm(x)) + x

        return self.norm(x)

# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(numer, denom):
    return numer / denom.clamp(min = 1e-10)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# gaussian diffusion

class NaturalSpeech2(nn.Module):

    @beartype
    def __init__(
        self,
        model: Model,
        codec: Optional[Union[SoundStream, EncodecWrapper]] = None,
        *,
        tokenizer: Optional[Tokenizer] = None,
        target_sample_hz = None,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule = 'sigmoid',
        objective = 'v',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        train_prob_self_cond = 0.9,
        rvq_cross_entropy_loss_weight = 0.,    # default this to off until we are sure it is working. not totally sold that this is critical
        scale = 1.                             # this will be set to < 1. for better convergence when training on higher resolution images
    ):
        super().__init__()
        self.model = model
        self.codec = codec

        assert exists(codec) or exists(target_sample_hz)

        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = None

        if exists(codec):
            self.target_sample_hz = codec.target_sample_hz
            self.seq_len_multiple_of = codec.seq_len_multiple_of

        assert not exists(codec) or model.dim == codec.codebook_dim, f'transformer model dimension {model.dim} must be equal to codec dimension {codec.codebook_dim}'

        self.dim = codec.codebook_dim if exists(codec) else model.dim

        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training

        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale

        # gamma schedules

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

        # probability for self conditioning during training

        self.train_prob_self_cond = train_prob_self_cond

        # min snr loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        # weight of the cross entropy loss to residual vq codebooks

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight

    @property
    def device(self):
        return next(self.model.parameters()).device

    def print(self, s):
        return self.accelerator.print(s)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, prompt = None, time_difference = None, cond_scale = 1.):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = time

            # get predicted x0

            model_output = self.model.forward_with_cond_scale(audio, noise_cond, prompt = prompt, cond_scale = cond_scale)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (audio * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(audio),
                torch.zeros_like(audio)
            )

            audio = mean + (0.5 * log_variance).exp() * noise

        return audio

    @torch.no_grad()
    def ddim_sample(self, shape, prompt = None, time_difference = None, cond_scale = 1.):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0

            model_output = self.model.forward_with_cond_scale(audio, times, prompt = prompt, cond_scale = cond_scale)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # get predicted noise

            pred_noise = safe_div(audio - alpha * x_start, sigma)

            # calculate x next

            audio = x_start * alpha_next + pred_noise * sigma_next

        return audio

    def process_prompt(self, prompt = None):
        if not exists(prompt):
            return None

        assert self.model.condition_on_prompt

        is_raw_prompt = prompt.ndim == 2
        assert not (is_raw_prompt and not exists(self.codec)), 'codec must be passed in if one were to train on raw prompt'

        if is_raw_prompt:
            with torch.no_grad():
                self.codec.eval()
                prompt, _, _ = self.codec(prompt, curtail_from_left = True, return_encoded = True)

        return prompt

    @torch.no_grad()
    def sample(
        self,
        *,
        length,
        prompt = None,
        batch_size = 1,
        cond_scale = 1.
    ):
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample

        prompt = self.process_prompt(prompt)

        if exists(prompt):
            batch_size = prompt.shape[0]

        audio = sample_fn((batch_size, length, self.dim), prompt = prompt, cond_scale = cond_scale)

        if exists(self.codec):
            audio = self.codec.decode(audio)

            if audio.ndim == 3:
                audio = rearrange(audio, 'b 1 n -> b n')

        return audio

    def forward(
        self,
        audio,
        codes = None,
        prompt = None,
        *args,
        **kwargs
    ):
        is_raw_audio = audio.ndim == 2

        assert not (is_raw_audio and not exists(self.codec)), 'codec must be passed in if one were to train on raw audio'

        if is_raw_audio:
            with torch.no_grad():
                self.codec.eval()
                audio, codes, _ = self.codec(audio, return_encoded = True)

        prompt = self.process_prompt(prompt)

        batch, n, d, device = *audio.shape, self.device

        assert d == self.dim, f'codec codebook dimension {d} must match model dimensions {self.dim}'

        # sample random times

        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)

        # noise sample

        noise = torch.randn_like(audio)

        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(audio, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)

        noised_audio = alpha * audio + sigma * noise

        # predict and take gradient step

        pred = self.model(noised_audio, times, prompt = prompt)

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = audio

        elif self.objective == 'v':
            target = alpha * noise - sigma * audio

        loss = F.mse_loss(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # min snr loss weight

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'x0':
            loss_weight = maybe_clipped_snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        loss =  (loss * loss_weight).mean()

        # cross entropy loss to codebooks

        if self.rvq_cross_entropy_loss_weight == 0 or not exists(codes):
            return loss

        if self.objective == 'x0':
            x_start = pred

        elif self.objective == 'eps':
            x_start = safe_div(audio - sigma * pred, alpha)

        elif self.objective == 'v':
            x_start = alpha * audio - sigma * pred

        _, ce_loss = self.codec.rq(x_start, codes)

        return loss + self.rvq_cross_entropy_loss_weight * ce_loss

# trainer

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        diffusion_model: NaturalSpeech2,
        *,
        dataset: Optional[Dataset] = None,
        folder = None,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 1,
        results_folder = './results',
        amp = False,
        fp16 = False,
        use_ema = True,
        split_batches = True,
        dataloader = None,
        data_max_length = None,
        data_max_length_seconds = 2,
        sample_length = None
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        assert exists(diffusion_model.codec)

        self.dim = diffusion_model.dim

        # training hyperparameters

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = dataloader

        if not exists(dl):
            assert exists(dataset) or exists(folder)

            if exists(dataset):
                self.ds = dataset
            elif exists(folder):
                # create dataset

                if exists(data_max_length_seconds):
                    assert not exists(data_max_length)
                    data_max_length = int(data_max_length_seconds * diffusion_model.target_sample_hz)
                else:
                    assert exists(data_max_length)

                self.ds = SoundDataset(
                    folder,
                    max_length = data_max_length,
                    target_sample_hz = diffusion_model.target_sample_hz,
                    seq_len_multiple_of = diffusion_model.seq_len_multiple_of
                )

                dl = DataLoader(
                    self.ds,
                    batch_size = train_batch_size,
                    shuffle = True,
                    pin_memory = True,
                    num_workers = cpu_count()
                )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.use_ema = use_ema
        self.ema = None

        if self.accelerator.is_main_process and use_ema:
            # make sure codec is not part of the EMA
            # encodec seems to be not deepcopyable, so this is a necessary hack

            codec = diffusion_model.codec
            diffusion_model.codec = None

            self.ema = EMA(
                diffusion_model,
                beta = ema_decay,
                update_every = ema_update_every,
                ignore_startswith_names = set(['codec.'])
            ).to(self.device)

            diffusion_model.codec = codec
            self.ema.ema_model.codec = codec

        # sampling hyperparameters

        self.sample_length = default(sample_length, data_max_length)
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        # results folder

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)
    
    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step % self.save_and_sample_every == 0:

                        models = [(self.unwrapped_model, str(self.step))]

                        if self.use_ema:
                            models.append((self.ema.ema_model, f'{self.step}.ema'))

                        for model, label in models:
                            model.eval()

                            with torch.no_grad():
                                generated = model.sample(
                                    batch_size = self.num_samples,
                                    length = self.sample_length
                                )

                            for ind, t in enumerate(generated):
                                filename = str(self.results_folder / f'sample_{label}.flac')
                                t = rearrange(t, 'n -> 1 n')
                                torchaudio.save(filename, t.cpu().detach(), self.unwrapped_model.target_sample_hz)

                        self.print(f'{steps}: saving to {str(self.results_folder)}')

                        self.save(milestone)

                pbar.update(1)

        self.print('training complete')
