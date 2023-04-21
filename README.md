<img src="./naturalspeech2.png" width="450px"></img>

## Natural Speech 2 - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2304.09116">Natural Speech 2</a>, Zero-shot Speech and Singing Synthesizer, in Pytorch

They simply apply latent diffusion to residual vector quantized latents for these results.

## Install

```bash
$ pip install naturalspeech2-pytorch
```

## Usage

```python
import torch
from naturalspeech2_pytorch import (
    EncodecWrapper,
    Transformer,
    NaturalSpeech2
)

# use encodec as an example

codec = EncodecWrapper()

model = Transformer(
    dim = codec.codebook_dim,
    depth = 12
)

# natural speech diffusion model

diffusion = NaturalSpeech2(
    model = model,
    codec = codec,
    timesteps = 1000
).cuda()

# mock raw audio data

raw_audio = torch.randn(4, 327680).cuda()

loss = diffusion(raw_audio)
loss.backward()

# do the above in a loop for a lot of raw audio data...
# then you can sample from your generative model as so

generated_audio = diffusion.sample(length = 1024) # (1, 327680)

```

## Appreciation

- <a href="https://stability.ai/">Stability</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for the amazing accelerate library

## Citations

```bibtex
@inproceedings{Shen2023NaturalSpeech2L,
    title   = {NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers},
    author  = {Kai Shen and Zeqian Ju and Xu Tan and Yanqing Liu and Yichong Leng and Lei He and Tao Qin and Sheng Zhao and Jiang Bian},
    year    = {2023}
}
```
