<img src="./naturalspeech2.png" width="450px"></img>

## Natural Speech 2 - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2304.09116">Natural Speech 2</a>, Zero-shot Speech and Singing Synthesizer, in Pytorch

NaturalSpeech 2 is a TTS system that leverages a neural audio codec with continuous latent vectors and a latent diffusion model with non-autoregressive generation to enable natural and zero-shot text-to-speech synthesis

## Appreciation

- <a href="https://stability.ai/">Stability</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for the amazing accelerate library

- <a href="https://github.com/manmay-nakhashi">Manmay</a> for submitting the initial code for phoneme, pitch, duration, and speech prompt encoders!

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

model = Model(
    dim = 128,
    depth = 6
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

Or if you want a `Trainer` class to take care of the training and sampling loop, just simply do

```python
from naturalspeech2_pytorch import Trainer

trainer = Trainer(
    diffusion_model = diffusion,     # diffusion model + codec from above
    folder = '/path/to/speech',
    train_batch_size = 16,
    gradient_accumulate_every = 2,
)

trainer.train()
```

## Citations

```bibtex
@inproceedings{Shen2023NaturalSpeech2L,
    title   = {NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers},
    author  = {Kai Shen and Zeqian Ju and Xu Tan and Yanqing Liu and Yichong Leng and Lei He and Tao Qin and Sheng Zhao and Jiang Bian},
    year    = {2023}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Salimans2022ProgressiveDF,
    title   = {Progressive Distillation for Fast Sampling of Diffusion Models},
    author  = {Tim Salimans and Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.00512}
}
```

```bibtex
@inproceedings{Hang2023EfficientDT,
    title   = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
    author  = {Tiankai Hang and Shuyang Gu and Chen Li and Jianmin Bao and Dong Chen and Han Hu and Xin Geng and Baining Guo},
    year    = {2023}
}
```
