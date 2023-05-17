import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from naturalspeech2_pytorch.naturalspeech2_pytorch import (
    NaturalSpeech2,
    Transformer,
    Wavenet,
    Model,
    Trainer,
    PhonemeEncoder,
    DurationPitchPredictor,
    SpeechPromptEncoder,
    Tokenizer,
    ESpeak
)

from audiolm_pytorch import (
    SoundStream,
    EncodecWrapper
)
