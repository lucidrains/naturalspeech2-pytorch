import torch
from torch import Tensor
from typing import Callable, List, Optional, Tuple

from torch.nn.utils.rnn import pad_sequence

from naturalspeech2_pytorch.utils.cleaner import TextProcessor
from naturalspeech2_pytorch.utils.phonemizers.espeak_wrapper import ESpeak

# default phoneme set

_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "'̃ˈˌːˑ. ,-"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
_diacrilics = "ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics

# default map

LANGUAGE_MAP = {
    'en-us': 'en',
    'fr-fr': 'es',
    'hi': 'hi'
}

# functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

class Tokenizer:
    def __init__(
        self,
        vocab = _phonemes,
        text_cleaner: Optional[Callable] = None,
        phonemizer: Optional[Callable] = None,
        default_lang = "en-us",
        add_blank: bool = False,
        use_eos_bos = False,
        pad_id = -1
    ):
        self.text_cleaner = default(text_cleaner, TextProcessor().phoneme_cleaners)
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.pad_id = pad_id

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        self.phonemizer = phonemizer
        if not exists(self.phonemizer):
            self.phonemizer = ESpeak(language = default_lang)

        self.language = self.phonemizer.language
        self.not_found_characters = []

    @property
    def espeak_language(self):
        return LANGUAGE_MAP.get(self.language, None)

    def encode(self, text: str) -> List[int]:
        """Encodes a string of text as a sequence of IDs."""
        token_ids = []
        for char in text:
            try:
                idx = self.char_to_id[char]
                token_ids.append(idx)
            except KeyError:
                # discard but store not found characters
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    print(text)
                    print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a sequence of IDs to a string of text."""
        text = ""
        for token_id in token_ids:
            text += self.id_to_char[token_id]
        return text

    def text_to_ids(
        self,
        text: str,
        language: str = None
    ) -> Tuple[List[int], str, str]:
        """Converts a string of text to a sequence of token IDs.

        Args:
            text(str):
                The text to convert to token IDs.

            language(str):
                The language code of the text. Defaults to None.

        TODO:
            - Add support for language-specific processing.

        1. Text normalizatin
        2. Phonemization (if use_phonemes is True)
        3. Add blank char between characters
        4. Add BOS and EOS characters
        5. Text to token IDs
        """

        language = default(language, self.espeak_language)

        cleaned_text = None
        if self.text_cleaner is not None:
            text = self.text_cleaner(text, language=language)
            cleaned_text = text
        phonemized = self.phonemizer.phonemize(text, separator="", language=language)
        if self.add_blank:
            phonemized = self.intersperse_blank_char(phonemized, True)
        if self.use_eos_bos:
            phonemized = self.pad_with_bos_eos(phonemized)

        return self.encode(phonemized), cleaned_text, phonemized

    def texts_to_tensor_ids(self, texts: List[str], language: str = None) -> Tensor:
        all_ids = []

        for text in texts:
            ids, *_ = self.text_to_ids(text, language = language)
            all_ids.append(torch.tensor(ids))

        return pad_sequence(all_ids, batch_first = True, padding_value = self.pad_id)

    def ids_to_text(self, id_sequence: List[int]) -> str:
        """Converts a sequence of token IDs to a string of text."""
        return self.decode(id_sequence)

    def pad_with_bos_eos(self, char_sequence: List[str]):
        """Pads a sequence with the special BOS and EOS characters."""
        return [self.characters.bos] + list(char_sequence) + [self.characters.eos]

    def intersperse_blank_char(self, char_sequence: List[str], use_blank_char: bool = False):
        """Intersperses the blank character between characters in a sequence.

        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        char_to_use = self.characters.blank if use_blank_char else self.characters.pad
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result

if __name__ == "__main__":
    txt_cleaner = TextProcessor()
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = txt_cleaner.phoneme_cleaners, phonemizer = ESpeak(language="en-us"))
    print(tokenizer.text_to_ids("Hello, Mr. Example, this is 9:30 am and  my number is 30.", language="en"))
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = txt_cleaner.phoneme_cleaners, phonemizer = ESpeak(language="fr-fr"))
    print(tokenizer.text_to_ids("Hola, Sr. Ejemplo, son las 9:30 am y mi número es el 30.", language="es"))
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = txt_cleaner.phoneme_cleaners, phonemizer = ESpeak(language="hi"))
    print(tokenizer.text_to_ids("हैलो, मिस्टर उदाहरण, यह सुबह 9:30 बजे है और मेरा नंबर 30 है।", language="hi"))
