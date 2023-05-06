""" from https://github.com/coqui-ai/TTS/"""
from typing import Callable, List

from cleaner import phoneme_cleaners
from utils.espeak_wrapper import ESpeak
class Tokenizer:
    def __init__(
        self,
        vocab,
        text_cleaner: Callable = None,
        phonemizer: Callable = None,
        add_blank: bool = False,
        use_eos_bos=False,
    ):
        self.text_cleaner = text_cleaner
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.vocab = vocab
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.phonemizer = phonemizer
        self.not_found_characters = []
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

    def text_to_ids(self, text: str, language: str = None) -> List[int]:  # pylint: disable=unused-argument
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
        # TODO: text cleaner should pick the right routine based on the language
        if self.text_cleaner is not None:
            text = self.text_cleaner(text)
        text = self.phonemizer.phonemize(text, separator="", language=language)
        if self.add_blank:
            text = self.intersperse_blank_char(text, True)
        if self.use_eos_bos:
            text = self.pad_with_bos_eos(text)
        return self.encode(text)

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
    #DEFAULT SET OF IPA PHONEMES
    # Phonemes definition (All IPA characters)
    _vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
    _non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
    _pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
    _suprasegmentals = "'̃ˈˌːˑ. "
    _other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
    _diacrilics = "ɚ˞ɫ"
    _phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = phoneme_cleaners, phonemizer = ESpeak(language="en-us"))
    print(tokenizer.text_to_ids("Hello this is a test."))
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = phoneme_cleaners, phonemizer = ESpeak(language="fr-fr"))
    print(tokenizer.text_to_ids("Bonjour c'est un essai."))
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = phoneme_cleaners, phonemizer = ESpeak(language="hi"))
    print(tokenizer.text_to_ids("नमस्ते यह एक परीक्षा है।"))