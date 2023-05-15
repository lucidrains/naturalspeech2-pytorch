import re
from pathlib import Path
from naturalspeech2_pytorch.utils.expand.abbreviations import AbbreviationExpander
from naturalspeech2_pytorch.utils.expand.number_norm import NumberNormalizer
from naturalspeech2_pytorch.utils.expand.time_norm import TimeExpander

CURRENT_DIR = Path(__file__).resolve().parent

class TextProcessor:
    def __init__(self, lang="en"):
        self.lang = lang
        self._whitespace_re = re.compile(r"\s+")
        # Example usage
        self.ab_expander = AbbreviationExpander(str(CURRENT_DIR / 'expand/abbreviations.csv'))
        self.time_expander = TimeExpander()
        self.num_normalizer = NumberNormalizer()
        # Add currency conversion rates
        symbol = '$'
        conversion_rates ={0.01: "cent", 0.02: "cents", 1: "dollar", 2: "dollars" }
        self.num_normalizer.add_currency(symbol, conversion_rates)
    def lowercase(self, text):
        return text.lower()

    def collapse_whitespace(self, text):
        return re.sub(self._whitespace_re, " ", text).strip()

    def remove_aux_symbols(self, text):
        text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
        return text

    def phoneme_cleaners(self, text, language = 'en'):
        text = self.time_expander.expand_time(text, language=language)
        text = self.num_normalizer.normalize_numbers(text, language=language)
        text = self.ab_expander.replace_text_abbreviations(text, language=language)
        text = self.remove_aux_symbols(text)
        text = self.collapse_whitespace(text)
        return text

if __name__ == "__main__":
    # Create an instance for English
    text_processor_en = TextProcessor(lang="en")

    # Process English text
    english_text = "Hello, Mr. Example, this is 9:30 am and  my number is 30."
    processed_english_text = text_processor_en.phoneme_cleaners(english_text, language='en')
    print(processed_english_text)

    # Create an instance for Spanish
    text_processor_es = TextProcessor(lang="es")

    # Process Spanish text
    spanish_text = "Hola, Sr. Ejemplo, son las 9:30 am y mi número es el 30."
    processed_spanish_text = text_processor_es.phoneme_cleaners(spanish_text, language='es')
    print(processed_spanish_text)
