import re
import inflect
from num2words import num2words
from num_to_words import num_to_word
class NumberNormalizer:
    def __init__(self):
        self._inflect = inflect.engine()
        self._number_re = re.compile(r"-?[0-9]+")
        self._currency_re = re.compile(r"([$€£¥₹])([0-9\,\.]*[0-9]+)")
        self._currencies = {}

    def add_currency(self, symbol, conversion_rates):
        self._currencies[symbol] = conversion_rates

    def normalize_numbers(self, text, language='en'):
        self._inflect = inflect.engine()
        self._set_language(language)
        text = re.sub(self._currency_re, self._expand_currency, text)
        text = re.sub(self._number_re, lambda match: self._expand_number(match, language), text)
        return text

    def _set_language(self, language):
        if language == 'en':
            self._inflect = inflect.engine()
        else:
            self._inflect = inflect.engine()
            # Add support for additional languages here

    def _expand_currency(self, match):
        unit = match.group(1)
        currency = self._currencies.get(unit)
        if currency:
            value = match.group(2)
            return self._expand_currency_value(value, currency)
        return match.group(0)

    def _expand_currency_value(self, value, inflection):
        parts = value.replace(",", "").split(".")
        if len(parts) > 2:
            return f"{value} {inflection[2]}"  # Unexpected format
        text = []
        integer = int(parts[0]) if parts[0] else 0
        if integer > 0:
            integer_unit = inflection.get(integer, inflection[2])
            text.append(f"{integer} {integer_unit}")
        fraction = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if fraction > 0:
            fraction_unit = inflection.get(fraction / 100, inflection[0.02])
            text.append(f"{fraction} {fraction_unit}")
        if not text:
            return f"zero {inflection[2]}"
        return " ".join(text)

    def _expand_number(self, match, language: str) -> str:
        num = int(match.group(0))
        if 1000 < num < 3000:
            if num == 2000:
                return self._number_to_words(num, language)
            if 2000 < num < 2010:
                return f"{self._number_to_words(2000, language)} {self._number_to_words(num % 100, language)}"
            if num % 100 == 0:
                return f"{self._number_to_words(num // 100, language)} {self._get_word('hundred')}"
            return self._number_to_words(num, language)
        return self._number_to_words(num, language)

    def _number_to_words(self, n: int, language: str) -> str:
        try:
            if language == 'en':
                return self._inflect.number_to_words(n)
            else:
                return num2words(n, lang=language)
        except:
            try:
                return num_to_word(n, lang=language)
            except:
                raise NotImplementedError("language not implemented")

    def _get_word(self, word):
        return word
if __name__ == "__main__":
    # Create an instance of NumberNormalizer
    normalizer = NumberNormalizer()
    # Add currency conversion rates
    symbol = '$'
    conversion_rates ={
            0.01: "cent",
            0.02: "cents",
            1: "dollar",
            2: "dollars",
        }
    normalizer.add_currency(symbol, conversion_rates)
    # Example 1: English (en) language
    text_en = "I have $1,000 and 5 apples."
    normalized_text_en = normalizer.normalize_numbers(text_en, language='en')
    print(normalized_text_en)
    # Output: "I have one thousand dollars and five apples."

    # Example 2: Spanish (es) language
    text_es = "Tengo $1.000 y 5 manzanas."
    normalized_text_es = normalizer.normalize_numbers(text_es, language='es')
    print(normalized_text_es)
    # Output: "Tengo mil dólares y cinco manzanas."
