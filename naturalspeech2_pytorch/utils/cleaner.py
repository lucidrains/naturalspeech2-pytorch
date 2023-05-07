import re
from utils.english.abbreviations import abbreviations_en
from utils.english.number_norm import normalize_numbers as en_normalize_numbers
from utils.english.time_norm import expand_time_english

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

def expand_abbreviations(text, lang="en"):
    if lang == "en":
        _abbreviations = abbreviations_en
    else:
        print("add new language abbrevations language not supported")
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text

def phoneme_cleaners(text):
    """Pipeline for phonemes mode, including number and abbreviation expansion."""
    text = expand_time_english(text)
    text = en_normalize_numbers(text)
    text = expand_abbreviations(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text