import csv
import re

class AbbreviationExpander:
    def __init__(self, abbreviations_file):
        self.abbreviations = {}
        self.patterns = {}
        self.load_abbreviations(abbreviations_file)

    def load_abbreviations(self, abbreviations_file):
        with open(abbreviations_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                abbreviation = row['abbreviation']
                expansion = row['expansion']
                language = row['language'].lower()
                self.abbreviations.setdefault(language, {})[abbreviation] = expansion

                if language not in self.patterns:
                    self.patterns[language] = re.compile(
                        r"\b(" + "|".join(re.escape(key) for key in self.abbreviations[language].keys()) + r")\b",
                        re.IGNORECASE
                    )

    def replace_abbreviations(self, match, language):
        return self.abbreviations[language][match.group(0).lower()]

    def replace_text_abbreviations(self, text, language):
        if language.lower() in self.patterns:
            return self.patterns[language.lower()].sub(
                lambda match: self.replace_abbreviations(match, language.lower()),
                text
            )
        else:
            return text
if __name__ == "__main__":
    # Example usage
    expander = AbbreviationExpander('abbreviations.csv')

    text_en = "Hello, Mr. Example. How are you today? I work at Intl. Corp."
    replaced_text_en = expander.replace_text_abbreviations(text_en, 'en')
    print(replaced_text_en)

    text_fr = "Bonjour, Sr. Example. Comment ça va aujourd'hui? Je travaille chez Intl. Corp."
    replaced_text_fr = expander.replace_text_abbreviations(text_fr, 'fr')
    print(replaced_text_fr)
