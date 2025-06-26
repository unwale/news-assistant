import re
import string

import nltk
from pymorphy3 import MorphAnalyzer
from razdel import tokenize

stopwords = set(nltk.corpus.stopwords.words("russian"))
analyzer = MorphAnalyzer()


def clean_text(text: str) -> str:
    text = text.lower()

    text = re.sub(r"\[.*?\|([^]]+)\]", r"\1", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text


def lemmatize_text(text: str) -> str:
    """
    Lemmatizes the text using spacy.
    """
    tokens = [token.text for token in tokenize(text)]
    return " ".join([analyzer.parse(token)[0].normal_form for token in tokens])
