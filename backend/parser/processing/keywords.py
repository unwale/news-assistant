from typing import Any, Dict, List

import nltk
from yake import KeywordExtractor

yake = KeywordExtractor(
    lan="ru", n=2, top=15, stopwords=nltk.corpus.stopwords.words("russian")
)


def extract_keywords(text: str) -> List[str]:
    """
    Extracts keywords from the text using YAKE.
    """
    keywords = yake.extract_keywords(text)
    return [keyword[0] for keyword in keywords if keyword[1] > 0.5]
