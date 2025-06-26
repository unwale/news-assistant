from typing import List

from processing.keywords import extract_keywords
from processing.lemmatization import clean_text, lemmatize_text
from processing.time import parse_with_duckling
from razdel import sentenize

MAX_WORDS_PER_CHUNK = 80


def chunk_text(text: str):
    sentences = [s.text for s in sentenize(text)]
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if current_word_count + word_count <= MAX_WORDS_PER_CHUNK:
            current_chunk.append(sentence)
            current_word_count += word_count
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_chunks(chunks: List[str], source_id: str, date: int):
    chunk_objects = []
    for chunk in chunks:
        chunk1 = clean_text(chunk)
        lemmatized_content = lemmatize_text(chunk1)
        lemmatized_keywords = [lemmatize_text(kw) for kw in extract_keywords(chunk1)]
        points, intervals = parse_with_duckling(chunk)

        chunk_object = {
            "content": chunk,
            "lemmatized_content": lemmatized_content,
            "lemmatized_keywords": lemmatized_keywords,
            "source_id": source_id,
            "date": date,
            "points": points,
            "intervals": intervals,
        }
        chunk_objects.append(chunk_object)
    return chunk_objects
