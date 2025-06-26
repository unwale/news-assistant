import json
import logging
import os
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import weaviate
from database.connection import initialize_weaviate
from processing.chunking import chunk_text, process_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def populate_weaviate(client: weaviate.WeaviateClient, jsonl_file: str):
    """Read JSONL file and populate News and Chunk collections."""
    if not os.path.exists(jsonl_file):
        logger.info(f"File {jsonl_file} does not exist. Skipping population.")
        return

    news_collection = client.collections.get("News")
    chunk_collection = client.collections.get("Chunk")

    references_to_add = []

    with client.batch.dynamic() as batch:
        with open(jsonl_file, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                try:
                    data = json.loads(line.strip())

                    date = data.get("date") / 1000
                    date = (
                        datetime.fromtimestamp(
                            date, tz=ZoneInfo("Europe/Moscow")
                        ).isoformat()
                        if date
                        else None
                    )
                    logger.info(f"Parsed date: {date} (line {line_number})")
                    text = data.get("text")
                    source_id = data.get("source_id")
                    news_url = data.get("news_url")

                    if not all([date, source_id, news_url]):
                        logger.warning(
                            f"Skipping line {line_number}: Missing required fields (date, source_id, news_url)"
                        )
                        continue

                    news_uuid = str(uuid.uuid4())
                    news_data = {
                        "date": date,
                        "url": news_url,
                        "source_id": source_id,
                    }
                    batch.add_object("News", news_data, uuid=news_uuid)
                    logger.info(
                        f"Created News object with UUID: {news_uuid} (line {line_number})"
                    )

                    chunks = chunk_text(text) if text else []
                    if not chunks:
                        logger.info(
                            f"No chunks generated for News UUID: {news_uuid} (line {line_number})"
                        )
                        continue

                    chunk_dicts = process_chunks(chunks, source_id=source_id, date=date)
                    for chunk_dict in chunk_dicts:
                        chunk_uuid = str(uuid.uuid4())
                        chunk_data = {
                            "content": chunk_dict["content"],
                            "lemmatized_content": chunk_dict["lemmatized_content"],
                            "lemmatized_keywords": chunk_dict["lemmatized_keywords"],
                            "temporal_points": chunk_dict["points"],
                            "temporal_intervals": chunk_dict["intervals"],
                        }
                        batch.add_object(
                            "Chunk",
                            chunk_data,
                            uuid=chunk_uuid,
                        )

                        references_to_add.append(
                            {
                                "from_uuid": chunk_uuid,
                                "to_uuid": news_uuid,
                            }
                        )
                        logger.info(
                            f"Created Chunk object with UUID: {chunk_uuid} for News UUID: {news_uuid}"
                        )

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at line {line_number}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_number}: {e}")
                    continue

        batch.flush()
        for reference in references_to_add:
            try:
                batch.add_reference(
                    from_uuid=reference["from_uuid"],
                    to=reference["to_uuid"],
                    from_property="news",
                    from_collection="Chunk",
                )
                logger.info(
                    f"Added reference from Chunk UUID: {reference['from_uuid']} to News UUID: {reference['to_uuid']}"
                )
            except Exception as e:
                logger.error(f"Error adding reference: {e}")
                continue
        batch.flush()
        logger.info("Batch processing completed")


def main():
    client = initialize_weaviate()
    jsonl_file = "/data/news.jsonl"

    try:
        # populate_weaviate(client, jsonl_file)
        logger.info("Database population completed")
    except Exception as e:
        logger.error(f"Database population failed: {e}")
        raise
    finally:
        client.close()
        logger.info("Weaviate connection closed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise
