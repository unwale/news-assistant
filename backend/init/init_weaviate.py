import logging
import os
import time

import weaviate
import weaviate.classes as wvc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://172.19.0.3:8080")


def initialize_weaviate():
    try:
        connection_params = weaviate.connect.ConnectionParams.from_url(
            url=WEAVIATE_URL, grpc_port=50051
        )
        client = weaviate.WeaviateClient(connection_params=connection_params)

        max_attempts = 70
        attempt = 1
        while attempt <= max_attempts:
            try:
                client.connect()
            except:
                pass
            if client.is_ready():
                logger.info("Successfully connected to Weaviate")
                return client
            logger.warning(f"Weaviate not ready, attempt {attempt}/{max_attempts}")
            time.sleep(5)
            attempt += 1

        logger.error("Failed to connect to Weaviate after max attempts")
        raise Exception("Weaviate connection failed")
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        raise


def create_schema(client: weaviate.WeaviateClient):
    try:
        existing_collections = [coll for coll in client.collections.list_all()]

        if "News" not in existing_collections:
            news_class = client.collections.create(
                name="News",
                description="A news article",
                properties=[
                    wvc.config.Property(
                        name="date",
                        data_type=wvc.config.DataType.DATE,
                        description="Date of the news article",
                    ),
                    wvc.config.Property(
                        name="url",
                        data_type=wvc.config.DataType.TEXT,
                        description="URL of the news article",
                    ),
                    wvc.config.Property(
                        name="source_id",
                        data_type=wvc.config.DataType.TEXT,
                        description="Identifier of the news source",
                    ),
                ],
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            )
            logger.info("Schema for 'News' class created successfully")
        else:
            logger.info("'News' class already exists")

        if "Chunk" not in existing_collections:
            chunk_class = client.collections.create(
                name="Chunk",
                description="A chunk of text with keywords",
                properties=[
                    wvc.config.Property(
                        name="content",
                        data_type=wvc.config.DataType.TEXT,
                        description="Original text content",
                    ),
                    wvc.config.Property(
                        name="lemmatized_content",
                        data_type=wvc.config.DataType.TEXT,
                        description="Lemmatized text content",
                    ),
                    wvc.config.Property(
                        name="lemmatized_keywords",
                        data_type=wvc.config.DataType.TEXT_ARRAY,
                        description="Lemmatized keywords",
                    ),
                    wvc.config.Property(
                        name="temporal_points",
                        data_type=wvc.config.DataType.OBJECT_ARRAY,
                        description="Array of temporal points with their granularity",
                        nested_properties=[
                            wvc.config.Property(
                                name="point",
                                data_type=wvc.config.DataType.DATE,
                                description="Specific temporal point (e.g., '2025-05-06T15:00:00Z')",
                            ),
                            wvc.config.Property(
                                name="grain",
                                data_type=wvc.config.DataType.TEXT,
                                description="Granularity of the temporal point (e.g., 'hour', 'day')",
                            ),
                        ],
                    ),
                    wvc.config.Property(
                        name="temporal_intervals",
                        data_type=wvc.config.DataType.OBJECT_ARRAY,
                        description="Array of temporal intervals with their granularity",
                        nested_properties=[
                            wvc.config.Property(
                                name="start",
                                data_type=wvc.config.DataType.DATE,
                                description="Start of the interval (e.g., '2025-05-06T00:00:00Z')",
                            ),
                            wvc.config.Property(
                                name="end",
                                data_type=wvc.config.DataType.DATE,
                                description="End of the interval (e.g., '2025-05-07T00:00:00Z')",
                            ),
                            wvc.config.Property(
                                name="grain",
                                data_type=wvc.config.DataType.TEXT,
                                description="Granularity of the interval (e.g., 'day', 'week')",
                            ),
                        ],
                    ),
                ],
                references=[
                    wvc.config.ReferenceProperty(name="news", target_collection="News")
                ],
                vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(
                    vectorize_collection_name=False, pooling_strategy="cls"
                ),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
            )
            logger.info("Schema for 'Chunk' class created successfully")
        else:
            logger.info("'Chunk' class already exists")
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        raise


def main():
    try:
        client = initialize_weaviate()
        create_schema(client)
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    finally:
        if client:
            client.close()
            logger.info("Weaviate client connection closed")


if __name__ == "__main__":
    main()
