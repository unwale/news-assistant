import logging
import os
import time

import weaviate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")


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
