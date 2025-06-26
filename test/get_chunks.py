import weaviate


def get_all_chunks(client: weaviate.Client) -> list:
    chunk_collection: weaviate.collections.Collection = client.collections.get("Chunk")
    reference = weaviate.classes.query.QueryReference(link_on="news")
    chunks = []

    for chunk in chunk_collection.iterator(return_references=reference):
        chunk_data = {
            "text": chunk.properties.get("content", ""),
            "date": chunk.references.get("news", {})
            .objects[0]
            .properties.get("date", "")
            .strftime("%Y-%m-%d %H:%M:%S"),
            "news_url": chunk.references.get("news", {})
            .objects[0]
            .properties.get("url", ""),
            "chunk_id": chunk.uuid.hex,
            "keywords": chunk.properties.get("lemmatized_keywords", []),
        }
        chunks.append(chunk_data)

    return chunks


connection_params = weaviate.connect.ConnectionParams.from_url(
    url="http://localhost:8080", grpc_port=50051
)
client = weaviate.WeaviateClient(connection_params=connection_params)
client.connect()
try:
    chunks = get_all_chunks(client)
    import json

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"Retrieved {len(chunks)} chunks and saved to 'chunks.json'.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.close()
