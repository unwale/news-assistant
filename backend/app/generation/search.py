from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import weaviate
from processing.lemmatization import lemmatize_text
from processing.time import parse_with_duckling
from weaviate.classes.query import Filter, MetadataQuery, QueryReference


def point_to_interval(point: datetime, grain: str) -> Tuple[datetime, datetime]:

    if grain == "day":
        end = point + timedelta(days=1)
    elif grain == "week":
        end = point + timedelta(weeks=1)
    elif grain == "month":
        end = point + timedelta(days=31)
    elif grain == "year":
        end = point + timedelta(days=365)
    else:
        end = point + timedelta(days=1)

    return (point, end)


def _to_datetime(date: str):
    return datetime.fromisoformat(date).replace(tzinfo=ZoneInfo("Europe/Moscow"))


def _calculate_temporal_score(
    query_intervals: List[Tuple[datetime]],
    query_points: List[datetime],
    query_date: datetime,
    chunk_intervals: List[Tuple[datetime]],
    chunk_points: List[datetime],
    chunk_date: datetime,
    gamma=0.2,
):
    temporal_score = 0.0

    reference_times = [query_date]
    if query_intervals:
        for start, end in query_intervals:
            midpoint = start + (end - start) / 2
            reference_times.append(midpoint)
    if query_points:
        reference_times.extend(query_points)

    durations = (
        [(end - start).total_seconds() for start, end in query_intervals]
        if query_intervals
        else []
    )
    max_duration = max(durations + [367 * 86400], default=367 * 86400)

    point_score = 0.0
    all_points = chunk_points + [chunk_date]
    max_point_score = 0.0
    for point in all_points:
        min_distance = min(
            abs((point - ref).total_seconds()) for ref in reference_times
        )
        point_score = max(0.0, 1.0 - min_distance / (max_duration * 2))
        max_point_score = max(max_point_score, point_score)
    temporal_score += max_point_score * 0.5

    interval_score = 0.0
    if chunk_intervals and query_intervals:
        max_overlap_score = 0.0
        for c_start, c_end in chunk_intervals:
            for q_start, q_end in query_intervals:
                intersect_start = max(c_start, q_start)
                intersect_end = min(c_end, q_end)
                intersection = max(0, (intersect_end - intersect_start).total_seconds())
                union = (max(c_end, q_end) - min(c_start, q_start)).total_seconds()
                overlap_score = intersection / union if union > 0 else 0.0
                max_overlap_score = max(max_overlap_score, overlap_score)
        interval_score = max_overlap_score
        temporal_score += interval_score * 0.5
    print(temporal_score)

    return temporal_score


def _search_dense(
    collection: weaviate.collections.Collection, query: str, filters: Filter
):
    bm25_response = collection.query.bm25(
        query=query,
        query_properties=["lemmatized_content", "lemmatized_keywords^2"],
        return_metadata=MetadataQuery(score=True),
        return_references=QueryReference(link_on="news"),
        filters=filters,
        limit=20,
    )

    bm25_results = []
    for obj in bm25_response.objects:
        bm25_results.append(
            {
                "uuid": obj.uuid,
                "content": obj.properties["content"],
                "score": obj.metadata.score,
                "news_url": obj.references["news"].objects[0].properties["url"],
                "date": obj.references["news"].objects[0].properties["date"],
            }
        )

    return bm25_results


def _search_sparse(
    collection: weaviate.collections.Collection, query: str, filters: Filter
):
    vector_response = collection.query.near_text(
        query=query,
        return_metadata=MetadataQuery(score=True, distance=True),
        return_references=QueryReference(link_on="news"),
        filters=filters,
        limit=20,
    )

    vector_results = []
    for obj in vector_response.objects:
        vector_results.append(
            {
                "uuid": obj.uuid,
                "content": obj.properties["content"],
                "distance": obj.metadata.distance,
                "news_url": obj.references["news"].objects[0].properties["url"],
                "date": obj.references["news"].objects[0].properties["date"],
            }
        )

    return vector_results


def _fuse_scores(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    query_intervals: List[Tuple[datetime]],
    query_points: List[datetime],
    query_date: datetime,
    k: int = 20,
    temporal_boost: float = 1.0,
) -> List[Dict[str, Any]]:
    scores = defaultdict(float)
    result_map = {}

    for result in bm25_results + vector_results:
        result_map[result["uuid"]] = {
            "content": result["content"],
            "news_url": result["news_url"],
            "date": result["date"],
        }

    for rank, result in enumerate(bm25_results):
        print(f"score: {bm25_results[rank]["score"]}")
        scores[result["uuid"]] += (
            (1 - vector_results[-1]["distance"])
            * bm25_results[rank]["score"]
            / bm25_results[0]["score"]
        )

    for rank, result in enumerate(vector_results):
        print(f"distance: {vector_results[rank]["distance"]}")
        scores[result["uuid"]] += 1 - vector_results[rank]["distance"]

    boosted_scores = {}
    for uuid, score in scores.items():
        result = result_map[uuid]
        # try:
        #     chunk_date = datetime.fromisoformat(result["date"]).replace(
        #         tzinfo=ZoneInfo("Europe/Moscow")
        #     )
        # except ValueError:
        #     chunk_date = datetime.now(tz=ZoneInfo("Europe/Moscow"))

        chunk_date = result["date"]
        chunk_intervals: List[Tuple[datetime]] = []
        chunk_points: List[datetime] = []

        temporal_score = _calculate_temporal_score(
            query_intervals=query_intervals,
            query_points=query_points,
            query_date=query_date,
            chunk_intervals=chunk_intervals,
            chunk_points=chunk_points,
            chunk_date=chunk_date,
            gamma=0.2,
        )

        boosted_score = score * (1.0 + temporal_boost * temporal_score)
        boosted_scores[uuid] = boosted_score

    combined_results = [
        {
            "uuid": uuid.hex,
            "content": result_map[uuid]["content"],
            "score": score,
            "news_url": result_map[uuid]["news_url"],
            "date": result_map[uuid]["date"],
        }
        for uuid, score in sorted(
            boosted_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]

    return combined_results[:10]


def search_weaviate(
    client: weaviate.Client, query: str, limit=10, alpha=0.5
) -> List[Dict[str, Any]]:
    now = datetime.now(tz=ZoneInfo("Europe/Moscow"))

    lemmatized_query = lemmatize_text(query)
    temporal_points, temporal_intervals = parse_with_duckling(query)

    temporal_filters = []
    intervals = temporal_intervals + [
        point_to_interval(
            datetime.fromisoformat(temporal_point["point"]), temporal_point["grain"]
        )
        for temporal_point in temporal_points
    ]
    if not intervals:
        intervals.append((now - timedelta(days=365), now))
    for interval in intervals:
        temporal_filters.append(
            Filter.all_of(
                [
                    Filter.by_ref("news")
                    .by_property("date")
                    .greater_or_equal(interval[0]),
                    Filter.by_ref("news")
                    .by_property("date")
                    .less_or_equal(interval[1]),
                ]
            )
        )
    temporal_filters = Filter.any_of(temporal_filters)

    temporal_points = [_to_datetime(point["point"]) for point in temporal_points]
    temporal_intervals = [
        (_to_datetime(interval["start"]), _to_datetime(interval["end"]))
        for interval in temporal_intervals
    ]

    chunk_collection = client.collections.get("Chunk")

    dense_results = _search_dense(
        collection=chunk_collection, query=lemmatized_query, filters=temporal_filters
    )
    sparse_results = _search_sparse(
        collection=chunk_collection, query=query, filters=temporal_filters
    )
    fused_results = _fuse_scores(
        dense_results,
        sparse_results,
        temporal_intervals,
        temporal_points,
        now,
        20,
        temporal_boost=1.0,
    )

    for i in range(len(fused_results)):
        fused_results[i]["date"] = fused_results[i]["date"].strftime("%Y.%m.%d")

    return fused_results
