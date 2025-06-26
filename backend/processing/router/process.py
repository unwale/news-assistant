from fastapi import APIRouter
from fastapi.responses import JSONResponse
from model.request import ProcessNewsRequest, ProcessQueryRequest
from processing.chunking import chunk_text, process_chunks
from processing.lemmatization import lemmatize_text
from processing.time import parse_with_duckling

router = APIRouter()


@router.post("/news")
def process_news(request: ProcessNewsRequest):
    chunks = chunk_text(request.content)
    return JSONResponse(content=process_chunks(chunks))


@router.post("/query")
def process_query(request: ProcessQueryRequest):
    lemmatized_query = lemmatize_text(request.text)
    temporal_points, temporal_intervals = parse_with_duckling(request.text)
    return JSONResponse(
        content={
            "lemmatized_query": lemmatized_query,
            "temporal_points": temporal_points,
            "temporal_intervals": temporal_intervals,
        }
    )
