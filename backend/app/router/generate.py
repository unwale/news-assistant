from database.connection import initialize_weaviate
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from generation.gigachat import generate
from generation.search import search_weaviate
from model.request import TextRequest

client = initialize_weaviate()
router = APIRouter()


@router.post("/retrieve")
async def retrieve(request: TextRequest):
    context_chunks = search_weaviate(client, request.text)
    return JSONResponse(content={"context": context_chunks})


@router.post("/generate_response")
async def generate_response(request: TextRequest):
    context_chunks = search_weaviate(client, request.text)
    print(context_chunks)
    context = ""
    if context_chunks:
        context = "\n".join(
            [
                f"{chunk["content"]}\nURL для встраивания в ответ: {chunk["news_url"]}\nДата публикации: {chunk["date"]}"
                for chunk in context_chunks
            ]
        )

    prompt = f"Контекст: \n{context}\n\n Запрос: {request.text}"

    response = generate(prompt)
    return JSONResponse(content={"message": response})


@router.post("/generate_with_context")
async def generate_response(request: TextRequest):
    context_chunks = search_weaviate(client, request.text)
    print(context_chunks)
    context = ""
    if context_chunks:
        context = "\n".join(
            [
                f"{chunk["content"]}\nURL для встраивания в ответ: {chunk["news_url"]}\nДата публикации: {chunk["date"]}"
                for chunk in context_chunks
            ]
        )

    prompt = f"Контекст: \n{context}\n\n Запрос: {request.text}"

    response = generate(prompt)
    return JSONResponse(content={"message": response, "context": context_chunks})
