from fastapi import FastAPI
from router.process import router as process_router

app = FastAPI()

app.include_router(process_router, prefix="/process")
