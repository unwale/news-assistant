import os
import tempfile

import gigaam
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

model = gigaam.load_model(model_name="rnnt", device="cpu", download_root="/models")

app = FastAPI()

PORT = os.getenv("PORT", 8003)
MODEL_TYPE = os.getenv("MODEL_TYPE", "rnnt")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp:
        temp_path = temp.name
        temp.write(await file.read())
        try:
            transcription = model.transcribe(temp_path)
            return JSONResponse(content={"transcription": transcription})
        finally:
            os.unlink(temp_path)


@app.get("/healthcheck")
async def healthcheck():
    return JSONResponse(status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
