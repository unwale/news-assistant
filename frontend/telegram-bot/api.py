import os

import requests

# from tts import SberSpeechAPI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

STT_URL = os.getenv("STT_URL", "http://localhost:8003")


async def generate_response(prompt: str) -> str:
    message = requests.post(
        f"{API_BASE_URL}/api/generate_response",
        json={"text": prompt},
    ).json()["message"]
    return message


async def transcribe_audio(audio_file: bytearray) -> str:
    response = requests.post(
        f"{STT_URL}/transcribe",
        files={"file": audio_file},
    )
    return response.json()["transcription"]


# async def synthesize_speech(text: str) -> bytearray:
