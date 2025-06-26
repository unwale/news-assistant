FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg git && apt-get clean

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE ${PORT}

CMD uvicorn main:app --host 0.0.0.0 --port $PORT