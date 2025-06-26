FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY init_weaviate.py .

CMD ["python", "init_weaviate.py"]