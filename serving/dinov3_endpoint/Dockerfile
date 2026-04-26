FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV MODEL_DIR=/model

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_server.py .

EXPOSE 8080
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8080"]
