FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir \
    lightgbm pandas numpy scikit-learn fastapi uvicorn pydantic \
    scikit-uplift scipy pyarrow

COPY src/ src/
COPY configs/ configs/
COPY main.py .

# Generate data + train on build (or mount volume with pre-trained artifacts)
RUN python main.py --n-clients 5000 --n-purchases 100000 --output-dir /app/artifacts

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
