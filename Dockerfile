# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcfitsio-dev libgfortran5 build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir numpy healpy pyyaml typer rich pydantic matplotlib pytest

COPY . .

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]
