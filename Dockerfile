FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default entrypoint runs inference.
# Override at runtime for server mode, e.g.:
# docker run --rm image-name python environment.py
CMD ["python", "inference.py"]
