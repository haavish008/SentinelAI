FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY backend-requirements.txt .

RUN pip install --no-cache-dir -r backend-requirements.txt

COPY backend ./backend
COPY models ./models
# The database is created/persisted through the Docker volume.

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
