FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ ./rag/
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

EXPOSE 8000

# Migrations are applied as a separate deploy step (alembic upgrade head),
# not on every container start — see docs/RAG.md / api_documentation.txt.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
