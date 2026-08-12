FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ ./rag/
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .
# Needed at runtime, not just migration time: alembic/versions/0004_seed_
# companions_tier1_characters.py reads data/raw/ during migration, and
# app/services/character_classification_sync.py reads data/generated/
# character_classifications.json on every backend startup.
COPY data/ ./data/

EXPOSE 8000

# Migrations are applied as a separate deploy step (alembic upgrade head),
# not on every container start — see docs/RAG.md / api_documentation.txt.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
