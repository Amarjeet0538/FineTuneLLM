# The log-triage API (FastAPI). Ollama runs in its own container (see docker-compose.yml).
FROM python:3.13-slim

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# only the five files the API needs: prompt + settings + parser + the API itself + its page
COPY src/config.py src/run_ollama.py src/score.py src/serve.py src/ui.html src/

# don't run as the all-powerful root user inside the container
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
