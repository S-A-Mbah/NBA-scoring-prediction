# 1. Lightweight official Python image. 3.13 (not the brief's 3.10) because the pinned
#    pandas 3.0 needs Python >= 3.11 and numpy 2.5 needs >= 3.12; matches the dev .venv.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. Working folder inside the container
WORKDIR /app

# 3. Requirements first so this layer is cached and libraries are not
#    reinstalled every time the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Run as a non-root user (least privilege if the app is ever compromised)
RUN useradd --create-home appuser

# 5. Copy the project: code, processed data, (later) models. Raw data and notebooks are
#    excluded by .dockerignore
COPY --chown=appuser:appuser . .
USER appuser

# 6. Port Streamlit listens on
EXPOSE 8501

# Lets `docker ps` show healthy/unhealthy; slim has no curl, so use Python
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# 7. 0.0.0.0 makes the app reachable from outside the container
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
