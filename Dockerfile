# ---- Stage 1: train the model ------------------------------------------------
# The model is trained while the image builds, so the image is self-contained and
# reproducible: same code + same data (verified by SHA-256) = same model.
FROM python:3.12-slim AS trainer
WORKDIR /app
ENV CHURN_ROOT=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && python -m churn.train

# ---- Stage 2: slim API runtime ----------------------------------------------
FROM python:3.12-slim
WORKDIR /app
ENV CHURN_ROOT=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY api ./api
COPY --from=trainer /app/models/churn_model.joblib ./models/churn_model.joblib
ENV MODEL_PATH=/app/models/churn_model.joblib
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
