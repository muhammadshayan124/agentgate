FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY examples ./examples
COPY data ./data
RUN useradd --create-home --uid 1000 app && mkdir -p /app/results && chown -R app /app
USER app
ENTRYPOINT ["agentgate"]
CMD ["--help"]

# Test image: `docker build --target test -t agentgate-test .`
FROM base AS test
USER root
RUN pip install ".[dev]"
COPY tests ./tests
COPY results ./results
RUN chown -R app /app
USER app
ENTRYPOINT ["python", "-m", "pytest", "-q"]
CMD []
