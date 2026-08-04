# First-party BuildML managed-serve image.
# Honesty: operator-built image for local/cluster recipes — not a managed cloud.
#
# Build from the repository root:
#   docker build -f deploy/serve/Dockerfile -t buildml-serve:local .
#
# Run (API key required for 0.0.0.0; insecure public bind override is off):
#   docker run --rm -p 8080:8080 \
#     -e BUILDML_API_KEY=rotate-me \
#     -e BUILDML_BUNDLE=/models/bundle \
#     -v /path/to/bundle:/models/bundle:ro \
#     buildml-serve:local
#
# Non-root, HEALTHCHECK on /health, trusted deserialize opt-in via BUILDML_SERVE_TRUSTED.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BUILDML_SERVE_HOST=0.0.0.0 \
    BUILDML_SERVE_PORT=8080 \
    BUILDML_BUNDLE=/models/bundle \
    BUILDML_SERVE_KIND=pipeline \
    BUILDML_SERVE_TRUSTED=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 buildml \
    && useradd --system --uid 10001 --gid buildml --home /home/buildml --create-home buildml \
    && mkdir -p /models /app \
    && chown -R buildml:buildml /models /app /home/buildml

WORKDIR /app

# Install the package from the build context (repo root).
COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY buildml ./buildml
RUN pip install --upgrade pip \
    && pip install --no-cache-dir ".[serve]" \
    && rm -rf /root/.cache

USER buildml

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${BUILDML_SERVE_PORT}/health" || exit 1

# Entrypoint requires BUILDML_API_KEY (or Basic auth env) for non-loopback bind.
# Intentionally does not enable the insecure public bind override.
ENTRYPOINT ["buildml-serve"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--trusted"]
