FROM aberger4/mouse-tracking-base:python3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=/usr/local/bin/python \
    PYTHONUNBUFFERED=1

# Copy metadata first for layer caching
COPY pyproject.toml uv.lock* README.md ./

# Only install runtime dependencies
RUN uv sync --frozen --no-group dev --no-group test --no-group lint --no-install-project

# Now add source and install the project itself
COPY src ./src

RUN uv pip install --system .

CMD ["mouse-tracking-runtime", "--help"]
