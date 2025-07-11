FROM us-docker.pkg.dev/colab-images/public/runtime:release-colab_20240626-060133_RC01

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Verify existing packages (optional, for debugging)
RUN python -m pip list

# Set working directory
WORKDIR /app

# Configure uv to use system Python and packages
ENV UV_SYSTEM_PYTHON=1
ENV UV_PYTHON=/usr/local/bin/python

# Copy dependency files first (better layer caching)
COPY pyproject.toml .
COPY uv.lock* .
COPY README.md .

# Install dependencies while respecting system packages
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY src .

# If you need to install your package in development mode
RUN uv pip install --system -e .

# Set Python to unbuffered mode
ENV PYTHONUNBUFFERED=1

# Reset the entrypoint to nothing
ENTRYPOINT []

# Entrypoint
CMD ["mouse-tracking-runtime"]
