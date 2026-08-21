FROM aberger4/mouse-tracking-base:python3.10-slim

# Install uv. Pinned rather than :latest so the resolver that reads uv.lock is
# the same one that wrote it.
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=/usr/local/bin/python \
    PYTHONUNBUFFERED=1

# Copy metadata first for layer caching
COPY pyproject.toml uv.lock* README.md support_code ./

# Install runtime dependencies from the lock file into the system interpreter.
# `uv pip install` does not read uv.lock, so export the locked resolution first;
# otherwise the dependencies that actually execute at runtime are an unlocked
# re-resolution against PyPI.
RUN uv export --frozen --no-group dev --no-group test --no-group lint \
      --no-emit-project -o /tmp/requirements.txt \
 && uv pip install --system -r /tmp/requirements.txt

# Now add source and install the project itself. `--no-deps` keeps the locked
# dependency set above from being re-resolved.
COPY src ./src

RUN uv pip install --system --no-deps .

COPY support_code ./support_code

# Fail the build (rather than a production batch) on import-time regressions.
RUN mouse-tracking --help

CMD ["mouse-tracking-runtime", "--help"]
