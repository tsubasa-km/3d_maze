FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv \
    && uv sync --no-dev

COPY 3d_maze.py ./

CMD ["uv", "run", "python", "3d_maze.py"]
