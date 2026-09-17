FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest

COPY src/ ./src/
COPY tests/ ./tests/
COPY run.py .

RUN mkdir -p /app/output

ENTRYPOINT ["python", "-m", "src"]
CMD ["all"]
