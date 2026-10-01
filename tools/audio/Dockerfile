FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY tools/audio/requirements.txt /tmp/audio-requirements.txt
RUN pip install --no-cache-dir -r /tmp/audio-requirements.txt

COPY tools/audio /app/tools/audio
RUN mkdir -p /app/tools/audio/results/events

EXPOSE 8000

ENTRYPOINT ["python", "tools/audio/backend/app.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
