# syntax=docker/dockerfile:1
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ENABLE_CUDA=false \
    ENVIRONMENT=production \
    HOME=/home/ocr \
    TORCH_HOME=/home/ocr/.cache/torch \
    MPLCONFIGDIR=/home/ocr/.cache/matplotlib

WORKDIR /app

# Shared libraries required by the OpenCV wheel bundled with ocr-tamil.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install CPU wheels first to avoid pulling CUDA libraries into this image.
RUN python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt \
    && python -m pip check

RUN useradd --create-home --uid 10001 ocr \
    && mkdir -p /home/ocr/.model_weights /home/ocr/.cache /app/temp_images \
    && chown -R ocr:ocr /home/ocr /app/temp_images

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY run.py ./

USER ocr

# Use the same checkpoints as the README setup; the package's legacy URLs
# are avoided by supplying these files before its first initialization.
RUN python - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import shutil

weights = Path.home() / ".model_weights"
base = "https://huggingface.co/GnanaPrasath/ocr_tamil/resolve/main/"
for remote, local in (
    ("parseq_tamil.pt", "parseq_tamil_v6.ckpt"),
    ("craft_mlt_25k.pth", "craft_mlt_25k.pth"),
):
    with urlopen(base + remote, timeout=120) as response:
        with (weights / local).open("wb") as output:
            shutil.copyfileobj(response, output)
PY

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=8).close()"

CMD ["python", "run.py"]
