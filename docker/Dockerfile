# PTF tahmin servisi (FastAPI, replay modu) için imaj.
# Derleme: repo KÖKÜNDEN çalıştırılır:  docker build -f docker/Dockerfile -t ptf-servis .
# Model paketi ve replay tablosu imaja GİRMEZ (büyük/özel veri); çalışırken volume (bağlanan klasör) ile verilir:
#   docker run -p 8000:8000 -v <paket klasörü>:/models:ro -v <tablo klasörü>:/data:ro ptf-servis
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/models \
    DATA_PATH=/data/dataset_weather_guncel.parquet

# LightGBM OpenMP kütüphanesini ister (libgomp1). CI deneyi (7 Kasım 2026): libgomp1 OLMADAN lightgbm içe aktarılamadı (python:3.13-slim, lightgbm 4.7.0), yani GEREKLİ.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Root olmayan kullanıcı
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# Önce yalnız requirements.txt: bağımlılık katmanı, kod değişince yeniden kurulmasın (katman önbelleği)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Kod. data/external/ (tavan ve tatil tabloları, küçük CSV) kod tarafından OKUNUR (price_caps.py, holidays_tr.py), bu yüzden imajda olmalı.
COPY --chown=app:app src/ ./src/
COPY --chown=app:app data/external/ ./data/external/

USER app
EXPOSE 8000

# curl kurmadan, Python ile sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["python", "-m", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
