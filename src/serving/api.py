"""Küçük web servisi (FastAPI): donmuş model paketini ve replay tablosunu HTTP ile sunar. Hesabı predictor.py yapar; burası sadece onu çağırır.

REPLAY MODU: yalnız C = 2026-09-22 .. tablonun son günü için tahmin verilir. Canlı gelecek tarih YOK
(BİLMİYORUZ: canlı veri hattı henüz yok). C'den önceki gün "model bu günü eğitimde gördü" (400), tablonun son gününden sonrası "tabloda yok" (404).

Ortam değişkenleri (ikisi de ZORUNLU; varsayılan yol yok):
  MODEL_DIR : model paketi klasörü (meta.json ve model dosyaları)
  DATA_PATH : replay tablosu (parquet)
Biri eksikse ya da yol yoksa uygulama AÇILMAZ (RuntimeError, anlaşılır mesaj).

Uç noktalar:
  GET /health               -> servis ve model bilgisi (app.state'ten okunur; yeni hesap yok)
  GET /predict?date=YYYY-MM-DD -> predictor.predict_day çıktısı AYNEN (alan eklenmez/silinmez)

Çalıştırma örneği (Faz 2'de): MODEL_DIR=... DATA_PATH=... python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
Kural: eksik değer doldurulmaz.
"""

import datetime as dt
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.serving import predictor as pr

ENV_HATA = "MODEL_DIR ve DATA_PATH ortam degiskenleri gerekli"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Açılışta paketi ve tabloyu BİR KEZ yükler, önbelleği ısıtır; kapanışta temizler."""
    model_dir, data_path = os.environ.get("MODEL_DIR"), os.environ.get("DATA_PATH")
    if not model_dir or not data_path:
        raise RuntimeError(ENV_HATA)
    if not Path(model_dir).is_dir():
        raise RuntimeError(f"MODEL_DIR klasoru yok: {model_dir}")
    if not Path(data_path).is_file():
        raise RuntimeError(f"DATA_PATH dosyasi yok: {data_path}")
    bundle = pr.load_bundle(model_dir)
    table = pr.load_table(data_path, bundle["c_day"])
    pr.quantile_cache(bundle, table)  # önbelleği ısıt: ilk istek yavaş olmasın
    app.state.bundle, app.state.table = bundle, table
    yield
    pr.clear_cache()


app = FastAPI(title="PTF tahmin servisi (replay)", lifespan=lifespan)


# Not: endpoint'ler normal `def` (async değil): hesap işlemci işi; FastAPI bunları iş parçacığı havuzunda çalıştırır, olay döngüsü kilitlenmez.
@app.get("/health")
def health():
    b, t = app.state.bundle, app.state.table
    return {"durum": "ok", "model_surumu": f"model_paketi_v1@{b['id'][:12]}", "egitim_sonu": b["c_day"].date().isoformat(),
            "tablo_son_gun": t["last_day"].date().isoformat(), "mod": "replay"}


@app.get("/predict")
def predict(date: dt.date):
    """date biçimi YYYY-MM-DD; biçim hatası otomatik 422."""
    try:
        return pr.predict_day(app.state.bundle, app.state.table, date)
    except ValueError as err:
        msg = str(err)
        if msg.startswith("egitimde goruldu"):
            raise HTTPException(status_code=400, detail=msg)
        if msg.startswith("tabloda yok"):
            raise HTTPException(status_code=404, detail=msg)
        raise  # başka ValueError: gizleme, 500 olarak kalsın
