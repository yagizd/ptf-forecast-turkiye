"""Servis tahmin modülü: dondurulmuş model paketini okur ve TEK bir günü tahmin eder (DURUMSUZ; kalıcı durum YOK).

Paket: data/processed/model_paketi_v1/ (src/serving/bundle.py üretir). Nokta modeli R1 (eğitim [TRAIN_START, C)),
aralık modelleri 4 LightGBM quantile (0.05, 0.1, 0.9, 0.95; eğitim [TRAIN_START, C-56 gün)). C = 2026-09-22.
Aralık = B3: ACI gamma 0,05; %80 (alpha çifti 0.1/0.9, hedef kayıp 0,2) ve %90 (0.05/0.95, 0,1). Gün D için kalibrasyon [D-56g, D).
alpha_C = hedef kayıp; alpha_D, C..D-1 günlerinin GERÇEK fiyatlarından HER İSTEKTE yeniden hesaplanır:
  her gün d için aralık alpha_d ile (kalibrasyon [d-56g, d)), e_d = o gün aralık DIŞI saat oranı (boş aralık = 1),
  alpha_(d+1) = alpha_d + 0,05 * (hedef - e_d).
ACI çekirdeği src/models/aralik_gunluk.py'den IMPORT edilir (scores, q_level, make_interval, aci_update); kopyalanmadı.
Skor ve aralık HAM (sıralanmamış) quantile çıktılarıyla; çıktıda aralık [0, tavan(D)] ile kırpılır. Q=+inf: aralık [0, tavan].
Boş aralık: lo/hi = None ve bos=True.
ZAMAN SIZINTISI YOK: D günü çıktısı yalnız D 00:00'dan ÖNCEKİ fiyatları kullanır. "gercek" yalnız bilgi amaçlıdır.
Replay penceresi: C .. tablonun son günü. D < C: ValueError("egitimde goruldu"); D > son gün: ValueError("tabloda yok").
Önbellek: aynı tablo (sha256) ve paket için quantile/nokta tahminleri [C-56g, son gün] satırlarında BİR KEZ hesaplanır (yalnız hız;
sonuç değişmez). Kural: eksik değer doldurulmaz.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import price_caps as pc
from src.features import ptf_features as pf
from src.features import ptf_features_v2 as v2
from src.models.aralik_gunluk import aci_update, make_interval, q_level, scores

ROOT = Path(__file__).resolve().parents[2]
TZ = "Europe/Istanbul"
C_DAY = pd.Timestamp("2026-09-22")
CAL_DAYS = 56
GAMMA = 0.05
GROUPS = ("H",)
LEVELS = {80: (0.1, 0.9, 0.2), 90: (0.05, 0.95, 0.1)}  # düzey -> (alpha_lo, alpha_hi, hedef kayıp)
MODEL_FILES = {"point": "point_r1.txt", "q05": "q05.txt", "q10": "q10.txt", "q90": "q90.txt", "q95": "q95.txt"}
Q_OF = {0.05: "q05", 0.1: "q10", 0.9: "q90", 0.95: "q95"}
EPOCH = pd.Timestamp("1970-01-01")
_CACHE: dict = {}


def clear_cache() -> None:
    _CACHE.clear()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def day_numbers(ts: pd.Series) -> np.ndarray:
    """Yerel (Europe/Istanbul) takvim günü -> 1970-01-01'den gün sayısı."""
    local = ts.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    return (local - EPOCH).dt.days.to_numpy().astype(int)


def validate_table(df: pd.DataFrame, c_day: pd.Timestamp) -> pd.Timestamp:
    """bundle.py ile AYNI doğrulamalar. Dönen: tablonun son günü (naif Timestamp). Hata: ValueError."""
    need = ["ts", pf.TARGET] + list(pf.WEATHER_FEATURES)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"tabloda sütun yok: {missing}")
    if str(df["ts"].dt.tz) != TZ:
        raise ValueError(f"ts saat dilimi {TZ} olmalı, bulunan: {df['ts'].dt.tz}")
    if df["ts"].duplicated().any():
        raise ValueError("ts tekrarı var")
    if not df["ts"].is_monotonic_increasing:
        raise ValueError("ts artan sırada değil")
    start = (pd.Timestamp(c_day) - pd.Timedelta(days=CAL_DAYS)).tz_localize(TZ)
    rng = df[df["ts"] >= start]
    if rng.empty or rng["ts"].min() != start:
        raise ValueError(f"tablo {start} (C-56 gün) başlangıcını içermiyor")
    day = rng["ts"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    counts = day.value_counts().sort_index()
    if not (counts == 24).all():
        bad = counts[counts != 24]
        raise ValueError(f"her gün tam 24 satır olmalı; sorunlu günler: {[d.date().isoformat() for d in bad.index[:5]]}")
    if not (pd.Series(counts.index).diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("C-56 gün .. son gün arasında eksik gün var")
    if rng[pf.TARGET].isna().any():
        raise ValueError("C-56 gün .. son gün arasında fiyat boş (NaN) satır var")
    return counts.index.max()


def make_table(df: pd.DataFrame, c_day: pd.Timestamp = C_DAY, sha: str | None = None) -> dict:
    """Doğrular ve özellikleri BİR KEZ üretir (grup H, normalizasyon yok). sha yoksa içerikten hesaplanır."""
    last_day = validate_table(df, c_day)
    feat = v2.build_features_v2(df, groups=GROUPS, normalize=False).reset_index(drop=True)
    assert len(feat) == len(df)
    if sha is None:
        sha = hashlib.sha256(pd.util.hash_pandas_object(df[["ts", pf.TARGET, *pf.WEATHER_FEATURES]], index=False).to_numpy().tobytes()).hexdigest()
    return {"sha256": sha, "c_day": pd.Timestamp(c_day), "last_day": last_day, "feat": feat,
            "day": day_numbers(feat["ts"])}


def load_table(path, c_day: pd.Timestamp = C_DAY) -> dict:
    """Parquet tabloyu okur (salt okunur), doğrular ve özellikleri üretir."""
    return make_table(pd.read_parquet(path), c_day, sha256_file(path))


def load_bundle(directory) -> dict:
    """Modelleri Booster(model_file=...) ile yükler. meta'daki lightgbm sürümü kurulu sürümle AYNI değilse RuntimeError."""
    import lightgbm as lgb

    d = Path(directory)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kurulu, kayitli = lgb.__version__, meta["surumler"]["lightgbm"]
    if kurulu != kayitli:
        raise RuntimeError(f"lightgbm sürümü uyuşmuyor: paket {kayitli}, kurulu {kurulu} (sürümler arası uyum belgelenmemiş)")
    models = {k: lgb.Booster(model_file=str(d / f)) for k, f in MODEL_FILES.items()}
    return {"dir": d, "meta": meta, "models": models, "id": sha256_file(meta_path), "c_day": pd.Timestamp(meta["C"])}


def quantile_cache(bundle: dict, table: dict) -> dict:
    """[C-56g, son gün] satırları için nokta ve HAM quantile tahminleri (önbellek anahtarı: tablo sha256 + paket)."""
    key = (table["sha256"], bundle["id"])
    if key in _CACHE:
        return _CACHE[key]
    if bundle["c_day"] != table["c_day"]:
        raise ValueError(f"paket C={bundle['c_day'].date()} ama tablo C={table['c_day'].date()} için doğrulandı")
    cols = list(bundle["meta"]["ozellikler"])
    feat = table["feat"]
    absent = [c for c in cols if c not in feat.columns]
    if absent:
        raise ValueError(f"tabloda özellik yok: {absent}")
    start = (bundle["c_day"] - pd.Timedelta(days=CAL_DAYS)).tz_localize(TZ)
    m = (feat["ts"] >= start).to_numpy()
    X = feat.loc[m, cols]
    out = {"ts": feat.loc[m, "ts"].reset_index(drop=True), "day": table["day"][m],
           "y": feat.loc[m, pf.TARGET].to_numpy(float)}
    for name, booster in bundle["models"].items():
        out[name] = np.asarray(booster.predict(X), dtype=float)
    _CACHE[key] = out
    return out


def level_state(cache: dict, level: int, c_no: int, d_no: int):
    """C..D-1 günlerinden alpha_D'yi hesaplar (kalıcı durum YOK) ve D günü için (alpha_D, Q, lo, hi, satır maskesi) döner."""
    a_lo, a_hi, target = LEVELS[level]
    qlo, qhi = cache[Q_OF[a_lo]], cache[Q_OF[a_hi]]
    day, y = cache["day"], cache["y"]
    alpha = target
    for d in range(c_no, d_no):  # yalnız D'den ÖNCEKİ günler
        cal = (day >= d - CAL_DAYS) & (day < d)
        Q = q_level(scores(y[cal], qlo[cal], qhi[cal]), alpha)
        m = day == d
        lo, hi = make_interval(qlo[m], qhi[m], Q)
        e = float(1.0 - ((y[m] >= lo) & (y[m] <= hi)).mean())
        alpha = aci_update(alpha, GAMMA, target, e)
        assert -GAMMA - 1e-12 <= alpha <= 1 + GAMMA + 1e-12, f"alpha sınır dışı: {alpha}"
    cal = (day >= d_no - CAL_DAYS) & (day < d_no)
    Q = q_level(scores(y[cal], qlo[cal], qhi[cal]), alpha)
    m = day == d_no
    lo, hi = make_interval(qlo[m], qhi[m], Q)
    return alpha, Q, lo, hi, m


def predict_day(bundle: dict, table: dict, date) -> dict:
    """Tek gün tahmini. D < C veya D > son gün: ValueError."""
    D = pd.Timestamp(date)
    if D.tzinfo is not None:
        D = D.tz_localize(None)
    D = D.normalize()
    C, last = bundle["c_day"], table["last_day"]
    if D < C:
        raise ValueError(f"egitimde goruldu: {D.date()} < {C.date()} (model {C.date()} oncesi verilerle egitildi)")
    if D > last:
        raise ValueError(f"tabloda yok: {D.date()} > tablonun son gunu {last.date()}")
    cache = quantile_cache(bundle, table)
    c_no, d_no = int((C - EPOCH).days), int((D - EPOCH).days)
    cap = pc.cap_for_day(D)
    res = {}
    for level in LEVELS:
        alpha, Q, lo, hi, m = level_state(cache, level, c_no, d_no)
        res[level] = (alpha, Q, lo, hi, m)
    m = res[80][4]
    assert int(m.sum()) == 24, "D günü tam 24 satır olmalı"
    point = cache["point"][m]
    y = cache["y"][m]
    ts = cache["ts"][m].reset_index(drop=True)
    hours = []
    for i in range(24):
        row = {"saat": int(ts.iloc[i].hour), "nokta": float(point[i])}
        for level in LEVELS:
            alpha, Q, lo, hi, _ = res[level]
            bos = bool(Q == -np.inf)
            row[f"lo{level}"] = None if bos else float(np.clip(lo[i], 0.0, cap))
            row[f"hi{level}"] = None if bos else float(np.clip(hi[i], 0.0, cap))
            row[f"bos{level}"] = bos
            row[f"alpha{level}"] = float(alpha)
        row["tavan"] = float(cap)
        row["gercek"] = float(y[i])  # yalnız bilgi amaçlı; tahmine girmez
        hours.append(row)
    return {"tarih": D.date().isoformat(), "egitim_sonu": C.date().isoformat(), "egitimde_goruldu": False,
            "model_surumu": f"model_paketi_v1@{bundle['id'][:12]}", "saatler": hours}
