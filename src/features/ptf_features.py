"""PTF tahmini için özellikler. SADECE tahmin anında (hedef günün bir gün öncesi ~10:00) bilinenler.

Hedef: D gününün her saatinin PTF'si. D günü fiyatları D-1 günü ~12:30'da belli olur; yani 10:00'da D-1 ve öncesinin
fiyatları bilinir, D'ninki bilinmez.

Özellikler:
- price_lag_d1 / d2 / d7 : aynı saatin 24 / 48 / 168 saat önceki fiyatı. ZAMANA göre eşlenir (ts - N saat);
  satır kaydırma YOK. O saat tabloda yoksa veya boşsa özellik NaN olur.
- price_d1_mean : D-1 takvim gününün 24 saatlik fiyat ortalaması (herhangi bir saat boşsa NaN).
- hour, dayofweek, month, is_weekend : takvim.
- Hava: build_weather_dataset.MODEL_FEATURES_PD2 (5 sütun, previous_day2 tahminleri), olduğu gibi; NaN kalır.

YASAK (özelliklerde olmaz): consumption, lep, tüm *_pd1 sütunları, hedef günün veya sonrasının herhangi bir fiyatı.
Sebep: yayın saatleri bilinmiyor, sızıntı riski var.

Kural: eksik değerler hiçbir şekilde doldurulmaz; NaN olarak kalır.
"""

import pandas as pd

from src.data.build_weather_dataset import MODEL_FEATURES_PD2

TARGET = "price"
PRICE_FEATURES = ["price_lag_d1", "price_lag_d2", "price_lag_d7", "price_d1_mean"]
CALENDAR_FEATURES = ["hour", "dayofweek", "month", "is_weekend"]
WEATHER_FEATURES = list(MODEL_FEATURES_PD2)

FEATURES_BASE = PRICE_FEATURES + CALENDAR_FEATURES  # hava YOK
FEATURES_WEATHER = FEATURES_BASE + WEATHER_FEATURES  # hava VAR (5 sütun)

# Özellik adlarında bu parçalar geçemez.
FORBIDDEN_NAME_PARTS = ("consumption", "lep", "pd1")


def assert_no_forbidden(columns) -> None:
    """Yasak bir sütun adı varsa hata ver."""
    bad = [c for c in columns if c == "lep" or any(p in c for p in FORBIDDEN_NAME_PARTS)]
    if bad:
        raise ValueError(f"özelliklerde yasak sütun var: {bad}")


# Modül yüklenirken de kontrol et: yanlışlıkla yasak sütun eklenirse hemen patlasın.
assert_no_forbidden(FEATURES_BASE)
assert_no_forbidden(FEATURES_WEATHER)


def _lag_by_time(price: pd.Series, ts: pd.Series, hours: int) -> pd.Series:
    """Her satır için (ts - hours saat) anındaki fiyat. Eşleşme zamana göre; yoksa NaN."""
    shifted = ts - pd.Timedelta(hours=hours)
    return pd.Series(price.reindex(pd.DatetimeIndex(shifted)).to_numpy(), index=ts.index)


def _d1_daily_mean(price: pd.Series, ts: pd.Series) -> pd.Series:
    """Her satır için, bir önceki takvim gününün 24 saatlik ortalaması.

    Önceki günde 24 saatten az satır varsa veya herhangi bir fiyat boşsa sonuç NaN (eksik saat doldurulmaz).
    """
    local = ts.dt.tz_localize(None)
    day = local.dt.normalize()
    p = pd.Series(price.to_numpy(), index=ts.index)
    rows = p.groupby(day).size()
    valid = p.notna().groupby(day).sum()
    mean = p.groupby(day).mean()
    mean[(rows != 24) | (valid != 24)] = float("nan")
    prev_day = day - pd.Timedelta(days=1)
    return pd.Series(mean.reindex(pd.DatetimeIndex(prev_day)).to_numpy(), index=ts.index)


def build_features(df: pd.DataFrame, include_weather: bool = True) -> pd.DataFrame:
    """Saatlik tablodan özellik tablosu üretir.

    df: ts (saat dilimli), price ve (include_weather ise) MODEL_FEATURES_PD2 sütunlarını içermeli.
        consumption / lep gibi diğer sütunlar varsa YOKSAYILIR (okunmaz, çıktıya girmez).
    Dönen: ts, price (hedef) ve özellikler. Satır sayısı ve sırası df ile aynı.
    """
    need = ["ts", TARGET] + (WEATHER_FEATURES if include_weather else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"gerekli sütunlar yok: {missing}")
    if df["ts"].duplicated().any():
        raise ValueError("ts tekrarı var; özellikler zamana göre eşlenemez")

    ts = df["ts"].reset_index(drop=True)
    price_by_ts = pd.Series(df[TARGET].to_numpy(), index=pd.DatetimeIndex(df["ts"]))

    out = pd.DataFrame({"ts": ts, TARGET: df[TARGET].to_numpy()})
    out["price_lag_d1"] = _lag_by_time(price_by_ts, ts, 24)
    out["price_lag_d2"] = _lag_by_time(price_by_ts, ts, 48)
    out["price_lag_d7"] = _lag_by_time(price_by_ts, ts, 168)
    out["price_d1_mean"] = _d1_daily_mean(df[TARGET].reset_index(drop=True), ts)
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    if include_weather:
        for c in WEATHER_FEATURES:
            out[c] = df[c].to_numpy()  # olduğu gibi; NaN kalır

    cols = ["ts", TARGET] + (FEATURES_WEATHER if include_weather else FEATURES_BASE)
    return out[cols]
