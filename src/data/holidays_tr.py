"""Türkiye tatil günlerinden günlük özellikler.

Kaynak: data/external/turkiye_tatil_gunleri.csv (tarih, tur, ad, kaynak_url, guven).
tur: resmi_tatil | arife_yarim_gun | idari_izin_kopru.

Özellikler (hedef günün takvim bilgisi; tatil takvimi önceden bilinir, fiyattan bağımsızdır):
- is_holiday          : resmi_tatil günü = 1
- is_arife            : arife_yarim_gun günü = 1
- is_bridge           : idari_izin_kopru günü = 1
- days_to_holiday     : bir sonraki resmi tatile kalan gün (tatilin kendisi 0), EN FAZLA 7
- days_since_holiday  : son resmi tatilden geçen gün (tatilin kendisi 0), EN FAZLA 7
Hafta sonu tek başına tatil değildir.

Üst sınır 7 bir TANIMDIR (doldurma değildir): "7 gün veya daha uzak" demektir.
Tablonun kapsadığı aralığın dışında (başlangıçtan 7 gün sonrası öncesi ve bitişten 7 gün öncesi sonrası)
günler için sınır anlamlı hesaplanamaz; bu günlerde ValueError fırlatılır. Eksik değer doldurulmaz.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOLIDAY_PATH = ROOT / "data" / "external" / "turkiye_tatil_gunleri.csv"
COVER_START = pd.Timestamp("2024-01-01")
COVER_END = pd.Timestamp("2026-12-31")
CAP_DAYS = 7
FEATURES = ["is_holiday", "is_arife", "is_bridge", "days_to_holiday", "days_since_holiday"]
TURLER = {"resmi_tatil", "arife_yarim_gun", "idari_izin_kopru"}


def load_holidays(path: Path = HOLIDAY_PATH) -> pd.DataFrame:
    t = pd.read_csv(path, parse_dates=["tarih"])
    bad = set(t["tur"]) - TURLER
    if bad:
        raise ValueError(f"bilinmeyen tatil türü: {bad}")
    return t


def daily_features(days, table: pd.DataFrame | None = None,
                   cover_start=COVER_START, cover_end=COVER_END) -> pd.DataFrame:
    """days: günlerin (tarih) listesi/serisi. Dönen: index=gün (normalize), sütunlar FEATURES."""
    t = load_holidays() if table is None else table
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(days)))).normalize()
    if d.tz is not None:
        d = d.tz_localize(None)
    lo = pd.Timestamp(cover_start) + pd.Timedelta(days=CAP_DAYS)
    hi = pd.Timestamp(cover_end) - pd.Timedelta(days=CAP_DAYS)
    if len(d) and (d.min() < lo or d.max() > hi):
        raise ValueError(f"gün aralığı tatil tablosunun güvenli kapsamı dışında ({lo.date()}..{hi.date()})")
    tur = t.drop_duplicates("tarih").set_index("tarih")["tur"]
    hol = np.sort(t.loc[t["tur"] == "resmi_tatil", "tarih"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    dv = d.to_numpy(dtype="datetime64[ns]")
    to_h = np.full(len(dv), CAP_DAYS, dtype=int)
    since_h = np.full(len(dv), CAP_DAYS, dtype=int)
    day_ns = np.timedelta64(1, "D")
    for i, x in enumerate(dv):
        j = np.searchsorted(hol, x, side="left")  # ilk hol >= x
        if j < len(hol):
            to_h[i] = min(int((hol[j] - x) / day_ns), CAP_DAYS)
        k = np.searchsorted(hol, x, side="right") - 1  # son hol <= x
        if k >= 0:
            since_h[i] = min(int((x - hol[k]) / day_ns), CAP_DAYS)
    kind = pd.Series(d).map(tur)
    out = pd.DataFrame({
        "is_holiday": (kind == "resmi_tatil").astype(int).to_numpy(),
        "is_arife": (kind == "arife_yarim_gun").astype(int).to_numpy(),
        "is_bridge": (kind == "idari_izin_kopru").astype(int).to_numpy(),
        "days_to_holiday": to_h,
        "days_since_holiday": since_h,
    }, index=d)
    return out


def features_for_ts(ts: pd.Series, table: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    """Saatlik ts serisi için özellikler (satır sırası ts ile aynı, index ts'in index'i)."""
    local = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts
    days = local.dt.normalize()
    f = daily_features(days.unique(), table, **kw)
    out = f.reindex(pd.DatetimeIndex(days))
    out.index = ts.index
    return out
