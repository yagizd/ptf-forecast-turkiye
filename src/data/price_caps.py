"""Resmî PTF azami fiyat (tavan) tablosu: gün -> tavan.

Kaynak: data/external/ptf_tavan_tablosu.csv (reports/tavan_arastirma.md'den, 2023-07-04 ve sonrası).
Tablonun başlangıcından önceki bir gün için hata fırlatır (tahmin yapılmaz).
Kural: eksik değer doldurulmaz.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAP_PATH = ROOT / "data" / "external" / "ptf_tavan_tablosu.csv"


def load_cap_table(path: Path = CAP_PATH) -> pd.DataFrame:
    t = pd.read_csv(path, parse_dates=["yururluk_ilk_teslim_gunu"])
    t = t.sort_values("yururluk_ilk_teslim_gunu").reset_index(drop=True)
    return t


def cap_for_day(day, table: pd.DataFrame | None = None) -> float:
    """Tek gün için tavan (TL/MWh). Tablo başlangıcından önceki gün: ValueError."""
    t = load_cap_table() if table is None else table
    d = pd.Timestamp(day)
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    d = d.normalize()
    ok = t[t["yururluk_ilk_teslim_gunu"] <= d]
    if ok.empty:
        raise ValueError(f"{d.date()} tavan tablosunun başlangıcından ({t['yururluk_ilk_teslim_gunu'].min().date()}) önce")
    return float(ok.iloc[-1]["tavan_tl_mwh"])


def caps_for_ts(ts: pd.Series, table: pd.DataFrame | None = None) -> pd.Series:
    """Saatlik ts serisi için (yerel takvim günü bazında) tavan serisi. Tablo öncesi gün varsa ValueError."""
    t = load_cap_table() if table is None else table
    local = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts
    days = local.dt.normalize()
    uniq = {d: cap_for_day(d, t) for d in days.unique()}
    return days.map(uniq).astype(float)
