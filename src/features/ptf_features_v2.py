"""PTF özellikleri v2: ptf_features.py'ye EK gruplar (ptf_features.py DEĞİŞTİRİLMEZ, import edilir).

Gruplar:
- Grup H (tatil)  : holidays_tr.FEATURES (5 sütun), hedef günün (D) takvim bilgisinden; fiyattan bağımsız.
- Grup Z (uçlar)  : price_d1_min, price_d1_max, zero_hours_d1 (D-1 gününde fiyatı 0 olan saat sayısı),
                    lag_d1_is_zero, lag_d7_is_zero. Sadece D-1 ve öncesinden; eksik saat varsa NaN (skipna=False mantığı).
- Varyant K (kapak normalizasyonu): her fiyat kendi gününün resmî tavanına bölünür (price_norm = price / tavan(gün)).
  Lag ve D-1 özellikleri (Grup Z dahil) price_norm'dan üretilir; hedef = price_norm (sütun "target");
  TL'ye dönüş: tahmin * "cap" (D günü tavanı). Eğitim örnek ağırlığı = cap (kayıp TL MAE ile hizalı kalır).
  Tavan tablosu 2023-07-04'ten başladığı için K modunda bu tarihten ÖNCEKİ satırlar atılır (tavanı bilinmeyen
  fiyat normalleştirilemez; doldurma yok).

Çıktı sütunları: ts, price (HAM TL; değerlendirme için), target (ham ya da norm), cap (K modunda), özellikler.
YASAK: consumption, lep, pd1 (ptf_features.assert_no_forbidden).
Kural: eksik değer hiçbir şekilde doldurulmaz. Tatil özellikleri tablo kapsamı dışındaki günlerde NaN'dır
(bilinmiyor; doldurulmaz).
"""

import numpy as np
import pandas as pd

from src.data import holidays_tr as ht
from src.data import price_caps as pc
from src.features import ptf_features as pf

FEATURES_H = list(ht.FEATURES)
FEATURES_Z = ["price_d1_min", "price_d1_max", "zero_hours_d1", "lag_d1_is_zero", "lag_d7_is_zero"]

pf.assert_no_forbidden(FEATURES_H)
pf.assert_no_forbidden(FEATURES_Z)


def feature_list(base: list, groups=()) -> list:
    """base listesine seçilen grupları ("H", "Z") ekler."""
    out = list(base)
    if "H" in groups:
        out += FEATURES_H
    if "Z" in groups:
        out += FEATURES_Z
    pf.assert_no_forbidden(out)
    return out


def _d1_stats(price: pd.Series, ts: pd.Series) -> pd.DataFrame:
    """Her satır için bir önceki takvim gününün min, max, sıfır saat sayısı. Gün eksik/boş saatliyse NaN."""
    day = ts.dt.tz_localize(None).dt.normalize()
    p = pd.Series(price.to_numpy(), index=ts.index)
    size = p.groupby(day).size()
    valid = p.notna().groupby(day).sum()
    full = (size == 24) & (valid == 24)
    g = pd.DataFrame({
        "mn": p.groupby(day).min(),
        "mx": p.groupby(day).max(),
        "zr": p.eq(0).groupby(day).sum().astype(float),
    })
    g.loc[~full, :] = np.nan  # eksik gün: üç değer de NaN (doldurma yok)
    prev = pd.DatetimeIndex(day - pd.Timedelta(days=1))
    r = g.reindex(prev)
    return pd.DataFrame({"price_d1_min": r["mn"].to_numpy(), "price_d1_max": r["mx"].to_numpy(),
                         "zero_hours_d1": r["zr"].to_numpy()}, index=ts.index)


def build_features_v2(df: pd.DataFrame, groups=(), normalize: bool = False, include_weather: bool = True,
                      holiday_table=None, cap_table=None,
                      cover=(ht.COVER_START, ht.COVER_END)) -> pd.DataFrame:
    """ptf_features.build_features + seçilen gruplar + isteğe bağlı K normalizasyonu.

    Dönen: ts, price (ham), target, cap, ardından özellikler (base + seçilen gruplar).
    """
    work = df
    cap = None
    if normalize:
        ct = pc.load_cap_table() if cap_table is None else cap_table
        start = ct["yururluk_ilk_teslim_gunu"].min().tz_localize(pf_tz(df))
        work = df[df["ts"] >= start].reset_index(drop=True)
        cap = pc.caps_for_ts(work["ts"], ct)
        work = work.copy()
        work["price"] = work["price"].to_numpy() / cap.to_numpy()
    base = pf.build_features(work, include_weather=include_weather)  # price = ham ya da norm
    ts = base["ts"]
    out = pd.DataFrame({"ts": ts})
    out["price"] = df.loc[df["ts"].isin(ts), "price"].to_numpy() if normalize else base["price"].to_numpy()
    out["target"] = base["price"].to_numpy()
    out["cap"] = cap.to_numpy() if normalize else np.nan
    feats = pf.FEATURES_WEATHER if include_weather else pf.FEATURES_BASE
    for c in feats:
        out[c] = base[c].to_numpy()
    nprice = pd.Series(base["price"].to_numpy(), index=ts.index)
    if "H" in groups:
        lo = pd.Timestamp(cover[0]) + pd.Timedelta(days=ht.CAP_DAYS)
        hi = pd.Timestamp(cover[1]) - pd.Timedelta(days=ht.CAP_DAYS)
        days = ts.dt.tz_localize(None).dt.normalize()
        ok = (days >= lo) & (days <= hi)
        h = pd.DataFrame(np.nan, index=ts.index, columns=FEATURES_H)
        if ok.any():
            hf = ht.daily_features(days[ok].unique(), holiday_table, cover_start=cover[0], cover_end=cover[1])
            h.loc[ok, :] = hf.reindex(pd.DatetimeIndex(days[ok])).to_numpy()
        for c in FEATURES_H:
            out[c] = h[c].to_numpy()
    if "Z" in groups:
        z = _d1_stats(nprice, ts)
        for c in ("price_d1_min", "price_d1_max", "zero_hours_d1"):
            out[c] = z[c].to_numpy()
        for c, lag in (("lag_d1_is_zero", "price_lag_d1"), ("lag_d7_is_zero", "price_lag_d7")):
            v = base[lag]
            out[c] = np.where(v.isna(), np.nan, (v == 0).astype(float))
    cols = ["ts", "price", "target", "cap"] + feature_list(feats, groups)
    return out[cols]


def pf_tz(df: pd.DataFrame):
    return df["ts"].dt.tz
