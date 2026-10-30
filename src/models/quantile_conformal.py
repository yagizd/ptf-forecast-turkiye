"""Tahmin aralığı: LightGBM quantile (alpha=0.1 ve 0.9) + conformalized quantile regression (CQR).

SADECE KOD: bu oturumda ÇALIŞTIRILMADI ve TEST EDİLMEDİ (Claude da incelemedi).

Tasarım (her blok için):
- Kalibrasyon penceresi = bloktan ÖNCEKİ son 8 hafta (56 gün; blok dahil değil).
- Quantile modeller sadece kalibrasyon penceresinden de ÖNCEKİ satırlarla eğitilir (kalibrasyon satırları eğitimde yok;
  CQR'nin geçerliliği için gerekli).
- Uyumsuzluk skoru E_i = max(q_lo_i - y_i, y_i - q_hi_i) kalibrasyon satırlarında; Q = skorların
  ceil((n+1)(1-a)) / n seviyesindeki kuantili (a = 0.2, hedef kapsama %80).
- Blok aralığı = [q_lo - Q, q_hi + Q]. Kapsama ve ortalama aralık genişliği raporlanır.
Kural: eksik değer doldurulmaz; hedefi boş satırlar eğitimden/kalibrasyondan atılır, özellik NaN'ları kalır.
"""

import math

import numpy as np
import pandas as pd

ALPHA_LO, ALPHA_HI = 0.1, 0.9
MISCOVERAGE = 0.2  # hedef kapsama %80
CAL_DAYS = 56  # 8 hafta


def conformal_quantile(scores, miscoverage: float = MISCOVERAGE) -> float:
    """Sonlu örneklem düzeltmeli kuantil: ceil((n+1)(1-a))-inci en küçük skor. n çok küçükse +inf (aralık sonsuz)."""
    s = np.sort(np.asarray(scores, dtype=float))
    s = s[~np.isnan(s)]
    n = len(s)
    k = math.ceil((n + 1) * (1 - miscoverage))
    if n == 0 or k > n:
        return float("inf")
    return float(s[k - 1])


def split_masks(ts: pd.Series, target: pd.Series, block_start, train_start, cal_days: int = CAL_DAYS):
    """(eğitim, kalibrasyon) maskeleri. Eğitim: train_start <= ts < block_start - cal_days; kalibrasyon: [block_start - cal_days, block_start).
    Her ikisinde hedef dolu olmalı. Blok ve sonrası hiçbirinde yok."""
    cal_start = block_start - pd.Timedelta(days=cal_days)
    ok = target.notna()
    train = (ts >= train_start) & (ts < cal_start) & ok
    cal = (ts >= cal_start) & (ts < block_start) & ok
    return train.to_numpy(), cal.to_numpy()


def cqr_interval(q_lo, q_hi, cal_lo, cal_hi, cal_y, miscoverage: float = MISCOVERAGE):
    """Blok için alt/üst sınır. cal_* kalibrasyon satırlarının quantile tahminleri ve gerçekleri."""
    scores = np.maximum(np.asarray(cal_lo) - np.asarray(cal_y), np.asarray(cal_y) - np.asarray(cal_hi))
    q = conformal_quantile(scores, miscoverage)
    return np.asarray(q_lo) - q, np.asarray(q_hi) + q, q


def coverage_and_width(y, lo, hi) -> dict:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    ok = ~np.isnan(y)
    inside = (y[ok] >= lo[ok]) & (y[ok] <= hi[ok])
    return {"n": int(ok.sum()), "kapsama": float(inside.mean()), "ortalama_genislik": float((hi[ok] - lo[ok]).mean())}


def run_cqr(feat: pd.DataFrame, features: list, blocks: list, train_start, params: dict | None = None,
            target_col: str = "price"):
    """Tüm bloklar için CQR aralıkları. feat: ts, target_col ve features içeren tablo. Dönen: ts, y, lo, hi, blok_no, Q."""
    import lightgbm as lgb

    base = dict(objective="quantile", learning_rate=0.05, n_estimators=400, num_leaves=31, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, deterministic=True,
                force_row_wise=True, verbosity=-1)
    base.update(params or {})
    parts = []
    for b in blocks:
        tr_m, cal_m = split_masks(feat["ts"], feat[target_col], b["start_ts"], train_start)
        te_m = ((feat["ts"] >= b["start_ts"]) & (feat["ts"] < b["end_ts"])).to_numpy()
        if tr_m.sum() == 0 or cal_m.sum() == 0 or te_m.sum() == 0:
            continue
        assert feat.loc[tr_m, "ts"].max() < feat.loc[cal_m, "ts"].min(), "eğitim kalibrasyondan önce olmalı"
        preds = {}
        for name, alpha in (("lo", ALPHA_LO), ("hi", ALPHA_HI)):
            m = lgb.LGBMRegressor(alpha=alpha, **base)
            m.fit(feat.loc[tr_m, features], feat.loc[tr_m, target_col])
            preds[name] = (m.predict(feat.loc[cal_m, features]), m.predict(feat.loc[te_m, features]))
        lo, hi, q = cqr_interval(preds["lo"][1], preds["hi"][1], preds["lo"][0], preds["hi"][0],
                                 feat.loc[cal_m, target_col].to_numpy())
        part = feat.loc[te_m, ["ts"]].copy()
        part["y"] = feat.loc[te_m, target_col].to_numpy()
        part["lo"], part["hi"], part["Q"], part["blok_no"] = lo, hi, q, b["no"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
