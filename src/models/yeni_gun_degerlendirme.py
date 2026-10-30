"""Test dönemi T'den SONRAKİ yeni günlerde (2026-10-21 ve sonrası) dondurulmuş R1 ve R3'ün değerlendirmesi.

R1 = reports/final_config.json'daki dondurulmuş yapılandırma (gruplar [H], ham fiyat hedefi, varsayılan parametreler).
R3 = aynı özellikler, K varyantı: hedef = fiyat/tavan, ağırlık = tavan, tahmin x tavan(D). Optuna parametresi YOK.
Eğitim: yeni günün öncesindeki tüm veri (TRAIN_START'tan); yeni günler 28 günden kısa olduğu için TEK blok.
Karşılaştırma: D-1, D-7, Kural, R1, R3 (MAE, rMAE = D-1'e göre). Bu sonuç bir karar değildir.
Çalıştırma: python -m src.models.yeni_gun_degerlendirme
Kural: eksik değer doldurulmaz.
"""

import json
import sys

import numpy as np
import pandas as pd

from src.features import ptf_features_v2 as v2
from src.models import baselines as bl
from src.models import validation_experiments as ve

ROOT = ve.ROOT
DATA_PATH = ROOT / "data" / "processed" / "dataset_weather_guncel.parquet"
CONFIG_PATH = ROOT / "reports" / "final_config.json"
PRED_PATH = ROOT / "data" / "processed" / "yeni_gun_tahminleri.parquet"
REPORT_PATH = ROOT / "reports" / "yeni_gun_degerlendirme.md"
NEW_START = pd.Timestamp("2026-10-21")


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["parametreler_kaynagi"].startswith("varsayilan"), "Optuna parametresi kullanılmaz"
    assert cfg["gruplar"] == ["H"] and cfg["kapak_normalizasyonu_K"] is False
    return cfg


def mae(d: pd.DataFrame, c: str) -> float:
    return float((d[c] - d["gercek"]).abs().mean())


def main() -> int:
    cfg = load_config()
    df = pd.read_parquet(DATA_PATH)
    last_day = df["ts"].max().tz_localize(None).normalize()
    groups = tuple(cfg["gruplar"])
    params = cfg["parametreler"]

    runs = {}
    for name, norm in (("R1", False), ("R3", True)):
        pred, cols, blocks = ve.walkforward_config(df, NEW_START, last_day, groups=groups, normalize=norm, params=params)
        assert cols == cfg["ozellikler"], "özellik listesi dondurulmuş yapılandırmayla uyuşmuyor"
        assert len(blocks) == 1
        runs[name] = pred

    bt = bl.run_backtest(bl.price_table(df), start=NEW_START, end=last_day)
    j = ve.join_preds(bt, runs)
    cm = j.dropna(subset=["gercek", *bl.MODELS.values(), "t_R1", "t_R3"]).copy()
    n_days = cm["gun"].nunique()

    out = pd.DataFrame({"ts": cm["gun"].dt.tz_localize("Europe/Istanbul") + pd.to_timedelta(cm["saat"], unit="h"),
                        "gercek": cm["gercek"].to_numpy(), "tahmin_d1": cm["tahmin_d1"].to_numpy(),
                        "tahmin_d7": cm["tahmin_d7"].to_numpy(), "tahmin_kural": cm["tahmin_kural"].to_numpy(),
                        "tahmin_R1": cm["t_R1"].to_numpy(), "tahmin_R3": cm["t_R3"].to_numpy()})
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_PATH, index=False)

    cols = {"D-1": "tahmin_d1", "D-7": "tahmin_d7", "Kural": "tahmin_kural", "R1": "t_R1", "R3": "t_R3"}
    d1 = mae(cm, "tahmin_d1")
    res = {n: mae(cm, c) for n, c in cols.items()}
    # ikinci, bağımsız hesap (numpy, kaydedilmiş parquet'ten)
    chk = pd.read_parquet(PRED_PATH)
    for n, c in (("D-1", "tahmin_d1"), ("D-7", "tahmin_d7"), ("Kural", "tahmin_kural"), ("R1", "tahmin_R1"), ("R3", "tahmin_R3")):
        v = float(np.mean(np.abs(chk[c].to_numpy() - chk["gercek"].to_numpy())))
        assert abs(v - res[n]) < 1e-9, (n, v, res[n])
    assert len(chk) == len(cm) == n_days * 24

    day_rows = []
    wins = 0
    for g, d in cm.groupby("gun"):
        m = {n: mae(d, c) for n, c in cols.items()}
        wins += int(m["R3"] < m["R1"])
        day_rows.append((g.date(), len(d), *[f"{m[n]:.1f}" for n in cols], "R3" if m["R3"] < m["R1"] else "R1"))

    cap_hours = int((cm["gercek"] == 4500).sum())
    L = [f"Bu sonuç bir KARAR DEĞİL; yaklaşık {n_days} gün var, R1/R3 seçimi için yetersiz. Önceden yazılmış kural (8/13 blok) değişmedi.",
         "", "# Yeni gün değerlendirmesi (2026-10-21 sonrası)", "",
         "**Claude (sohbet) bu kodu ve sonucu İNCELEMEDİ.**", "",
         f"- Veri: {DATA_PATH.name}; yeni günler {cm['gun'].min().date()}..{cm['gun'].max().date()}; {n_days} gün, {len(cm)} saat "
         "(tüm modellerin ve gerçeğin dolu olduğu aynı satırlar). Bu günler test dönemi T'nin DIŞINDADIR.",
         f"- R1: dondurulmuş final (gruplar {list(groups)}, ham fiyat hedefi, varsayılan parametreler, Optuna YOK). "
         "R3: aynı özellikler, hedef = fiyat/tavan, ağırlık = tavan, tahmin x tavan(D) (tavan 4500 TL/MWh, 2026-04-04 kararı).",
         f"- Eğitim: {cfg['TRAIN_START']} .. 2026-10-20 23:00 (yeni günün öncesindeki tüm veri). Yeni günler {n_days} gün (<28) olduğundan "
         "tek blok kullanıldı: model bir kez eğitildi, 9 günün hepsi bu modelle tahmin edildi (eğitimin bitişinden 1..9 gün sonrası).",
         f"- Özellikler ({len(cfg['ozellikler'])}): {', '.join(cfg['ozellikler'])}",
         f"- Fiyatı 4500 (tavan) olan saat sayısı: {cap_hours}", "",
         "## Toplam", "",
         bl._md_table(["model", "MAE (TL/MWh)", "rMAE (D-1'e göre)"], [(n, f"{v:.1f}", f"{v / d1:.3f}") for n, v in res.items()]), "",
         "## Gün gün MAE", "",
         bl._md_table(["gün", "saat", *[f"MAE {n}" for n in cols], "R1/R3 hangisi düşük"], day_rows), "",
         f"R3, {n_days} günün {wins}'inde R1'den düşük MAE verdi (bilgi amaçlı; {n_days} gün, 28 günlük tek bloktan bile kısa).", "",
         "## Notlar", "",
         f"- {n_days} gün çok az; R3'ün ya da R1'in 'daha iyi' olduğu SONUCU ÇIKARILMAZ.",
         "- Seçim kuralı (V üzerinde, ≥%1,0 göreli kazanç ve 13 tam bloktan ≥8'inde daha iyi) değişmedi; bu tablo o kuralın yerini almaz.",
         "- Sızıntı testi: tests/test_yeni_gun_degerlendirme.py (D gününden sonraki fiyatlar değişince D'nin özellikleri değişmez).", ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"gun": n_days, "saat": len(cm), **{k: round(v, 2) for k, v in res.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
