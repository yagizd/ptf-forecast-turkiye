"""Dondurulmuş yapılandırmayla test dönemi T'yi TEK SEFER çalıştırır.

T = 2025-10-21..2026-10-20. Yapılandırma reports/final_config.json'dan okunur (ADIM 7a, `--freeze` ile yazılır).
Çıktı: data/processed/final_tahminleri.parquet (ts, price, tahmin_final, blok_no) ve reports/final_model_raporu.md.
Çıktılar zaten varsa HİÇBİR ŞEY çalıştırılmaz (tek çalıştırma koruması).

Çalıştırma: python -m src.models.final_test_run --freeze   (yapılandırmayı dondurur, T'ye dokunmaz)
            python -m src.models.final_test_run            (T'yi tek sefer çalıştırır)
Kural: eksik değer doldurulmaz.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import price_caps as pc
from src.features import ptf_features as pf
from src.features import ptf_features_v2 as v2
from src.models import baselines as bl
from src.models import lgbm_walkforward as lw
from src.models import validation_experiments as ve
from src.models import tune_optuna as to

ROOT = ve.ROOT
CONFIG_PATH = ROOT / "reports" / "final_config.json"
PRED_PATH = ROOT / "data" / "processed" / "final_tahminleri.parquet"
REPORT_PATH = ROOT / "reports" / "final_model_raporu.md"
FIRST_PATH = ROOT / "data" / "processed" / "lgbm_tahminleri.parquet"
T_START = bl.TEST_START
T_END = bl.TEST_END
CODE_FILES = ["src/features/ptf_features.py", "src/features/ptf_features_v2.py", "src/models/lgbm_walkforward.py",
              "src/models/validation_experiments.py", "src/models/final_test_run.py", "src/data/price_caps.py",
              "src/data/holidays_tr.py", "src/models/baselines.py"]
INTRO = ("Test dönemi daha önce bir kez görüldü (ilk model). Bu yüzden bu sonuç tam temiz bir son sınav DEĞİLDİR; "
         "yapılandırma yalnızca doğrulama dönemi V'ye bakılarak seçildi.")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze() -> int:
    sec = json.loads(ve.SEC_PATH.read_text(encoding="utf-8"))
    par = json.loads(to.PARAMS_PATH.read_text(encoding="utf-8")) if to.PARAMS_PATH.exists() else None
    params = par["parametreler"] if par else lw.LGBM_PARAMS
    cfg = {
        "ozellikler": sec["ozellikler"], "gruplar": sec["gruplar"], "kapak_normalizasyonu_K": sec["kapak_normalizasyonu_K"],
        "hedef_bicimi": sec["hedef_bicimi"], "parametreler": params,
        "parametreler_kaynagi": ("optuna benimsendi" if par and par["benimsendi"] else
                                 "varsayilan (optuna benimsenmedi veya calismadi)"),
        "seed": {"lightgbm_random_state": params.get("random_state", 42), "optuna_tpe": 42},
        "TRAIN_START": str(lw.TRAIN_START), "blok_gunu": lw.BLOCK_DAYS,
        "test_donemi": [str(T_START.date()), str(T_END.date())],
        "kod_sha256": {f: sha(ROOT / f) for f in CODE_FILES},
    }
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("final_config.json sha256:", sha(CONFIG_PATH))
    return 0


def _mae(d, c):
    return float((d[c] - d["gercek"]).abs().mean())


def _table(cm: pd.DataFrame, key: pd.Series, name: str, cols: dict) -> str:
    rows = []
    for k, g in cm.groupby(key.loc[cm.index], sort=True):
        d1 = _mae(g, "tahmin_d1")
        rows.append((k, len(g), *[f"{_mae(g, c):.1f}" for c in cols.values()],
                     *[f"{_mae(g, c) / d1:.3f}" if d1 > 0 else "-" for c in list(cols.values())[1:]]))
    return bl._md_table([name, "n", *[f"MAE {n}" for n in cols], *[f"rMAE {n}" for n in list(cols)[1:]]], rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args(argv)
    if args.freeze:
        return freeze()
    if PRED_PATH.exists() or REPORT_PATH.exists():
        print("DURDU: çıktılar zaten var; tek çalıştırma kuralı gereği tekrar çalıştırılmaz")
        return 1
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    df = pd.read_parquet(ve.DATA_PATH)  # T burada ilk kez (tek sefer) yüklenir
    groups, norm = tuple(cfg["gruplar"]), bool(cfg["kapak_normalizasyonu_K"])
    pred, cols, blocks = ve.walkforward_config(df, T_START, T_END, groups=groups, normalize=norm,
                                               params=cfg["parametreler"])
    assert cols == cfg["ozellikler"], "özellik listesi dondurulmuş yapılandırmayla uyuşmuyor"
    out = pred.rename(columns={"tahmin": "tahmin_final"})[["ts", "price", "tahmin_final", "blok_no"]]
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_PATH, index=False)

    bt = bl.run_backtest(bl.price_table(df), start=T_START, end=T_END)
    first = pd.read_parquet(FIRST_PATH)
    gun, saat = lw._day_hour(first["ts"])
    fq = pd.DataFrame({"gun": gun.astype(bt["gun"].dtype), "saat": saat, "ilk_havasiz": first["tahmin_havasiz"].to_numpy(),
                       "ilk_havali": first["tahmin_havali"].to_numpy()})
    j = bt.merge(fq, on=["gun", "saat"], how="left", validate="one_to_one")
    j = ve.join_preds(j, {"final": pred})
    j = j.rename(columns={"t_final": "tahmin_final"})
    cm = j.dropna(subset=["gercek", *bl.MODELS.values(), "ilk_havasiz", "ilk_havali", "tahmin_final"])
    cols_m = {"D-1": "tahmin_d1", "D-7": "tahmin_d7", "Kural": "tahmin_kural", "İlk havasız": "ilk_havasiz",
              "İlk havalı": "ilk_havali", "FİNAL": "tahmin_final"}
    d1 = _mae(cm, "tahmin_d1")
    tot = bl._md_table(["model", "MAE (TL/MWh)", "rMAE (D-1'e göre)"],
                       [(n, f"{_mae(cm, c):.1f}", f"{_mae(cm, c) / d1:.3f}") for n, c in cols_m.items()])
    g = cm["gun"]
    wk = pd.Series(np.where(g.dt.dayofweek < 5, "Hafta içi", "Hafta sonu"), index=cm.index)
    lab_old = bl.label_special(cm, bl.month_caps(df))
    capd = cm["gun"].map({d: pc.cap_for_day(d) for d in cm["gun"].unique()})
    lab_new = pd.Series(np.where(cm["gercek"] == 0, "Fiyat 0", np.where(cm["gercek"] == capd, "Tavanda (resmî)", "Diğer")),
                        index=cm.index)
    ay = g.dt.strftime("%Y-%m")
    mt = _table(cm, ay, "ay", cols_m)
    vurgu = [m for m in ("2026-04", "2026-05") if m in set(ay)]
    blk = _table(cm, cm["blok_no"].astype(int), "blok", cols_m)
    L = [INTRO, "", "# Final model raporu (test dönemi T, TEK çalıştırma)", "",
         "**Claude (sohbet) bu kodu ve sonucu İNCELEMEDİ.**", "",
         f"- Test dönemi {T_START.date()}..{T_END.date()}; ölçüm satırı {len(cm)} (tüm modellerin ve gerçeğin dolu olduğu aynı satırlar).",
         f"- Yapılandırma: gruplar {list(groups)}, K={norm}, parametre kaynağı: {cfg['parametreler_kaynagi']}.",
         f"- Özellikler: {', '.join(cfg['ozellikler'])}", "",
         "## Toplam", "", tot, "", "## Hafta içi / hafta sonu", "", _table(cm, wk, "grup", cols_m), "",
         "## Saat türleri (eski kural: ayın en yüksek fiyatı, en az 5 saat)", "", _table(cm, lab_old, "saat türü", cols_m), "",
         "## Saat türleri (resmî tavan tablosu, yeni)", "", _table(cm, lab_new, "saat türü", cols_m), "",
         "## Aylık", "", mt, "",
         "## Vurgulu aylar (2026-04 ve 2026-05)", "",
         _table(cm[ay.isin(vurgu)], ay[ay.isin(vurgu)], "ay", cols_m) if vurgu else "(yok)", "",
         "## Blok bazında", "", blk, ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Tamam. FINAL MAE %.1f, D-1 %.1f" % (_mae(cm, "tahmin_final"), d1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
