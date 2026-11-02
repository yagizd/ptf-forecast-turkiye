"""Servis için DONDURULMUŞ model paketi üretir: data/processed/model_paketi_v1/ (git dışı; pakette HAM VERİ YOK).

Tasarım: C = 2026-09-22 (T'nin 13. bloğunun ilk günü).
- Nokta modeli R1: LGBMRegressor, reports/final_config.json parametreleri, hedef = ham fiyat, eğitim [TRAIN_START, C).
- Aralık modelleri: 4 LightGBM quantile (alpha 0.05, 0.1, 0.9, 0.95), parametreler aralik_v2.BASE, eğitim [TRAIN_START, C-56 gün).
- Özellikler reports/final_config.json "ozellikler" (grup H, K yok); özellik listesi config ile AYNI olmalı (assert).
- Veri: data/processed/dataset_weather_guncel.parquet (salt okunur). T kilidi (load_locked) BURADA KULLANILMAZ: üretim paketi,
  hiçbir ayar/seçim yapılmıyor; parametreler ve özellikler config'ten DONDURULMUŞ gelir (Optuna yok).
- Modeller LightGBM metin biçiminde (Booster.save_model, .txt); pickle/joblib YOK.
- Kaydettikten sonra her model YENİDEN YÜKLENİR ve eğitimdeki tahminle np.array_equal karşılaştırılır; değilse DURULUR.
- Çıktı ÖNCE dosyaya: data/processed/model_paketi_v1/olusturma_raporu.md; konsola kısa özet.
SÜRE TAHMİNİ (ölçüm DEĞİL): 5 eğitim (1 nokta modeli 400 ağaç + 4 quantile); referans aralik_gunluk 70 eğitim + döngü = 37 sn (ölçüm);
bu betik için ~10-20 sn "tahmin" (veri yükleme ve özellik üretimi dahil).
Çalıştırma (ana .venv): PYTHONUTF8=1 python -m src.serving.bundle
Kural: eksik değer doldurulmaz.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import holidays_tr as ht
from src.data import price_caps as pc
from src.features import ptf_features as pf
from src.features import ptf_features_v2 as v2
from src.models import aralik_v2 as av2
from src.models import lgbm_walkforward as lw
from src.serving import predictor as pr

ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = ROOT / "data" / "processed" / "dataset_weather_guncel.parquet"
CONFIG_PATH = ROOT / "reports" / "final_config.json"
OUT_DIR = ROOT / "data" / "processed" / "model_paketi_v1"
CODE_FILES = ["src/serving/bundle.py", "src/serving/predictor.py", "src/models/aralik_gunluk.py",
              "src/features/ptf_features.py", "src/features/ptf_features_v2.py", "src/data/holidays_tr.py", "src/data/price_caps.py"]
Q_ALPHAS = {"q05": 0.05, "q10": 0.1, "q90": 0.9, "q95": 0.95}


def train_models(feat: pd.DataFrame, cols: list, c_day: pd.Timestamp, train_start: pd.Timestamp, point_params: dict, q_base: dict) -> dict:
    """1 nokta modeli [train_start, C) + 4 quantile model [train_start, C-56g). Hedef = ham fiyat; boş hedefli satırlar eğitimde yok."""
    import lightgbm as lgb

    c_ts = pd.Timestamp(c_day).tz_localize(pr.TZ)
    ok = feat[pf.TARGET].notna()
    m_point = ((feat["ts"] >= train_start) & (feat["ts"] < c_ts) & ok).to_numpy()
    m_q = ((feat["ts"] >= train_start) & (feat["ts"] < c_ts - pd.Timedelta(days=pr.CAL_DAYS)) & ok).to_numpy()
    assert m_point.any() and m_q.any()
    assert feat.loc[m_point, "ts"].max() < c_ts and feat.loc[m_q, "ts"].max() < c_ts - pd.Timedelta(days=pr.CAL_DAYS)
    models = {}
    point = lgb.LGBMRegressor(**point_params)
    point.fit(feat.loc[m_point, cols], feat.loc[m_point, pf.TARGET])
    models["point"] = point
    for name, a in Q_ALPHAS.items():
        m = lgb.LGBMRegressor(alpha=a, **q_base)
        m.fit(feat.loc[m_q, cols], feat.loc[m_q, pf.TARGET])
        models[name] = m
    models["_n"] = {"nokta_egitim_satir": int(m_point.sum()), "aralik_egitim_satir": int(m_q.sum()),
                    "nokta_egitim_ilk": str(feat.loc[m_point, "ts"].min()), "nokta_egitim_son": str(feat.loc[m_point, "ts"].max()),
                    "aralik_egitim_ilk": str(feat.loc[m_q, "ts"].min()), "aralik_egitim_son": str(feat.loc[m_q, "ts"].max())}
    return models


def build_bundle(table: dict, out_dir: Path, cols: list, point_params: dict, q_base: dict, train_start: pd.Timestamp,
                 table_sha256: str | None = None) -> tuple:
    """Eğit, kaydet, yeniden yükle, birebir karşılaştır. Dönen: (meta, bilgi). Var olan paketin üzerine YAZMAZ."""
    import lightgbm as lgb

    out_dir = Path(out_dir)
    if (out_dir / "point_r1.txt").exists() or (out_dir / "meta.json").exists():
        raise FileExistsError(f"paket zaten var, üzerine yazılmaz: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    c_day = table["c_day"]
    feat = table["feat"]
    assert feat["ts"].min() <= train_start, "tablo TRAIN_START'ı kapsamıyor"

    t0 = time.time()
    models = train_models(feat, cols, c_day, train_start, point_params, q_base)
    train_s = time.time() - t0
    n_info = models.pop("_n")

    start = (c_day - pd.Timedelta(days=pr.CAL_DAYS)).tz_localize(pr.TZ)
    X = feat.loc[feat["ts"] >= start, cols]
    files, checks = {}, {}
    for name, model in models.items():
        path = out_dir / pr.MODEL_FILES[name]
        ref = np.asarray(model.predict(X), dtype=float)
        model.booster_.save_model(str(path))
        reloaded = lgb.Booster(model_file=str(path))
        same = bool(np.array_equal(reloaded.predict(X), ref))
        checks[name] = same
        if not same:
            raise RuntimeError(f"yeniden yüklenen {name} modeli eğitimdeki tahminle AYNI DEĞİL; DURULDU")
        files[name] = {"dosya": path.name, "bayt": path.stat().st_size, "sha256": pr.sha256_file(path)}

    root_rel = lambda f: pr.sha256_file(ROOT / f)  # noqa: E731
    meta = {
        "C": str(c_day.date()),
        "nokta_egitim_bitis": str(c_day.date()) + " 00:00 (hariç)",
        "aralik_egitim_bitis": str((c_day - pd.Timedelta(days=pr.CAL_DAYS)).date()) + " 00:00 (hariç)",
        "train_start": str(train_start),
        "kalibrasyon_gun": pr.CAL_DAYS,
        "gamma": pr.GAMMA,
        "hedef_kayiplar": {str(k): v[2] for k, v in pr.LEVELS.items()},
        "ozellikler": list(cols),
        "parametreler": {"nokta": point_params, "aralik": q_base},
        "replay_ilk_gun": str(c_day.date()),
        "replay_son_gun": str(table["last_day"].date()),
        "tablo_sha256": table_sha256 or table["sha256"],
        "tavan_tablosu_sha256": pr.sha256_file(pc.CAP_PATH),
        "tatil_tablosu_sha256": pr.sha256_file(ht.HOLIDAY_PATH),
        "kod_sha256": {f: root_rel(f) for f in CODE_FILES},
        "surumler": {"python": platform.python_version(), "lightgbm": lgb.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "pyarrow": __import__("pyarrow").__version__},
        "model_dosyalari": {n: f["dosya"] for n, f in files.items()},
        "olusturma_zamani": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {"egitim_suresi_sn": train_s, "dosyalar": files, "kontrol_birebir": checks, **n_info, "satir_tablo": len(feat)}
    return meta, info


def write_report(meta: dict, info: dict, total_s: float, path: Path) -> None:
    from src.models.baselines import _md_table

    L = ["# Model paketi v1: oluşturma raporu", "",
         "**Claude (sohbet) bu kodu ve çıktıyı HENÜZ İNCELEMEDİ.**", "",
         f"- Oluşturma zamanı: {meta['olusturma_zamani']}; toplam süre (ölçüm): {total_s:.1f} sn; eğitim süresi (ölçüm): {info['egitim_suresi_sn']:.1f} sn.",
         f"- Tablo: dataset_weather_guncel.parquet, sha256 {meta['tablo_sha256']}; satır {info['satir_tablo']}; replay {meta['replay_ilk_gun']} .. {meta['replay_son_gun']}.",
         f"- C = {meta['C']}. Nokta modeli eğitim: {info['nokta_egitim_ilk']} .. {info['nokta_egitim_son']} ({info['nokta_egitim_satir']} satır).",
         f"- Aralık modelleri eğitim: {info['aralik_egitim_ilk']} .. {info['aralik_egitim_son']} ({info['aralik_egitim_satir']} satır). Kalibrasyon {meta['kalibrasyon_gun']} gün, gamma {meta['gamma']}, hedef kayıplar {meta['hedef_kayiplar']}.",
         f"- Özellikler ({len(meta['ozellikler'])}): {', '.join(meta['ozellikler'])}",
         f"- Sürümler: {meta['surumler']}", "",
         "## Dosyalar", "",
         _md_table(["model", "dosya", "bayt", "sha256", "yeniden yükleme = eğitim tahmini (np.array_equal)"],
                   [(n, f["dosya"], f["bayt"], f["sha256"], "EVET" if info["kontrol_birebir"][n] else "HAYIR") for n, f in info["dosyalar"].items()]), "",
         f"- meta.json sha256: {pr.sha256_file(path.parent / 'meta.json')}", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.time()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["gruplar"] == ["H"] and cfg["kapak_normalizasyonu_K"] is False
    assert cfg["parametreler_kaynagi"].startswith("varsayilan"), "Optuna parametresi kullanılmaz"
    groups = tuple(cfg["gruplar"])
    cols = list(cfg["ozellikler"])
    assert cols == v2.feature_list(pf.FEATURES_WEATHER, groups), "özellik listesi config ile aynı olmalı"
    df = pd.read_parquet(TABLE_PATH)
    table = pr.make_table(df, pr.C_DAY, pr.sha256_file(TABLE_PATH))
    assert all(c in table["feat"].columns for c in cols)
    meta, info = build_bundle(table, OUT_DIR, cols, cfg["parametreler"], av2.BASE, lw.TRAIN_START)
    write_report(meta, info, time.time() - t0, OUT_DIR / "olusturma_raporu.md")
    print(f"paket yazıldı: {OUT_DIR.name}; C {meta['C']}; replay {meta['replay_ilk_gun']}..{meta['replay_son_gun']}; "
          f"5 model yeniden yüklendi ve birebir aynı; süre {time.time() - t0:.1f} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
