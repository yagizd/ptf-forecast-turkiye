"""Geçmiş deney sonuçlarını MLflow'a AKTARIR (yeni model eğitimi YOK; sayılar kayıtlı tahmin dosyalarından HESAPLANIR).

Ayrı ortamla çalışır: .venv-mlflow\\Scripts\\python.exe -m src.models.mlflow_aktar [--kuru]
Yalnızca pandas, numpy, mlflow ve standart kütüphane kullanır; lightgbm/optuna/sklearn/proje modülleri İMPORT EDİLMEZ.
--kuru : hiçbir şey yazmaz (MLflow'a da dosyaya da); neyi kaydedeceğini ve hesaplanan sayıları ekrana basar.
Normal : takip deposu sqlite:///mlflow/mlflow.db (repo kökünde mlflow/, git'e girmez), deney "epias-ptf-forecast".
Aynı run adı varsa atlanır (tekrar çalıştırmaya güvenli). Kural: eksik değer doldurulmaz.

İNCELENEN DOSYA SÜTUNLARI (betik yazılırken okundu, tahmin edilmedi):
- data/processed/lgbm_tahminleri.parquet (8760 satır, T): ts[us, Europe/Istanbul], price, tahmin_havasiz, tahmin_havali, blok_no; NaN yok.
- data/processed/final_tahminleri.parquet (8760, T): ts, price, tahmin_final, blok_no; NaN yok.
- data/processed/yeni_gun_tahminleri.parquet (216, 2026-10-21..29): ts, gercek, tahmin_d1, tahmin_d7, tahmin_kural, tahmin_R1, tahmin_R3; NaN yok.
- data/processed/egitim_oturumu/aralik_v_tahminleri.parquet (8760, V): ts, y, lo, hi, Q, blok_no, tavan, lo_klip, hi_klip; NaN yok.
- reports/baselines_gunluk.csv (8760, T): gun, saat, gercek, tahmin_d1, tahmin_d7, tahmin_kural; NaN yok.
  (lgbm/final dosyalarında kıyaslar yok; D-1/D-7/Kural buradan gelir, (gun, saat) ile eşlenir.)
- reports/final_config.json: ozellikler, gruplar, kapak_normalizasyonu_K, hedef_bicimi, parametreler, parametreler_kaynagi, ...

Notlar:
- İlk LightGBM modelinin özellik listesi/parametresi bu dosyalarda kayıtlı DEĞİL; run'da "BİLMİYORUM (kayıtlı değil)" yazılır.
- R3'ün (K varyantı) özellikleri R1 ile aynı, fark hedef biçiminde (fiyat/tavan); bu da etiket olarak yazılır.
- kayit_zamani_commit = aktarım anındaki HEAD; deneyin gerçekte çalıştığı commit OLMAYABİLİR.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EXPERIMENT = "epias-ptf-forecast"
TRACKING_URI = "sqlite:///" + (ROOT / "mlflow" / "mlflow.db").as_posix()
ARTIFACT_DIR = ROOT / "mlflow" / "artifacts"
LOG_PATH = PROC / "mlflow_oturumu" / "ilerleme.log"
TOL = 0.05
TZ = "Europe/Istanbul"

SRC = {
    "baselines": ROOT / "reports" / "baselines_gunluk.csv",
    "lgbm": PROC / "lgbm_tahminleri.parquet",
    "final": PROC / "final_tahminleri.parquet",
    "yeni": PROC / "yeni_gun_tahminleri.parquet",
    "aralik": PROC / "egitim_oturumu" / "aralik_v_tahminleri.parquet",
    "config": ROOT / "reports" / "final_config.json",
}
DATASET = PROC / "dataset_weather.parquet"

# Bilinen değerler (YALNIZCA çapraz kontrol için; kaydedilen sayılar buradan GELMEZ)
KNOWN = {
    "T_kiyaslar": {"mae_tl_mwh_d1": 519.2, "mae_tl_mwh_d7": 598.6, "mae_tl_mwh_kural": 497.5},
    "T_ilk_lightgbm": {"mae_tl_mwh_havasiz": 433.5, "mae_tl_mwh_havali": 419.3},
    "T_final_R1": {"mae_tl_mwh": 420.5},
    "Yeni9gun": {"mae_tl_mwh_d1": 495.2, "mae_tl_mwh_d7": 720.3, "mae_tl_mwh_kural": 448.6,
                 "mae_tl_mwh_R1": 433.9, "mae_tl_mwh_R3": 422.6},
    "V_aralik_CQR": {"kapsama_kirpilmamis_yuzde": 75.43, "genislik_kirpilmamis": 854.33},
}
SKIPPED = [
    "V dönemi R0, R0b, R1, R2, R3 doğrulama deneyleri: satır düzeyinde tahmin dosyası KAYITLI DEĞİL "
    "(yalnızca reports/dogrulama_deneyleri.md'de özet sayılar var; elle yazılmadı) - kaydı yok, atlandı",
    "Optuna denemeleri (optuna_v.db): tahmin dosyası yok; sayılar tahmin dosyalarından hesaplanamaz - kaydı yok, atlandı",
]


def log(msg: str, kuru: bool = False) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [mlflow_aktar] {msg}"
    print(line, flush=True)
    if not kuru and LOG_PATH.parent.exists():
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as err:  # git yoksa
        return f"BİLMİYORUM ({type(err).__name__})"


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def with_gun_saat(df: pd.DataFrame) -> pd.DataFrame:
    """ts (Europe/Istanbul) -> (gun, saat) anahtarı; baselines_gunluk.csv ile eşlemek için."""
    out = df.copy()
    local = out["ts"].dt.tz_convert(TZ)
    out["gun"] = local.dt.strftime("%Y-%m-%d")
    out["saat"] = local.dt.hour
    return out


def load_baselines() -> pd.DataFrame:
    b = pd.read_csv(SRC["baselines"])
    b["gun"] = b["gun"].astype(str)
    return b


def cfg_params(cfg: dict) -> dict:
    p = {f"param_{k}": v for k, v in cfg["parametreler"].items()}
    p.update(ozellikler=", ".join(cfg["ozellikler"]), gruplar=",".join(cfg["gruplar"]),
             kapak_normalizasyonu_K=cfg["kapak_normalizasyonu_K"], hedef_bicimi=cfg["hedef_bicimi"],
             parametreler_kaynagi=cfg["parametreler_kaynagi"])
    return p


def build_runs() -> list:
    cfg = json.loads(SRC["config"].read_text(encoding="utf-8"))
    base = load_baselines()
    runs = []

    # 1) kıyaslar (T)
    d1 = mae(base["tahmin_d1"], base["gercek"])
    m = {"mae_tl_mwh_d1": d1, "mae_tl_mwh_d7": mae(base["tahmin_d7"], base["gercek"]),
         "mae_tl_mwh_kural": mae(base["tahmin_kural"], base["gercek"]),
         "n_saat": float(len(base))}
    m["rmae_d1_d7"] = m["mae_tl_mwh_d7"] / d1
    m["rmae_d1_kural"] = m["mae_tl_mwh_kural"] / d1
    runs.append(dict(name="T_kiyaslar", donem="T", kaynak=[SRC["baselines"]], metrics=m, params={"model": "model yok (kural tabanlı kıyaslar)"}))

    # 2) ilk LightGBM (T)
    lg = with_gun_saat(pd.read_parquet(SRC["lgbm"])).merge(base, on=["gun", "saat"], how="left", validate="one_to_one")
    assert lg["tahmin_d1"].notna().all() and len(lg) == len(base)
    d1l = mae(lg["tahmin_d1"], lg["price"])
    m = {"mae_tl_mwh_havasiz": mae(lg["tahmin_havasiz"], lg["price"]), "mae_tl_mwh_havali": mae(lg["tahmin_havali"], lg["price"]),
         "mae_tl_mwh_d1_ayni_satirlar": d1l, "n_saat": float(len(lg))}
    m["rmae_d1_havasiz"] = m["mae_tl_mwh_havasiz"] / d1l
    m["rmae_d1_havali"] = m["mae_tl_mwh_havali"] / d1l
    runs.append(dict(name="T_ilk_lightgbm", donem="T", kaynak=[SRC["lgbm"], SRC["baselines"]], metrics=m,
                     params={"parametreler_ve_ozellikler": "BİLMİYORUM (kayıtlı değil; bu dosyalarda yok)"}))

    # 3) final R1 (T)
    fn = with_gun_saat(pd.read_parquet(SRC["final"])).merge(base, on=["gun", "saat"], how="left", validate="one_to_one")
    assert fn["tahmin_d1"].notna().all() and len(fn) == len(base)
    d1f = mae(fn["tahmin_d1"], fn["price"])
    m = {"mae_tl_mwh": mae(fn["tahmin_final"], fn["price"]), "mae_tl_mwh_d1_ayni_satirlar": d1f, "n_saat": float(len(fn))}
    m["rmae_d1"] = m["mae_tl_mwh"] / d1f
    runs.append(dict(name="T_final_R1", donem="T", kaynak=[SRC["final"], SRC["baselines"], SRC["config"]], metrics=m,
                     params=cfg_params(cfg)))

    # 4) yeni 9 gün (T dışı)
    yn = pd.read_parquet(SRC["yeni"])
    d1y = mae(yn["tahmin_d1"], yn["gercek"])
    m = {f"mae_tl_mwh_{k}": mae(yn[c], yn["gercek"]) for k, c in
         (("d1", "tahmin_d1"), ("d7", "tahmin_d7"), ("kural", "tahmin_kural"), ("R1", "tahmin_R1"), ("R3", "tahmin_R3"))}
    for k in ("d7", "kural", "R1", "R3"):
        m[f"rmae_d1_{k}"] = m[f"mae_tl_mwh_{k}"] / d1y
    m["n_saat"] = float(len(yn))
    m["n_gun"] = float(yn["ts"].dt.tz_convert(TZ).dt.normalize().nunique())
    p = cfg_params(cfg)
    p["R3_notu"] = "R3 = aynı özellikler, hedef fiyat/tavan, ağırlık tavan, tahmin x tavan(D)"
    runs.append(dict(name="Yeni9gun_D1_D7_Kural_R1_R3", donem="T_disi_yeni_gunler", kaynak=[SRC["yeni"], SRC["config"]], metrics=m,
                     params=p, extra_tags={"not": "KARAR DEĞİL; 9 gün yetersiz"}))

    # 5) aralık CQR (V)
    ar = pd.read_parquet(SRC["aralik"])
    y = ar["y"].to_numpy()
    m = {}
    for ad, lo, hi in (("kirpilmamis", "lo", "hi"), ("kirpilmis", "lo_klip", "hi_klip")):
        l, h = ar[lo].to_numpy(), ar[hi].to_numpy()
        m[f"kapsama_{ad}_yuzde"] = float(np.mean((y >= l) & (y <= h)) * 100)
        m[f"genislik_{ad}"] = float(np.mean(h - l))
        m[f"alt_iskalama_{ad}_yuzde"] = float(np.mean(y < l) * 100)
        m[f"ust_iskalama_{ad}_yuzde"] = float(np.mean(y > h) * 100)
    m["n_saat"] = float(len(ar))
    runs.append(dict(name="V_aralik_CQR", donem="V", kaynak=[SRC["aralik"], SRC["config"]], metrics=m,
                     params={**cfg_params(cfg), "yontem": "LightGBM quantile 0.1/0.9 + CQR, kalibrasyon 56 gün, hedef kapsama %80"}))
    return runs


def crosscheck(run: dict) -> list:
    """Bilinen değerlerle karşılaştır. Tutmayanların listesi döner (SAYI DEĞİŞTİRİLMEZ)."""
    bad = []
    for key, known in KNOWN[_known_key(run["name"])].items():
        got = run["metrics"][key]
        if abs(got - known) > TOL + 1e-6:
            bad.append((key, got, known))
    return bad


def _known_key(name: str) -> str:
    return "Yeni9gun" if name.startswith("Yeni9gun") else name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kuru", action="store_true", help="hiçbir şey yazma; sadece neyi kaydedeceğini ve sayıları yazdır")
    args = ap.parse_args(argv)
    runs = build_runs()
    ds_sha, head, now = sha256(DATASET), git_head(), time.strftime("%Y-%m-%d %H:%M:%S")

    client = None
    exp_id = None
    existing = set()
    if not args.kuru:
        import mlflow
        from mlflow.tracking import MlflowClient

        (ROOT / "mlflow").mkdir(exist_ok=True)
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT)
        exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
        for r in client.search_runs([exp_id], max_results=1000):
            existing.add(r.data.tags.get("mlflow.runName"))

    log(f"{'KURU MOD' if args.kuru else 'NORMAL MOD'}: {len(runs)} run hazır; dataset_weather sha256 {ds_sha[:12]}…; HEAD {head[:10]}", args.kuru)
    for run in runs:
        bad = crosscheck(run)
        tags = {"donem": run["donem"], "kaynak_dosya": "; ".join(str(p.relative_to(ROOT)) for p in run["kaynak"]),
                "dataset_weather_sha256": ds_sha, "kayit_zamani_commit": head,
                "kayit_zamani_commit_notu": "aktarım anındaki HEAD; deneyin gerçekte çalıştığı commit olmayabilir",
                "kayit_zamani": now, "uyusmuyor": "evet" if bad else "hayir", **run.get("extra_tags", {})}
        if run["donem"] == "T":
            tags["T_daha_once_goruldu"] = "evet"
        log(f"run '{run['name']}' etiketler: {json.dumps({k: v for k, v in tags.items() if k not in ('dataset_weather_sha256', 'kayit_zamani_commit_notu')}, ensure_ascii=False)}", args.kuru)
        for k, v in run["metrics"].items():
            log(f"    metrik {k} = {v:.4f}", args.kuru)
        if bad:
            for key, got, known in bad:
                log(f"    UYUSMUYOR: {key} hesaplanan {got:.4f} bilinen {known} (sayı DEĞİŞTİRİLMEDİ)", args.kuru)
        if args.kuru:
            continue
        if run["name"] in existing:
            log(f"    '{run['name']}' zaten kayıtlı, atlandı")
            continue
        import mlflow

        with mlflow.start_run(experiment_id=exp_id, run_name=run["name"]):
            mlflow.log_params({k: str(v)[:500] for k, v in run["params"].items()})
            mlflow.log_metrics(run["metrics"])
            mlflow.set_tags(tags)
        log(f"    '{run['name']}' kaydedildi")
    log("Atlanan deneyler (kaydı yok):", args.kuru)
    for s in SKIPPED:
        log("  - " + s, args.kuru)
    return 0


if __name__ == "__main__":
    sys.exit(main())
