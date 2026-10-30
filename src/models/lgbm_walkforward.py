"""LightGBM ilk deneme: yürüyen pencere (walk-forward) ile havasız ve havalı model.

Walk-forward: her blokta SADECE geçmişle eğitip sonraki bloğu tahmin etmek.
- Eğitim başlangıcı: 2024-03-08 00:00 (Europe/Istanbul); hava verisi ancak bu tarihten sonra tam dolu.
- Test dönemi: baselines.TEST_START .. baselines.TEST_END (aynı 8.760 saat).
- 28 günlük bloklar (son blok kısa olabilir). Her blokta eğitim = hedef günü blok başlangıcından ÖNCE olan,
  2024-03-08'den itibaren TÜM satırlar (genişleyen pencere). Hedefi boş olan satırlar atılır; özellik NaN'ları KALIR.
- İki model: "havasiz" (pf.FEATURES_BASE) ve "havali" (pf.FEATURES_WEATHER).

PARAMETRELER BİR AYAR DEĞİLDİR: ilk deneme için sabit tahmindir; test sonucuna bakıp DEĞİŞTİRİLMEYECEK.
Ayar (Optuna) Hafta 4-5'te AYRI doğrulama setiyle yapılacak. Erken durdurma yok, test verisi eğitimde/ayarda kullanılmaz.

Girdi : data/processed/dataset_weather.parquet
Çıktı : data/processed/lgbm_tahminleri.parquet (ts, price, tahmin_havasiz, tahmin_havali, blok_no)
        reports/lgbm_ilk_model.md
Çalıştırma (repo kökünden): python -m src.models.lgbm_walkforward   (üzerine yazmak için --overwrite)
Gerekli paket: lightgbm (fonksiyon içinde içe aktarılır; kurulu değilse sadece eğitim adımı çalışmaz).

Çalışma süresi TAHMİNİ (ölçüm değil): 14 blok x 2 model = 28 eğitim. Her biri en çok ~22 bin satır x 9-13 özellik,
400 ağaç. Ağaç başına kabaca 0,5-3 ms varsayımıyla eğitim başına ~0,2-1,2 sn + ek yük => toplam kabaca 10 sn - 2 dk.

Kural: eksik değerler hiçbir şekilde doldurulmaz; LightGBM NaN'ı kendisi işler.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import ptf_features as pf
from src.models import baselines as bl

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "dataset_weather.parquet"
PRED_PATH = ROOT / "data" / "processed" / "lgbm_tahminleri.parquet"
REPORT_PATH = ROOT / "reports" / "lgbm_ilk_model.md"

TZ = "Europe/Istanbul"
TRAIN_START = pd.Timestamp("2024-03-08 00:00", tz=TZ)
BLOCK_DAYS = 28

MODEL_SETS = {"havasiz": pf.FEATURES_BASE, "havali": pf.FEATURES_WEATHER}
LGBM_COLS = {"havasiz": "tahmin_havasiz", "havali": "tahmin_havali"}

# SABİT parametreler (ayar değil). Hata ölçümüz MAE olduğu için objective="regression_l1".
LGBM_PARAMS = dict(
    objective="regression_l1",
    learning_rate=0.05,
    n_estimators=400,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,  # sadece uyarı yazılarını sustur; modeli etkilemez
)

# reports/baselines.md'deki (2026-10-21) kıyas MAE değerleri: tutarlılık kontrolü için.
BASELINE_REF = {"d1": 519.2, "kural": 497.5}


# ------------------------------------------------------------------ bloklar ve bölme
def make_blocks(test_start, test_end, block_days: int = BLOCK_DAYS) -> list:
    """Test dönemini block_days günlük bloklara böler. Son blok kısa olabilir. Bloklar dönemi TAM 1 kez kaplar."""
    start = pd.Timestamp(test_start).normalize()
    end = pd.Timestamp(test_end).normalize()
    blocks, no, cur = [], 1, start
    while cur <= end:
        last = min(cur + pd.Timedelta(days=block_days - 1), end)
        blocks.append({
            "no": no,
            "first_day": cur,
            "last_day": last,
            "start_ts": cur.tz_localize(TZ),
            "end_ts": (last + pd.Timedelta(days=1)).tz_localize(TZ),  # bitiş HARİÇ
        })
        no += 1
        cur = last + pd.Timedelta(days=1)
    return blocks


def split_for_block(feat: pd.DataFrame, block: dict, train_start=TRAIN_START):
    """(eğitim maskesi, test maskesi) döner.

    Eğitim: train_start <= ts < blok başlangıcı, ve hedefi (price) dolu olanlar.
    Test  : blok başlangıcı <= ts < blok bitişi (hedefi boş olanlar dahil, satır sayısı korunur).
    """
    ts = feat["ts"]
    train = (ts >= train_start) & (ts < block["start_ts"]) & feat[pf.TARGET].notna()
    test = (ts >= block["start_ts"]) & (ts < block["end_ts"])
    return train.to_numpy(), test.to_numpy()


# ------------------------------------------------------------------ eğitim ve tahmin
def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, params: dict | None = None):
    """Tek model eğit, test satırlarını tahmin et. (tahminler, {özellik: gain}) döner. NaN doldurulmaz."""
    import lightgbm as lgb  # burada: paket kurulu değilse sadece bu adım çalışmaz

    model = lgb.LGBMRegressor(**(params or LGBM_PARAMS))
    model.fit(train_df[features], train_df[pf.TARGET])
    pred = model.predict(test_df[features])
    gain = model.booster_.feature_importance(importance_type="gain")
    return pred, dict(zip(features, [float(g) for g in gain]))


def run_walkforward(df: pd.DataFrame, test_start, test_end, train_start=TRAIN_START,
                    block_days: int = BLOCK_DAYS, params: dict | None = None, model_sets: dict | None = None):
    """Yürüyen pencere. (tahmin tablosu, özellik önemi tablosu, bloklar) döner.

    Tahmin tablosu: ts, price, blok_no, tahmin_<model>. Satır sayısı = test dönemindeki satır sayısı.
    """
    model_sets = model_sets or MODEL_SETS
    feat = pf.build_features(df, include_weather=True)
    blocks = make_blocks(test_start, test_end, block_days)
    parts, imps = [], []
    for b in blocks:
        train_m, test_m = split_for_block(feat, b, train_start)
        assert not (train_m & test_m).any(), "eğitim ve test satırları kesişiyor"
        train, test = feat[train_m], feat[test_m]
        if len(train) == 0:
            raise ValueError(f"blok {b['no']}: eğitim satırı yok")
        assert train["ts"].max() < b["start_ts"], "ZAMAN KURALI İHLALİ: eğitimde blok başlangıcından sonrası var"
        assert train["ts"].min() >= train_start, "eğitim başlangıcından önceki satır var"
        if len(test) == 0:
            continue
        part = test[["ts", pf.TARGET]].copy()
        part["blok_no"] = b["no"]
        for name, cols in model_sets.items():
            pred, gain = fit_predict(train, test, cols, params)
            part[f"tahmin_{name}"] = pred
            imps.append({"model": name, "blok_no": b["no"], **gain})
        parts.append(part)
    if not parts:
        raise ValueError("test dönemi için hiç satır yok")
    return pd.concat(parts, ignore_index=True), pd.DataFrame(imps), blocks


# ------------------------------------------------------------------ ölçüm (kıyaslar baselines.py'den)
def _day_hour(ts: pd.Series):
    local = ts.dt.tz_localize(None)
    return local.dt.normalize(), local.dt.hour


def merge_predictions(bt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """baselines.run_backtest çıktısına (gun, saat) üzerinden LightGBM tahminlerini ekler."""
    gun, saat = _day_hour(pred["ts"])
    gun = gun.astype(bt["gun"].dtype)  # ns/us farkı birleştirmeyi bozmasın
    p = pd.DataFrame({"gun": gun, "saat": saat, "blok_no": pred["blok_no"].to_numpy(),
                      **{c: pred[c].to_numpy() for c in LGBM_COLS.values()}})
    return bt.merge(p, on=["gun", "saat"], how="left", validate="one_to_one")


def common_rows_all(bt_all: pd.DataFrame) -> pd.DataFrame:
    """Gerçek + 3 kıyas + 2 LightGBM tahmininin hepsi dolu satırlar (hepsi aynı satır kümesinde kıyaslanır)."""
    return bt_all.dropna(subset=["gercek", *bl.MODELS.values(), *LGBM_COLS.values()])


def summarize_all(sub: pd.DataFrame) -> dict:
    """baselines.summarize (kıyaslar) + LightGBM MAE ve rMAE (D-1'e göre)."""
    s = bl.summarize(sub)
    for m, c in LGBM_COLS.items():
        mae = float((sub[c] - sub["gercek"]).abs().mean())
        s[f"mae_{m}"] = mae
        s[f"rmae_{m}"] = mae / s["mae_d1"] if s["mae_d1"] > 0 else float("nan")
    return s


def grouped(common: pd.DataFrame, key: pd.Series, key_name: str) -> pd.DataFrame:
    rows = [{key_name: k, **summarize_all(g)} for k, g in common.groupby(key.loc[common.index], sort=True)]
    return pd.DataFrame(rows)


def nan_summary(df: pd.DataFrame, test_start, test_end) -> dict:
    """Test döneminde özellikleri NaN olan satır sayıları (NaN doldurulmaz, sadece sayılır)."""
    feat = pf.build_features(df, include_weather=True)
    s = pd.Timestamp(test_start).normalize().tz_localize(TZ)
    e = (pd.Timestamp(test_end).normalize() + pd.Timedelta(days=1)).tz_localize(TZ)
    t = feat[(feat["ts"] >= s) & (feat["ts"] < e)]
    return {
        "test_rows": len(t),
        "base_any_nan": int(t[pf.FEATURES_BASE].isna().any(axis=1).sum()),
        "weather_any_nan": int(t[pf.WEATHER_FEATURES].isna().any(axis=1).sum()),
    }


def importance_table(imps: pd.DataFrame, model: str) -> pd.DataFrame:
    """Blok başına normalize edilmiş gain payı (%), bloklar üzerinde ortalama."""
    sub = imps[imps["model"] == model].drop(columns=["model", "blok_no"]).dropna(axis=1, how="all")
    total = sub.sum(axis=1).replace(0, np.nan)
    share = sub.div(total, axis=0) * 100
    out = share.mean().sort_values(ascending=False, kind="stable")  # eşit paylarda sıra bozulmasın
    return pd.DataFrame({"özellik": out.index, "ortalama gain payı (%)": out.round(1).to_numpy()})


# ------------------------------------------------------------------ rapor
def _f(x, nd=1) -> str:
    return "-" if pd.isna(x) else f"{x:.{nd}f}"


def _row(first, r) -> tuple:
    return (first, r["n"], _f(r["mae_d1"]), _f(r["mae_d7"]), _f(r["mae_kural"]), _f(r["mae_havasiz"]),
            _f(r["mae_havali"]), _f(r["rmae_d7"], 3), _f(r["rmae_kural"], 3), _f(r["rmae_havasiz"], 3),
            _f(r["rmae_havali"], 3))


HEAD = ["n", "MAE D-1", "MAE D-7", "MAE Kural", "MAE havasız", "MAE havalı",
        "rMAE D-7", "rMAE Kural", "rMAE havasız", "rMAE havalı"]


def _group_table(df: pd.DataFrame, first_col: str, header: str) -> str:
    return bl._md_table([header, *HEAD], [_row(r[first_col], r) for r in df.to_dict("records")])


def build_report(bt_all, common, imps, blocks, caps, nan_info, baseline_rows, params) -> str:
    tot = summarize_all(common)
    gun = common["gun"]
    haftalik = grouped(common, pd.Series(np.where(gun.dt.dayofweek < 5, "Hafta içi", "Hafta sonu"), index=common.index), "grup")
    ozel = grouped(common, bl.label_special(common, caps), "saat türü")
    aylik = grouped(common, gun.dt.strftime("%Y-%m"), "ay")
    blok = grouped(common, common["blok_no"].astype(int), "blok")

    ref_ok = {m: round(tot[f"mae_{m}"], 1) == BASELINE_REF[m] for m in BASELINE_REF}
    rows_ok = len(common) == baseline_rows
    L = ["# LightGBM ilk deneme (yürüyen pencere)", "",
         "**Bu bir İLK DENEME'dir. Parametreler sabit bir tahmindir, AYAR DEĞİLDİR; bu sonuçlara bakıp parametre "
         "değiştirilmemelidir.** Ayar (Optuna) ayrı bir doğrulama setiyle yapılacak. Erken durdurma ve test verisiyle ayar yok.", "",
         "## Protokol", "",
         f"- Eğitim başlangıcı: {TRAIN_START} (hava verisi bundan sonra tam dolu).",
         f"- Test dönemi: {bl.TEST_START.date()} .. {bl.TEST_END.date()}; {len(blocks)} blok x {BLOCK_DAYS} gün (son blok kısa olabilir).",
         "- Her blokta eğitim: hedef zamanı blok başlangıcından ÖNCE olan tüm satırlar (genişleyen pencere); hedefi boş olanlar atılır, özellik NaN'ları kalır.",
         "- Özellikler: price_lag_d1/d2/d7, price_d1_mean, hour, dayofweek, month, is_weekend (+ havalı modelde 5 hava sütunu, previous_day2). "
         "consumption, lep ve *_pd1 sütunları KULLANILMADI.",
         f"- Parametreler: {params}", "",
         f"- Ölçüm satır sayısı: **{tot['n']}** (kıyaslar, havasız ve havalı tahminlerin hepsinin ve gerçeğin dolu olduğu satırlar). "
         f"baselines.common_rows ile aynı küme mi: **{'evet' if rows_ok else 'HAYIR'}** ({tot['n']} / {baseline_rows}).",
         f"- Test döneminde en az bir özelliği NaN olan satır: temel özellikler {nan_info['base_any_nan']}, hava özellikleri {nan_info['weather_any_nan']} "
         f"(toplam {nan_info['test_rows']}). NaN'lar doldurulmadı; LightGBM kendisi işledi.", "",
         "## Tutarlılık kontrolü (reports/baselines.md ile)", "",
         bl._md_table(["model", "baselines.md'deki MAE", "bu çalışmadaki MAE", "sonuç"],
                      [("D-1", BASELINE_REF["d1"], _f(tot["mae_d1"]), "UYUŞUYOR" if ref_ok["d1"] else "UYUŞMUYOR"),
                       ("Kural", BASELINE_REF["kural"], _f(tot["mae_kural"]), "UYUŞUYOR" if ref_ok["kural"] else "UYUŞMUYOR")]),
         "", "## Toplam", "",
         bl._md_table(["model", "MAE (TL/MWh)", "rMAE (D-1'e göre)"],
                      [("D-1", _f(tot["mae_d1"]), "1.000"), ("D-7", _f(tot["mae_d7"]), _f(tot["rmae_d7"], 3)),
                       ("Kural", _f(tot["mae_kural"]), _f(tot["rmae_kural"], 3)),
                       ("LightGBM havasız", _f(tot["mae_havasiz"]), _f(tot["rmae_havasiz"], 3)),
                       ("LightGBM havalı", _f(tot["mae_havali"]), _f(tot["rmae_havali"], 3))]),
         "", "## Hafta içi / hafta sonu", "", _group_table(haftalik, "grup", "grup"),
         "", "## Özel saatler (fiyat 0 / ayın tavanında / diğer)", "",
         f"'Tavanda' = fiyat, o ayın en yüksek fiyatına eşit ve o ayda bu fiyatta en az {bl.MIN_CAP_HOURS} saat var (baselines.label_special). "
         "Resmî tavan değil; sadece değerlendirme etiketi.", "", _group_table(ozel, "saat türü", "saat türü"),
         "", "## Blok bazında", "",
         bl._md_table(["blok", "tarih aralığı", *HEAD],
                      [(r["blok"], f"{blocks[int(r['blok']) - 1]['first_day'].date()} .. {blocks[int(r['blok']) - 1]['last_day'].date()}",
                        *_row(r["blok"], r)[1:]) for r in blok.to_dict("records")]),
         "", "## Aylık", "", _group_table(aylik, "ay", "ay"),
         "", "## Özellik önemi (gain, bloklar üzerinde ortalama pay)", ""]
    for m in MODEL_SETS:
        t = importance_table(imps, m)
        L += [f"### LightGBM {m}", "",
              bl._md_table(["özellik", "ortalama gain payı (%)"], [(r["özellik"], r["ortalama gain payı (%)"]) for r in t.to_dict("records")]), ""]

    yorum = [f"Toplam MAE: D-1 {_f(tot['mae_d1'])}, Kural {_f(tot['mae_kural'])}, LightGBM havasız {_f(tot['mae_havasiz'])}, "
             f"havalı {_f(tot['mae_havali'])} TL/MWh."]
    for m, ad in (("havasiz", "havasız"), ("havali", "havalı")):
        r = tot[f"rmae_{m}"]
        yorum.append(f"LightGBM {ad}: rMAE {_f(r, 3)} -> D-1'i " + ("yendi" if r < 1 else "yenemedi") +
                     "; Kural'a göre " + ("daha iyi" if tot[f"mae_{m}"] < tot["mae_kural"] else "daha kötü veya eşit") + ".")
        if r < 0.5:
            yorum.append(f"DİKKAT: {ad} modelde rMAE {_f(r, 3)} çok düşük; plana göre bu bir sızıntı belirtisi olabilir, kontrol edilmeli.")
    diff = tot["mae_havali"] - tot["mae_havasiz"]
    yorum.append(f"Hava eklenince MAE farkı: {diff:+.1f} TL/MWh (negatif = hava yardımcı oldu). "
                 "Tek bir test döneminde tek çalıştırma; fark istatistiksel olarak sınanmadı.")
    yorum += ["Bayram günleri ayrıca ele alınmadı.",
              "Bu sonuçlara bakarak parametre veya özellik seti değiştirmek test dönemini 'ayar' için kullanmak olur; yapılmamalı."]
    L += ["## Yorum (sayılardan otomatik üretildi)", ""] + [f"- {y}" for y in yorum]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ yazma ve çalıştırma
def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(data_path: Path = DATA_PATH, pred_path: Path = PRED_PATH, report_path: Path = REPORT_PATH,
        overwrite: bool = False, params: dict | None = None) -> dict:
    """Tüm akış. Çıktı dosyaları varsa ve overwrite yoksa FileExistsError (hiçbir şey çalıştırılmadan)."""
    for p in (pred_path, report_path):
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} zaten var; üzerine yazmak için --overwrite verin")
    if not lightgbm_available():
        raise RuntimeError("lightgbm kurulu değil (pip install lightgbm). Kurulum izni gerekir.")

    df = pd.read_parquet(data_path)
    prices = bl.price_table(df)
    bt = bl.run_backtest(prices)  # D-1, D-7, Kural (baselines.py)
    baseline_rows = len(bl.common_rows(bt))

    pred, imps, blocks = run_walkforward(df, bl.TEST_START, bl.TEST_END, params=params)
    bt_all = merge_predictions(bt, pred)
    common = common_rows_all(bt_all)
    caps = bl.month_caps(df)  # sadece değerlendirme etiketi için
    nan_info = nan_summary(df, bl.TEST_START, bl.TEST_END)

    report = build_report(bt_all, common, imps, blocks, caps, nan_info, baseline_rows, params or LGBM_PARAMS)
    out = pred[["ts", pf.TARGET, "tahmin_havasiz", "tahmin_havali", "blok_no"]]
    _write_parquet_atomic(out, pred_path)
    _write_text_atomic(report, report_path)
    tot = summarize_all(common)
    return {"rows": len(out), "common": len(common), "blocks": len(blocks), "totals": tot,
            "pred_path": pred_path, "report_path": report_path}


def _say(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LightGBM yürüyen pencere ilk deneme.")
    ap.add_argument("--overwrite", action="store_true", help="çıktı dosyaları varsa üzerine yaz")
    args = ap.parse_args(argv)
    try:
        r = run(overwrite=args.overwrite)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, AssertionError) as err:
        _say(f"DURDU: {type(err).__name__}: {str(err)[:500]}")
        return 1
    t = r["totals"]
    _say(f"Tamam: {r['rows']} tahmin satiri, {r['blocks']} blok, olcum satiri {r['common']}")
    _say(f"MAE D-1 {t['mae_d1']:.1f} | Kural {t['mae_kural']:.1f} | havasiz {t['mae_havasiz']:.1f} | havali {t['mae_havali']:.1f}")
    _say(f"Rapor: {r['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
