"""Hava tahminlerini ana saatlik tabloya ekler -> data/processed/dataset_weather.parquet

Girdi : data/processed/dataset.parquet  (ts, price, consumption, lep; 50.856 satır)
        data/raw/weather/<sehir>/weather_<sehir>_YYYY-MM.csv  (8 şehir, _previous_day1/2 tahminleri)
Çıktı : data/processed/dataset_weather.parquet  ve  data/processed/dataset_weather_kalite.md

Hava verisi Open-Meteo.com (CC BY 4.0), ticari olmayan kullanım.

Kurallar:
- Eksik değerler (NaN) hiçbir şekilde doldurulmaz. Şehir ortalaması: tek şehirde bile NaN varsa sonuç NaN.
- Birleştirme ts üzerinden, SOL TARAF dataset.parquet (satır sayısı değişmez).
- price, consumption, lep eski dataset.parquet ile BİREBİR aynı olmalı (NaN'lar dahil); değilse yazmadan durulur.
- Modele sadece MODEL_FEATURES_PD2 girer (previous_day2). previous_day1 sütunları tabloda durur ama modele girmez.
Çalıştırma (repo kökünden): python -m src.data.build_weather_dataset   (üzerine yazmak için --overwrite)
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.data.weather import BASE_VARS, CITIES
from src.data.weather import RAW_DIR as WEATHER_DIR

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.parquet"
OUT_PATH = PROCESSED_DIR / "dataset_weather.parquet"
REPORT_PATH = PROCESSED_DIR / "dataset_weather_kalite.md"

TZ = "Europe/Istanbul"
EXPECTED_DATASET_ROWS = 50856  # dataset.parquet
EXPECTED_CITY_ROWS = 24576  # her şehir: 2024-01-01 .. 2026-10-20, 1024 gün x 24

# Değişkenler weather.py'deki BASE_VARS'tan gelir (sıra: sıcaklık, rüzgâr, güneş, bulut, yağış).
# Sıra ve adlar tests/test_weather_dataset.py içinde ayrıca denetlenir.
TEMP, WIND, SOLAR, CLOUD, PRECIP = BASE_VARS
WIND_CITY = "canakkale"  # rüzgâr: sadece Çanakkale
SOLAR_CITY = "konya"  # güneş: sadece Konya

BASE_COLUMNS = ["ts", "price", "consumption", "lep"]


def _src_col(var: str, day: int) -> str:
    """API'deki sütun adı. Örn: temperature_2m_previous_day2"""
    return f"{var}_previous_day{day}"


def _specs(day: int, suffix: str) -> dict:
    """çıktı sütunu -> (kaynak değişken, gün, şehir). şehir None ise 8 şehrin ortalaması."""
    return {
        f"temp_mean_{suffix}": (TEMP, day, None),
        f"cloud_mean_{suffix}": (CLOUD, day, None),
        f"precip_mean_{suffix}": (PRECIP, day, None),
        f"wind_{WIND_CITY}_{suffix}": (WIND, day, WIND_CITY),
        f"solar_{SOLAR_CITY}_{suffix}": (SOLAR, day, SOLAR_CITY),
    }


FEATURE_SPECS = {**_specs(2, "pd2"), **_specs(1, "pd1")}
# Modele girecek 5 sütun (SADECE previous_day2). pd1 sütunları burada OLMAMALI.
MODEL_FEATURES_PD2 = [c for c in FEATURE_SPECS if c.endswith("_pd2")]
# Tabloda durup modele girmeyecek 5 sütun (previous_day1).
PD1_COLUMNS = [c for c in FEATURE_SPECS if c.endswith("_pd1")]
WEATHER_COLUMNS = MODEL_FEATURES_PD2 + PD1_COLUMNS


# ------------------------------------------------------------------ hava dosyalarını okuma
def prepare_city_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Ham şehir tablosunu hazırlar: 'time' (saat dilimsiz yerel saat) -> 'ts' (Europe/Istanbul).

    Belirsiz/olmayan saat gelirse hata verir. Değerlere dokunulmaz.
    """
    value_cols = [_src_col(v, d) for v in BASE_VARS for d in (1, 2)]
    missing = [c for c in ["time", *value_cols] if c not in raw.columns]
    if missing:
        raise ValueError(f"hava tablosunda beklenen sütunlar yok: {missing}")
    t = pd.to_datetime(raw["time"])
    if t.dt.tz is not None:
        raise ValueError("'time' sütununda saat dilimi olmamalı (yerel saat bekleniyor)")
    ts = t.dt.tz_localize(TZ, ambiguous="raise", nonexistent="raise")
    df = raw[value_cols].copy()
    df.insert(0, "ts", ts)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def load_city_frames(weather_dir: Path = WEATHER_DIR):
    """8 şehrin aylık CSV'lerini okur. ({şehir: tablo}, {şehir: aylık dosya sayısı}) döner."""
    frames, counts = {}, {}
    for city in CITIES:
        files = sorted((weather_dir / city).glob("weather_*.csv"))
        if not files:
            raise FileNotFoundError(f"{city}: hava CSV'si yok ({weather_dir / city})")
        raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        frames[city] = prepare_city_frame(raw)
        counts[city] = len(files)
    return frames, counts


def validate_city_frames(frames: dict, counts: dict, expected_rows: int | None = EXPECTED_CITY_ROWS) -> None:
    """Şehir kontrolleri. Biri tutmazsa ValueError."""
    for needed in (WIND_CITY, SOLAR_CITY):
        if needed not in CITIES:
            raise ValueError(f"'{needed}' CITIES içinde yok")
    absent = [c for c in CITIES if c not in frames]
    if absent:
        raise ValueError(f"eksik şehir: {absent}")
    for city in CITIES:
        f = frames[city]
        if int(f["ts"].duplicated().sum()) > 0:
            raise ValueError(f"{city}: tekrar eden zaman var ({int(f['ts'].duplicated().sum())})")
        if expected_rows is not None and len(f) != expected_rows:
            raise ValueError(f"{city}: {len(f)} satır var, beklenen {expected_rows}")
    ref_city = next(iter(CITIES))
    ref = frames[ref_city]["ts"].reset_index(drop=True)
    for city in CITIES:
        if not ref.equals(frames[city]["ts"].reset_index(drop=True)):
            raise ValueError(f"{city}: zaman dizisi {ref_city} ile birebir aynı değil")
    if len({counts[c] for c in CITIES}) != 1:
        raise ValueError(f"şehirlerin aylık dosya sayıları farklı: { {c: counts[c] for c in CITIES} }")


# ------------------------------------------------------------------ özellikler ve birleştirme
def build_weather_features(frames: dict) -> pd.DataFrame:
    """10 hava sütununu üretir (index = ts).

    Şehir ortalaması: şehirler yan yana, mean(axis=1, skipna=False) -> tek şehirde NaN varsa sonuç NaN.
    """
    out = {}
    for name, (var, day, city) in FEATURE_SPECS.items():
        col = _src_col(var, day)
        if city is None:
            wide = pd.concat([frames[c].set_index("ts")[col].rename(c) for c in CITIES], axis=1)
            out[name] = wide.mean(axis=1, skipna=False)
        else:
            out[name] = frames[city].set_index("ts")[col]
    return pd.DataFrame(out)[WEATHER_COLUMNS]


def check_base(base: pd.DataFrame, expected_rows: int | None = EXPECTED_DATASET_ROWS) -> None:
    """dataset.parquet kontrolleri. Biri tutmazsa ValueError."""
    if list(base.columns) != BASE_COLUMNS:
        raise ValueError(f"dataset.parquet sütunları beklenenden farklı: {list(base.columns)}")
    if str(base["ts"].dt.tz) != TZ:
        raise ValueError(f"dataset.parquet ts saat dilimi {TZ} değil: {base['ts'].dt.tz}")
    if base["ts"].duplicated().any():
        raise ValueError("dataset.parquet'te ts tekrarı var")
    if expected_rows is not None and len(base) != expected_rows:
        raise ValueError(f"dataset.parquet {len(base)} satır, beklenen {expected_rows}")


def merge_weather(base: pd.DataFrame, features: pd.DataFrame):
    """ts üzerinden LEFT JOIN (sol: base). Satır sayısı değişirse hata. (tablo, bilgi) döner.

    bilgi['outside']       : hava tarafında olup base'de karşılığı olmayan saat (dışarıda kalan)
    bilgi['unmatched_base']: base'de olup hava tarafında karşılığı olmayan saat (hava sütunları NaN kalır)
    """
    feat = features.copy()
    feat.index = feat.index.astype(base["ts"].dtype)  # ns/us farkı birleştirmeyi bozmasın
    feat = feat.rename_axis("ts").reset_index()
    outside = int((~feat["ts"].isin(base["ts"])).sum())
    unmatched_base = int((~base["ts"].isin(feat["ts"])).sum())
    out = base.merge(feat, on="ts", how="left", validate="one_to_one")
    if len(out) != len(base):
        raise ValueError(f"birleştirme satır sayısını değiştirdi: {len(base)} -> {len(out)}")
    return out[BASE_COLUMNS + WEATHER_COLUMNS], {"outside": outside, "unmatched_base": unmatched_base}


def check_unchanged(base: pd.DataFrame, out: pd.DataFrame) -> None:
    """ts, price, consumption, lep eski tabloyla BİREBİR aynı mı (NaN'lar dahil, tam eşitlik)."""
    for c in BASE_COLUMNS:
        pd.testing.assert_series_equal(
            base[c].reset_index(drop=True), out[c].reset_index(drop=True), check_exact=True)


# ------------------------------------------------------------------ rapor için ölçümler
def first_filled(ts: pd.Series, s: pd.Series):
    """İlk null-olmayan değerin zamanı (yoksa None)."""
    filled = ts[s.notna()]
    return filled.min() if len(filled) else None


def null_runs_after_first(ts: pd.Series, s: pd.Series) -> list:
    """İlk dolu zamandan SONRAKİ ardışık null aralıkları: [(başlangıç, bitiş, saat sayısı), ...]"""
    first = first_filled(ts, s)
    if first is None:
        return []
    t = ts[s.isna() & (ts > first)]
    if t.empty:
        return []
    grp = (t.diff() != pd.Timedelta(hours=1)).cumsum()
    return [(g.iloc[0], g.iloc[-1], len(g)) for _, g in t.groupby(grp)]


def column_stats(ts: pd.Series, s: pd.Series) -> dict:
    first = first_filled(ts, s)
    after = 0 if first is None else int((s.isna() & (ts > first)).sum())
    return {"null": int(s.isna().sum()), "first": first, "after": after,
            "runs": null_runs_after_first(ts, s)}


def check_expectations(out: pd.DataFrame) -> list:
    """Yağız'ın varsayımlarını ölçer. [(beklenti, ölçülen, uyuştu_mu)] döner. Sonuç güzelleştirilmez."""
    ts = out["ts"]
    res = []
    for col in WEATHER_COLUMNS:
        wind_solar = col.startswith(("wind_", "solar_"))
        ym = (2024, 3) if wind_solar else (2024, 2)
        first = first_filled(ts, out[col])
        ok = first is not None and (first.year, first.month) == ym and first.day <= 10
        res.append((f"{col}: ilk dolu {ym[0]}-{ym[1]:02d} başında (gün <= 10)",
                    "yok" if first is None else str(first), ok))
    exp = [(pd.Timestamp("2026-04-17 18:00", tz=TZ), pd.Timestamp("2026-04-23 02:00", tz=TZ), 129)]
    runs = null_runs_after_first(ts, out["cloud_mean_pd2"])
    res.append(("cloud_mean_pd2: ortada tek delik 2026-04-17 18:00 -> 2026-04-23 02:00 (129 saat)",
                "; ".join(f"{a} -> {b} ({n} saat)" for a, b, n in runs) or "delik yok", runs == exp))
    for col in WEATHER_COLUMNS:
        if col == "cloud_mean_pd2":
            continue
        runs = null_runs_after_first(ts, out[col])
        res.append((f"{col}: ilk doludan sonra null yok", f"{sum(n for _, _, n in runs)} null", not runs))
    return res


def _md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def build_report(out: pd.DataFrame, info: dict, sha256: str) -> str:
    ts = out["ts"]
    L = ["# Hava eklenmiş veri seti kalite raporu", "",
         "Hava verisi Open-Meteo.com (CC BY 4.0), ticari olmayan kullanım. Hava sütunları gerçekleşen hava değil, "
         "`_previous_day1/2` TAHMİNLERİDİR. Hiçbir NaN doldurulmadı.", "",
         "## 1. Genel", "",
         f"- Çıktı dosyası SHA-256: `{sha256}`",
         f"- Satır sayısı: **{len(out)}** (dataset.parquet: {info['base_rows']})",
         f"- Sütunlar: {', '.join(out.columns)}",
         f"- Hava verisinin zaman aralığı: {info['weather_first']} .. {info['weather_last']}",
         "", "## 2. price / consumption / lep eşitlik kontrolü", "",
         f"- Sonuç: **{info['equality']}** (assert_series_equal, check_exact=True; ts dahil, NaN'lar dahil)",
         f"- lep NaN sayısı (çıktıda): {int(out['lep'].isna().sum())}", "",
         "## 3. Birleştirme (LEFT JOIN, sol: dataset.parquet)", "",
         f"- Dışarıda kalan hava saati (hava tarafında var, dataset'te yok): **{info['outside']}**",
         f"- Hava tarafında karşılığı olmayan dataset saati (hava sütunları NaN): **{info['unmatched_base']}** "
         "(hava 2024-01-01'de başlıyor; öncesi beklenen şekilde NaN)", "",
         "## 4. Şehir kontrolleri", ""]
    L.append(_md_table(["şehir", "satır", "aylık dosya sayısı"],
                       [(c, info["city_rows"][c], info["city_files"][c]) for c in CITIES]))
    L += ["", "## 5. Hava sütunları", "",
          "'ilk dolu sonrası null' = ilk dolu zamandan SONRAKİ satırlardaki null sayısı. Toplam null, "
          "hava verisi öncesindeki satırları da içerir.", ""]
    rows = []
    for col in WEATHER_COLUMNS:
        st = column_stats(ts, out[col])
        rows.append((col, "evet" if col in MODEL_FEATURES_PD2 else "hayır (pd1)", st["null"],
                     "yok" if st["first"] is None else st["first"], st["after"]))
    L.append(_md_table(["sütun", "modele girer mi", "toplam null", "ilk dolu ts", "ilk dolu sonrası null"], rows))
    L += ["", "### İlk dolu sonrası null aralıkları", ""]
    any_run = False
    for col in WEATHER_COLUMNS:
        for a, b, n in null_runs_after_first(ts, out[col]):
            any_run = True
            L.append(f"- {col}: {a} -> {b} ({n} saat)")
    if not any_run:
        L.append("Hiçbir sütunda yok.")
    exp = check_expectations(out)
    L += ["", "## 6. Beklenti kontrolü (varsayım; sonuç güzelleştirilmedi)", ""]
    L.append(_md_table(["beklenti", "ölçülen", "sonuç"],
                       [(a, b, "UYUŞUYOR" if ok else "UYUŞMUYOR") for a, b, ok in exp]))
    bad = [a for a, _, ok in exp if not ok]
    L += ["", ("Tüm beklentiler uyuştu." if not bad else
               f"**UYUŞMAYAN beklenti sayısı: {len(bad)}** (yukarıdaki tabloda 'UYUŞMUYOR' olanlar).")]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ yazma
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_parquet_atomic(df: pd.DataFrame, path: Path, overwrite: bool = False) -> None:
    """Önce .tmp'ye yazar, sonra yer değiştirir. Dosya varsa ve overwrite yoksa FileExistsError."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} zaten var; üzerine yazmak için --overwrite verin")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run(
    base_path: Path = DATASET_PATH,
    weather_dir: Path = WEATHER_DIR,
    out_path: Path = OUT_PATH,
    report_path: Path = REPORT_PATH,
    overwrite: bool = False,
    expected_dataset_rows: int | None = EXPECTED_DATASET_ROWS,
    expected_city_rows: int | None = EXPECTED_CITY_ROWS,
) -> dict:
    """Tüm akış. Herhangi bir kontrol tutmazsa istisna fırlatır ve HİÇBİR çıktı yazılmaz."""
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} zaten var; üzerine yazmak için --overwrite verin")

    base = pd.read_parquet(base_path)
    check_base(base, expected_dataset_rows)
    frames, counts = load_city_frames(weather_dir)
    validate_city_frames(frames, counts, expected_city_rows)

    features = build_weather_features(frames)
    out, merge_info = merge_weather(base, features)
    check_unchanged(base, out)  # farklıysa AssertionError -> yazmadan durur

    write_parquet_atomic(out, out_path, overwrite)
    sha = _sha256(out_path)

    info = {
        **merge_info,
        "base_rows": len(base),
        "equality": "BİREBİR AYNI",
        "weather_first": features.index.min(),
        "weather_last": features.index.max(),
        "city_rows": {c: len(frames[c]) for c in CITIES},
        "city_files": counts,
    }
    report = build_report(out, info, sha)
    _write_text_atomic(report, report_path)  # önce dosyaya, ekrana sonra
    return {"rows": len(out), "sha256": sha, "report": report_path, "out": out_path, **merge_info,
            "expectations_bad": [a for a, _, ok in check_expectations(out) if not ok]}


def _say(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:  # konsol Türkçe harfte çökmesin
        print(msg.encode("ascii", "replace").decode("ascii"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hava tahminlerini ana tabloya ekler.")
    ap.add_argument("--overwrite", action="store_true", help="dataset_weather.parquet varsa üzerine yaz")
    args = ap.parse_args(argv)
    try:
        r = run(overwrite=args.overwrite)
    except (ValueError, FileExistsError, FileNotFoundError, AssertionError) as err:
        _say(f"DURDU: {type(err).__name__}: {str(err)[:500]}")
        return 1
    _say(f"Yazıldı: {r['out']} ({r['rows']} satır)")
    _say(f"SHA-256: {r['sha256']}")
    _say(f"Rapor: {r['report']}")
    _say(f"Dışarıda kalan hava saati: {r['outside']} | hava karşılığı olmayan dataset saati: {r['unmatched_base']}")
    _say(f"Uyuşmayan beklenti sayısı: {len(r['expectations_bad'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
