"""İndirilen hava dosyalarının kalite raporu -> data/processed/weather_kalite.md

Hava verisi Open-Meteo.com (CC BY 4.0), ticari olmayan kullanım.

Şehir x sütun için: toplam satır, null sayısı, İLK DOLU zaman, TAM DOLU başlangıç günü
(o günden sonra hiç null yok), ayrıca şehir başına tekrar eden 'time' ve eksik saat sayısı.
Hiçbir değer doldurulmaz. Şehirler arasında başlangıç farkı varsa raporda vurgulanır.
Çalıştırma (repo kökünden): python -m src.data.weather_quality
"""

from pathlib import Path

import pandas as pd

from src.data.weather import CITIES, HOURLY_VARS, RAW_DIR

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def read_city(city_dir: Path) -> pd.DataFrame:
    """Bir şehrin aylık CSV'lerini okur. 'time' sütunu zamana çevrilir (saat dilimi yok, yerel saat)."""
    files = sorted(city_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV yok: {city_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def column_stats(time: pd.Series, values: pd.Series) -> dict:
    """Tek sütun için: satır, null, ilk dolu zaman, tam dolu başlangıç günü.

    - ilk_dolu : ilk null-olmayan değerin zamanı (hiç yoksa None)
    - tam_dolu : o günden itibaren hiç null olmayan ilk gün (son null'ın ertesi günü).
                 Hiç null yoksa verinin ilk günü; son satır bile null ise None.
    - ilk_dolu_sonrasi_null : ilk dolu zamandan SONRAKİ satırlardaki null sayısı
                 (baştaki beklenen boşluğu ayırır; asıl önemli olan ortadaki delikler). Hiç dolu yoksa None.
    """
    n = len(values)
    is_null = values.isna()
    filled_times = time[~is_null]
    null_times = time[is_null]
    ilk_dolu = filled_times.min() if len(filled_times) else None
    if n == 0:
        tam_dolu = None
    elif len(null_times) == 0:
        tam_dolu = time.min().normalize()
    else:
        sonraki = null_times.max().normalize() + pd.Timedelta(days=1)
        tam_dolu = sonraki if sonraki <= time.max().normalize() else None
    sonrasi = None if ilk_dolu is None else int(is_null[time > ilk_dolu].sum())
    return {"satir": n, "null": int(is_null.sum()), "ilk_dolu": ilk_dolu, "tam_dolu": tam_dolu,
            "ilk_dolu_sonrasi_null": sonrasi}


def city_stats(df: pd.DataFrame) -> dict:
    """Şehir düzeyinde: satır, tekrar eden time, eksik saat, ilk/son zaman."""
    grid = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    eksik = len(grid.difference(pd.DatetimeIndex(df["time"].unique())))
    return {
        "satir": len(df),
        "tekrar": int(df["time"].duplicated().sum()),
        "eksik_saat": eksik,
        "ilk": df["time"].min(),
        "son": df["time"].max(),
        "null_toplam": int(df[HOURLY_VARS].isna().sum().sum()),
    }


def _fmt_ts(t) -> str:
    return "yok" if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def _fmt_day(t) -> str:
    return "yok" if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d")


def _md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def build_report(frames: dict, absent: list | None = None) -> str:
    """frames: {şehir: tablo}. absent: klasörü/CSV'si olmayan şehirler."""
    out = ["# Hava verisi kalite raporu", "",
           "Hava verisi Open-Meteo.com (CC BY 4.0), ticari olmayan kullanım. "
           "Değerler gerçekleşen hava değil, `_previous_day1/2` TAHMİNLERİDİR. Hiçbir null doldurulmadı.", ""]
    if absent:
        out += [f"**Dosyası olmayan şehirler:** {', '.join(absent)}", ""]

    cstats = {c: city_stats(df) for c, df in frames.items()}
    out += ["## 1. Şehir özeti", ""]
    out.append(_md_table(
        ["şehir", "satır", "ilk zaman", "son zaman", "tekrar eden time", "eksik saat", "toplam null"],
        [(c, s["satir"], _fmt_ts(s["ilk"]), _fmt_ts(s["son"]), s["tekrar"], s["eksik_saat"], s["null_toplam"])
         for c, s in cstats.items()],
    ))

    stats = {c: {v: column_stats(df["time"], df[v]) for v in HOURLY_VARS} for c, df in frames.items()}
    out += ["", "## 2. Şehir x sütun", "",
            "'ilk dolu' = ilk null olmayan değerin zamanı. 'tam dolu başlangıç' = o günden sonra hiç null yok. "
            "'ilk dolu sonrası null' = ilk dolu zamandan SONRAKİ satırlardaki null sayısı "
            "(baştaki boşluk beklenen; bu sayı ortadaki delikleri gösterir).", ""]
    rows = []
    for c in frames:
        for v in HOURLY_VARS:
            s = stats[c][v]
            sonra = "-" if s["ilk_dolu_sonrasi_null"] is None else s["ilk_dolu_sonrasi_null"]
            rows.append((c, v, s["satir"], s["null"], _fmt_ts(s["ilk_dolu"]), sonra, _fmt_day(s["tam_dolu"])))
    out.append(_md_table(["şehir", "sütun", "satır", "null", "ilk dolu", "ilk dolu sonrası null", "tam dolu başlangıç"], rows))

    out += ["", "## 3. Şehirler arası başlangıç farkı", ""]
    farklar = []
    for v in HOURLY_VARS:
        for anahtar, ad, fmt in (("ilk_dolu", "ilk dolu", _fmt_ts), ("tam_dolu", "tam dolu başlangıç", _fmt_day)):
            degerler = {c: fmt(stats[c][v][anahtar]) for c in frames}
            if len(set(degerler.values())) > 1:
                farklar.append(f"- **{v} / {ad}:** " + ", ".join(f"{c}={d}" for c, d in degerler.items()))
    if farklar:
        out += ["**⚠️ ŞEHİRLER ARASI FARK VAR:**", ""] + farklar
    else:
        out.append("Şehirler arasında başlangıç tarihi farkı yok.")
    return "\n".join(out) + "\n"


def main() -> None:
    frames, absent = {}, []
    for city in CITIES:
        try:
            frames[city] = read_city(RAW_DIR / city)
        except FileNotFoundError:
            absent.append(city)
    if not frames:
        raise RuntimeError(f"Hiç hava dosyası yok: {RAW_DIR}")
    report = build_report(frames, absent)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "weather_kalite.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
