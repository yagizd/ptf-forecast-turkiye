"""Aylık ham CSV'leri tek saatlik tabloya birleştirir ve kalite raporu üretir.

Girdi : data/raw/ptf/, data/raw/rt_cons/, data/raw/load_plan/ (aylık CSV'ler)
Çıktı : data/processed/dataset.parquet  ve  data/processed/kalite_raporu.md

Kural: eksik değerler ASLA doldurulmaz. Boş kalır (NaN).
Çalıştırma (repo kökünden): python -m src.data.build_dataset
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TZ = "Europe/Istanbul"  # EPİAŞ teslimat saatleri bu saat diliminde (UTC+3 sabit)

# çıktı sütunu -> (ham klasör, kaynaktaki değer sütunu, kaynaktaki saat sütunu)
SOURCES = {
    "price": ("ptf", "price", "hour"),
    "consumption": ("rt_cons", "consumption", "time"),
    "lep": ("load_plan", "lep", "time"),
}

# Bu veri kesiti (2021-01-01 .. 2026-10-20) için bizim beklentimiz.
# Veri büyüyünce bu sayılar eskir; rapor sadece gösterir, hata fırlatmaz.
EXPECTED = {"satır": 50856, "price eksik": 0, "consumption eksik": 0, "lep eksik": 24}


def read_monthly(folder: Path, value_col: str, hour_col: str, out_name: str):
    """Bir klasördeki aylık CSV'leri okur. (ts, out_name) tablosu ve saat uyumsuzluğu sayısı döner."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV bulunamadı: {folder}")
    raw = pd.concat(
        (pd.read_csv(f, usecols=["date", hour_col, value_col]) for f in files),
        ignore_index=True,
    )
    # 'date' = "2021-01-01T00:00:00+03:00" -> saat dilimli zaman damgası
    ts = pd.to_datetime(raw["date"], utc=True).dt.tz_convert(TZ)
    # Güvenlik kontrolü: kaynaktaki "00:00" gibi saat yazısı, ts'in saatiyle aynı mı?
    mismatch = int((ts.dt.strftime("%H:%M") != raw[hour_col].astype(str)).sum())
    table = pd.DataFrame({"ts": ts, out_name: raw[value_col].astype(float)})
    return table, mismatch


def read_all(raw_dir: Path = RAW_DIR):
    """Üç kaynağı okur. ({ad: tablo}, {ad: saat_uyumsuzluğu}) döner."""
    frames, mismatches = {}, {}
    for name, (folder, value_col, hour_col) in SOURCES.items():
        frames[name], mismatches[name] = read_monthly(raw_dir / folder, value_col, hour_col, name)
    return frames, mismatches


def combine(frames: dict):
    """(ts, değer) tablolarını tek saatlik tabloda birleştirir.

    - Aynı ts bir kaynakta iki kez varsa sonuncusu tutulur; kaç tane olduğu döner (rapora girer).
    - İlk ve son saat arasındaki HER saat için bir satır olur. Hiçbir kaynakta olmayan saat
      bile satır olarak yer alır, değerleri NaN kalır. Hiçbir şey doldurulmaz.
    Dönen: (tablo, {ad: kaynaktaki tekrar sayısı})
    """
    dups, merged = {}, None
    for name, f in frames.items():
        dups[name] = int(f.duplicated(subset="ts").sum())
        f = f.drop_duplicates(subset="ts", keep="last")
        merged = f if merged is None else merged.merge(f, on="ts", how="outer")
    grid = pd.date_range(merged["ts"].min(), merged["ts"].max(), freq="h")
    out = merged.set_index("ts").sort_index().reindex(grid)
    out.index.name = "ts"
    return out.reset_index(), dups


def _md_table(headers, rows) -> str:
    """Küçük Markdown tablosu (ek paket gerektirmesin diye elle yazıldı)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _missing_days(df: pd.DataFrame, col: str):
    """(tamamı eksik günler, kısmen eksik gün sayısı) döner."""
    local = df["ts"].dt.tz_localize(None)
    per_day = df[col].isna().groupby(local.dt.normalize()).sum()
    full = [d.strftime("%Y-%m-%d") for d in per_day.index[per_day == 24]]
    partial = int(((per_day > 0) & (per_day < 24)).sum())
    return full, partial


def quality_report(df: pd.DataFrame, dups: dict, mismatches: dict) -> str:
    """Kalite raporunu Markdown metni olarak üretir."""
    n = len(df)
    sirali = bool(df["ts"].is_monotonic_increasing)
    benzersiz = bool(df["ts"].is_unique)
    missing = {c: int(df[c].isna().sum()) for c in SOURCES}

    out = ["# Veri kalite raporu", ""]
    out += [
        "## 1. Genel",
        "",
        f"- Toplam satır: **{n}**",
        f"- İlk saat: {df['ts'].iloc[0]}  |  Son saat: {df['ts'].iloc[-1]}",
        f"- ts sıralı mı: {sirali}  |  her saat bir kez mi: {benzersiz}",
        f"- Tekrar eden ts (birleşik tabloda): {int(df['ts'].duplicated().sum())}",
        "",
    ]

    checks = [
        ("satır sayısı", EXPECTED["satır"], n),
        ("price eksik saat", EXPECTED["price eksik"], missing["price"]),
        ("consumption eksik saat", EXPECTED["consumption eksik"], missing["consumption"]),
        ("lep eksik saat", EXPECTED["lep eksik"], missing["lep"]),
    ]
    out += ["## 2. Beklenti kontrolü (bu veri kesiti için)", ""]
    out.append(_md_table(["kontrol", "beklenen", "gerçekleşen", "tamam mı"],
                         [(a, b, c, "evet" if b == c else "HAYIR") for a, b, c in checks]))
    out += ["", "## 3. Eksik değerler (hiçbiri doldurulmadı)", ""]
    rows = []
    for c in SOURCES:
        full, partial = _missing_days(df, c)
        shown = ", ".join(full[:10]) + (" ..." if len(full) > 10 else "")
        rows.append((c, missing[c], f"{100 * missing[c] / n:.3f}%", len(full), shown or "-", partial))
    out.append(_md_table(["sütun", "eksik saat", "oran", "tamamı eksik gün", "hangi günler", "kısmen eksik gün"], rows))

    out += ["", "## 4. Kaynak dosyalarda kontroller", ""]
    out.append(_md_table(
        ["kaynak", "tekrar eden ts (kaynakta)", "saat sütunu ile ts uyumsuz"],
        [(c, dups[c], mismatches[c]) for c in SOURCES],
    ))

    out += ["", "## 5. Fiyat: yıla göre", "",
            "'Yıl içi en yüksek' resmî tavan değil, o yıl veride görülen en yüksek fiyattır.", ""]
    rows = []
    for y, g in df.groupby(df["ts"].dt.year):
        p = g["price"]
        mx = p.max()
        rows.append((y, int(p.notna().sum()), int((p == 0).sum()), f"{p.mean():.1f}", mx, int((p == mx).sum())))
    out.append(_md_table(["yıl", "fiyatı olan saat", "fiyat 0 olan saat", "ortalama", "yıl içi en yüksek", "o fiyatta kaç saat"], rows))

    out += ["", "## 6. Fiyat: aya göre", "",
            "'Ayın en yüksek fiyatı' resmî tavan değil, o ay veride görülen en yüksek fiyattır.", ""]
    rows = []
    for ay, g in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        p = g["price"]
        mx = p.max()
        rows.append((ay, mx, int((p == mx).sum()), int((p == 0).sum())))
    out.append(_md_table(["ay", "ayın en yüksek fiyatı", "kaç saat o fiyatta", "fiyat 0 olan saat"], rows))
    return "\n".join(out) + "\n"


def main() -> None:
    frames, mismatches = read_all()
    df, dups = combine(frames)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    report = quality_report(df, dups, mismatches)
    (PROCESSED_DIR / "kalite_raporu.md").write_text(report, encoding="utf-8")
    print(report)

    try:
        df.to_parquet(PROCESSED_DIR / "dataset.parquet", index=False)
    except ImportError as err:
        raise RuntimeError("Parquet yazmak için 'pyarrow' paketi gerekli (henüz kurulu değil).") from err
    print(f"Yazıldı: {PROCESSED_DIR / 'dataset.parquet'}  ({len(df)} satır)")


if __name__ == "__main__":
    main()
