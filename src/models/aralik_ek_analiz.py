"""V aralık tahminleri için SADECE AÇIKLAYICI ek analiz (kırpılmamış aralık). Model eğitimi yok, ağ yok.

Girdi : data/processed/egitim_oturumu/aralik_v_tahminleri.parquet (yalnızca okunur)
Çıktı : reports/aralik_dogrulama_ek.md
Çalıştırma: PYTHONUTF8=1 python -m src.models.aralik_ek_analiz
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "processed" / "egitim_oturumu" / "aralik_v_tahminleri.parquet"
OUT_PATH = ROOT / "reports" / "aralik_dogrulama_ek.md"
LAST_BLOCK = 14


def md(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x) -> str:
    return f"{100 * x:.1f}%"


def group_table(d: pd.DataFrame, key: pd.Series, name: str, with_width: bool = False, mark_small: bool = False) -> str:
    rows = []
    for k, g in d.groupby(key, sort=True):
        row = [k, len(g), pct(g["ic"].mean())]
        if with_width:
            row.append(f"{g['genislik'].mean():.1f}")
        rows.append(row)
    return md([name, "n", "kapsama"] + (["ortalama genişlik"] if with_width else []), rows)


def main() -> int:
    d = pd.read_parquet(IN_PATH)
    d = d[d["y"].notna()].copy()
    d["ic"] = (d["y"] >= d["lo"]) & (d["y"] <= d["hi"])
    d["alt"] = d["y"] < d["lo"]
    d["ust"] = d["y"] > d["hi"]
    d["genislik"] = d["hi"] - d["lo"]
    assert (d["ic"].astype(int) + d["alt"].astype(int) + d["ust"].astype(int) == 1).all()

    hour = d["ts"].dt.hour
    wk = pd.Series(np.where(d["ts"].dt.dayofweek < 5, "Hafta içi", "Hafta sonu"), index=d.index)
    grp = pd.Series(np.where(d["y"] == 0, "y = 0", np.where(d["y"] == d["tavan"], "y = günün tavanı", "diğer")), index=d.index)
    ay = d["ts"].dt.strftime("%Y-%m")

    rows_e = []
    for no, g in d.groupby("blok_no"):
        rows_e.append((int(no), len(g) if no != LAST_BLOCK else f"{len(g)} (n küçük, yorumlanmaz)",
                       pct(g["alt"].mean()), pct(g["ust"].mean()), pct(g["ic"].mean())))

    L = ["Bu tablolar açıklayıcıdır. Bunlara bakarak model, parametre veya kural DEĞİŞTİRİLMEZ.", "",
         "# Aralık doğrulaması: ek analiz (V, kırpılmamış aralık)", "",
         f"Kaynak: {IN_PATH.name}; satır {len(d)}. Hepsi açıklayıcıdır, karar değildir.",
         f"Blok {LAST_BLOCK} yalnızca 1 gün (24 saat): n küçük, yorumlanmaz.", "",
         "## a) Saate göre kapsama ve ortalama genişlik", "", group_table(d, hour, "saat", with_width=True), "",
         "## b) Hafta içi / hafta sonu", "", group_table(d, wk, "grup"), "",
         "## c) Fiyat grubuna göre kapsama", "", group_table(d, grp, "grup"), "",
         "## d) Aya göre kapsama", "", group_table(d, ay, "ay"), "",
         "## e) Iskalama yönü", "",
         f"Toplam: aralığın altında (y < lo) {pct(d['alt'].mean())}; üstünde (y > hi) {pct(d['ust'].mean())}; içinde {pct(d['ic'].mean())}.", "",
         md(["blok", "n", "altında (y < lo)", "üstünde (y > hi)", "içinde"], rows_e), ""]
    OUT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("yazıldı", OUT_PATH.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
