"""Fan chart (SALT-OKUNUR; model yok). Girdi: data/processed/egitim_oturumu/aralik_v2_tahminleri.parquet.

İki grafik: blok 7'den (zayıf kapsama) bir hafta -> reports/figures/fan_blok7.png, blok 11'den (iyi kapsama) bir hafta -> fan_blok11.png.
Her grafikte üst satır 56 günlük, alt satır 28 günlük varyant: gerçek fiyat, q50, %80 ve %90 CQR bantları (kırpılmamış).
Hafta = ilgili bloğun ilk 7 günü. Yalnızca pandas, numpy, matplotlib import eder; .venv-mlflow ile çalışacak şekilde yazıldı.
Çalıştırma: PYTHONUTF8=1 .venv-mlflow\\Scripts\\python.exe -m src.models.aralik_grafik
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "processed" / "egitim_oturumu" / "aralik_v2_tahminleri.parquet"
OUT_DIR = ROOT / "reports" / "figures"
JOBS = [(7, "fan_blok7.png", "zayıf kapsama"), (11, "fan_blok11.png", "iyi kapsama")]
VARIANTS = [("cal56", "kalibrasyon 56 gün"), ("cal28", "kalibrasyon 28 gün")]
C_Y, C_Q50, C_B80, C_B90 = "#222222", "#1f4e9c", "#4f86d6", "#a9c5ee"


def main() -> int:
    d = pd.read_parquet(IN_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for blok, fname, etiket in JOBS:
        b = d[d["blok_no"] == blok]
        start = b["ts"].min()
        win = b[b["ts"] < start + pd.Timedelta(days=7)]
        span = f"{win['ts'].min():%Y-%m-%d}..{win['ts'].max():%Y-%m-%d}"
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
        for ax, (v, vad) in zip(axes, VARIANTS):
            w = win[win["varyant"] == v].sort_values("ts")
            x = w["ts"].dt.tz_localize(None)
            ax.fill_between(x, w["lo90"], w["hi90"], color=C_B90, label="%90 aralık (CQR)")
            ax.fill_between(x, w["lo80"], w["hi80"], color=C_B80, label="%80 aralık (CQR)")
            ax.plot(x, w["q50"], color=C_Q50, lw=1.4, label="q50 (sıralanmış)")
            ax.plot(x, w["y"], color=C_Y, lw=1.2, label="gerçek fiyat")
            in80 = float(((w["y"] >= w["lo80"]) & (w["y"] <= w["hi80"])).mean()) * 100
            in90 = float(((w["y"] >= w["lo90"]) & (w["y"] <= w["hi90"])).mean()) * 100
            ax.set_title(f"Blok {blok} ({etiket}), {span}, {vad}   |   bu haftada kapsama: %80 -> {in80:.0f}%, %90 -> {in90:.0f}%", fontsize=10)
            ax.set_ylabel("PTF (TL/MWh)")
            ax.grid(alpha=0.25)
        axes[0].legend(loc="upper right", fontsize=8, ncol=4)
        fig.savefig(OUT_DIR / fname, dpi=130)
        plt.close(fig)
        print("yazıldı", fname, span)
    return 0


if __name__ == "__main__":
    sys.exit(main())
