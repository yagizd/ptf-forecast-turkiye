"""Kıyas (baseline) modelleri ve walk-forward backtest.

Walk-forward: geçmişte her gün sanki o günmüşüz gibi tahmin yaparız.
D gününü tahmin ederken SADECE D-1 ve öncesinin PTF'sine bakılır.
consumption ve lep (yük tahmini) kullanılmaz: yayın saatlerini bilmiyoruz, sızıntı riski var.

Girdi : data/processed/dataset.parquet
Çıktı : reports/baselines.md  ve  reports/baselines_gunluk.csv
Çalıştırma (repo kökünden): python -m src.models.baselines
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "dataset.parquet"
REPORTS_DIR = ROOT / "reports"

# Test dönemi: son 12 ay. Kıyas modellerinde ayarlanacak parametre yok, bu yüzden
# test dönemine bakıp ayar yapma tehlikesi yok.
TEST_START = pd.Timestamp("2025-10-21")
TEST_END = pd.Timestamp("2026-10-20")

MODELS = {"d1": "tahmin_d1", "d7": "tahmin_d7", "kural": "tahmin_kural"}
MODEL_NAMES = {
    "d1": "1. Dünün aynı saati (D-1)",
    "d7": "2. Geçen haftanın aynı günü (D-7)",
    "kural": "3. Kural (Sal-Cum: D-1, Pzt/Cmt/Paz: D-7)",
}


# ------------------------------------------------------------------ veri hazırlığı
def price_table(df: pd.DataFrame) -> pd.DataFrame:
    """Saatlik tablodan 'gün x 24 saat' fiyat tablosu üretir. Sadece price kullanılır."""
    local = df["ts"].dt.tz_localize(None)  # yerel duvar saati (UTC+3), saat dilimi bilgisi atılır
    t = pd.DataFrame({
        "gun": local.dt.normalize(),
        "saat": local.dt.hour,
        "price": df["price"].to_numpy(),
    })
    wide = t.pivot(index="gun", columns="saat", values="price").reindex(columns=range(24))
    # Eksik gün olursa satır NaN olarak görünsün (kaymasın)
    return wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))


# ------------------------------------------------------------------ kıyas modelleri
def _check_history(history: pd.DataFrame, d: pd.Timestamp) -> None:
    """ZAMAN KURALI: geçmiş tablosunda D günü veya sonrası ASLA olmamalı."""
    assert len(history) == 0 or history.index.max() < d, (
        f"ZAMAN KURALI İHLALİ: {d.date()} tahmin edilirken {history.index.max().date()} günü görünüyor"
    )


def _pick(history: pd.DataFrame, d: pd.Timestamp, lag_days: int) -> pd.Series:
    """D-lag gününün 24 saatlik fiyatı. Gün yoksa 24 tane NaN (doldurma yok)."""
    day = d - pd.Timedelta(days=lag_days)
    if day in history.index:
        return pd.Series(history.loc[day].to_numpy(dtype=float), index=range(24))
    return pd.Series(np.nan, index=range(24))


def forecast_d1(history: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
    """1. model: dünün aynı saati."""
    _check_history(history, d)
    return _pick(history, d, 1)


def forecast_d7(history: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
    """2. model: geçen haftanın aynı günü ve saati."""
    _check_history(history, d)
    return _pick(history, d, 7)


def forecast_kural(history: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
    """3. model: Salı-Cuma D-1; Pazartesi, Cumartesi, Pazar D-7."""
    _check_history(history, d)
    # dayofweek: Pazartesi=0 ... Pazar=6. Salı..Cuma = 1..4
    return _pick(history, d, 1 if d.dayofweek in (1, 2, 3, 4) else 7)


FORECASTERS = {"d1": forecast_d1, "d7": forecast_d7, "kural": forecast_kural}


# ------------------------------------------------------------------ walk-forward
def forecast_day(prices: pd.DataFrame, d: pd.Timestamp) -> dict:
    """D günü için 3 modelin 24 saatlik tahmini. Modele SADECE D'den önceki günler verilir."""
    history = prices.loc[prices.index < d]  # D ve sonrası kesilir
    _check_history(history, d)
    return {name: f(history, d) for name, f in FORECASTERS.items()}


def run_backtest(prices: pd.DataFrame, start=TEST_START, end=TEST_END) -> pd.DataFrame:
    """start..end her gün için tahmin yap. Uzun tablo döner: gun, saat, gercek, tahmin_*."""
    parts = []
    for d in pd.date_range(start, end, freq="D"):
        preds = forecast_day(prices, d)
        actual = prices.loc[d].to_numpy(dtype=float) if d in prices.index else np.full(24, np.nan)
        part = pd.DataFrame({"gun": d, "saat": range(24), "gercek": actual})
        for name, col in MODELS.items():
            part[col] = preds[name].to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ hata ölçüleri
def common_rows(bt: pd.DataFrame) -> pd.DataFrame:
    """Gerçek ve 3 tahminin hepsi dolu olan satırlar (modeller aynı satırlarda kıyaslanır)."""
    return bt.dropna(subset=["gercek", *MODELS.values()])


def summarize(sub: pd.DataFrame) -> dict:
    """n, her modelin MAE'si ve rMAE (model MAE / D-1 MAE; 1'in altı = D-1'i yendi)."""
    mae = {m: float((sub[c] - sub["gercek"]).abs().mean()) for m, c in MODELS.items()}
    base = mae["d1"]
    out = {"n": int(len(sub)), **{f"mae_{m}": v for m, v in mae.items()}}
    for m in ("d7", "kural"):
        out[f"rmae_{m}"] = mae[m] / base if base > 0 else float("nan")
    return out


def group_summary(common: pd.DataFrame, key: pd.Series, key_name: str) -> pd.DataFrame:
    """key'e göre gruplayıp her grup için summarize."""
    rows = [{key_name: k, **summarize(g)} for k, g in common.groupby(key.loc[common.index], sort=True)]
    return pd.DataFrame(rows)


MIN_CAP_HOURS = 5  # bir fiyatın 'tavan' sayılması için ayda en az bu kadar saat tekrar etmesi gerekir


def month_caps(df: pd.DataFrame) -> pd.Series:
    """Her ay için 'tavan' fiyatı. Index: 'YYYY-MM'.

    Tavan = o ayın en yüksek fiyatı, ama SADECE o ayda tam o fiyatta en az 5 saat varsa.
    Yoksa NaN (o ay için tavan yok). Resmî tavan değildir; veriden çıkarılan bir göstergedir.
    Bu sadece DEĞERLENDİRME içindir, tahmin yapılırken kullanılmaz.
    """
    caps = {}
    for ay, g in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        mx = g["price"].max()
        caps[ay] = mx if (g["price"] == mx).sum() >= MIN_CAP_HOURS else np.nan
    return pd.Series(caps, dtype=float)


def label_special(common: pd.DataFrame, caps: pd.Series) -> pd.Series:
    """Her saati etiketle: fiyat 0 / ayın tavanında / diğer.

    'Tavanda' = fiyat, o ayın tavan fiyatına (month_caps) eşit. NaN tavanlı aylarda hiç 'Tavanda' olmaz.
    """
    cap = common["gun"].dt.strftime("%Y-%m").map(caps)
    lab = np.where(common["gercek"] == 0, "Fiyat 0",
                   np.where(common["gercek"] == cap, "Tavanda", "Diğer"))
    return pd.Series(lab, index=common.index)


# ------------------------------------------------------------------ rapor
def _md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _table_from(df: pd.DataFrame, first_col: str, first_header: str) -> str:
    rows = [
        (r[first_col], r["n"], f"{r['mae_d1']:.1f}", f"{r['mae_d7']:.1f}", f"{r['mae_kural']:.1f}",
         f"{r['rmae_d7']:.3f}", f"{r['rmae_kural']:.3f}")
        for r in df.to_dict("records")  # iterrows tamsayıları float yapıyordu (0 -> 0.0); records türü korur
    ]
    return _md_table([first_header, "saat sayısı", "MAE D-1", "MAE D-7", "MAE Kural", "rMAE D-7", "rMAE Kural"], rows)


def build_report(bt: pd.DataFrame, caps: pd.Series) -> str:
    common = common_rows(bt)
    tot = summarize(common)
    gun = common["gun"]
    aylik = group_summary(common, gun.dt.strftime("%Y-%m"), "ay")
    saatlik = group_summary(common, common["saat"], "saat")
    haftalik = group_summary(common, pd.Series(np.where(gun.dt.dayofweek < 5, "Hafta içi", "Hafta sonu"), index=common.index), "grup")
    ozel = group_summary(common, label_special(common, caps), "saat türü")

    best = min(MODELS, key=lambda m: tot[f"mae_{m}"])
    L = ["# Kıyas modelleri (baseline) sonuçları", ""]
    L += [
        f"Test dönemi: {TEST_START.date()} .. {TEST_END.date()} ({bt['gun'].nunique()} gün, {len(bt)} saat). "
        f"Üç modelin de tahmini ve gerçek fiyatı dolu olan {tot['n']} saat değerlendirildi "
        f"({len(bt) - tot['n']} saat boş olduğu için dışarıda kaldı).",
        "",
        "MAE = ortalama kaç TL/MWh yanıldık. rMAE = modelin MAE'si ÷ D-1'in MAE'si (1'in altı = D-1'i yendi).",
        "",
        "## Toplam",
        "",
        _md_table(["model", "MAE (TL/MWh)", "rMAE"],
                  [(MODEL_NAMES["d1"], f"{tot['mae_d1']:.1f}", "1.000"),
                   (MODEL_NAMES["d7"], f"{tot['mae_d7']:.1f}", f"{tot['rmae_d7']:.3f}"),
                   (MODEL_NAMES["kural"], f"{tot['mae_kural']:.1f}", f"{tot['rmae_kural']:.3f}")]),
        "",
        "## Hafta içi / hafta sonu", "", _table_from(haftalik, "grup", "grup"), "",
        "## Aylık", "", _table_from(aylik, "ay", "ay"), "",
        "## Saate göre (0-23)", "", _table_from(saatlik, "saat", "saat"), "",
        "## Özel saatler (fiyat 0 veya ayın tavanında)", "",
        f"'Tavanda' = fiyat, o ayın en yüksek fiyatına eşit VE o ayda tam o fiyatta en az {MIN_CAP_HOURS} saat var. "
        "Bu resmî tavan değil, ayın veriden çıkan en yüksek fiyatıdır (tüm veri setinden hesaplandı, "
        "sadece değerlendirme içindir). 'Diğer' = ikisi de değil.", "",
        _table_from(ozel, "saat türü", "saat türü"), "",
    ]

    # Yorum: cümleler sayılardan üretilir, elle yazılmış iddia yok.
    yorum = [
        f"Toplam MAE: D-1 {tot['mae_d1']:.1f}, D-7 {tot['mae_d7']:.1f}, Kural {tot['mae_kural']:.1f} TL/MWh. "
        f"En düşük hata: {MODEL_NAMES[best]}.",
    ]
    for m, ad in (("d7", "D-7"), ("kural", "Kural")):
        r = tot[f"rmae_{m}"]
        yorum.append(f"{ad} için rMAE {r:.3f}: " + ("D-1'i yendi." if r < 1 else "D-1'i yenemedi."))
    wk = haftalik.set_index("grup")
    if {"Hafta içi", "Hafta sonu"} <= set(wk.index):
        yorum.append(
            "Hafta sonu en iyi model "
            f"{MODEL_NAMES[min(MODELS, key=lambda m: wk.loc['Hafta sonu', f'mae_{m}'])]}, "
            "hafta içi "
            f"{MODEL_NAMES[min(MODELS, key=lambda m: wk.loc['Hafta içi', f'mae_{m}'])]}."
        )
    sp = ozel.set_index("saat türü")
    if "Diğer" in sp.index and len(sp.index) > 1:
        ozel_n = int(sp.loc[[i for i in sp.index if i != "Diğer"], "n"].sum())
        yorum.append(
            f"D-1'in MAE'si diğer saatlerde {sp.loc['Diğer', 'mae_d1']:.1f}; "
            + "; ".join(f"'{i}' saatlerinde {sp.loc[i, 'mae_d1']:.1f} ({int(sp.loc[i, 'n'])} saat)"
                        for i in sp.index if i != "Diğer")
            + f". Özel saat toplamı {ozel_n}."
        )
    yorum += [
        "Bayram günleri ayrıca ele alınmadı; tatil etkisi hatalara karışmış olabilir.",
        "consumption ve lep (yük tahmini) bu modellerde kullanılmadı. Sebep: yayın saatlerini bilmiyoruz, "
        "tahmin anında elimizde olmayabilirler (sızıntı riski).",
        "Zaman kuralı: D günü tahmin edilirken sadece D-1 ve öncesinin fiyatı görüldü (kodda assert ve testle denetleniyor).",
    ]
    L += ["## Yorum", ""] + [f"- {y}" for y in yorum]
    return "\n".join(L) + "\n"


def main() -> None:
    try:
        df = pd.read_parquet(DATASET)
    except ImportError as err:
        raise RuntimeError("Parquet okumak için 'pyarrow' paketi gerekli (henüz kurulu değil).") from err
    except FileNotFoundError as err:
        raise RuntimeError(f"{DATASET} yok. Önce: python -m src.data.build_dataset") from err

    prices = price_table(df)  # sadece price sütunu kullanılır
    caps = month_caps(df)  # sadece değerlendirme için

    bt = run_backtest(prices)
    n_days = (TEST_END - TEST_START).days + 1
    assert len(bt) == n_days * 24, "backtest satır sayısı beklenenden farklı"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = bt.copy()
    out["gun"] = out["gun"].dt.strftime("%Y-%m-%d")
    out.to_csv(REPORTS_DIR / "baselines_gunluk.csv", index=False, encoding="utf-8")
    report = build_report(bt, caps)
    (REPORTS_DIR / "baselines.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
