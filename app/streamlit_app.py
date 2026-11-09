"""PTF tahmin paneli (Streamlit): YALNIZ saklanmış hazır sonuçları gösterir.

Ne yapmaz: EPİAŞ'a bağlanmaz, .env okumaz, model eğitmez, tahmin üretmez.
Ne okur (hepsi test dönemi T = 2025-10-21..2026-10-20, 8760 saat):
  - data/processed/final_tahminleri.parquet               : ts, price, tahmin_final, blok_no   (final nokta tahmini)
  - data/processed/oturum_T_final/aralik_T_tahminleri.parquet : ts, y, varyant, tavan, lo80, hi80, lo90, hi90 (B3 aralığı)
  - reports/baselines_gunluk.csv                           : gun, saat, gercek, tahmin_d1, tahmin_d7, tahmin_kural
Bu dosyalar GitHub'da YOK (.gitignore); yoksa panel açık bir mesaj gösterir.
T daha önce bir kez görüldü: temiz sınav değil; tek dönem, anlamlılık sınanmadı.

Çalıştırma (ayrı ortam .venv-app): python -m streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = ROOT / "data" / "processed" / "final_tahminleri.parquet"
ARALIK_PATH = ROOT / "data" / "processed" / "oturum_T_final" / "aralik_T_tahminleri.parquet"
BASE_PATH = ROOT / "reports" / "baselines_gunluk.csv"

st.set_page_config(page_title="PTF tahmin paneli (T dönemi)", layout="wide")
st.title("PTF tahmin paneli: saklanmış sonuçlar (test dönemi T)")
st.caption(
    "Bu panel canlı DEĞİLDİR ve tahmin üretmez; yalnız daha önce hesaplanıp saklanmış sonuçları gösterir. "
    "T dönemi daha önce bir kez görüldü: temiz bir son sınav değildir. Tek dönem, tek koşu; anlamlılık sınanmadı."
)

# ---------------------------------------------------------------- veri var mı?
eksik = [str(p.relative_to(ROOT)) for p in (FINAL_PATH, ARALIK_PATH, BASE_PATH) if not p.is_file()]
if eksik:
    st.error(
        "Veri dosyaları bulunamadı. Bu dosyalar GitHub'da YOKTUR (veri paylaşım şartları belli değil), "
        "yalnız geliştiricinin bilgisayarında bulunur."
    )
    st.write("Eksik dosyalar:")
    for e in eksik:
        st.code(e)
    st.stop()


@st.cache_data
def yukle():
    f = pd.read_parquet(FINAL_PATH, columns=["ts", "price", "tahmin_final"])
    a = pd.read_parquet(ARALIK_PATH, columns=["ts", "y", "varyant", "tavan", "lo80", "hi80", "lo90", "hi90"])
    a = a[a["varyant"] == "B3"]
    b = pd.read_csv(BASE_PATH)
    f = f.sort_values("ts").reset_index(drop=True)
    a = a.sort_values("ts").reset_index(drop=True)
    b = b.reset_index(drop=True)
    loc = f["ts"].dt.tz_localize(None)  # yerel duvar saati
    d = pd.DataFrame({
        "gun": loc.dt.normalize(),
        "saat": loc.dt.hour,
        "gercek": f["price"].to_numpy(),
        "final": f["tahmin_final"].to_numpy(),
    })
    uyum = (
        len(f) == len(a) == len(b)
        and bool((f["ts"].to_numpy() == a["ts"].to_numpy()).all())
        and bool(((f["price"] - a["y"]).abs() < 1e-9).all())
        and bool((pd.to_datetime(b["gun"]).to_numpy() == d["gun"].to_numpy()).all())
        and bool((b["saat"].to_numpy() == d["saat"].to_numpy()).all())
        and bool(((b["gercek"] - f["price"]).abs() < 1e-9).all())
    )
    d["d1"] = b["tahmin_d1"].to_numpy()
    d["d7"] = b["tahmin_d7"].to_numpy()
    d["kural"] = b["tahmin_kural"].to_numpy()
    d["lo80"] = a["lo80"].clip(lower=0.0).to_numpy()  # raporlardaki gibi [0, günün tavanı] kırpması
    d["hi80"] = a[["hi80", "tavan"]].min(axis=1).to_numpy()
    d["lo90"] = a["lo90"].clip(lower=0.0).to_numpy()
    d["hi90"] = a[["hi90", "tavan"]].min(axis=1).to_numpy()
    return d, uyum


d, uyum = yukle()
if not uyum:
    st.warning("Uyarı: üç dosyanın satırları birbiriyle uyuşmuyor (zaman, saat ya da gerçek fiyat). Sonuçlara güvenmeyin.")


def ozet(x: pd.DataFrame) -> dict:
    out = {
        "MAE final (TL/MWh)": (x["gercek"] - x["final"]).abs().mean(),
        "MAE D-1": (x["gercek"] - x["d1"]).abs().mean(),
        "MAE D-7": (x["gercek"] - x["d7"]).abs().mean(),
        "MAE Kural": (x["gercek"] - x["kural"]).abs().mean(),
    }
    return out


def kapsama(x: pd.DataFrame, lvl: str) -> float:
    return float(((x["gercek"] >= x["lo" + lvl]) & (x["gercek"] <= x["hi" + lvl])).mean() * 100)


# ---------------------------------------------------------------- gün seçimi
gunler = sorted(d["gun"].dt.date.unique())
secili = st.sidebar.date_input("Gün", value=gunler[0], min_value=gunler[0], max_value=gunler[-1])
st.sidebar.write("Seçilebilir aralık: " + str(gunler[0]) + " .. " + str(gunler[-1]))
st.sidebar.write("Aralık yöntemi: B3 (günlük kalibrasyon + ACI).")
x = d[d["gun"].dt.date == secili].copy()
if x.empty:
    st.error("Seçilen gün için veri yok.")
    st.stop()

# ---------------------------------------------------------------- grafik
st.subheader(f"{secili}: saatlik gerçek fiyat, final tahmin, kıyaslar ve B3 aralığı")
w = x.rename(columns={"gercek": "Gerçek", "final": "Final tahmin", "d1": "D-1", "d7": "D-7", "kural": "Kural"})
long = w.melt(id_vars=["saat"], value_vars=["Gerçek", "Final tahmin", "D-1", "D-7", "Kural"], var_name="seri", value_name="TL/MWh")
x_enc = alt.X("saat:Q", title="Saat (0-23)", scale=alt.Scale(domain=[0, 23]))
band90 = alt.Chart(w).mark_area(opacity=0.15).encode(x=x_enc, y=alt.Y("lo90:Q", title="TL/MWh"), y2="hi90:Q")
band80 = alt.Chart(w).mark_area(opacity=0.30).encode(x=x_enc, y="lo80:Q", y2="hi80:Q")
lines = alt.Chart(long).mark_line(point=True).encode(
    x=x_enc, y=alt.Y("TL/MWh:Q"), color=alt.Color("seri:N", title=""),
    tooltip=["saat", "seri", alt.Tooltip("TL/MWh:Q", format=".1f")],
)
st.altair_chart((band90 + band80 + lines).properties(height=420), use_container_width=True)
st.caption("Açık gölge: %90 aralık; koyu gölge: %80 aralık (B3; alt sınır 0'a, üst sınır günün tavanına kırpıldı).")

# ---------------------------------------------------------------- seçilen gün özeti
c1, c2 = st.columns(2)
with c1:
    st.subheader("Seçilen gün")
    g = ozet(x)
    g["B3 %80 kapsama (%)"] = kapsama(x, "80")
    g["B3 %90 kapsama (%)"] = kapsama(x, "90")
    st.table(pd.Series(g, name="değer").round(1).to_frame())
with c2:
    st.subheader("Tüm T dönemi (8760 saat)")
    t = ozet(d)
    t["B3 %80 kapsama (%)"] = kapsama(d, "80")
    t["B3 %90 kapsama (%)"] = kapsama(d, "90")
    t["B3 %80 ort. genişlik"] = float((d["hi80"] - d["lo80"]).mean())
    t["B3 %90 ort. genişlik"] = float((d["hi90"] - d["lo90"]).mean())
    st.table(pd.Series(t, name="değer").round(2).to_frame())

st.subheader("Saatlik değerler")
st.dataframe(
    w[["saat", "Gerçek", "Final tahmin", "D-1", "D-7", "Kural", "lo80", "hi80", "lo90", "hi90"]].round(1),
    use_container_width=True, hide_index=True,
)

st.caption(
    "Sınırlamalar: T daha önce görüldü (temiz sınav değil); tek dönem, tek koşu, anlamlılık sınanmadı; "
    "aralık genişliği doğrulama dönemine göre büyük, sebebi bilinmiyor. Canlı veri hattı yok."
)
