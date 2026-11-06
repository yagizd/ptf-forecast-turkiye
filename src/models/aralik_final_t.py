"""Test dönemi T için final aralık koşusu (TEK KOŞU): B3 = günlük güncellenen 56 günlük kalibrasyon + ACI (gamma 0,05); %80 ve %90.

TASLAK: bu dosya Proje şefi okuyup onaylamadan ÇALIŞTIRILMAZ. T = 2025-10-21 .. 2026-10-20 (8.760 saat).
T daha önce bir kez görüldü (ilk model); bu temiz bir sınav DEĞİL.

AYAR YOK: gamma 0,05, kalibrasyon 56 gün, düzeyler (80, 90) ve model parametreleri aralik_v2.py / aralik_gunluk.py ile AYNI
(aralik_v2.BASE ve aralik_v2.LEVELS import edilir). Komut satırından değiştirilebilen hiçbir ayar yok.
TEK KOŞU koruması: çıktı klasörü (data/processed/oturum_T_final/) ya da rapor dosyası zaten varsa betik HİÇBİR ŞEY yapmadan hata verip çıkar.
(Çıktı klasörü koşunun başında oluşturulur ve kilit görevi görür; koşu yarıda kesilirse klasörü silmek Yağız'ın kararıdır.)

Veri: data/processed/dataset_weather_guncel.parquet, ts < 2026-10-21 00:00 ile KESİLİR (T sonrası günler, 21-29 Ekim 2026, KULLANILMAZ).
Yöntem V'deki 6B ile aynıdır: her 28 günlük blok için 4 LightGBM quantile modeli (alpha 0.05, 0.1, 0.9, 0.95), eğitim
[TRAIN_START, blok_başı - 56 gün); aralık HAM (sıralanmamış) quantile çıktılarıyla; B0 = blok başında bir kez kalibrasyon (kıyas).
ACI alpha'sı T'nin ilk gününde hedef kayıpla (0,2 / 0,1) başlar ve bloklar arasında sıfırlanmaz (V koşusundan devam ETMEZ).

aralik_gunluk.py'den IMPORT edilenler (kopya yok): run_series, metrics, pct, md (q_level vb. run_series içinde kullanılır).
predictor.py'den: day_numbers, EPOCH.
Çıktılar ÖNCE dosyaya: data/processed/oturum_T_final/aralik_T_tahminleri.parquet ve reports/aralik_final_t.md; ekrana sonra.
SÜRE TAHMİNİ (ölçüm DEĞİL): V koşusu aralik_gunluk.md'de "Betik süresi (ölçüm): 30 sn" idi (70 eğitim); burada 14 blok x 4 model = 56 eğitim,
günlük döngü benzer; T için ~30-60 sn "tahmin". Gerçek süre rapora yazılır.
Çalıştırma (Faz 2'de, onaydan sonra, TEK KEZ): PYTHONUTF8=1 python -m src.models.aralik_final_t
Kural: eksik değer doldurulmaz.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import aralik_gunluk as ag

ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = ROOT / "data" / "processed" / "dataset_weather_guncel.parquet"
CONFIG_PATH = ROOT / "reports" / "final_config.json"
V_PRED_PATH = ROOT / "data" / "processed" / "oturum_6B" / "aralik_gunluk_tahminleri.parquet"
V_REPORT_PATH = ROOT / "reports" / "aralik_gunluk.md"
OUT_DIR = ROOT / "data" / "processed" / "oturum_T_final"
PRED_NAME = "aralik_T_tahminleri.parquet"
LOG_NAME = "ilerleme.log"
REPORT_PATH = ROOT / "reports" / "aralik_final_t.md"

TZ = "Europe/Istanbul"
T_START = pd.Timestamp("2025-10-21")
T_END = pd.Timestamp("2026-10-20")  # son gün DAHİL
T_END_EXCL = pd.Timestamp("2026-10-21 00:00", tz=TZ)  # bu ve sonrası KULLANILMAZ
T_HOURS = 365 * 24  # 8.760
GAMMA = 0.05
CAL_DAYS = 56
LEVELS_PCT = (80, 90)
HEARTBEAT_S = 600
ALPHAS_USED = (0.05, 0.1, 0.9, 0.95)  # %80 için (0.1, 0.9), %90 için (0.05, 0.95); medyan gerekmez
TARGET_TOL_PUAN = 2.0

TOP_LINE = "T daha önce bir kez görüldü (ilk model); bu temiz bir sınav DEĞİL."
INCELENMEDI = "Bu rapor Claude (sohbet) tarafından HENÜZ İNCELENMEDİ."
TK_RULES = ("T KOŞUSU KABUL KURALLARI (sonuçtan ÖNCE yazıldı, DEĞİŞMEZ): TK1. Koşu tek kez; tekrar yok. "
            "TK2. Hiçbir ayar/parametre/özellik T sonucuna bakarak değiştirilmez. "
            "TK3. Bilgi eşiği (ret kuralı DEĞİL): iki düzeyde de |kapsama - hedef| <= 2,0 puan ise \"hedef tuttu\", değilse \"hedef tutmadı\"; "
            "sonuç olduğu gibi yazılır, kötü çıkan aylar güzelleştirilmez. "
            "TK4. V sonuçları (79,86 / 89,92) ile T sonuçları yan yana verilir; fark yorumlanırken \"tek T dönemi, tek koşu, anlamlılık sınanmadı\" notu eklenir.")
NOTE_T4 = "Tek T dönemi, tek koşu, anlamlılık sınanmadı."


# =====================================================================  küçük saf fonksiyonlar (testler bunları kullanır)
def check_single_run(out_dir: Path = OUT_DIR, report_path: Path = REPORT_PATH) -> None:
    """Çıktı klasörü ya da rapor zaten varsa HİÇBİR ŞEY yapmadan FileExistsError."""
    if Path(out_dir).exists() or Path(report_path).exists():
        raise FileExistsError(f"TEK KOŞU koruması: çıktı zaten var ({out_dir} veya {report_path}); hiçbir şey yapılmadı")


def truncate_to_t(df: pd.DataFrame) -> pd.DataFrame:
    """ts >= 2026-10-21 00:00 olan satırları (T sonrası) atar. Kalan tablo T'nin son saatinde (2026-10-20 23:00) biter."""
    out = df[df["ts"] < T_END_EXCL].reset_index(drop=True)
    assert out["ts"].max() < T_END_EXCL
    return out


def t_rows_mask(ts: pd.Series) -> np.ndarray:
    """T penceresindeki (2025-10-21 00:00 .. 2026-10-20 23:00) satırların maskesi."""
    start = T_START.tz_localize(TZ)
    return ((ts >= start) & (ts < T_END_EXCL)).to_numpy()


def run_level(blocks: list, target: float, variant: str):
    """Tek düzey ve varyant için seri. variant: 'B0' (blok başında bir kez) ya da 'B3' (günlük + ACI, gamma 0,05). Ayar yok."""
    if variant == "B0":
        return ag.run_series(blocks, target, "B0", 0.0, cal_days=CAL_DAYS)
    if variant == "B3":
        return ag.run_series(blocks, target, "ACI", GAMMA, cal_days=CAL_DAYS)
    raise ValueError(variant)


pct = ag.pct  # aralik_gunluk.py'deki yardımcılar (kopya yok)
md = ag.md


class RunLog:
    """ilerleme.log: her önemli adımda ve en az 10 dakikada bir tarihli satır."""

    def __init__(self, path: Path):
        self.path, self.last = Path(path), time.time()

    def write(self, msg: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        self.last = time.time()

    def tick(self, msg: str) -> None:
        if time.time() - self.last >= HEARTBEAT_S:
            self.write("(10 dk kalp atışı) " + msg)


# =====================================================================  veri ve model eğitimi (hafif import'lar burada)
def build_t_data(log: RunLog):
    """T için blok verisi: 4 quantile modeli blok başına eğitir. Dönen: (blok listesi, meta tablosu, ham quantile sırası sayıları)."""
    import lightgbm as lgb

    from src.features import ptf_features as pf
    from src.features import ptf_features_v2 as v2
    from src.models import aralik_v2 as av2
    from src.models import baselines as bl
    from src.models import lgbm_walkforward as lw
    from src.models import quantile_conformal as qc
    from src.serving import predictor as pr

    assert (bl.TEST_START, bl.TEST_END) == (T_START, T_END), "T tarihleri baselines.py ile aynı olmalı"
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    groups = tuple(cfg["gruplar"])
    assert groups == ("H",) and cfg["kapak_normalizasyonu_K"] is False
    assert cfg["parametreler_kaynagi"].startswith("varsayilan"), "Optuna parametresi kullanılmaz"
    cols = list(cfg["ozellikler"])
    assert cols == v2.feature_list(pf.FEATURES_WEATHER, groups)
    assert set(ALPHAS_USED) <= set(av2.ALPHAS) and av2.LEVELS == {80: ((0.1, 0.9), 0.2, 0.80), 90: ((0.05, 0.95), 0.1, 0.90)}

    df = truncate_to_t(pd.read_parquet(TABLE_PATH))  # 21 Ekim 2026 ve sonrası ATILIR
    feat = v2.build_features_v2(df, groups=groups, normalize=False).reset_index(drop=True)
    assert feat["ts"].is_monotonic_increasing and feat["ts"].max() < T_END_EXCL
    feat["day"] = pr.day_numbers(feat["ts"])
    t_first = int((T_START - pr.EPOCH).days)
    t_last = int((T_END - pr.EPOCH).days)
    rng = feat[(feat["day"] >= t_first - CAL_DAYS) & (feat["day"] <= t_last)]
    cnt = rng.groupby("day").size()
    assert len(cnt) == t_last - (t_first - CAL_DAYS) + 1 and (cnt == 24).all(), "T (ve 56 gün öncesi) içinde her gün tam 24 satır olmalı"
    assert rng[pf.TARGET].notna().all(), "y'de boş değer var"
    assert int(t_rows_mask(feat["ts"]).sum()) == T_HOURS, "T tam 8.760 saat olmalı"

    blocks = lw.make_blocks(T_START, T_END)
    log.write(f"veri hazır: {len(feat)} satır, {len(blocks)} blok; eğitim başlıyor")
    data, metas, cross = [], [], {"q10>q90": 0, "q05>q95": 0, "sirasi_bozuk": 0}
    for b in blocks:
        bs, be = b["start_ts"], b["end_ts"]
        tr_m, _ = qc.split_masks(feat["ts"], feat[pf.TARGET], bs, lw.TRAIN_START, CAL_DAYS)
        pm = ((feat["ts"] >= bs - pd.Timedelta(days=CAL_DAYS)) & (feat["ts"] < be)).to_numpy()
        assert feat.loc[tr_m, "ts"].max() < bs - pd.Timedelta(days=CAL_DAYS), "kalibrasyon satırları eğitimde olmamalı"
        sub = feat[pm]
        fd = int((b["first_day"] - pr.EPOCH).days)
        ld = int((b["last_day"] - pr.EPOCH).days)
        assert (sub["day"] >= fd).sum() == 24 * (ld - fd + 1), f"blok {b['no']} tam günlerden oluşmalı"
        preds = {}
        for a in ALPHAS_USED:
            m = lgb.LGBMRegressor(alpha=a, **av2.BASE)
            m.fit(feat.loc[tr_m, cols], feat.loc[tr_m, pf.TARGET])
            preds[a] = m.predict(sub[cols])
        test = (sub["day"] >= fd).to_numpy()
        q = np.column_stack([preds[a][test] for a in ALPHAS_USED])  # 0.05, 0.1, 0.9, 0.95 (HAM)
        cross["q10>q90"] += int((preds[0.1][test] > preds[0.9][test]).sum())
        cross["q05>q95"] += int((preds[0.05][test] > preds[0.95][test]).sum())
        cross["sirasi_bozuk"] += int((np.diff(q, axis=1) < 0).any(axis=1).sum())
        data.append({"no": b["no"], "day": sub["day"].to_numpy(), "y": sub[pf.TARGET].to_numpy(), "preds": preds, "first_day": fd, "last_day": ld})
        t = sub[test]
        metas.append(pd.DataFrame({"ts": t["ts"].reset_index(drop=True), "y": t[pf.TARGET].to_numpy(), "blok_no": b["no"], "day": t["day"].to_numpy()}))
        log.write(f"blok {b['no']} bitti")
        log.tick(f"blok {b['no']}")
    meta = pd.concat(metas, ignore_index=True)
    assert len(meta) == T_HOURS
    assert meta["ts"].min() == T_START.tz_localize(TZ) and meta["ts"].max() == T_END_EXCL - pd.Timedelta(hours=1)
    return data, meta, cross, av2


def v_summary() -> dict:
    """V sonuçları (kıyas): kaydedilmiş V parquet'inden HESAPLANIR (elle yazılmış sayı yok)."""
    v = pd.read_parquet(V_PRED_PATH)
    out = {}
    for var in ("B0", "B3"):
        d = v[v["varyant"] == var]
        for lvl in LEVELS_PCT:
            out[(var, lvl)] = ag.metrics(d["y"], d[f"lo{lvl}"], d[f"hi{lvl}"], d["tavan"], d[f"Q{lvl}"])
    return out


def v_time_text() -> str:
    try:
        m = re.search(r"Betik süresi \(ölçüm\): (\d+) sn", V_REPORT_PATH.read_text(encoding="utf-8"))
        return f"V koşusu {m.group(1)} sn sürmüştü (ölçüm); T için süre tahmini benzer: 30-60 sn (tahmin)" if m else "V süresi okunamadı: BİLMİYORUM"
    except OSError:
        return "V raporu okunamadı: BİLMİYORUM"


# =====================================================================  main
def main() -> int:
    t0 = time.time()
    check_single_run()  # varsa HİÇBİR ŞEY yapmadan hata
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    log = RunLog(OUT_DIR / LOG_NAME)
    log.write("BAŞLADI. " + v_time_text())
    from src.data import price_caps as pc

    data, meta, cross, av2 = build_t_data(log)
    meta["ts"] = pd.to_datetime(meta["ts"], utc=True).dt.tz_convert(TZ)
    meta["tavan"] = pc.caps_for_ts(meta["ts"]).to_numpy()
    meta["ay"] = meta["ts"].dt.strftime("%Y-%m")

    R = {}
    for var in ("B0", "B3"):
        cols = {}
        for lvl, ((alo, ahi), target, _tgt) in av2.LEVELS.items():
            bl_ = [{"no": d["no"], "day": d["day"], "y": d["y"], "qlo": d["preds"][alo], "qhi": d["preds"][ahi],
                    "first_day": d["first_day"], "last_day": d["last_day"]} for d in data]
            outs, _recs = run_level(bl_, target, var)
            for key in ("lo", "hi", "alpha", "Q"):
                cols[f"{key}{lvl}"] = np.concatenate([o[key] for o in outs])
        R[var] = pd.concat([meta, pd.DataFrame(cols)], axis=1).assign(varyant=var)
        log.write(f"{var} bitti")

    # SONUÇ ÖNCE DOSYAYA
    long = pd.concat(R.values(), ignore_index=True)
    long[["ts", "y", "blok_no", "varyant", "tavan", "lo80", "hi80", "lo90", "hi90", "alpha80", "alpha90", "Q80", "Q90"]].to_parquet(
        OUT_DIR / PRED_NAME, index=False)
    log.write("parquet yazıldı")

    def st(sub, lvl):
        return ag.metrics(sub["y"], sub[f"lo{lvl}"], sub[f"hi{lvl}"], sub["tavan"], sub[f"Q{lvl}"])

    tot = {(v, lvl): st(R[v], lvl) for v in R for lvl in LEVELS_PCT}
    vsum = v_summary()
    tgt = {80: 0.80, 90: 0.90}
    hit = all(abs(tot[("B3", lvl)]["kapsama"] - tgt[lvl]) * 100 <= TARGET_TOL_PUAN + 1e-9 for lvl in LEVELS_PCT)

    L = [TOP_LINE, "", INCELENMEDI, "", TK_RULES, "",
         f"# Final aralık koşusu: B3 (günlük kalibrasyon + ACI, gamma {GAMMA}) T = {T_START.date()}..{T_END.date()} ({T_HOURS} saat)", "",
         "- Ayar yok: V'de seçilen yöntem aynen uygulandı. Tablo 21-29 Ekim 2026 günlerini İÇERMEZ.",
         "- Sebep: BİLMİYORUZ; tavan değişimi hipotezi sınanmadı (zayıf dönemlerin nedeni araştırılmadı).",
         f"- Ham quantile sırası bozuk satır (T test satırlarında): q10 > q90 {cross['q10>q90']}; q05 > q95 {cross['q05>q95']}; "
         f"4 quantile'ın sırası bozuk (herhangi biri) {cross['sirasi_bozuk']}.", "",
         "## Bilgi eşiği (TK3, ret kuralı DEĞİL)", "",
         f"B3 için: %80 |kapsama - hedef| = {abs(tot[('B3', 80)]['kapsama'] - 0.8) * 100:.2f} puan; "
         f"%90 |kapsama - hedef| = {abs(tot[('B3', 90)]['kapsama'] - 0.9) * 100:.2f} puan. **Sonuç: {'hedef tuttu' if hit else 'hedef tutmadı'}** (eşik {TARGET_TOL_PUAN} puan).", "",
         "## Toplam (T, n=8760 / varyant)", ""]
    rows = []
    for v in ("B0", "B3"):
        for lvl in LEVELS_PCT:
            s = tot[(v, lvl)]
            unb = R[v].loc[R[v][f"Q{lvl}"] == np.inf, "day"].nunique()
            rows.append((v, f"%{lvl}", s["n"], pct(s["kapsama"]), f"{s['genislik_klip']:.1f}", pct(s["alt"]), pct(s["ust"]), s["bos_saat"], unb))
    L += [md(["varyant", "düzey", "n", "kapsama", "kırpılmış genişlik", "alt ıskalama", "üst ıskalama", "boş aralık saat", "sınırsız-Q gün"], rows), "",
          "B0 = günlük güncellemesiz 56 günlük kalibrasyon (blok başında bir kez); B3 = günlük güncelleme + ACI.", "",
          "## V ile T yan yana (TK4)", ""]
    rows = []
    for v in ("B0", "B3"):
        for lvl in LEVELS_PCT:
            sv, stt = vsum[(v, lvl)], tot[(v, lvl)]
            rows.append((v, f"%{lvl}", pct(sv["kapsama"]), f"{sv['genislik_klip']:.1f}", pct(stt["kapsama"]), f"{stt['genislik_klip']:.1f}"))
    L += [md(["varyant", "düzey", "V kapsama", "V genişlik", "T kapsama", "T genişlik"], rows), "", NOTE_T4, "",
          "## Blok başına kapsama", ""]
    rows = []
    for no in sorted(R["B3"]["blok_no"].unique()):
        row = [f"{int(no)}" + (" (n küçük, yorumlanmaz)" if int(no) == 14 else ""), int((R["B3"]["blok_no"] == no).sum())]
        for lvl in LEVELS_PCT:
            for v in ("B0", "B3"):
                row.append(pct(st(R[v][R[v]["blok_no"] == no], lvl)["kapsama"]))
        rows.append(row)
    L += [md(["blok", "n", "B0 %80", "B3 %80", "B0 %90", "B3 %90"], rows), "", "## Ay başına kapsama", ""]
    rows, months = [], sorted(meta["ay"].unique())
    for ay in months:
        row = [ay, int((meta["ay"] == ay).sum())]
        for lvl in LEVELS_PCT:
            for v in ("B0", "B3"):
                row.append(pct(st(R[v][R[v]["ay"] == ay], lvl)["kapsama"]))
        rows.append(row)
    L += [md(["ay", "n", "B0 %80", "B3 %80", "B0 %90", "B3 %90"], rows), "", "## Ayrı satır: 2026-04 ve 2026-05", ""]
    rows = []
    for ay in ("2026-04", "2026-05"):
        for lvl in LEVELS_PCT:
            row = [f"**{ay}**", f"%{lvl}"]
            for v in ("B0", "B3"):
                s = st(R[v][R[v]["ay"] == ay], lvl)
                row.append(f"{pct(s['kapsama'])} / {s['genislik_klip']:.1f}")
            rows.append(row)
    L += [md(["ay", "düzey", "B0 (kapsama / genişlik)", "B3 (kapsama / genişlik)"], rows), "",
          "## Fiyat 0 saatleri ve tavan saatleri için kapsama", ""]
    rows = []
    for ad, sel in (("fiyat 0", lambda d: d["y"] == 0), ("fiyat = günün tavanı", lambda d: d["y"] == d["tavan"])):
        for lvl in LEVELS_PCT:
            row = [ad, f"%{lvl}"]
            for v in ("B0", "B3"):
                d = R[v][sel(R[v])]
                row.append(f"{len(d)} saat; {pct(st(d, lvl)['kapsama'])}" if len(d) else "0 saat; -")
            rows.append(row)
    L += [md(["saat türü", "düzey", "B0 (n; kapsama)", "B3 (n; kapsama)"], rows), "", "## B3 alpha izi (gün başına kullanılan alpha)", ""]
    rows = []
    for lvl in LEVELS_PCT:
        a = R["B3"].groupby("day")[f"alpha{lvl}"].first()
        rows.append((f"%{lvl}", f"{a.min():.4f}", f"{a.mean():.4f}", f"{a.max():.4f}", int((a < 0).sum()), int((a > 1).sum()), len(a)))
    L += [md(["düzey", "min", "ortalama", "maks", "alpha<0 gün", "alpha>1 gün", "gün"], rows), "",
          f"Betik süresi (ölçüm): {time.time() - t0:.0f} sn.", ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.write(f"rapor yazıldı; süre {time.time() - t0:.0f} sn")
    print(f"B3: %80 {pct(tot[('B3', 80)]['kapsama'])} %90 {pct(tot[('B3', 90)]['kapsama'])}; {'hedef tuttu' if hit else 'hedef tutmadı'}; süre {time.time() - t0:.0f} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
