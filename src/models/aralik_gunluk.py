"""Hafta 6B: V üzerinde günlük güncellenen kalibrasyon (B1) ve ACI (B2, B3) ile tahmin aralığı; kıyas B0 = mevcut cal56.

B0 = blok başında bir kez, kalibrasyon [blok_başı-56g, blok_başı).
B1 = her gün D için kalibrasyon [D-56g, D), Q sabit hedef kayıp (0,2 / 0,1) ile.
B2 = B1 + ACI gamma 0,01;  B3 = B1 + ACI gamma 0,05.  Her düzey için AYRI tek alpha; alpha_1 = hedef kayıp;
     alpha_(D+1) = alpha_D + gamma * (hedef kayıp - e_D), e_D = D günü aralık DIŞINDA kalan saat oranı (boş aralık = 1);
     V'nin ilk gününden başlar, bloklar arasında SIFIRLANMAZ, burn-in yok, alpha KIRPILMAZ (assert: [-gamma, 1+gamma]).
Skor (HAM, sıralanmamış quantile çıktılarıyla): s = max(q_lo - y, y - q_hi); aralık [q_lo - Q, q_hi + Q].
Q fonksiyonu burada yeni yazıldı (quantile_conformal.py'ye dokunulmadı): k = ceil((n+1)(1-alpha));
  k > n -> Q = +inf (aralık sınırsız); k < 1 -> Q = -inf (BOŞ aralık: kapsama yok, genişlik 0); aksi: sıralı skorların k-1. indeksi.
Zaman sızıntısı yok: D günü aralığı yalnız D 00:00'dan ÖNCEKİ y değerlerini kullanır.
Veri, özellikler, bloklar, BASE, ALPHAS, LEVELS: aralik_v2.py ile AYNI (oradan import edilir). T kilidi: ve.load_locked().
Eğitim cal56 ile AYNI: [TRAIN_START, blok_başı-56g), 5 LightGBM quantile; tahminler [blok_başı-56g, blok_sonu) TÜM satırları için.

Kod ikiye ayrılır: ÇEKİRDEK (saf numpy; model yok; testler bunları kullanır) ve main (model eğitimi, rapor).
Çıktılar: reports/aralik_gunluk.md ve data/processed/oturum_6B/aralik_gunluk_tahminleri.parquet
SÜRE TAHMİNİ (ölçüm DEĞİL): 14 blok x 5 model = 70 eğitim; aralik_v2 140 eğitim = 115 sn (ölçüm) olduğundan ~60-90 sn "tahmin".
Çalıştırma (ana .venv): PYTHONUTF8=1 python -m src.models.aralik_gunluk
Kural: eksik değer doldurulmaz.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

INF = float("inf")
CAL_DAYS = 56
GAMMAS = {"B0": None, "B1": 0.0, "B2": 0.01, "B3": 0.05}
CANDIDATES = ("B1", "B2", "B3")
FULL_BLOCKS = 13
LAST_BLOCK = 14
TOL_PUAN = 2.0
GENISLIK_CARPAN = 1.25
MIN_BLOK = 8
ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "aralik_gunluk.md"
PRED_PATH = ROOT / "data" / "processed" / "oturum_6B" / "aralik_gunluk_tahminleri.parquet"
CONFIG_PATH = ROOT / "reports" / "final_config.json"
INCELENMEDI = "Bu rapor Claude (sohbet) tarafından HENÜZ İNCELENMEDİ."
RULE = ("KABUL KURALI (sonuçtan ÖNCE yazıldı, DEĞİŞTİRİLMEZ): Aday B1, B2, B3 \"GEÇTİ\" ancak ve ancak, HEM %80 HEM %90 düzeyinde "
        "tüm V için (n=8760): (a) |kapsama - hedef| <= 2,0 puan (%80: 78,00-82,00; %90: 88,00-92,00); "
        "(b) kırpılmış ortalama genişlik <= 1,25 x B0'ın aynı düzeydeki kırpılmış genişliği; "
        "(c) 13 tam bloğun en az 8'inde |blok kapsaması - hedef| < B0'ınki (iki düzeyde de ayrı ayrı). "
        "Geçenler arasında en küçük %90 kırpılmış genişlik seçilir; eşitlikte küçük gamma. Hiçbiri geçmezse B0 (56 gün sabit) KALIR. "
        "Yeni deney, gamma ayarı, kural değişikliği YOK. Blok 14 (1 gün) kuralda sayılmaz.")
CAUSE = "Sebep: BİLMİYORUZ; tavan değişimi hipotezi sınanmadı."


# =====================================================================  ÇEKİRDEK (saf numpy; model yok)
def scores(y, q_lo, q_hi) -> np.ndarray:
    return np.maximum(np.asarray(q_lo, float) - np.asarray(y, float), np.asarray(y, float) - np.asarray(q_hi, float))


def q_level(sc, alpha: float) -> float:
    """Sonlu örneklem düzeltmeli Q. k>n: +inf (sınırsız); k<1: -inf (BOŞ aralık); aksi sıralı skorların (k-1). indeksi."""
    s = np.sort(np.asarray(sc, dtype=float))
    s = s[~np.isnan(s)]
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return INF
    if k < 1:
        return -INF
    return float(s[k - 1])


def make_interval(q_lo, q_hi, Q: float):
    """[q_lo - Q, q_hi + Q]. Q=+inf: (-inf, +inf); Q=-inf: (+inf, -inf) yani boş."""
    return np.asarray(q_lo, float) - Q, np.asarray(q_hi, float) + Q


def aci_update(alpha: float, gamma: float, target: float, e: float) -> float:
    return alpha + gamma * (target - e)


def run_series(blocks: list, target: float, mode: str, gamma: float = 0.0, cal_days: int = CAL_DAYS):
    """Bir düzey için B0/B1/ACI serisi.

    blocks: her blok {no, day (satır başına tamsayı gün), y, qlo, qhi, first_day, last_day}; diziler [first_day-cal_days, last_day] kapsar.
    mode: 'B0' (blok başında bir kez), 'B1' (günlük), 'ACI' (günlük + alpha izleyici; bloklar arası sıfırlanmaz).
    Dönen: (blok başına dict(lo, hi, alpha, Q) yalnız TEST satırları için, gün kayıtları listesi).
    """
    alpha = target
    outs, recs = [], []
    for b in blocks:
        day, y, qlo, qhi = np.asarray(b["day"]), np.asarray(b["y"], float), np.asarray(b["qlo"], float), np.asarray(b["qhi"], float)
        test = (day >= b["first_day"]) & (day <= b["last_day"])
        td, ty, tlo_q, thi_q = day[test], y[test], qlo[test], qhi[test]
        lo, hi = np.full(len(td), np.nan), np.full(len(td), np.nan)
        al, Qr = np.full(len(td), np.nan), np.full(len(td), np.nan)
        if mode == "B0":
            cal = (day >= b["first_day"] - cal_days) & (day < b["first_day"])
            assert cal.any(), "B0 kalibrasyon boş"
            Q = q_level(scores(y[cal], qlo[cal], qhi[cal]), target)
            lo, hi = make_interval(tlo_q, thi_q, Q)
            al[:], Qr[:] = target, Q
        else:
            for D in range(int(b["first_day"]), int(b["last_day"]) + 1):
                cal = (day >= D - cal_days) & (day < D)  # YALNIZ D gününden ÖNCE
                assert cal.any(), "kalibrasyon boş"
                a = alpha if mode == "ACI" else target
                Q = q_level(scores(y[cal], qlo[cal], qhi[cal]), a)
                m = td == D
                lo[m], hi[m] = make_interval(tlo_q[m], thi_q[m], Q)
                al[m], Qr[m] = a, Q
                inside = (ty[m] >= lo[m]) & (ty[m] <= hi[m])
                e = float(1.0 - inside.mean())
                recs.append({"blok": b["no"], "gun": D, "alpha": a, "Q": Q, "e": e, "n_kal": int(cal.sum())})
                if mode == "ACI":
                    alpha = aci_update(alpha, gamma, target, e)
                    assert -gamma - 1e-12 <= alpha <= 1 + gamma + 1e-12, f"alpha sınır dışı: {alpha}"
        outs.append({"lo": lo, "hi": hi, "alpha": al, "Q": Qr})
    return outs, recs


def metrics(y, lo, hi, cap, Q) -> dict:
    """Kapsama (boş aralık = hayır), kırpılmış genişlik ([0, tavan]; boş aralık 0), kırpılmamış genişlik (yalnız sonlu satırlar)."""
    y, lo, hi, cap, Q = (np.asarray(v, float) for v in (y, lo, hi, cap, Q))
    empty = Q == -INF
    inside = (y >= lo) & (y <= hi) & ~empty
    wc = np.maximum(np.clip(hi, 0.0, cap) - np.clip(lo, 0.0, cap), 0.0)
    wc[empty] = 0.0
    fin = np.isfinite(lo) & np.isfinite(hi)
    return {"n": int(len(y)), "kapsama": float(inside.mean()), "genislik_klip": float(wc.mean()),
            "genislik_klipsiz_sonlu": float((hi[fin] - lo[fin]).mean()) if fin.any() else float("nan"),
            "alt": float(((y < lo) & ~empty).mean()), "ust": float(((y > hi) & ~empty).mean()),
            "bos_saat": int(empty.sum()), "ters_aralik_saat": int(((lo > hi) & ~empty).sum())}


def decide(stats: dict, blockcov: dict) -> dict:
    """KABUL KURALI. stats[(varyant, düzey)] = metrics(...); blockcov[(varyant, düzey)] = {blok_no: kapsama} (blok 1..13).
    Dönen: aday -> {'a':..,'b':..,'c':..,'gecti':bool}, 'secilen'."""
    tgt = {80: 0.80, 90: 0.90}
    res = {}
    for v in CANDIDATES:
        r = {"gecti": True}
        for lvl in (80, 90):
            s, s0 = stats[(v, lvl)], stats[("B0", lvl)]
            d = abs(s["kapsama"] - tgt[lvl]) * 100
            a_ok = d <= TOL_PUAN + 1e-9
            b_ok = s["genislik_klip"] <= GENISLIK_CARPAN * s0["genislik_klip"]
            better = sum(int(abs(blockcov[(v, lvl)][k] - tgt[lvl]) < abs(blockcov[("B0", lvl)][k] - tgt[lvl]))
                         for k in range(1, FULL_BLOCKS + 1))
            c_ok = better >= MIN_BLOK
            r[lvl] = {"a": (a_ok, f"|kapsama−hedef| {d:.2f} puan (≤2,0 gerekir)"),
                      "b": (b_ok, f"kırpılmış genişlik {s['genislik_klip']:.1f} / B0 {s0['genislik_klip']:.1f} = "
                                  f"{s['genislik_klip'] / s0['genislik_klip']:.3f} (≤1,25 gerekir)"),
                      "c": (c_ok, f"B0'dan hedefe daha yakın tam blok: {better}/13 (≥8 gerekir)")}
            r["gecti"] = r["gecti"] and a_ok and b_ok and c_ok
        res[v] = r
    passers = [v for v in CANDIDATES if res[v]["gecti"]]
    if passers:
        # en küçük %90 kırpılmış genişlik; eşitlikte küçük gamma
        res["secilen"] = min(passers, key=lambda v: (stats[(v, 90)]["genislik_klip"], GAMMAS[v]))
    else:
        res["secilen"] = "B0"
    return res


# =====================================================================  main (model eğitimi + rapor)
def md(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def build_blocks_data():
    """Veri + 5 quantile model eğitimi (blok başına). Dönen: ham blok verisi listesi, meta tablosu (test satırları)."""
    import json

    import lightgbm as lgb

    from src.features import ptf_features as pf
    from src.features import ptf_features_v2 as v2
    from src.models import aralik_v2 as av2
    from src.models import lgbm_walkforward as lw
    from src.models import quantile_conformal as qc
    from src.models import validation_experiments as ve

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    groups = tuple(cfg["gruplar"])
    assert cfg["kapak_normalizasyonu_K"] is False
    df = ve.load_locked()  # T kilidi
    assert df["ts"].max() < ve.T_LOCK
    feat = v2.build_features_v2(df, groups=groups, normalize=False).reset_index(drop=True)
    cols = list(cfg["ozellikler"])
    assert cols == v2.feature_list(pf.FEATURES_WEATHER, groups)
    assert feat["ts"].is_monotonic_increasing
    blocks = lw.make_blocks(ve.V_START, ve.V_END)
    local = feat["ts"].dt.tz_convert(lw.TZ).dt.tz_localize(None).dt.normalize()
    feat["day"] = (local - pd.Timestamp("1970-01-01")).dt.days.astype(int)
    v_first = int((pd.Timestamp(ve.V_START) - pd.Timestamp("1970-01-01")).days)
    v_last = int((pd.Timestamp(ve.V_END) - pd.Timestamp("1970-01-01")).days)
    rng = feat[(feat["day"] >= v_first - CAL_DAYS) & (feat["day"] <= v_last)]
    cnt = rng.groupby("day").size()
    assert len(cnt) == v_last - (v_first - CAL_DAYS) + 1 and (cnt == 24).all(), "V (ve 56 gün öncesi) içinde her gün tam 24 satır olmalı"
    assert rng["price"].notna().all(), "y'de boş değer var"

    data, metas = [], []
    for b in blocks:
        bs, be = b["start_ts"], b["end_ts"]
        tr_m, _ = qc.split_masks(feat["ts"], feat["price"], bs, lw.TRAIN_START, CAL_DAYS)
        pm = ((feat["ts"] >= bs - pd.Timedelta(days=CAL_DAYS)) & (feat["ts"] < be)).to_numpy()
        assert feat.loc[tr_m, "ts"].max() < bs - pd.Timedelta(days=CAL_DAYS), "kalibrasyon satırları eğitimde olmamalı"
        sub = feat[pm]
        fd, ld = int((b["first_day"] - pd.Timestamp("1970-01-01")).days), int((b["last_day"] - pd.Timestamp("1970-01-01")).days)
        assert (sub["day"] >= fd).sum() == 24 * (ld - fd + 1), f"blok {b['no']} tam günlerden oluşmalı"
        preds = {}
        for a in av2.ALPHAS:
            m = lgb.LGBMRegressor(alpha=a, **av2.BASE)
            m.fit(feat.loc[tr_m, cols], feat.loc[tr_m, "price"])
            preds[a] = m.predict(sub[cols])
        data.append({"no": b["no"], "day": sub["day"].to_numpy(), "y": sub["price"].to_numpy(), "preds": preds,
                     "first_day": fd, "last_day": ld})
        t = sub[sub["day"] >= fd]
        metas.append(pd.DataFrame({"ts": t["ts"].reset_index(drop=True), "y": t["price"].to_numpy(), "blok_no": b["no"], "day": t["day"].to_numpy()}))
    meta = pd.concat(metas, ignore_index=True)
    return data, meta, av2


def main() -> int:
    t0 = time.time()
    from src.data import price_caps as pc
    from src.models import validation_experiments as ve

    data, meta, av2 = build_blocks_data()
    meta["ts"] = pd.to_datetime(meta["ts"], utc=True).dt.tz_convert("Europe/Istanbul")
    meta["tavan"] = pc.caps_for_ts(meta["ts"]).to_numpy()
    meta["ay"] = meta["ts"].dt.strftime("%Y-%m")

    # her varyant x düzey
    R = {}
    day_recs = {}
    for var, gamma in GAMMAS.items():
        cols = {}
        for lvl, ((alo, ahi), target, _) in av2.LEVELS.items():
            bl = [{"no": d["no"], "day": d["day"], "y": d["y"], "qlo": d["preds"][alo], "qhi": d["preds"][ahi],
                   "first_day": d["first_day"], "last_day": d["last_day"]} for d in data]
            mode = "B0" if var == "B0" else ("B1" if var == "B1" else "ACI")
            outs, recs = run_series(bl, target, mode, gamma or 0.0)
            for key in ("lo", "hi", "alpha", "Q"):
                cols[f"{key}{lvl}"] = np.concatenate([o[key] for o in outs])
            day_recs[(var, lvl)] = recs
        R[var] = pd.concat([meta, pd.DataFrame(cols)], axis=1).assign(varyant=var)

    # ÇAPRAZ KONTROL (rapordan önce)
    ref = pd.read_parquet(ve.OUT_DIR / "aralik_v2_tahminleri.parquet")
    ref = ref[ref["varyant"] == "cal56"].reset_index(drop=True)
    b0 = R["B0"]
    fmt = "%Y-%m-%d %H:%M:%S"
    ok1 = bool(len(ref) == len(b0)
               and (pd.to_datetime(ref["ts"], utc=True).dt.strftime(fmt) == pd.to_datetime(b0["ts"], utc=True).dt.strftime(fmt)).all()
               and all(np.allclose(ref[c].to_numpy(), b0[c].to_numpy(), atol=1e-6) for c in ("lo80", "hi80", "lo90", "hi90")))
    ok2 = True
    first = meta.groupby("blok_no")["day"].transform("min") == meta["day"]
    for c in ("lo80", "hi80", "lo90", "hi90"):
        ok2 = ok2 and bool(np.allclose(R["B1"].loc[first, c].to_numpy(), b0.loc[first, c].to_numpy(), atol=1e-9))
    if not (ok1 and ok2):
        REPORT_PATH.write_text("\n".join(["ÇAPRAZ KONTROL TUTMADI", "", INCELENMEDI, "", RULE, "",
                                          f"- (1) B0 = aralik_v2 cal56 (np.allclose atol 1e-6): {'TUTTU' if ok1 else 'TUTMADI'}",
                                          f"- (2) B1'in her bloğun ilk günü = B0: {'TUTTU' if ok2 else 'TUTMADI'}",
                                          "B1-B3 sonuçları YAZILMADI.", ""]) + "\n", encoding="utf-8")
        print("ÇAPRAZ KONTROL TUTMADI; B1-B3 yazılmadı")
        return 1

    long = pd.concat(R.values(), ignore_index=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    long[["ts", "y", "blok_no", "varyant", "tavan", "lo80", "hi80", "lo90", "hi90", "alpha80", "alpha90", "Q80", "Q90"]].to_parquet(PRED_PATH, index=False)

    def st(sub, lvl):
        return metrics(sub["y"], sub[f"lo{lvl}"], sub[f"hi{lvl}"], sub["tavan"], sub[f"Q{lvl}"])

    stats, blockcov = {}, {}
    for v in GAMMAS:
        for lvl in (80, 90):
            stats[(v, lvl)] = st(R[v], lvl)
            blockcov[(v, lvl)] = {int(k): st(g, lvl)["kapsama"] for k, g in R[v].groupby("blok_no")}
    dec = decide(stats, blockcov)

    L = [RULE, "", INCELENMEDI, "", f"# Hafta 6B: günlük kalibrasyon ve ACI (V = {ve.V_START.date()}..{ve.V_END.date()})", "",
         f"- {CAUSE}",
         "- Çapraz kontrol: (1) B0 = aralik_v2 cal56 satırları (np.allclose atol 1e-6): TUTTU; (2) B1'in her bloğun ilk günü = B0: TUTTU.",
         "- Blok 14 yalnızca 1 gün (24 saat): n küçük, yorumlanmaz; kuralda sayılmaz.", "", "## Toplam (n=8760 / varyant)", ""]
    rows = []
    for v in GAMMAS:
        for lvl in (80, 90):
            s = stats[(v, lvl)]
            unb = R[v].loc[R[v][f"Q{lvl}"] == INF, "day"].nunique()
            rows.append((v, f"%{lvl}", s["n"], pct(s["kapsama"]), f"{s['genislik_klip']:.1f}", f"{s['genislik_klipsiz_sonlu']:.1f}",
                         pct(s["alt"]), pct(s["ust"]), s["bos_saat"], unb, s["ters_aralik_saat"]))
    L += [md(["varyant", "düzey", "n", "kapsama", "kırpılmış genişlik (KARAR)", "kırpılmamış genişlik (yalnız sonlu satırlar)", "alt ıskalama",
              "üst ıskalama", "boş aralık saat", "sınırsız-Q gün", "ters aralık (lo>hi) saat"], rows), "", "## Kabul kuralı sonucu", ""]
    for v in CANDIDATES:
        L.append(f"### {v} (gamma {GAMMAS[v]})")
        for lvl in (80, 90):
            for k, ad in (("a", "(a)"), ("b", "(b)"), ("c", "(c)")):
                ok, txt = dec[v][lvl][k]
                L.append(f"- %{lvl} {ad}: **{'DOĞRU' if ok else 'YANLIŞ'}**; {txt}")
        L.append(f"- Sonuç: **{'GEÇTİ' if dec[v]['gecti'] else 'GEÇMEDİ'}**")
        L.append("")
    L += [f"**Seçilen: {dec['secilen']}**" + (" (hiçbir aday geçmedi: B0, 56 gün sabit, KALIR)" if dec["secilen"] == "B0" else ""), "",
          "## Blok başına kapsama", ""]
    rows = []
    for no in range(1, LAST_BLOCK + 1):
        row = [f"{no}" + (" (n küçük, yorumlanmaz)" if no == LAST_BLOCK else ""), 24 if no == LAST_BLOCK else 672]
        for lvl in (80, 90):
            for v in GAMMAS:
                row.append(pct(st(R[v][R[v]["blok_no"] == no], lvl)["kapsama"]))
        rows.append(row)
    L += [md(["blok", "n"] + [f"{v} %{lvl}" for lvl in (80, 90) for v in GAMMAS], rows), "", "## Ay başına kapsama", ""]
    rows = []
    for ay in sorted(meta["ay"].unique()):
        row = [ay, int((meta["ay"] == ay).sum())]
        for lvl in (80, 90):
            for v in GAMMAS:
                row.append(pct(st(R[v][R[v]["ay"] == ay], lvl)["kapsama"]))
        rows.append(row)
    L += [md(["ay", "n"] + [f"{v} %{lvl}" for lvl in (80, 90) for v in GAMMAS], rows), "", "## Vurgulu: blok 7 ve 2025-04 (kapsama | kırpılmış genişlik)", ""]
    rows = []
    for ad, sel in (("blok 7", lambda d: d["blok_no"] == 7), ("2025-04", lambda d: d["ay"] == "2025-04")):
        for lvl in (80, 90):
            row = [f"**{ad}**", f"%{lvl}"]
            for v in GAMMAS:
                s = st(R[v][sel(R[v])], lvl)
                row.append(f"{pct(s['kapsama'])} / {s['genislik_klip']:.1f}")
            rows.append(row)
    L += [md(["bölge", "düzey"] + list(GAMMAS), rows), "", "## Alpha izi (gün başına kullanılan alpha)", ""]
    rows = []
    for v in GAMMAS:
        for lvl in (80, 90):
            a = R[v].groupby("day")[f"alpha{lvl}"].first()
            rows.append((v, f"%{lvl}", f"{a.min():.4f}", f"{a.mean():.4f}", f"{a.max():.4f}", int((a < 0).sum()), int((a > 1).sum()), len(a)))
    L += [md(["varyant", "düzey", "min", "ortalama", "maks", "alpha<0 gün", "alpha>1 gün", "gün"], rows), "",
          "## Bilgi amaçlı: ilk 28 gün hariç toplam kapsama", ""]
    cut = pd.Timestamp(ve.V_START).tz_localize("Europe/Istanbul") + pd.Timedelta(days=28)
    rows = []
    for v in GAMMAS:
        d = R[v][R[v]["ts"] >= cut]
        rows.append((v, len(d), pct(st(d, 80)["kapsama"]), pct(st(d, 90)["kapsama"])))
    L += [md(["varyant", "n", "%80 kapsama", "%90 kapsama"], rows), "", f"Betik süresi (ölçüm): {time.time() - t0:.0f} sn.", ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"seçilen {dec['secilen']}; " + "; ".join(f"{v}: %80 {pct(stats[(v, 80)]['kapsama'])} %90 {pct(stats[(v, 90)]['kapsama'])}" for v in GAMMAS)
          + f"; süre {time.time() - t0:.0f} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
