"""Hafta 6: V üzerinde tahmin aralığı v2 (CQR): kalibrasyon penceresi 56 gün ve 28 gün, %80 ve %90 düzeyleri.

Veri, özellikler, bloklar aralik_dogrulama.py ile AYNI: ve.load_locked() (T kilidi), reports/final_config.json "ozellikler"/"gruplar",
v2.build_features_v2, lw.make_blocks(V_START, V_END), lw.TRAIN_START. Parametre/özellik/kural AYARI yok.
Her blok ve varyant için eğitim = [train_start, blok_başı - cal_days), kalibrasyon = [blok_başı - cal_days, blok_başı).
Her blok/varyantta 5 LightGBM quantile modeli (alpha 0.05, 0.1, 0.5, 0.9, 0.95); parametreler quantile_conformal.run_cqr içindeki
`base` ile AYNI (oradan kopyalandı; o dosya değiştirilmedi).
CQR: (0.1, 0.9) çifti miscoverage 0.2 (%80); (0.05, 0.95) çifti miscoverage 0.1 (%90). CQR HAM (sıralanmamış) quantile çıktılarıyla yapılır
(önceki koşuyla aynı). np.sort ile sıralanmış quantile'lar SADECE q50, pinball loss ve fan chart içindir.
Çıktılar: reports/probabilistic.md ve data/processed/egitim_oturumu/aralik_v2_tahminleri.parquet (önce dosyalar, sonra kısa konsol özeti).

SÜRE TAHMİNİ (ölçüm DEĞİL): önceki koşu 14 blok x 2 model = 28 eğitim 11 sn sürdü; burada 14 x 2 varyant x 5 model = 140 eğitim,
doğrusal ölçekle ~55 sn; "tahmin" olarak 1-3 dk. Gerçek süre rapora yazılır.

Çalıştırma (repo kökünden, ana .venv): PYTHONUTF8=1 python -m src.models.aralik_v2
Kural: eksik değer doldurulmaz.
"""

import json
import sys
import time

import numpy as np
import pandas as pd

from src.data import price_caps as pc
from src.features import ptf_features as pf
from src.features import ptf_features_v2 as v2
from src.models import lgbm_walkforward as lw
from src.models import quantile_conformal as qc
from src.models import validation_experiments as ve

ROOT = ve.ROOT
CONFIG_PATH = ROOT / "reports" / "final_config.json"
REPORT_PATH = ROOT / "reports" / "probabilistic.md"
PRED_PATH = ve.OUT_DIR / "aralik_v2_tahminleri.parquet"

VARIANTS = {"cal56": 56, "cal28": 28}
ALPHAS = (0.05, 0.1, 0.5, 0.9, 0.95)
LEVELS = {80: ((0.1, 0.9), 0.2, 0.80), 90: ((0.05, 0.95), 0.1, 0.90)}  # düzey -> (alpha çifti, miscoverage, hedef)
FULL_BLOCKS = 13
LAST_BLOCK = 14
PREV = {"kapsama_yuzde": 75.43, "genislik": 854.33, "tol_kapsama": 0.05, "tol_genislik": 0.5}
INCELENMEDI = "Kodu ve toplam sonuçları Claude (sohbet) 31 Ekim'de inceledi: toplam kapsama, genişlik, karar kuralı girdileri ve pinball parquet'ten bağımsız yeniden hesaplandı, aynı çıktı. Blok, saat ve ay tabloları satır satır doğrulanmadı."
RULE = ("KARAR KURALI (sonuçtan önce yazıldı, değiştirilmez): \"28 gün SEÇİLİR ancak ve ancak şunların HEPSİ doğruysa: "
        "(a) hem %80 hem %90 düzeyinde |kapsama − hedef| 56 günlüğe göre en az 1,5 puan KÜÇÜK; "
        "(b) her iki düzeyde ortalama genişlik 56 günlüğe göre en çok %15 büyük; "
        "(c) her iki düzeyde 13 TAM bloğun en az 8'inde |blok kapsaması − hedef| 28 günlükte daha küçük. "
        "Aksi hâlde 56 gün KALIR.\"")
CAUSE = "Zayıf kapsamanın sebebi: BİLMİYORUZ; tavan değişimi hipotezi sınanmadı."

# quantile_conformal.run_cqr içindeki `base` ile aynı
BASE = dict(objective="quantile", learning_rate=0.05, n_estimators=400, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, deterministic=True,
            force_row_wise=True, verbosity=-1)


def md(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def run_variant(feat: pd.DataFrame, cols: list, blocks: list, cal_days: int, name: str) -> tuple:
    """Bir varyant için tüm bloklar. Dönen: tahmin tablosu, {blok: crossing sayıları}."""
    import lightgbm as lgb

    parts, cross = [], {}
    for b in blocks:
        tr_m, cal_m = qc.split_masks(feat["ts"], feat["price"], b["start_ts"], lw.TRAIN_START, cal_days)
        te_m = ((feat["ts"] >= b["start_ts"]) & (feat["ts"] < b["end_ts"])).to_numpy()
        assert tr_m.sum() > 0 and cal_m.sum() > 0 and te_m.sum() > 0, f"{name} blok {b['no']}: boş küme"
        assert feat.loc[tr_m, "ts"].max() < feat.loc[cal_m, "ts"].min(), "eğitim kalibrasyondan önce olmalı"
        assert feat.loc[cal_m, "ts"].max() < b["start_ts"], "kalibrasyon blok başından önce olmalı"
        assert feat.loc[te_m, "ts"].min() >= b["start_ts"], "test blok içinde olmalı"
        pc_, pt_ = {}, {}
        for a in ALPHAS:
            m = lgb.LGBMRegressor(alpha=a, **BASE)
            m.fit(feat.loc[tr_m, cols], feat.loc[tr_m, "price"])
            pc_[a], pt_[a] = m.predict(feat.loc[cal_m, cols]), m.predict(feat.loc[te_m, cols])
        y_cal = feat.loc[cal_m, "price"].to_numpy()
        part = feat.loc[te_m, ["ts"]].copy()
        part["y"] = feat.loc[te_m, "price"].to_numpy()
        part["blok_no"], part["varyant"] = b["no"], name
        raw = np.column_stack([pt_[a] for a in ALPHAS])
        srt = np.sort(raw, axis=1)  # SADECE q50, pinball ve grafik için
        for i, c in enumerate(("q05", "q10", "q50", "q90", "q95")):
            part[c] = srt[:, i]
        for lvl, ((alo, ahi), mis, _) in LEVELS.items():
            lo, hi, _q = qc.cqr_interval(pt_[alo], pt_[ahi], pc_[alo], pc_[ahi], y_cal, mis)  # HAM çıktılarla
            part[f"lo{lvl}"], part[f"hi{lvl}"] = lo, hi
        cross[b["no"]] = {"c80": int((pt_[0.1] > pt_[0.9]).sum()), "c90": int((pt_[0.05] > pt_[0.95]).sum()),
                          "sira_bozuk": int((np.diff(raw, axis=1) < 0).any(axis=1).sum()), "n": int(te_m.sum())}
        parts.append(part)
    return pd.concat(parts, ignore_index=True), cross


def level_stats(d: pd.DataFrame, lvl: int) -> dict:
    lo, hi, y, cap = d[f"lo{lvl}"].to_numpy(), d[f"hi{lvl}"].to_numpy(), d["y"].to_numpy(), d["tavan"].to_numpy()
    a = qc.coverage_and_width(y, lo, hi)
    c = qc.coverage_and_width(y, np.clip(lo, 0.0, cap), np.clip(hi, 0.0, cap))
    return {"n": a["n"], "kap": a["kapsama"], "gen": a["ortalama_genislik"], "kap_klip": c["kapsama"], "gen_klip": c["ortalama_genislik"],
            "alt": float(np.mean(y < lo)), "ust": float(np.mean(y > hi))}


def pinball(y, q, a) -> float:
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(a * e, (a - 1) * e)))


def evaluate_rule(res: dict) -> dict:
    """KARAR KURALI otomatik. res = {varyant: DataFrame}. Dönen: a, b, c ayrı ayrı + sayılar + nihai karar."""
    out = {"a": {}, "b": {}, "c": {}}
    for lvl, (_, _, tgt) in LEVELS.items():
        s56, s28 = level_stats(res["cal56"], lvl), level_stats(res["cal28"], lvl)
        d56, d28 = abs(s56["kap"] - tgt) * 100, abs(s28["kap"] - tgt) * 100
        out["a"][lvl] = (d56 - d28 >= 1.5, f"|kap−hedef| 56g {d56:.2f} puan, 28g {d28:.2f} puan, fark {d56 - d28:.2f} (≥1,5 gerekir)")
        out["b"][lvl] = (s28["gen"] <= 1.15 * s56["gen"], f"genişlik 56g {s56['gen']:.1f}, 28g {s28['gen']:.1f}, oran {s28['gen'] / s56['gen']:.3f} (≤1,15 gerekir)")
        better = 0
        for no in range(1, FULL_BLOCKS + 1):
            e56 = abs(level_stats(res["cal56"][res["cal56"]["blok_no"] == no], lvl)["kap"] - tgt)
            e28 = abs(level_stats(res["cal28"][res["cal28"]["blok_no"] == no], lvl)["kap"] - tgt)
            better += int(e28 < e56)
        out["c"][lvl] = (better >= 8, f"28g daha yakın olan tam blok: {better}/13 (≥8 gerekir)")
    out["hepsi"] = all(v[0] for k in ("a", "b", "c") for v in out[k].values())
    return out


def main() -> int:
    t0 = time.time()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    groups = tuple(cfg["gruplar"])
    assert cfg["kapak_normalizasyonu_K"] is False
    df = ve.load_locked()  # T kilidi
    assert df["ts"].max() < ve.T_LOCK
    feat = v2.build_features_v2(df, groups=groups, normalize=False)
    cols = list(cfg["ozellikler"])
    assert cols == v2.feature_list(pf.FEATURES_WEATHER, groups)
    blocks = lw.make_blocks(ve.V_START, ve.V_END)

    res, cross = {}, {}
    for name, cal_days in VARIANTS.items():
        res[name], cross[name] = run_variant(feat, cols, blocks, cal_days, name)
        res[name]["tavan"] = pc.caps_for_ts(res[name]["ts"]).to_numpy()
    allp = pd.concat(res.values(), ignore_index=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    allp[["ts", "y", "blok_no", "varyant", "q05", "q10", "q50", "q90", "q95", "lo80", "hi80", "lo90", "hi90", "tavan"]].to_parquet(PRED_PATH, index=False)

    # çapraz kontrol: 56 gün + %80 önceki koşuyla
    s = level_stats(res["cal56"], 80)
    ok_kap = abs(s["kap"] * 100 - PREV["kapsama_yuzde"]) <= PREV["tol_kapsama"]
    ok_gen = abs(s["gen"] - PREV["genislik"]) <= PREV["tol_genislik"]
    cross_ok = bool(ok_kap and ok_gen)
    rule = evaluate_rule(res)

    L = []
    if not cross_ok:
        L += ["ÇAPRAZ KONTROL TUTMADI", ""]
    L += [INCELENMEDI, "", RULE, "",
          "# Olasılıksal tahmin (aralık) v2: kalibrasyon 56 gün ve 28 gün, %80 ve %90 (V = "
          f"{ve.V_START.date()}..{ve.V_END.date()})", "",
          f"- Özellikler ({len(cols)}): {', '.join(cols)}; gruplar {list(groups)}; K yok; parametreler quantile_conformal.run_cqr `base` ile aynı.",
          f"- Bloklar: {len(blocks)} (13 tam + blok {LAST_BLOCK} yalnızca 1 gün: n küçük, yorumlanmaz). Bloklar yalnızca V içinde; T'ye dokunulmadı.",
          f"- {CAUSE}", "",
          "## Çapraz kontrol (56 gün + %80 önceki koşuyla)", "",
          f"- Kapsama {s['kap'] * 100:.4f}% (önceki {PREV['kapsama_yuzde']}, tolerans {PREV['tol_kapsama']} puan): {'TUTTU' if ok_kap else 'TUTMADI'}",
          f"- Genişlik {s['gen']:.4f} (önceki {PREV['genislik']}, tolerans {PREV['tol_genislik']}): {'TUTTU' if ok_gen else 'TUTMADI'}", "",
          "## Karar kuralı sonucu (otomatik)", ""]
    for k, ad in (("a", "(a) kapsama hedefe ≥1,5 puan daha yakın"), ("b", "(b) genişlik ≤ %15 büyük"), ("c", "(c) ≥8/13 blokta hedefe daha yakın")):
        for lvl in LEVELS:
            v = rule[k][lvl]
            L.append(f"- {ad}, %{lvl}: **{'DOĞRU' if v[0] else 'YANLIŞ'}**; {v[1]}")
    L += ["", f"**Sonuç: {'28 gün SEÇİLİR' if rule['hepsi'] else '56 gün KALIR'}** (a, b, c'nin hepsi doğru olmadıkça 56 gün kalır).", ""]

    # toplam tablo
    rows = []
    for name in VARIANTS:
        for lvl, (_, _, tgt) in LEVELS.items():
            t = level_stats(res[name], lvl)
            plan = ("evet" if 0.88 <= t["kap"] <= 0.92 else "hayır") if lvl == 90 else "-"
            rows.append((name, f"%{lvl}", t["n"], pct(t["kap"]), f"{t['gen']:.1f}", pct(t["kap_klip"]), f"{t['gen_klip']:.1f}",
                         pct(t["alt"]), pct(t["ust"]), plan))
    L += ["## Toplam (kırpılmamış | [0, günün tavanı] kırpılmış)", "",
          md(["varyant", "düzey", "n", "kapsama", "genişlik", "kapsama (kırpılmış)", "genişlik (kırpılmış)", "alt ıskalama", "üst ıskalama",
              "plan kontrolü: %90 kapsama %88-%92 içinde mi"], rows), "",
          "Plan kontrolü açıklayıcıdır; sonuca göre ayar yapılmaz.", ""]

    # crossing
    rows = []
    for name in VARIANTS:
        tot = {k: sum(v[k] for v in cross[name].values()) for k in ("c80", "c90", "sira_bozuk", "n")}
        rows.append((name, tot["n"], tot["c80"], tot["c90"], tot["sira_bozuk"]))
    L += ["## Ham quantile crossing (CQR öncesi, sıralanmamış çıktılar)", "",
          md(["varyant", "satır", "q10 > q90", "q05 > q95", "5 quantile'ın sırası bozuk satır"], rows), ""]

    # pinball
    rows = []
    for name in VARIANTS:
        d = res[name]
        pb = [pinball(d["y"], d[c], a) for c, a in zip(("q05", "q10", "q50", "q90", "q95"), ALPHAS)]
        rows.append((name, *[f"{x:.2f}" for x in pb], f"{np.mean(pb):.2f}", f"{float(np.mean(np.abs(d['y'] - d['q50']))):.2f}"))
    L += ["## Pinball loss (sıralanmış quantile'lar) ve q50 MAE (açıklayıcı)", "",
          md(["varyant", "q05", "q10", "q50", "q90", "q95", "ortalama", "q50 MAE (TL/MWh)"], rows), ""]

    # blok, saat, ay
    def grp_table(key_fn, key_name, mark_last=False):
        keys = sorted(set(key_fn(res["cal56"]).unique()) | set(key_fn(res["cal28"]).unique()))
        rows = []
        for k in keys:
            row = [f"{k}" + (" (n küçük, yorumlanmaz)" if mark_last and k == LAST_BLOCK else "")]
            n = 0
            for name in VARIANTS:
                d = res[name][key_fn(res[name]) == k]
                n = len(d)
                for lvl in LEVELS:
                    t = level_stats(d, lvl)
                    row += [pct(t["kap"]), f"{t['gen']:.1f}"]
            rows.insert(len(rows), [row[0], n, *row[1:]])
        hdr = [key_name, "n"] + [f"{nm} %{lvl} {w}" for nm in VARIANTS for lvl in LEVELS for w in ("kapsama", "genişlik")]
        return md(hdr, rows)

    L += ["## Blok başına", "", grp_table(lambda d: d["blok_no"], "blok", mark_last=True), "",
          "## Saate göre", "", grp_table(lambda d: d["ts"].dt.hour, "saat"), "",
          "## Aya göre", "", grp_table(lambda d: d["ts"].dt.strftime("%Y-%m"), "ay"), "",
          f"Betik süresi (ölçüm): {time.time() - t0:.0f} sn.", ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"çapraz kontrol {'TUTTU' if cross_ok else 'TUTMADI'}; 56g %80 kapsama {s['kap'] * 100:.2f}% genişlik {s['gen']:.0f}; "
          f"karar: {'28 gün SEÇİLİR' if rule['hepsi'] else '56 gün KALIR'}; süre {time.time() - t0:.0f} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
