"""Doğrulama dönemi V üzerinde özellik deneyleri (R0..R3).

V = 2024-10-21..2025-10-20. TEST DÖNEMİ KİLİDİ: veri yüklenir yüklenmez ts >= 2025-10-21 00:00 (Europe/Istanbul)
satırları atılır; test dönemi T'ye bu modülde hiç dokunulmaz.
Yürüyen pencere: lgbm_walkforward.make_blocks / split_for_block (28 gün, genişleyen pencere), sabit LGBM_PARAMS.

SIRA ve KURAL önceden sabit (sonuca bakıp değiştirilmez):
  R0 = FEATURES_WEATHER (referans), R0b = FEATURES_BASE (sadece bilgi),
  R1 = R0 + Grup H, R2 = (R0 veya seçilmişse R1) + Grup Z, R3 = seçilen kümenin K varyantı.
Seçme kuralı: V-MAE bir önceki seçiliye göre en az %1,0 (göreli) iyi VE 13 tam bloktan en az 8'inde daha iyi.

Çalıştırma (repo kökünden): python -m src.models.validation_experiments
Kural: eksik değer doldurulmaz.
"""

import json
from pathlib import Path

import pandas as pd

from src.features import ptf_features as pf
from src.features import ptf_features_v2 as v2
from src.models import baselines as bl
from src.models import lgbm_walkforward as lw

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "dataset_weather.parquet"
HOLIDAY_CSV = ROOT / "data" / "external" / "turkiye_tatil_gunleri.csv"
OUT_DIR = ROOT / "data" / "processed" / "egitim_oturumu"
REPORT_PATH = ROOT / "reports" / "dogrulama_deneyleri.md"
SEC_PATH = OUT_DIR / "secilen_ozellikler.json"

TZ = lw.TZ
V_START = pd.Timestamp("2024-10-21")
V_END = pd.Timestamp("2025-10-20")
T_LOCK = pd.Timestamp("2025-10-21 00:00", tz=TZ)
MIN_REL_GAIN = 0.01
MIN_BLOCKS_BETTER = 8
FULL_BLOCKS = 13


def load_locked(path: Path = DATA_PATH) -> pd.DataFrame:
    """Veriyi yükler ve HEMEN test dönemini (ts >= 2025-10-21) atar."""
    df = pd.read_parquet(path)
    df = df[df["ts"] < T_LOCK].reset_index(drop=True)
    assert df["ts"].max() < T_LOCK
    return df


def walkforward_config(df, start, end, groups=(), normalize=False, base_features=None,
                       params=None, train_start=lw.TRAIN_START, feat=None):
    """Tek yapılandırma için yürüyen pencere. Dönen: DataFrame(ts, price, tahmin, blok_no), özellik listesi, bloklar.

    base_features: pf.FEATURES_WEATHER (varsayılan) ya da pf.FEATURES_BASE.
    normalize=True (K): hedef = price/tavan, ağırlık = tavan, tahmin tavanla çarpılıp TL'ye çevrilir.
    """
    import lightgbm as lgb

    base_features = list(base_features or pf.FEATURES_WEATHER)
    include_weather = base_features == list(pf.FEATURES_WEATHER)
    if feat is None:
        feat = v2.build_features_v2(df, groups=groups, normalize=normalize, include_weather=include_weather)
    cols = v2.feature_list(base_features, groups)
    blocks = lw.make_blocks(start, end)
    parts = []
    for b in blocks:
        train_m, test_m = lw.split_for_block(feat, b, train_start)
        assert not (train_m & test_m).any()
        tr, te = feat[train_m], feat[test_m]
        assert tr["ts"].max() < b["start_ts"], "ZAMAN KURALI İHLALİ"
        if len(te) == 0:
            continue
        model = lgb.LGBMRegressor(**(params or lw.LGBM_PARAMS))
        if normalize:
            model.fit(tr[cols], tr["target"], sample_weight=tr["cap"])
            pred = model.predict(te[cols]) * te["cap"].to_numpy()
        else:
            model.fit(tr[cols], tr["target"])
            pred = model.predict(te[cols])
        part = te[["ts", "price"]].copy()
        part["tahmin"] = pred
        part["blok_no"] = b["no"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True), cols, blocks


def join_preds(bt: pd.DataFrame, runs: dict) -> pd.DataFrame:
    """bt'ye (gun, saat) üzerinden her yapılandırmanın tahminini t_<ad> olarak ve blok_no'yu ekler."""
    out = bt
    for i, (name, p) in enumerate(runs.items()):
        gun, saat = lw._day_hour(p["ts"])
        q = pd.DataFrame({"gun": gun.astype(bt["gun"].dtype), "saat": saat, f"t_{name}": p["tahmin"].to_numpy()})
        if i == 0:
            q["blok_no"] = p["blok_no"].to_numpy()
        out = out.merge(q, on=["gun", "saat"], how="left", validate="one_to_one")
    return out


def common(joined: pd.DataFrame, names) -> pd.DataFrame:
    return joined.dropna(subset=["gercek", *bl.MODELS.values(), *[f"t_{n}" for n in names]])


def mae_of(cm: pd.DataFrame, col: str) -> float:
    return float((cm[col] - cm["gercek"]).abs().mean())


def block_mae(cm: pd.DataFrame, col: str) -> pd.Series:
    return (cm[col] - cm["gercek"]).abs().groupby(cm["blok_no"]).mean()


def decide(cur_mae, new_mae, cur_blk: pd.Series, new_blk: pd.Series) -> dict:
    rel = (cur_mae - new_mae) / cur_mae
    full = [b for b in cur_blk.index if b <= FULL_BLOCKS]
    better = int((new_blk.loc[full] < cur_blk.loc[full]).sum())
    return {"rel_gain": float(rel), "blocks_better": better, "of": len(full),
            "select": bool(rel >= MIN_REL_GAIN and better >= MIN_BLOCKS_BETTER)}


def main() -> int:
    df = load_locked()
    prices = bl.price_table(df)
    bt = bl.run_backtest(prices, start=V_START, end=V_END)
    has_h = HOLIDAY_CSV.exists()

    runs, blocks = {}, None
    for name, kw in (("R0", {}), ("R0b", dict(base_features=pf.FEATURES_BASE))):
        runs[name], _, blocks = walkforward_config(df, V_START, V_END, **kw)

    steps = ([("R1", "H")] if has_h else []) + [("R2", "Z"), ("R3", "K")]
    cur_groups, cur_norm, cur_key, status = (), False, "R0", {}
    for name, g in steps:
        groups = cur_groups + (g,) if g in ("H", "Z") else cur_groups
        norm = True if g == "K" else cur_norm
        runs[name], _, blocks = walkforward_config(df, V_START, V_END, groups=groups, normalize=norm)
        cm = common(join_preds(bt, runs), runs)  # o ana kadar çalışan her şeyin ortak satırları
        d = decide(mae_of(cm, f"t_{cur_key}"), mae_of(cm, f"t_{name}"),
                   block_mae(cm, f"t_{cur_key}"), block_mae(cm, f"t_{name}"))
        d.update(vs=cur_key, groups=groups, normalize=norm, cur_mae=mae_of(cm, f"t_{cur_key}"),
                 new_mae=mae_of(cm, f"t_{name}"))
        status[name] = d
        if d["select"]:
            cur_groups, cur_norm, cur_key = groups, norm, name

    cm = common(join_preds(bt, runs), runs)
    base_d1 = mae_of(cm, "tahmin_d1")
    tab = [(n, len(cm), round(mae_of(cm, f"t_{n}"), 2), round(mae_of(cm, f"t_{n}") / base_d1, 3)) for n in runs]
    tab += [(nm, len(cm), round(mae_of(cm, bl.MODELS[m]), 2), round(mae_of(cm, bl.MODELS[m]) / base_d1, 3))
            for m, nm in (("d1", "D-1"), ("d7", "D-7"), ("kural", "Kural"))]
    cm = cm.assign(ay=cm["gun"].dt.strftime("%Y-%m"))
    monthly = pd.DataFrame({n: (cm[f"t_{n}"] - cm["gercek"]).abs().groupby(cm["ay"]).mean() for n in runs})
    monthly["D-1"] = (cm["tahmin_d1"] - cm["gercek"]).abs().groupby(cm["ay"]).mean()
    blkt = pd.DataFrame({n: block_mae(cm, f"t_{n}") for n in runs})

    final_cols = v2.feature_list(pf.FEATURES_WEATHER, cur_groups)
    sec = {
        "secilen_yapilandirma": cur_key,
        "gruplar": list(cur_groups),
        "kapak_normalizasyonu_K": bool(cur_norm),
        "ozellikler": final_cols,
        "hedef_bicimi": ("price / tavan(gun); TL'ye cevirme: tahmin * tavan(D); egitim agirligi = tavan(D)"
                         if cur_norm else "price (TL/MWh), ham"),
        "tatil_tablosu_var": has_h,
        "claude_incelemedi": True,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEC_PATH.write_text(json.dumps(sec, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# Doğrulama deneyleri (V = 2024-10-21..2025-10-20)", "",
         "**Claude (sohbet) bu kodu ve sonucu İNCELEMEDİ.** Test dönemi T'ye dokunulmadı "
         "(ts >= 2025-10-21 veri yüklenir yüklenmez atıldı).", "",
         f"Sabit varsayılan parametreler (lgbm_walkforward.LGBM_PARAMS). {len(blocks)} blok (13 tam + son kısa blok). "
         f"Seçme kuralı: göreli V-MAE iyileşmesi >= %{MIN_REL_GAIN * 100:.1f} VE 13 tam bloktan >= {MIN_BLOCKS_BETTER}'inde daha iyi.",
         f"Tatil tablosu (Grup H) var mı: {'evet' if has_h else 'HAYIR, R1 atlandı'}.", "",
         "## V-MAE (tüm yapılandırmalar aynı ortak satırlarda)", "",
         bl._md_table(["yapılandırma", "n", "MAE (TL/MWh)", "rMAE (D-1'e göre)"], tab), "",
         "## Seçme kuralı sonuçları", ""]
    rows = [(n, s["vs"], list(s["groups"]), s["normalize"], f"{s['cur_mae']:.2f}", f"{s['new_mae']:.2f}",
             f"{s['rel_gain'] * 100:+.2f}%", f"{s['blocks_better']}/{s['of']}", "SEÇİLDİ" if s["select"] else "seçilmedi")
            for n, s in status.items()]
    L += [bl._md_table(["aday", "karşı", "gruplar", "K", "önceki MAE", "yeni MAE", "göreli kazanç", "iyi blok", "karar"], rows), "",
          "(Her karar, o adımda o ana kadar çalışmış yapılandırmaların ortak satırlarında verildi; satır sayısı üstteki tablodan az farklı olabilir.)", "",
          f"**Seçilen: {cur_key}** — gruplar {list(cur_groups)}, K={cur_norm}.", "",
          "## Aylık MAE", "",
          bl._md_table(["ay", *monthly.columns], [(i, *[f"{v:.1f}" for v in r]) for i, r in monthly.iterrows()]), "",
          "## Blok başına MAE", "",
          bl._md_table(["blok", *blkt.columns], [(int(i), *[f"{v:.1f}" for v in r]) for i, r in blkt.iterrows()]), ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"secilen": cur_key, "tab": tab}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
