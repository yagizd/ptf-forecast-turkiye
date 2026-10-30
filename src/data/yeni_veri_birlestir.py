"""Eski + yeni ham dosyalardan data/processed/dataset_weather_guncel.parquet üretir.

Eski dataset_weather.parquet'e dokunulmaz. 2026-10-20'ye kadar olan satırlar eskisiyle BİREBİR karşılaştırılır.
Tabloya yalnızca PTF, tüketim, yük planı VE hava sütunlarının hepsinin dolu olduğu, kesintisiz ilk yeni günler girer.
Kural: eksik değer doldurulmaz (skipna=False).
Çalıştırma: python -m src.data.yeni_veri_birlestir
"""

import time

import pandas as pd

from src.data import build_dataset as bd
from src.data import build_weather_dataset as bw
from src.data.weather import CITIES

ROOT = bd.ROOT
OLD = ROOT / "data" / "processed" / "dataset_weather.parquet"
OUT = ROOT / "data" / "processed" / "dataset_weather_guncel.parquet"
RPT = ROOT / "data" / "processed" / "egitim_oturumu_30Ekim" / "veri_guncelleme_raporu.md"
LAST_OLD = pd.Timestamp("2026-10-20 23:00", tz=bd.TZ)
LOG = ROOT / "data" / "processed" / "egitim_oturumu_30Ekim" / "ilerleme.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    old = pd.read_parquet(OLD)
    frames, mism = bd.read_all()
    base, dups = bd.combine(frames)  # ts, price, consumption, lep (tüm ham dosyalar, eski + yeni)
    wf = {}
    for city in CITIES:
        files = sorted((bw.WEATHER_DIR / city).glob("weather_*.csv"))
        wf[city] = bw.prepare_city_frame(pd.concat((pd.read_csv(f) for f in files), ignore_index=True))
        assert not wf[city]["ts"].duplicated().any(), city
    feats = bw.build_weather_features(wf)
    full, info = bw.merge_weather(base, feats)

    # gün bazında tamlık (yeni günler)
    cols = [c for c in full.columns if c != "ts"]
    day = full["ts"].dt.tz_localize(None).dt.normalize()
    new = full[full["ts"] > LAST_OLD].copy()
    nday = day[new.index]
    per = new[cols].notna().groupby(nday).sum()
    cnt = new.groupby(nday).size()
    rows = []
    last_ok = None
    prefix_ok = True
    for d in per.index:
        full_day = bool((per.loc[d] == 24).all() and cnt.loc[d] == 24)
        only_weather_missing = bool((per.loc[d, ["price", "consumption", "lep"]] == 24).all() and not full_day)
        rows.append((d.date(), int(cnt.loc[d]), *[int(per.loc[d, c]) for c in cols], full_day, only_weather_missing))
        if full_day and prefix_ok:
            last_ok = d
        else:
            prefix_ok = False
    assert last_ok is not None, "tam yeni gün yok"
    cut = pd.Timestamp(last_ok.date().isoformat() + " 23:00", tz=bd.TZ)
    out = full[full["ts"] <= cut].reset_index(drop=True)

    # eski tabloyla birebir karşılaştırma
    head = out[out["ts"] <= LAST_OLD].reset_index(drop=True)
    diffs = []
    same_shape = list(head.columns) == list(old.columns) and len(head) == len(old)
    if same_shape:
        for c in old.columns:
            try:
                pd.testing.assert_series_equal(old[c].reset_index(drop=True), head[c], check_exact=True)
            except AssertionError as e:
                diffs.append((c, str(e)[:200]))
    else:
        diffs.append(("şekil", f"eski {old.shape} {list(old.columns)} / yeni baş {head.shape}"))

    nulls = {c: int(out[c].isna().sum()) for c in out.columns}
    new_out = out[out["ts"] > LAST_OLD]
    nulls_new = {c: int(new_out[c].isna().sum()) for c in out.columns}
    ndays = new_out["ts"].dt.normalize().nunique()
    if not diffs:
        if OUT.exists():
            raise FileExistsError(OUT)
        out.to_parquet(OUT, index=False)
    L = ["# Veri güncelleme raporu", "", "**Claude (sohbet) incelemedi.**", "",
         f"- Yeni tablo: {OUT.name}; satır {len(out)}; ilk {out['ts'].iloc[0]}; son {out['ts'].iloc[-1]}",
         f"- Eski tablo satır {len(old)}; son {old['ts'].iloc[-1]}",
         f"- Yeni gün: {ndays}; yeni saat: {len(new_out)}",
         f"- Eski bölümle birebir (tüm sütunlar, NaN dahil, tam eşitlik): {'EVET' if not diffs else 'HAYIR'}; fark: {diffs or 'yok'}",
         f"- Hava birleştirme bilgisi: {info}",
         f"- null sayıları (tüm tablo): {nulls}",
         f"- null sayıları (yalnız yeni günler): {nulls_new}",
         f"- Kaynak dosyalarda tekrar eden ts: {dups}; saat uyumsuzluğu: {mism}", "",
         "## Yeni günlerde kaynak başına dolu saat (24 olmalı; tabloya yalnızca hepsi tam olanlar girer)", "",
         bd._md_table(["gün", "satır", *cols, "hepsi tam", "yalnız hava eksik"], rows), ""]
    RPT.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"birlestirme: yeni gün {ndays}, yeni saat {len(new_out)}, satır {len(out)}, son {out['ts'].iloc[-1]}, "
        f"eski bölümle birebir {'EVET' if not diffs else 'HAYIR ' + str(diffs)}")
    log(f"null (tüm): {nulls}")
    log(f"null (yeni): {nulls_new}")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
