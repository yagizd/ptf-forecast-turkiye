"""Gerçek paket ve gerçek veriyle servis testleri. Paket (data/processed/model_paketi_v1/) veya veri yoksa ATLANIR.

Faz 2'de paket üretildikten sonra çalışır: python -m unittest tests.test_serving_gercek -v   (unittest discover YOK)
(e) bilgi çıktısı assert DEĞİLDİR: data/processed/model_paketi_v1/gercek_test_bilgi.txt dosyasına ve ekrana yazılır.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data" / "processed" / "model_paketi_v1"
TABLE_PATH = ROOT / "data" / "processed" / "dataset_weather_guncel.parquet"
FINAL_PATH = ROOT / "data" / "processed" / "final_tahminleri.parquet"  # sütunlar (final_test_run.py): ts, price, tahmin_final, blok_no
INFO_PATH = BUNDLE_DIR / "gercek_test_bilgi.txt"
READY = (BUNDLE_DIR / "meta.json").exists() and TABLE_PATH.exists() and FINAL_PATH.exists()
TZ = "Europe/Istanbul"


@unittest.skipUnless(READY, "paket/veri yok (Faz 2'de paket üretildikten sonra çalışır)")
class TestGercek(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.features import ptf_features_v2 as v2
        from src.models import aralik_gunluk as ag
        from src.serving import predictor as pr

        cls.v2, cls.ag, cls.pr = v2, ag, pr
        pr.clear_cache()
        cls.bundle = pr.load_bundle(BUNDLE_DIR)
        cls.df = pd.read_parquet(TABLE_PATH)
        cls.table = pr.make_table(cls.df, cls.bundle["c_day"], pr.sha256_file(TABLE_PATH))
        cls.cache = pr.quantile_cache(cls.bundle, cls.table)
        cls.C = cls.bundle["c_day"]
        cls.last = cls.table["last_day"]
        cls.days = list(pd.date_range(cls.C, cls.last, freq="D"))

    def test_a_nokta_final_tahminleriyle_ayni(self):
        fin = pd.read_parquet(FINAL_PATH)
        fin = fin[fin["blok_no"] == 13]  # blok 13: 2026-09-22 .. 2026-10-19, eğitim [TRAIN_START, 2026-09-22)
        self.assertEqual(len(fin), 28 * 24)
        mine = pd.Series(self.cache["point"], index=pd.DatetimeIndex(self.cache["ts"]))
        got = mine.reindex(pd.DatetimeIndex(fin["ts"])).to_numpy()
        self.assertFalse(np.isnan(got).any())
        np.testing.assert_allclose(got, fin["tahmin_final"].to_numpy(), atol=1e-6)

    def test_b_aci_run_series_ile_ayni(self):
        ag, pr = self.ag, self.pr
        c_no, last_no = int((self.C - pr.EPOCH).days), int((self.last - pr.EPOCH).days)
        day = self.cache["day"]
        test_days = day[day >= c_no]
        for level, (a_lo, a_hi, target) in pr.LEVELS.items():
            block = {"no": 1, "day": day, "y": self.cache["y"], "qlo": self.cache[pr.Q_OF[a_lo]], "qhi": self.cache[pr.Q_OF[a_hi]],
                     "first_day": c_no, "last_day": last_no}
            outs, _ = ag.run_series([block], target, "ACI", 0.05, cal_days=56)
            for D in self.days:
                out = pr.predict_day(self.bundle, self.table, D)
                m = test_days == int((D - pr.EPOCH).days)
                cap = pr.pc.cap_for_day(D)
                lo, hi = np.clip(outs[0]["lo"][m], 0.0, cap), np.clip(outs[0]["hi"][m], 0.0, cap)
                bos = outs[0]["Q"][m] == -np.inf
                np.testing.assert_array_equal([h[f"bos{level}"] for h in out["saatler"]], bos)
                if not bos.any():
                    np.testing.assert_allclose([h[f"lo{level}"] for h in out["saatler"]], lo, atol=1e-9, err_msg=f"{D.date()} lo{level}")
                    np.testing.assert_allclose([h[f"hi{level}"] for h in out["saatler"]], hi, atol=1e-9, err_msg=f"{D.date()} hi{level}")
                np.testing.assert_allclose([h[f"alpha{level}"] for h in out["saatler"]], outs[0]["alpha"][m], atol=1e-12)

    def test_c_maskelenmis_tabloda_ozellikler_ayni(self):
        v2, pr = self.v2, self.pr
        for D in (self.C, self.C + pd.Timedelta(days=10), self.last):
            cut = D.tz_localize(TZ)
            masked = self.df.copy()
            masked.loc[masked["ts"] >= cut, "price"] = np.nan
            f_full = v2.build_features_v2(self.df, groups=pr.GROUPS, normalize=False)
            f_mask = v2.build_features_v2(masked, groups=pr.GROUPS, normalize=False)
            cols = list(self.bundle["meta"]["ozellikler"])
            day = (f_full["ts"] >= cut) & (f_full["ts"] < cut + pd.Timedelta(days=1))
            self.assertEqual(int(day.sum()), 24)
            pd.testing.assert_frame_equal(f_full.loc[day, cols].reset_index(drop=True), f_mask.loc[day, cols].reset_index(drop=True),
                                          check_exact=True)

    def test_d_pencere_hatalari(self):
        pr = self.pr
        with self.assertRaises(ValueError):
            pr.predict_day(self.bundle, self.table, self.C - pd.Timedelta(days=1))
        with self.assertRaises(ValueError):
            pr.predict_day(self.bundle, self.table, self.last + pd.Timedelta(days=1))

    def test_e_bilgi(self):
        """Assert DEĞİL: nokta tahminin aralık dışında kalan saat sayısı, boş aralık ve sınırsız-Q sayıları."""
        ag, pr = self.ag, self.pr
        dis80 = dis90 = n = bos80 = bos90 = 0
        sinirsiz = {80: 0, 90: 0}
        for D in self.days:
            out = pr.predict_day(self.bundle, self.table, D)
            for h in out["saatler"]:
                n += 1
                if h["bos80"]:
                    bos80 += 1
                elif not (h["lo80"] <= h["nokta"] <= h["hi80"]):
                    dis80 += 1
                if h["bos90"]:
                    bos90 += 1
                elif not (h["lo90"] <= h["nokta"] <= h["hi90"]):
                    dis90 += 1
            c_no, d_no = int((self.C - pr.EPOCH).days), int((D - pr.EPOCH).days)
            for level in pr.LEVELS:
                if pr.level_state(self.cache, level, c_no, d_no)[1] == np.inf:
                    sinirsiz[level] += 1
        lines = [f"replay {self.C.date()} .. {self.last.date()}: {len(self.days)} gün, {n} saat",
                 f"nokta tahmin %80 aralığın DIŞINDA: {dis80} saat; %90 aralığın dışında: {dis90} saat",
                 f"boş aralık saat: %80 {bos80}, %90 {bos90}",
                 f"sınırsız-Q (Q=+inf) gün: %80 {sinirsiz[80]}, %90 {sinirsiz[90]}"]
        INFO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
