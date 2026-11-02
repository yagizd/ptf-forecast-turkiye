"""Servis tahmin modülü testleri: SENTETİK veri, küçük LightGBM (n_estimators<=5). Gerçek veri/paket KULLANILMAZ.

Çalıştırma (yalnız bu dosya): python -m unittest tests.test_serving_predictor -v
LightGBM kurulu değilse testler atlanır.
"""

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HAS_LGB = importlib.util.find_spec("lightgbm") is not None
TZ = "Europe/Istanbul"
C = pd.Timestamp("2025-05-01")
TRAIN_START = pd.Timestamp("2024-12-08", tz=TZ)

if HAS_LGB:
    from src.features import ptf_features as pf
    from src.features import ptf_features_v2 as v2
    from src.models import aralik_gunluk as ag
    from src.models import aralik_v2 as av2
    from src.serving import bundle as bd
    from src.serving import predictor as pr
    from src.data import price_caps as pc

SMALL = dict(n_estimators=5, num_leaves=7, min_child_samples=5)


def synth_df(start="2024-12-01", days=230, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 24, freq="h", tz=TZ)
    hour = ts.hour.to_numpy()
    price = np.clip(1500 + 700 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 250, len(ts)), 0, 3400)
    df = pd.DataFrame({"ts": ts, "price": price})
    for c in pf.WEATHER_FEATURES:
        df[c] = rng.normal(size=len(ts))
    return df


def small_params():
    point = dict(objective="regression_l1", learning_rate=0.1, random_state=42, deterministic=True,
                 force_row_wise=True, verbosity=-1, subsample_freq=1, subsample=0.8, colsample_bytree=0.8, **SMALL)
    q = dict(av2.BASE)
    q.update(SMALL)
    return point, q


def cols_h():
    return v2.feature_list(pf.FEATURES_WEATHER, ("H",))


@unittest.skipUnless(HAS_LGB, "lightgbm yok")
class TestServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.df = synth_df()
        cls.table = pr.make_table(cls.df, C)
        cls.point_params, cls.q_base = small_params()
        cls.dir = Path(cls.tmp.name) / "paket"
        cls.meta, cls.info = bd.build_bundle(cls.table, cls.dir, cols_h(), cls.point_params, cls.q_base, TRAIN_START)
        cls.bundle = pr.load_bundle(cls.dir)
        cls.last = cls.table["last_day"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        pr.clear_cache()

    @staticmethod
    def strip(out):
        o = copy.deepcopy(out)
        for h in o["saatler"]:
            h.pop("gercek")
        return o

    def test_kaydet_yukle_birebir(self):
        self.assertTrue(all(self.info["kontrol_birebir"].values()))
        cache = pr.quantile_cache(self.bundle, self.table)
        start = (C - pd.Timedelta(days=56)).tz_localize(TZ)
        X = self.table["feat"].loc[self.table["feat"]["ts"] >= start, cols_h()]
        for name, booster in self.bundle["models"].items():
            np.testing.assert_array_equal(cache[name], booster.predict(X))

    def test_sizinti_yok(self):
        D = C + pd.Timedelta(days=10)
        df2 = self.df.copy()
        late = df2["ts"] >= D.tz_localize(TZ)
        rng = np.random.default_rng(77)
        df2.loc[late, "price"] = rng.uniform(0, 3400, int(late.sum()))
        t2 = pr.make_table(df2, C)
        self.assertNotEqual(self.table["sha256"], t2["sha256"])
        a = pr.predict_day(self.bundle, self.table, D)
        b = pr.predict_day(self.bundle, t2, D)
        self.assertNotEqual([h["gercek"] for h in a["saatler"]], [h["gercek"] for h in b["saatler"]])  # yalnız bilgi alanı değişir
        self.assertEqual(self.strip(a), self.strip(b))

    def test_pencere_hatalari(self):
        with self.assertRaisesRegex(ValueError, "egitimde goruldu"):
            pr.predict_day(self.bundle, self.table, C - pd.Timedelta(days=1))
        with self.assertRaisesRegex(ValueError, "tabloda yok"):
            pr.predict_day(self.bundle, self.table, self.last + pd.Timedelta(days=1))
        out = pr.predict_day(self.bundle, self.table, self.last)  # son gün geçerli
        self.assertEqual(len(out["saatler"]), 24)
        self.assertFalse(out["egitimde_goruldu"])

    def test_durumsuzluk(self):
        D, D2 = C + pd.Timedelta(days=12), C + pd.Timedelta(days=3)
        pr.clear_cache()
        t1 = pr.make_table(self.df, C)
        p2_once = pr.predict_day(self.bundle, t1, D2)
        p1_sonra = pr.predict_day(self.bundle, t1, D)
        pr.clear_cache()
        t2 = pr.make_table(self.df, C)
        p1_tek = pr.predict_day(self.bundle, t2, D)
        p2_sonra = pr.predict_day(self.bundle, t2, D2)
        self.assertEqual(p1_sonra, p1_tek)
        self.assertEqual(p2_once, p2_sonra)

    def test_aci_esdegerligi(self):
        D = C + pd.Timedelta(days=9)
        cache = pr.quantile_cache(self.bundle, self.table)
        c_no, d_no = int((C - pr.EPOCH).days), int((D - pr.EPOCH).days)
        day = cache["day"]
        out = pr.predict_day(self.bundle, self.table, D)
        for level, (a_lo, a_hi, target) in pr.LEVELS.items():
            block = {"no": 1, "day": day, "y": cache["y"], "qlo": cache[pr.Q_OF[a_lo]], "qhi": cache[pr.Q_OF[a_hi]],
                     "first_day": c_no, "last_day": int((self.last - pr.EPOCH).days)}
            outs, _ = ag.run_series([block], target, "ACI", 0.05, cal_days=56)
            test_days = day[day >= c_no]
            m = test_days == d_no
            cap = pc.cap_for_day(D)
            lo = np.clip(outs[0]["lo"][m], 0.0, cap)
            hi = np.clip(outs[0]["hi"][m], 0.0, cap)
            al = outs[0]["alpha"][m]
            np.testing.assert_allclose([h[f"lo{level}"] for h in out["saatler"]], lo, atol=1e-9)
            np.testing.assert_allclose([h[f"hi{level}"] for h in out["saatler"]], hi, atol=1e-9)
            np.testing.assert_allclose([h[f"alpha{level}"] for h in out["saatler"]], al, atol=1e-12)

    def test_surum_uyusmazligi(self):
        d2 = Path(self.tmp.name) / "paket_bozuk"
        shutil.copytree(self.dir, d2)
        meta = json.loads((d2 / "meta.json").read_text(encoding="utf-8"))
        meta["surumler"]["lightgbm"] = "0.0.0"
        (d2 / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            pr.load_bundle(d2)

    def test_var_olan_paketin_ustune_yazmaz(self):
        with self.assertRaises(FileExistsError):
            bd.build_bundle(self.table, self.dir, cols_h(), self.point_params, self.q_base, TRAIN_START)

    def test_tablo_dogrulama(self):
        bad = self.df.drop(index=self.df.index[-5])  # son gün eksik saat
        with self.assertRaises(ValueError):
            pr.make_table(bad, C)
        naive = self.df.copy()
        naive["ts"] = naive["ts"].dt.tz_localize(None)
        with self.assertRaises(Exception):
            pr.make_table(naive, C)


if __name__ == "__main__":
    unittest.main()
