"""ptf_features_v2 için testler (sentetik veri, ağ yok)."""

import unittest

import numpy as np
import pandas as pd

from src.data import price_caps as pc
from src.features import ptf_features as pf
from src.features import ptf_features_v2 as v2

TZ = "Europe/Istanbul"
HOL = pd.DataFrame({
    "tarih": pd.to_datetime(["2025-06-06", "2025-06-07", "2025-06-30", "2025-07-15"]),
    "tur": ["arife_yarim_gun", "resmi_tatil", "idari_izin_kopru", "resmi_tatil"],
})
COVER = ("2025-05-01", "2025-09-30")
KW = dict(holiday_table=HOL, cover=COVER)


def synth(days=70, seed=0, start="2025-05-20"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 24, freq="h", tz=TZ)
    price = rng.uniform(0, 3400, len(ts))
    price[rng.random(len(ts)) < 0.1] = 0.0
    price[rng.random(len(ts)) < 0.05] = 3400.0
    d = pd.DataFrame({"ts": ts, "price": price})
    for c in pf.WEATHER_FEATURES:
        d[c] = rng.normal(size=len(ts))
    return d


def feats(df, groups, normalize):
    return v2.build_features_v2(df, groups=groups, normalize=normalize, **KW)


class TestLeak(unittest.TestCase):
    def check(self, groups, normalize):
        df = synth()
        D = pd.Timestamp("2025-07-10", tz=TZ)
        a = feats(df, groups, normalize)
        df2 = df.copy()
        m = df2["ts"] >= D
        df2.loc[m, "price"] = np.random.default_rng(99).uniform(0, 3400, m.sum())
        b = feats(df2, groups, normalize)
        day = (a["ts"] >= D) & (a["ts"] < D + pd.Timedelta(days=1))
        cols = [c for c in a.columns if c not in ("price", "target")]
        pd.testing.assert_frame_equal(a.loc[day, cols].reset_index(drop=True), b.loc[day, cols].reset_index(drop=True))
        self.assertEqual(int(day.sum()), 24)

    def test_H(self):
        self.check(("H",), False)

    def test_Z(self):
        self.check(("Z",), False)

    def test_HZ(self):
        self.check(("H", "Z"), False)

    def test_K(self):
        self.check(("H", "Z"), True)
        self.check((), True)


class TestKRoundtrip(unittest.TestCase):
    def test_gidis_donus(self):
        df = synth()
        f = feats(df, ("Z",), True)
        np.testing.assert_allclose(f["target"].to_numpy() * f["cap"].to_numpy(), f["price"].to_numpy())
        self.assertTrue(((f["target"] >= 0) & (f["target"] <= 1)).all())
        self.assertTrue((f["cap"] == 3400).all())  # 2025-05-20..2025-07-28: 3.400 dönemi

    def test_ham_modda_target_price(self):
        f = feats(synth(), ("H",), False)
        np.testing.assert_array_equal(f["target"].to_numpy(), f["price"].to_numpy())


class TestMissing(unittest.TestCase):
    def test_eksik_saatte_Z_nan(self):
        df = synth()
        D = pd.Timestamp("2025-07-10", tz=TZ)
        bad = pd.Timestamp("2025-07-09 13:00", tz=TZ)
        d2 = df[df["ts"] != bad].reset_index(drop=True)
        f = v2.build_features_v2(d2, groups=("Z",), **KW)
        day = (f["ts"] >= D) & (f["ts"] < D + pd.Timedelta(days=1))
        for c in ("price_d1_min", "price_d1_max", "zero_hours_d1"):
            self.assertTrue(f.loc[day, c].isna().all(), c)
        ok = (f["ts"] >= D + pd.Timedelta(days=2)) & (f["ts"] < D + pd.Timedelta(days=3))
        self.assertTrue(f.loc[ok, "price_d1_min"].notna().all())

    def test_Z_degerleri(self):
        df = synth()
        f = v2.build_features_v2(df, groups=("Z",), **KW)
        D = pd.Timestamp("2025-07-10", tz=TZ)
        prev = df[(df["ts"] >= D - pd.Timedelta(days=1)) & (df["ts"] < D)]["price"]
        row = f[f["ts"] == D].iloc[0]
        self.assertEqual(row["price_d1_min"], prev.min())
        self.assertEqual(row["price_d1_max"], prev.max())
        self.assertEqual(row["zero_hours_d1"], (prev == 0).sum())
        r2 = f[f["ts"] == D + pd.Timedelta(hours=5)].iloc[0]
        self.assertEqual(r2["lag_d1_is_zero"], float(df.loc[df["ts"] == D + pd.Timedelta(hours=5) - pd.Timedelta(hours=24), "price"].iloc[0] == 0))


class TestHolidayCols(unittest.TestCase):
    def test_H_sutunlari(self):
        f = feats(synth(), ("H",), False)
        r = f[f["ts"] == pd.Timestamp("2025-06-07 12:00", tz=TZ)].iloc[0]
        self.assertEqual(r["is_holiday"], 1)
        r = f[f["ts"] == pd.Timestamp("2025-06-06 12:00", tz=TZ)].iloc[0]
        self.assertEqual((r["is_arife"], r["days_to_holiday"]), (1, 1))

    def test_yasak_sutun_yok(self):
        pf.assert_no_forbidden(v2.feature_list(pf.FEATURES_WEATHER, ("H", "Z")))


class TestKaynak(unittest.TestCase):
    def test_doldurma_yok(self):
        for mod in (v2,):
            src = open(mod.__file__, encoding="utf-8").read()
            for w in ("fillna", "ffill", "bfill", "interpolate"):
                self.assertNotIn(w + "(", src)


if __name__ == "__main__":
    unittest.main()
