"""quantile_conformal testleri (YAZILDI, bu oturumda ÇALIŞTIRILMADI). LightGBM gerektiren test paket yoksa atlanır."""

import importlib.util
import unittest

import numpy as np
import pandas as pd

from src.models import quantile_conformal as qc

TZ = "Europe/Istanbul"


class TestConformal(unittest.TestCase):
    def test_kuantil(self):
        self.assertEqual(qc.conformal_quantile(np.arange(1, 10), 0.2), 8.0)  # ceil(10*0.8)=8 -> 8. küçük
        self.assertEqual(qc.conformal_quantile([], 0.2), float("inf"))
        self.assertEqual(qc.conformal_quantile([1.0], 0.2), float("inf"))  # k=2 > n=1

    def test_aralik_ve_kapsama(self):
        lo, hi, q = qc.cqr_interval([0, 0], [10, 10], [0, 0, 0, 0, 0], [10, 10, 10, 10, 10], [5, 5, 5, 5, 5])
        self.assertEqual(q, -5.0)
        self.assertEqual(hi.tolist(), [5.0, 5.0])
        c = qc.coverage_and_width([1, 20], [0, 0], [10, 10])
        self.assertEqual(c["kapsama"], 0.5)
        self.assertEqual(c["ortalama_genislik"], 10.0)

    def test_bolme_zaman_kurali(self):
        ts = pd.Series(pd.date_range("2025-01-01", periods=24 * 120, freq="h", tz=TZ))
        y = pd.Series(np.arange(len(ts), dtype=float))
        bs = pd.Timestamp("2025-04-01", tz=TZ)
        tr, cal = qc.split_masks(ts, y, bs, pd.Timestamp("2025-01-01", tz=TZ))
        self.assertFalse((tr & cal).any())
        self.assertTrue(ts[tr].max() < ts[cal].min())
        self.assertTrue(ts[cal].max() < bs)
        self.assertEqual(int(cal.sum()), 56 * 24)

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "lightgbm yok")
    def test_run_cqr_kapsama_makul(self):
        rng = np.random.default_rng(0)
        ts = pd.Series(pd.date_range("2025-01-01", periods=24 * 200, freq="h", tz=TZ))
        x = rng.normal(size=len(ts))
        feat = pd.DataFrame({"ts": ts, "price": x + rng.normal(size=len(ts)), "x": x})
        blocks = [{"no": 1, "start_ts": pd.Timestamp("2025-06-01", tz=TZ), "end_ts": pd.Timestamp("2025-06-15", tz=TZ)}]
        r = qc.run_cqr(feat, ["x"], blocks, pd.Timestamp("2025-01-01", tz=TZ), params={"n_estimators": 50})
        c = qc.coverage_and_width(r["y"], r["lo"], r["hi"])
        self.assertGreater(c["kapsama"], 0.6)


if __name__ == "__main__":
    unittest.main()
