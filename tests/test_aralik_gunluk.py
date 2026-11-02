"""aralik_gunluk çekirdek fonksiyonları için testler (saf numpy; LightGBM EĞİTMEZ, ağ yok).

Çalıştırma (yalnız bu dosya): python -m unittest tests.test_aralik_gunluk -v
"""

import unittest

import numpy as np

from src.models import aralik_gunluk as ag
from src.models import quantile_conformal as qc

INF = float("inf")


def synth_block(seed=0, cal_days=5, n_test_days=6, no=1, offset=0):
    """Sentetik blok: gün başına 24 satır; first_day = offset + cal_days."""
    rng = np.random.default_rng(seed)
    total = cal_days + n_test_days
    day = np.repeat(np.arange(offset, offset + total), 24)
    base = rng.uniform(1000, 3000, len(day))
    y = base + rng.normal(0, 150, len(day))
    return {"no": no, "day": day, "y": y, "qlo": base - 120, "qhi": base + 120,
            "first_day": offset + cal_days, "last_day": offset + total - 1}


class TestQ(unittest.TestCase):
    def test_sonsuz_ve_bos(self):
        sc = np.arange(10, dtype=float)  # n=10
        self.assertEqual(ag.q_level(sc, 0.05), INF)       # k = ceil(11*0.95) = 11 > 10
        self.assertEqual(ag.q_level(sc, -0.01), INF)      # alpha<0 -> k>n
        self.assertEqual(ag.q_level([], 0.2), INF)        # skor yok
        self.assertEqual(ag.q_level(sc, 1.0), -INF)       # k = 0 < 1 -> boş
        self.assertEqual(ag.q_level(sc, 1.2), -INF)

    def test_ortada_conformal_quantile_ile_ayni(self):
        rng = np.random.default_rng(3)
        for n in (30, 100, 672):
            sc = rng.normal(0, 100, n)
            for a in (0.05, 0.1, 0.2, 0.5, 0.9):
                k = int(np.ceil((n + 1) * (1 - a)))
                if 1 <= k <= n:
                    self.assertEqual(ag.q_level(sc, a), qc.conformal_quantile(sc, a))

    def test_bos_ve_sinirsiz_aralik(self):
        lo, hi = ag.make_interval(np.array([100.0]), np.array([200.0]), INF)
        self.assertEqual((lo[0], hi[0]), (-INF, INF))
        lo, hi = ag.make_interval(np.array([100.0]), np.array([200.0]), -INF)
        self.assertEqual((lo[0], hi[0]), (INF, -INF))


class TestTracker(unittest.TestCase):
    def test_formul_elle(self):
        self.assertAlmostEqual(ag.aci_update(0.1, 0.01, 0.1, 0.0), 0.101, places=12)
        self.assertAlmostEqual(ag.aci_update(0.1, 0.01, 0.1, 1.0), 0.091, places=12)

    def test_alpha_sinirda_kalir(self):
        # e, aralığın gerçek durumuyla TUTARLI: alpha<0 sınırsız (e=0), alpha>1 boş (e=1), arada rastgele
        rng = np.random.default_rng(5)
        for gamma in (0.01, 0.05):
            for target in (0.2, 0.1):
                alpha = target
                for _ in range(1000):
                    self.assertTrue(-gamma - 1e-12 <= alpha <= 1 + gamma + 1e-12)
                    e = 0.0 if alpha < 0 else (1.0 if alpha > 1 else float(rng.random()))
                    alpha = ag.aci_update(alpha, gamma, target, e)
                self.assertTrue(-gamma - 1e-12 <= alpha <= 1 + gamma + 1e-12)


class TestMetrik(unittest.TestCase):
    def test_normal(self):
        y = np.array([5.0, 15.0, 25.0, 35.0])
        lo, hi = np.full(4, 10.0), np.full(4, 30.0)
        m = ag.metrics(y, lo, hi, np.full(4, 100.0), np.zeros(4))
        self.assertAlmostEqual(m["kapsama"], 0.5)
        self.assertAlmostEqual(m["genislik_klip"], 20.0)
        self.assertAlmostEqual(m["alt"], 0.25)
        self.assertAlmostEqual(m["ust"], 0.25)
        self.assertEqual(m["bos_saat"], 0)

    def test_sinirsiz(self):
        y = np.array([5.0, 500.0])
        lo, hi = np.full(2, -INF), np.full(2, INF)
        m = ag.metrics(y, lo, hi, np.full(2, 100.0), np.full(2, INF))
        self.assertEqual(m["kapsama"], 1.0)
        self.assertAlmostEqual(m["genislik_klip"], 100.0)  # [0, tavan] kırpılınca tavan
        self.assertTrue(np.isnan(m["genislik_klipsiz_sonlu"]))  # sonlu satır yok

    def test_bos(self):
        y = np.array([5.0, 50.0])
        lo, hi = np.full(2, INF), np.full(2, -INF)
        m = ag.metrics(y, lo, hi, np.full(2, 100.0), np.full(2, -INF))
        self.assertEqual(m["kapsama"], 0.0)
        self.assertEqual(m["genislik_klip"], 0.0)
        self.assertEqual(m["bos_saat"], 2)
        self.assertEqual(m["alt"], 0.0)
        self.assertEqual(m["ust"], 0.0)


class TestSeri(unittest.TestCase):
    def test_sizinti_yok(self):
        """D ve sonrasındaki y'ler değişince D günü aralığı ve alpha_D DEĞİŞMEZ (ACI dahil)."""
        for mode, gamma in (("B1", 0.0), ("ACI", 0.05)):
            b = synth_block(seed=1)
            D = b["first_day"] + 3
            b2 = {**b, "y": b["y"].copy()}
            rng = np.random.default_rng(99)
            late = b2["day"] >= D
            b2["y"][late] = rng.uniform(0, 5000, int(late.sum()))
            o1, r1 = ag.run_series([b], 0.2, mode, gamma, cal_days=5)
            o2, r2 = ag.run_series([b2], 0.2, mode, gamma, cal_days=5)
            test_day = b["day"][(b["day"] >= b["first_day"])]
            m = test_day <= D  # D dahil, D günü ve öncesi
            for key in ("lo", "hi", "alpha", "Q"):
                np.testing.assert_array_equal(o1[0][key][m], o2[0][key][m], err_msg=f"{mode} {key}")
            a1 = [r["alpha"] for r in r1 if r["gun"] <= D]
            a2 = [r["alpha"] for r in r2 if r["gun"] <= D]
            self.assertEqual(a1, a2)

    def test_b1_ilk_gun_b0_ile_ayni(self):
        b = synth_block(seed=2)
        o0, _ = ag.run_series([b], 0.1, "B0", cal_days=5)
        o1, _ = ag.run_series([b], 0.1, "B1", cal_days=5)
        first = b["day"][b["day"] >= b["first_day"]] == b["first_day"]
        for key in ("lo", "hi", "Q"):
            np.testing.assert_array_equal(o0[0][key][first], o1[0][key][first])

    def test_aci_gamma_sifir_b1_ile_ayni(self):
        b = synth_block(seed=4)
        o1, _ = ag.run_series([b], 0.2, "B1", cal_days=5)
        o2, _ = ag.run_series([b], 0.2, "ACI", 0.0, cal_days=5)
        for key in ("lo", "hi", "Q"):
            np.testing.assert_array_equal(o1[0][key], o2[0][key])

    def test_aci_bloklar_arasi_sifirlanmaz(self):
        b1 = synth_block(seed=6, no=1, offset=0)
        b2 = synth_block(seed=7, no=2, offset=b1["last_day"] + 1 - 5)  # ikinci bloğun kalibrasyonu birincinin sonunu kapsar
        _, recs = ag.run_series([b1, b2], 0.2, "ACI", 0.05, cal_days=5)
        last_b1 = [r for r in recs if r["blok"] == 1][-1]
        first_b2 = [r for r in recs if r["blok"] == 2][0]
        expected = ag.aci_update(last_b1["alpha"], 0.05, 0.2, last_b1["e"])
        self.assertAlmostEqual(first_b2["alpha"], expected, places=12)

    def test_karar_kurali_b0_kalir(self):
        stats = {}
        blockcov = {}
        for v in ("B0", "B1", "B2", "B3"):
            for lvl, t in ((80, 0.8), (90, 0.9)):
                stats[(v, lvl)] = {"kapsama": t - 0.05, "genislik_klip": 100.0}
                blockcov[(v, lvl)] = {k: t - 0.05 for k in range(1, 14)}
        self.assertEqual(ag.decide(stats, blockcov)["secilen"], "B0")  # hiçbiri (a)'yı geçemez (5 puan)


if __name__ == "__main__":
    unittest.main()
