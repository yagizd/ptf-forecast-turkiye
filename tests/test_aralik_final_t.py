"""aralik_final_t için testler (SENTETİK veri; gerçek veri/paket GEREKTİRMEZ; LightGBM EĞİTMEZ). TASLAK: Faz 1'de ÇALIŞTIRILMADI.

Çalıştırma (yalnız bu dosya, Proje şefi onaylarsa): python -m unittest tests.test_aralik_final_t -v   (unittest discover YOK)
"""

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import aralik_final_t as ft

TZ = "Europe/Istanbul"


def synth_hourly(start="2025-08-01", end="2026-11-05 23:00") -> pd.DataFrame:
    ts = pd.date_range(start, end, freq="h", tz=TZ)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ts": ts, "price": rng.uniform(0, 3000, len(ts))})


def synth_block(seed=0, cal_days=5, n_test_days=6, no=1, offset=0):
    rng = np.random.default_rng(seed)
    total = cal_days + n_test_days
    day = np.repeat(np.arange(offset, offset + total), 24)
    base = rng.uniform(1000, 3000, len(day))
    y = base + rng.normal(0, 150, len(day))
    return {"no": no, "day": day, "y": y, "qlo": base - 120, "qhi": base + 120,
            "first_day": offset + cal_days, "last_day": offset + total - 1}


class TestTekKosu(unittest.TestCase):
    def test_cikti_klasoru_varsa_reddeder(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "oturum_T_final"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                ft.check_single_run(out, Path(d) / "yok_rapor.md")

    def test_rapor_varsa_reddeder(self):
        with tempfile.TemporaryDirectory() as d:
            rep = Path(d) / "rapor.md"
            rep.write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ft.check_single_run(Path(d) / "olmayan_klasor", rep)

    def test_hicbiri_yoksa_gecer(self):
        with tempfile.TemporaryDirectory() as d:
            ft.check_single_run(Path(d) / "olmayan_klasor", Path(d) / "olmayan_rapor.md")  # hata vermemeli

    def test_reddedince_hicbir_sey_yazmaz(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "oturum_T_final"
            out.mkdir()
            before = sorted(p.name for p in Path(d).iterdir())
            with self.assertRaises(FileExistsError):
                ft.check_single_run(out, Path(d) / "r.md")
            self.assertEqual(before, sorted(p.name for p in Path(d).iterdir()))
            self.assertEqual(list(out.iterdir()), [])


class TestPencere(unittest.TestCase):
    def test_t_penceresi_tam_8760_saat(self):
        df = ft.truncate_to_t(synth_hourly())
        m = ft.t_rows_mask(df["ts"])
        self.assertEqual(int(m.sum()), 8760)
        self.assertEqual(ft.T_HOURS, 8760)
        t = df.loc[m, "ts"]
        self.assertEqual(t.iloc[0], pd.Timestamp("2025-10-21 00:00", tz=TZ))
        self.assertEqual(t.iloc[-1], pd.Timestamp("2026-10-20 23:00", tz=TZ))
        self.assertTrue((t.diff().dropna() == pd.Timedelta(hours=1)).all())

    def test_21_ekim_2026_ve_sonrasi_kullanilmaz(self):
        full = synth_hourly()
        self.assertTrue((full["ts"] >= pd.Timestamp("2026-10-21", tz=TZ)).any())  # veride VAR
        df = ft.truncate_to_t(full)
        self.assertFalse((df["ts"] >= pd.Timestamp("2026-10-21", tz=TZ)).any())
        self.assertEqual(df["ts"].max(), pd.Timestamp("2026-10-20 23:00", tz=TZ))
        self.assertFalse(ft.t_rows_mask(full["ts"])[(full["ts"] >= pd.Timestamp("2026-10-21", tz=TZ)).to_numpy()].any())

    def test_t_oncesi_kalibrasyon_gunleri_korunur(self):
        df = ft.truncate_to_t(synth_hourly())
        self.assertLess(df["ts"].min(), pd.Timestamp("2025-08-26", tz=TZ))  # T başı - 56 gün mevcut


class TestAyarYok(unittest.TestCase):
    def test_sabitler(self):
        self.assertEqual(ft.GAMMA, 0.05)
        self.assertEqual(ft.CAL_DAYS, 56)
        self.assertEqual(ft.LEVELS_PCT, (80, 90))

    def test_komut_satiri_ayari_yok(self):
        self.assertEqual(len(inspect.signature(ft.main).parameters), 0)
        self.assertFalse(hasattr(ft, "argparse"))
        src = Path(ft.__file__).read_text(encoding="utf-8")
        self.assertNotIn("sys.argv", src)
        self.assertNotIn("argparse", src)


class TestGelecegeBakmaYok(unittest.TestCase):
    def test_d_gunu_kalibrasyonu_yalniz_oncesinden(self):
        for variant in ("B0", "B3"):
            b = synth_block(seed=1)
            D = b["first_day"] + 3
            b2 = {**b, "y": b["y"].copy()}
            late = b2["day"] >= D
            b2["y"][late] = np.random.default_rng(99).uniform(0, 5000, int(late.sum()))
            # run_level sabit kalibrasyon 56 gün kullanır; testte kısa sentetik blok için aynı çekirdeği doğrudan çağırıyoruz
            from src.models import aralik_gunluk as ag

            mode, gamma = ("B0", 0.0) if variant == "B0" else ("ACI", ft.GAMMA)
            o1, r1 = ag.run_series([b], 0.2, mode, gamma, cal_days=5)
            o2, r2 = ag.run_series([b2], 0.2, mode, gamma, cal_days=5)
            test_day = b["day"][b["day"] >= b["first_day"]]
            if variant == "B3":
                m = test_day <= D  # D günü ve öncesi
            else:
                m = np.ones(len(test_day), dtype=bool)  # B0 blok başında bir kez; blok içi y değişimi etkilemez (kalibrasyon blok öncesi)
            for key in ("lo", "hi", "alpha", "Q"):
                np.testing.assert_array_equal(o1[0][key][m], o2[0][key][m], err_msg=f"{variant} {key}")
            if variant == "B3":
                self.assertEqual([r["alpha"] for r in r1 if r["gun"] <= D], [r["alpha"] for r in r2 if r["gun"] <= D])

    def test_run_level_b0_ve_b3_calisir_ve_dogru_modu_kullanir(self):
        b = synth_block(seed=2, cal_days=56, n_test_days=3)
        o_b0, _ = ft.run_level([b], 0.2, "B0")
        o_b3, recs = ft.run_level([b], 0.2, "B3")
        self.assertEqual(len(o_b0[0]["lo"]), 72)
        self.assertEqual(len(recs), 3)  # B3 günlük kayıt tutar
        np.testing.assert_array_equal(o_b0[0]["alpha"], np.full(72, 0.2))  # B0'da alpha sabit
        with self.assertRaises(ValueError):
            ft.run_level([b], 0.2, "B1")  # tanımsız varyant (ayar yok)


if __name__ == "__main__":
    unittest.main()
