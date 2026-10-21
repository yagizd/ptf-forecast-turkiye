"""baselines için testler. Sahte fiyat tablosuyla çalışır; gerçek veri gerekmez.

En önemli test: zaman kuralı. D gününün ve sonrasının fiyatı değişince D'nin tahmini DEĞİŞMEMELİ.
Çalıştırma (repo kökünden): python -m unittest discover -s tests   (veya pytest)
"""

import unittest

import numpy as np
import pandas as pd

from src.models import baselines as b


def _prices():
    """2024-01-01 (Pazartesi) .. 2024-02-29, 60 gün. Fiyat = gün_sırası*100 + saat (her hücre farklı)."""
    days = pd.date_range("2024-01-01", periods=60, freq="D")
    data = [[i * 100 + h for h in range(24)] for i in range(len(days))]
    return pd.DataFrame(data, index=days, columns=range(24), dtype=float)


def _row(prices, day):
    return prices.loc[pd.Timestamp(day)].to_numpy()


class TestKiyasModelleri(unittest.TestCase):
    def setUp(self):
        self.p = _prices()

    def test_d1_dunun_ayni_saati(self):
        f = b.forecast_day(self.p, pd.Timestamp("2024-02-13"))
        np.testing.assert_array_equal(f["d1"].to_numpy(), _row(self.p, "2024-02-12"))

    def test_d7_gecen_haftanin_ayni_gunu(self):
        f = b.forecast_day(self.p, pd.Timestamp("2024-02-13"))
        np.testing.assert_array_equal(f["d7"].to_numpy(), _row(self.p, "2024-02-06"))

    def test_kural_gunlere_gore(self):
        # 12 Şub 2024 Pazartesi. Pzt/Cmt/Paz -> D-7 ; Sal..Cum -> D-1
        beklenen = {"2024-02-12": 7, "2024-02-13": 1, "2024-02-14": 1, "2024-02-15": 1,
                    "2024-02-16": 1, "2024-02-17": 7, "2024-02-18": 7}
        for gun, lag in beklenen.items():
            d = pd.Timestamp(gun)
            f = b.forecast_day(self.p, d)
            np.testing.assert_array_equal(
                f["kural"].to_numpy(), _row(self.p, d - pd.Timedelta(days=lag)), err_msg=gun)

    def test_gecmis_yoksa_nan_doldurma_yok(self):
        f = b.forecast_day(self.p, pd.Timestamp("2024-01-03"))  # D-7 tabloda yok
        self.assertTrue(f["d7"].isna().all())
        self.assertFalse(f["d1"].isna().any())


class TestZamanKurali(unittest.TestCase):
    def test_D_ve_sonrasi_degisince_tahmin_degismez(self):
        d = pd.Timestamp("2024-02-14")
        orijinal = _prices()
        bozuk = orijinal.copy()
        bozuk.loc[bozuk.index >= d] = 1e9  # D ve sonrasının fiyatını boz
        a, c = b.forecast_day(orijinal, d), b.forecast_day(bozuk, d)
        for m in b.MODELS:
            np.testing.assert_array_equal(a[m].to_numpy(), c[m].to_numpy(), err_msg=m)

    def test_D_minus_1_degisince_tahmin_degisir(self):
        # Testin boş olmadığını gösterir: geçmiş değişince tahmin gerçekten değişiyor.
        d = pd.Timestamp("2024-02-14")
        orijinal = _prices()
        degisik = orijinal.copy()
        degisik.loc[d - pd.Timedelta(days=1)] = 1e9
        self.assertFalse(np.array_equal(
            b.forecast_day(orijinal, d)["d1"].to_numpy(), b.forecast_day(degisik, d)["d1"].to_numpy()))

    def test_assert_D_gorunurse_hata_verir(self):
        d = pd.Timestamp("2024-02-14")
        tum_tablo = _prices()  # D günü de içinde
        for f in b.FORECASTERS.values():
            with self.assertRaises(AssertionError):
                f(tum_tablo, d)


class TestBacktest(unittest.TestCase):
    def test_sekil_ve_sutunlar(self):
        bt = b.run_backtest(_prices(), "2024-02-01", "2024-02-10")
        self.assertEqual(len(bt), 10 * 24)
        self.assertEqual(list(bt.columns), ["gun", "saat", "gercek", "tahmin_d1", "tahmin_d7", "tahmin_kural"])

    def test_gercek_deger_dogru_gunun(self):
        bt = b.run_backtest(_prices(), "2024-02-01", "2024-02-02")
        ilk = bt[bt["gun"] == pd.Timestamp("2024-02-01")]
        np.testing.assert_array_equal(ilk["gercek"].to_numpy(), _row(_prices(), "2024-02-01"))


class TestOlculer(unittest.TestCase):
    def test_mae_ve_rmae(self):
        sub = pd.DataFrame({
            "gercek": [10.0, 10.0],
            "tahmin_d1": [11.0, 9.0],     # MAE 1
            "tahmin_d7": [12.0, 8.0],     # MAE 2
            "tahmin_kural": [10.0, 10.0],  # MAE 0
        })
        s = b.summarize(sub)
        self.assertAlmostEqual(s["mae_d1"], 1.0)
        self.assertAlmostEqual(s["rmae_d7"], 2.0)
        self.assertAlmostEqual(s["rmae_kural"], 0.0)

    def test_bos_satirlar_disarida_kalir(self):
        bt = pd.DataFrame({
            "gercek": [1.0, np.nan], "tahmin_d1": [1.0, 1.0],
            "tahmin_d7": [1.0, 1.0], "tahmin_kural": [1.0, 1.0],
        })
        self.assertEqual(len(b.common_rows(bt)), 1)

    def test_ozel_saat_etiketi(self):
        common = pd.DataFrame({
            "gun": pd.to_datetime(["2025-11-01", "2025-11-01", "2025-11-01", "2025-12-01"]),
            "gercek": [0.0, 3400.0, 100.0, 3400.0],
        })
        # Kasım'ın tavanı 3400; Aralık'ın tavanı yok (NaN) -> 3400 orada 'Diğer' kalır
        caps = pd.Series({"2025-11": 3400.0, "2025-12": np.nan})
        lab = b.label_special(common, caps)
        self.assertEqual(list(lab), ["Fiyat 0", "Tavanda", "Diğer", "Diğer"])

    def test_aylik_tavan_en_az_5_saat_kurali(self):
        def ay(yil_ay, tavan_saat):
            ts = pd.date_range(f"{yil_ay}-01", periods=20, freq="h", tz="Europe/Istanbul")
            fiyat = [100.0] * (20 - tavan_saat) + [500.0] * tavan_saat  # en yüksek = 500
            return pd.DataFrame({"ts": ts, "price": fiyat})

        df = pd.concat([ay("2025-01", 5), ay("2025-02", 4)], ignore_index=True)
        caps = b.month_caps(df)
        self.assertEqual(caps["2025-01"], 500.0)   # 5 saat -> tavan var
        self.assertTrue(np.isnan(caps["2025-02"]))  # 4 saat -> tavan yok

    def test_price_table_sekli(self):
        ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="Europe/Istanbul")
        df = pd.DataFrame({"ts": ts, "price": np.arange(48.0)})
        t = b.price_table(df)
        self.assertEqual(t.shape, (2, 24))
        self.assertEqual(t.iloc[1, 0], 24.0)


if __name__ == "__main__":
    unittest.main()
