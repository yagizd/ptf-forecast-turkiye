"""holidays_tr için testler (sahte tablo, ağ yok)."""

import unittest

import pandas as pd

from src.data import holidays_tr as ht

T = pd.DataFrame({
    "tarih": pd.to_datetime(["2024-03-20", "2024-03-21", "2024-03-22", "2024-03-25", "2024-03-30"]),
    "tur": ["arife_yarim_gun", "resmi_tatil", "resmi_tatil", "idari_izin_kopru", "resmi_tatil"],
})
KW = dict(cover_start="2024-03-01", cover_end="2024-04-30")


def f(*days):
    return ht.daily_features(days, T, **KW)


class TestHolidays(unittest.TestCase):
    def test_bayraklar(self):
        r = f("2024-03-20", "2024-03-21", "2024-03-25", "2024-03-23")
        self.assertEqual(r["is_arife"].tolist(), [1, 0, 0, 0])
        self.assertEqual(r["is_holiday"].tolist(), [0, 1, 0, 0])
        self.assertEqual(r["is_bridge"].tolist(), [0, 0, 1, 0])

    def test_hafta_sonu_tatil_degil(self):
        r = f("2024-03-23", "2024-03-24")  # Cumartesi, Pazar
        self.assertEqual(r["is_holiday"].tolist(), [0, 0])

    def test_gun_sayilari(self):
        r = f("2024-03-21", "2024-03-20", "2024-03-27", "2024-03-26")
        self.assertEqual(r["days_to_holiday"].tolist(), [0, 1, 3, 4])  # 03-26 -> 03-30
        self.assertEqual(r["days_since_holiday"].tolist(), [0, 7, 5, 4])  # 03-20: önceki tatil tabloda yok -> sınır 7

    def test_ust_sinir_7(self):
        r = f("2024-04-15")
        self.assertEqual(r["days_to_holiday"].iloc[0], 7)
        self.assertEqual(r["days_since_holiday"].iloc[0], 7)
        self.assertTrue((r[["days_to_holiday", "days_since_holiday"]] <= 7).all().all())

    def test_kapsam_disi_hata(self):
        with self.assertRaises(ValueError):
            f("2024-03-02")
        with self.assertRaises(ValueError):
            f("2024-04-29")

    def test_ts_serisi(self):
        ts = pd.Series(pd.date_range("2024-03-21 22:00", periods=4, freq="h", tz="Europe/Istanbul"))
        r = ht.features_for_ts(ts, T, **KW)
        self.assertEqual(r["is_holiday"].tolist(), [1, 1, 1, 1])  # 21 Mart 22:00,23:00; 22 Mart 00:00,01:00 (22 Mart de tatil)
        self.assertEqual(r["days_to_holiday"].tolist(), [0, 0, 0, 0])

    def test_kaynakta_doldurma_yok(self):
        src = open(ht.__file__, encoding="utf-8").read()
        for w in ("fillna", "ffill", "bfill", "interpolate"):
            self.assertNotIn(w + "(", src)


if __name__ == "__main__":
    unittest.main()
