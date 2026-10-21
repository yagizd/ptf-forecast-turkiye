"""build_dataset için testler. Gerçek veri gerekmez; küçük sahte tablolarla çalışır.

Çalıştırma (repo kökünden): python -m unittest discover -s tests   (veya pytest)
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.build_dataset import SOURCES, TZ, combine, quality_report, read_monthly


def _frames(hours=72):
    """3 günlük sahte (ts, değer) tabloları."""
    ts = pd.date_range("2024-01-01", periods=hours, freq="h", tz=TZ)
    return {
        "price": pd.DataFrame({"ts": ts, "price": range(hours)}).astype({"price": float}),
        "consumption": pd.DataFrame({"ts": ts, "consumption": [1000.0 + i for i in range(hours)]}),
        "lep": pd.DataFrame({"ts": ts, "lep": [900.0 + i for i in range(hours)]}),
    }


class TestCombine(unittest.TestCase):
    def test_her_saat_bir_kez_ve_sirali(self):
        fr = _frames()
        # karıştır ve bir satırı iki kez ekle
        fr["price"] = pd.concat([fr["price"], fr["price"].iloc[[5]]]).sample(frac=1, random_state=1)
        df, dups = combine(fr)
        self.assertTrue(df["ts"].is_unique)
        self.assertTrue(df["ts"].is_monotonic_increasing)
        self.assertEqual(len(df), 72)
        self.assertEqual(dups["price"], 1)

    def test_satir_sayisi_beklenen(self):
        df, _ = combine(_frames(hours=31 * 24))
        self.assertEqual(len(df), 744)

    def test_eksikler_doldurulmaz(self):
        fr = _frames()
        # lep'te 2. günün 24 saatini sil
        day2 = fr["lep"]["ts"].dt.day == 2
        fr["lep"] = fr["lep"][~day2]
        df, _ = combine(fr)
        self.assertEqual(int(df["lep"].isna().sum()), 24)
        self.assertEqual(int(df["price"].isna().sum()), 0)
        self.assertEqual(int(df["consumption"].isna().sum()), 0)

    def test_hicbir_kaynakta_olmayan_saat_bos_satir_olur(self):
        fr = _frames()
        for k in fr:
            fr[k] = fr[k].drop(index=10)  # 10. saat her yerde yok
        df, _ = combine(fr)
        self.assertEqual(len(df), 72)  # satır yine de var
        self.assertTrue(df.loc[10, ["price", "consumption", "lep"]].isna().all())


class TestReadMonthly(unittest.TestCase):
    def test_okuma_ve_saat_dilimi(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "a_2021-01.csv").write_text(
                "date,hour,price,priceUsd,priceEur\n"
                "2021-01-01T00:00:00+03:00,00:00,263.99,35.96,29.31\n"
                "2021-01-01T01:00:00+03:00,01:00,236.99,32.29,26.31\n",
                encoding="utf-8",
            )
            t, mismatch = read_monthly(p, "price", "hour", "price")
        self.assertEqual(list(t.columns), ["ts", "price"])
        self.assertEqual(mismatch, 0)
        self.assertEqual(str(t["ts"].dt.tz), TZ)
        self.assertEqual(t["ts"].iloc[0].hour, 0)


class TestQualityReport(unittest.TestCase):
    def test_rapor_uretilir(self):
        df, dups = combine(_frames())
        text = quality_report(df, dups, {c: 0 for c in SOURCES})
        self.assertIn("Toplam satır", text)
        self.assertIn("lep", text)
        self.assertIn("72", text)


if __name__ == "__main__":
    unittest.main()
