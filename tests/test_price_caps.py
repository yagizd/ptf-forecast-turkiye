"""price_caps için testler (ağ yok)."""

import unittest

import pandas as pd

from src.data import price_caps as pc


class TestPriceCaps(unittest.TestCase):
    def test_gecis_gunleri(self):
        self.assertEqual(pc.cap_for_day("2023-07-04"), 2700)
        self.assertEqual(pc.cap_for_day("2024-06-30"), 2700)
        self.assertEqual(pc.cap_for_day("2024-07-01"), 3000)
        self.assertEqual(pc.cap_for_day("2025-04-04"), 3000)
        self.assertEqual(pc.cap_for_day("2025-04-05"), 3400)
        self.assertEqual(pc.cap_for_day("2026-04-03"), 3400)
        self.assertEqual(pc.cap_for_day("2026-04-04"), 4500)
        self.assertEqual(pc.cap_for_day("2026-10-20"), 4500)

    def test_tablo_oncesi_hata(self):
        with self.assertRaises(ValueError):
            pc.cap_for_day("2023-07-03")

    def test_ts_serisi(self):
        ts = pd.Series(pd.date_range("2025-04-04 22:00", periods=4, freq="h", tz="Europe/Istanbul"))
        self.assertEqual(pc.caps_for_ts(ts).tolist(), [3000, 3000, 3400, 3400])


if __name__ == "__main__":
    unittest.main()
