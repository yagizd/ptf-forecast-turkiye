"""Sızıntı testi: D gününden SONRAKİ fiyatlar değişince D gününe (ve öncesine) ait satırlar/özellikler değişmemeli."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import ptf_features_v2 as v2
from tests.test_ptf_features_v2 import KW, TZ, synth

REAL = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset_weather_guncel.parquet"


def _perturb_after(df: pd.DataFrame, day: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    out = df.copy()
    after = out["ts"] >= (day + pd.Timedelta(days=1)).tz_localize(TZ)
    rng = np.random.default_rng(seed)
    out.loc[after, "price"] = rng.uniform(0, 3400, int(after.sum()))
    assert after.any() and (out.loc[after, "price"] != df.loc[after, "price"]).any()
    return out


class TestSizintiYok(unittest.TestCase):
    def _check(self, df, day, **kw):
        f1 = v2.build_features_v2(df, **kw)
        f2 = v2.build_features_v2(_perturb_after(df, day), **kw)
        cut = (day + pd.Timedelta(days=1)).tz_localize(TZ)
        a, b = f1[f1["ts"] < cut].reset_index(drop=True), f2[f2["ts"] < cut].reset_index(drop=True)
        self.assertGreater(len(a), 0)
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    def test_sentetik_ham(self):
        self._check(synth(), pd.Timestamp("2025-06-20"), groups=("H",), **KW)

    def test_sentetik_h_z(self):
        self._check(synth(), pd.Timestamp("2025-06-20"), groups=("H", "Z"), **KW)

    def test_sentetik_k(self):
        self._check(synth(), pd.Timestamp("2025-06-20"), groups=("H",), normalize=True, **KW)

    @unittest.skipUnless(REAL.exists(), "güncel tablo yok")
    def test_gercek_yeni_gunler(self):
        df = pd.read_parquet(REAL)
        for norm in (False, True):
            self._check(df, pd.Timestamp("2026-10-25"), groups=("H",), normalize=norm)


if __name__ == "__main__":
    unittest.main()
