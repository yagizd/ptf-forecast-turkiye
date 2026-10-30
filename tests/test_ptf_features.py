"""ptf_features için AĞSIZ testler (sahte veri). LightGBM gerekmez.

En önemli test: SIZINTI TESTİ. D gününün 00:00'ından itibaren tüm fiyatlar değişse bile D gününün
özellikleri DEĞİŞMEMELİ (D günü fiyatı tahmin anında bilinmez).

Çalıştırma (repo kökünden): python -m unittest discover -s tests   (veya pytest)
"""

import inspect
import unittest

import numpy as np
import pandas as pd

from src.data.build_weather_dataset import MODEL_FEATURES_PD2
from src.features import ptf_features as pf

TZ = "Europe/Istanbul"
H = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def synth(days=40, start="2024-03-01", seed=0):
    """Sahte saatlik tablo. consumption, lep ve bir pd1 sütunu bilerek de var: özelliklere GİRMEMELİ."""
    ts = pd.date_range(start, periods=days * 24, freq="h", tz=TZ)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ts": ts,
        "price": rng.uniform(500, 3000, len(ts)).round(2),
        "consumption": rng.uniform(2e4, 5e4, len(ts)),
        "lep": rng.uniform(2e4, 5e4, len(ts)),
    })
    for c in MODEL_FEATURES_PD2:
        df[c] = rng.uniform(0, 30, len(ts))
    df["temp_mean_pd1"] = 1.0
    return df


class TestSizinti(unittest.TestCase):
    def test_D_ve_sonrasi_fiyat_degisince_D_ozellikleri_degismez(self):
        df = synth()
        D = pd.Timestamp("2024-03-20 00:00", tz=TZ)
        mod = df.copy()
        m = mod["ts"] >= D
        mod.loc[m, "price"] = np.random.default_rng(99).uniform(1e5, 2e5, int(m.sum()))  # D ve sonrasını boz
        f1, f2 = pf.build_features(df), pf.build_features(mod)

        gun = (f1["ts"] >= D) & (f1["ts"] < D + DAY)
        pd.testing.assert_frame_equal(
            f1.loc[gun, pf.FEATURES_WEATHER].reset_index(drop=True),
            f2.loc[gun, pf.FEATURES_WEATHER].reset_index(drop=True))
        # D'den önceki satırlar (hedef dahil) tamamen aynı
        once = f1["ts"] < D
        pd.testing.assert_frame_equal(f1[once].reset_index(drop=True), f2[once].reset_index(drop=True))
        # Testin boş olmadığını gösteren kontroller: hedef değişti, ertesi günün lag'i değişti
        self.assertFalse(f1.loc[gun, "price"].reset_index(drop=True).equals(f2.loc[gun, "price"].reset_index(drop=True)))
        ertesi = (f1["ts"] >= D + DAY) & (f1["ts"] < D + 2 * DAY)
        self.assertFalse(f1.loc[ertesi, "price_lag_d1"].reset_index(drop=True)
                         .equals(f2.loc[ertesi, "price_lag_d1"].reset_index(drop=True)))

    def test_sadece_D_gunu_fiyati_degisince_D_ozellikleri_degismez(self):
        df = synth()
        D = pd.Timestamp("2024-03-15 00:00", tz=TZ)
        mod = df.copy()
        m = (mod["ts"] >= D) & (mod["ts"] < D + DAY)
        mod.loc[m, "price"] = 123456.0
        f1, f2 = pf.build_features(df), pf.build_features(mod)
        gun = (f1["ts"] >= D) & (f1["ts"] < D + DAY)
        pd.testing.assert_frame_equal(
            f1.loc[gun, pf.FEATURES_WEATHER].reset_index(drop=True),
            f2.loc[gun, pf.FEATURES_WEATHER].reset_index(drop=True))


class TestLaglar(unittest.TestCase):
    def test_lag_degerleri_dogru(self):
        df = synth(days=30)
        f = pf.build_features(df).set_index("ts")
        p = df.set_index("ts")["price"]
        t = pd.Timestamp("2024-03-15 13:00", tz=TZ)
        self.assertEqual(f.loc[t, "price_lag_d1"], p[t - 24 * H])
        self.assertEqual(f.loc[t, "price_lag_d2"], p[t - 48 * H])
        self.assertEqual(f.loc[t, "price_lag_d7"], p[t - 168 * H])

    def test_eksik_saat_zamana_gore_NaN_kayma_yok(self):
        df = synth(days=30)
        X = pd.Timestamp("2024-03-10 05:00", tz=TZ)
        dm = df[df["ts"] != X].reset_index(drop=True)  # X saati tabloda yok
        f = pf.build_features(dm).set_index("ts")
        p = dm.set_index("ts")["price"]
        self.assertTrue(np.isnan(f.loc[X + 24 * H, "price_lag_d1"]))     # tam X'i işaret eden lag NaN
        self.assertTrue(np.isnan(f.loc[X + 48 * H, "price_lag_d2"]))
        self.assertTrue(np.isnan(f.loc[X + 168 * H, "price_lag_d7"]))
        # Komşu satırlar kaymadı: X+25h'nin lag'i X+1h'nin fiyatı
        self.assertEqual(f.loc[X + 25 * H, "price_lag_d1"], p[X + 1 * H])
        self.assertEqual(f.loc[X + 49 * H, "price_lag_d2"], p[X + 1 * H])
        self.assertEqual(f.loc[X + 23 * H, "price_lag_d1"], p[X - 1 * H])
        self.assertEqual(f.loc[X + 100 * H, "price_lag_d1"], p[X + 76 * H])

    def test_ilk_gunlerde_lag_NaN(self):
        f = pf.build_features(synth(days=10))
        self.assertTrue(f["price_lag_d1"].iloc[:24].isna().all())
        self.assertTrue(f["price_lag_d7"].iloc[:168].isna().all())
        self.assertFalse(f["price_lag_d7"].iloc[168:].isna().any())


class TestGunlukOrtalama(unittest.TestCase):
    def test_d1_ortalamasi_dogru(self):
        df = synth(days=20)
        f = pf.build_features(df)
        D = pd.Timestamp("2024-03-10 00:00", tz=TZ)
        beklenen = df[(df["ts"] >= D - DAY) & (df["ts"] < D)]["price"].mean()
        satirlar = f[(f["ts"] >= D) & (f["ts"] < D + DAY)]["price_d1_mean"]
        self.assertEqual(len(satirlar), 24)
        np.testing.assert_allclose(satirlar.to_numpy(), beklenen)
        self.assertTrue(f["price_d1_mean"].iloc[:24].isna().all())  # ilk günün D-1'i yok

    def test_bir_fiyat_bossa_ortalama_NaN_skipna_false(self):
        df = synth(days=20)
        D = pd.Timestamp("2024-03-10 00:00", tz=TZ)
        df.loc[df["ts"] == D - 5 * H, "price"] = float("nan")
        f = pf.build_features(df)
        gun = f[(f["ts"] >= D) & (f["ts"] < D + DAY)]["price_d1_mean"]
        self.assertTrue(gun.isna().all())
        ertesi = f[(f["ts"] >= D + DAY) & (f["ts"] < D + 2 * DAY)]["price_d1_mean"]
        self.assertFalse(ertesi.isna().any())  # sadece D-1'in boşluğu etkiler

    def test_eksik_satir_varsa_ortalama_NaN(self):
        df = synth(days=20)
        D = pd.Timestamp("2024-03-10 00:00", tz=TZ)
        dm = df[df["ts"] != D - 3 * H].reset_index(drop=True)
        f = pf.build_features(dm)
        gun = f[(f["ts"] >= D) & (f["ts"] < D + DAY)]["price_d1_mean"]
        self.assertTrue(gun.isna().all())


class TestYasaklarVeKapsam(unittest.TestCase):
    def test_ozelliklerde_yasak_ad_yok(self):
        for cols in (pf.FEATURES_BASE, pf.FEATURES_WEATHER):
            for c in cols:
                for bad in ("consumption", "lep", "pd1"):
                    self.assertNotIn(bad, c)

    def test_weather_listesi_base_arti_5_hava(self):
        self.assertEqual(pf.FEATURES_WEATHER, pf.FEATURES_BASE + list(MODEL_FEATURES_PD2))
        self.assertEqual(len(MODEL_FEATURES_PD2), 5)
        self.assertFalse(set(pf.FEATURES_BASE) & set(MODEL_FEATURES_PD2))

    def test_cikti_sutunlari_ve_girdideki_yasak_sutunlar_yok_sayilir(self):
        df = synth()  # consumption, lep, temp_mean_pd1 içeriyor
        f = pf.build_features(df)
        self.assertEqual(list(f.columns), ["ts", "price", *pf.FEATURES_WEATHER])
        for yasak in ("consumption", "lep", "temp_mean_pd1"):
            self.assertNotIn(yasak, f.columns)
        self.assertEqual(len(f), len(df))

    def test_assert_no_forbidden_hata_verir(self):
        for kotu in (["price_lag_d1", "consumption"], ["lep"], ["temp_mean_pd1"]):
            with self.assertRaises(ValueError):
                pf.assert_no_forbidden(kotu)
        pf.assert_no_forbidden(pf.FEATURES_WEATHER)  # temiz liste hata vermez

    def test_hava_olmadan_da_calisir_ve_hava_gerekiyorsa_hata_verir(self):
        df = synth().drop(columns=list(MODEL_FEATURES_PD2))
        f = pf.build_features(df, include_weather=False)
        self.assertEqual(list(f.columns), ["ts", "price", *pf.FEATURES_BASE])
        with self.assertRaises(ValueError):
            pf.build_features(df, include_weather=True)

    def test_ts_tekrari_hata(self):
        df = synth(days=5)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            pf.build_features(df)


class TestNaNVeTakvim(unittest.TestCase):
    def test_hava_nan_kalir_doldurulmaz(self):
        df = synth(days=10)
        col = MODEL_FEATURES_PD2[1]
        df.loc[[3, 4, 50], col] = float("nan")
        f = pf.build_features(df)
        self.assertEqual(list(f.index[f[col].isna()]), [3, 4, 50])
        dolu = ~f[col].isna().to_numpy()
        np.testing.assert_array_equal(f[col].to_numpy()[dolu], df[col].to_numpy()[dolu])
        for c in MODEL_FEATURES_PD2:
            if c != col:
                self.assertEqual(int(f[c].isna().sum()), 0)

    def test_takvim_ozellikleri(self):
        df = synth(days=30)
        f = pf.build_features(df).set_index("ts")
        cumartesi = pd.Timestamp("2024-03-23 14:00", tz=TZ)
        cuma = pd.Timestamp("2024-03-22 14:00", tz=TZ)
        self.assertEqual(f.loc[cumartesi, "hour"], 14)
        self.assertEqual(f.loc[cumartesi, "dayofweek"], 5)
        self.assertEqual(f.loc[cumartesi, "is_weekend"], 1)
        self.assertEqual(f.loc[cuma, "is_weekend"], 0)
        self.assertEqual(f.loc[cumartesi, "month"], 3)

    def test_kaynakta_doldurma_yontemi_yok(self):
        src = inspect.getsource(pf)
        for yasak in ("fillna", "ffill", "bfill", "interpolate"):
            self.assertNotIn(yasak, src)


if __name__ == "__main__":
    unittest.main()
