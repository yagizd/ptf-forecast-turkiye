"""lgbm_walkforward için AĞSIZ testler (sahte veri).

LightGBM gerektiren testler, paket kurulu değilse atlanır (unittest.skipUnless).
Gerçek veri gerektiren tek test (kıyas değerleri) dataset_weather.parquet yoksa atlanır.

Çalıştırma (repo kökünden): python -m unittest discover -s tests   (veya pytest)
"""

import importlib.util
import inspect
import unittest

import numpy as np
import pandas as pd

from src.data.build_weather_dataset import MODEL_FEATURES_PD2
from src.features import ptf_features as pf
from src.models import baselines as bl
from src.models import lgbm_walkforward as lw

TZ = "Europe/Istanbul"
H = pd.Timedelta(hours=1)
HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def synth(days=100, start="2024-03-01", seed=0):
    ts = pd.date_range(start, periods=days * 24, freq="h", tz=TZ)
    rng = np.random.default_rng(seed)
    hour = ts.hour.to_numpy()
    df = pd.DataFrame({
        "ts": ts,
        "price": (1500 + 300 * np.sin(hour / 24 * 2 * np.pi) + 20 * ts.dayofweek.to_numpy()
                  + rng.normal(0, 40, len(ts))).round(2),
        "consumption": rng.uniform(2e4, 5e4, len(ts)),
        "lep": rng.uniform(2e4, 5e4, len(ts)),
    })
    for c in MODEL_FEATURES_PD2:
        df[c] = rng.uniform(0, 30, len(ts))
    df["temp_mean_pd1"] = 1.0
    return df


class TestBloklar(unittest.TestCase):
    def test_test_donemi_tam_1_kez_kaplanir(self):
        blocks = lw.make_blocks(bl.TEST_START, bl.TEST_END)
        gunler = []
        for b in blocks:
            gunler += list(pd.date_range(b["first_day"], b["last_day"], freq="D"))
        self.assertEqual(gunler, list(pd.date_range(bl.TEST_START, bl.TEST_END, freq="D")))  # sıralı, tekrarsız, boşluksuz
        self.assertEqual(len(gunler), 365)
        self.assertEqual(len(blocks), 14)
        uzunluk = [(b["last_day"] - b["first_day"]).days + 1 for b in blocks]
        self.assertTrue(all(u <= 28 for u in uzunluk))
        self.assertEqual(uzunluk[-1], 1)  # son blok kısa

    def test_blok_zaman_sinirlari_bitisik_ve_8760_saat(self):
        blocks = lw.make_blocks(bl.TEST_START, bl.TEST_END)
        for a, b in zip(blocks, blocks[1:]):
            self.assertEqual(a["end_ts"], b["start_ts"])
        toplam = sum((b["end_ts"] - b["start_ts"]) / H for b in blocks)
        self.assertEqual(toplam, 8760)
        self.assertEqual(blocks[0]["start_ts"], pd.Timestamp("2025-10-21 00:00", tz=TZ))
        self.assertEqual(blocks[-1]["end_ts"], pd.Timestamp("2026-10-21 00:00", tz=TZ))

    def test_kisa_donem(self):
        blocks = lw.make_blocks("2024-05-20", "2024-06-02", block_days=7)
        self.assertEqual([b["no"] for b in blocks], [1, 2])
        self.assertEqual(blocks[1]["last_day"], pd.Timestamp("2024-06-02"))


class TestBolme(unittest.TestCase):
    def _feat(self, df):
        return pf.build_features(df, include_weather=True)

    def test_egitim_hep_blok_baslangicindan_once_ve_kesisim_yok(self):
        feat = self._feat(synth())
        ts0 = pd.Timestamp("2024-03-08 00:00", tz=TZ)
        for b in lw.make_blocks("2024-05-20", "2024-06-02", block_days=7):
            tr, te = lw.split_for_block(feat, b, ts0)
            self.assertFalse((tr & te).any())
            self.assertLess(feat.loc[tr, "ts"].max(), b["start_ts"])
            self.assertGreaterEqual(feat.loc[tr, "ts"].min(), ts0)
            self.assertTrue((feat.loc[te, "ts"] >= b["start_ts"]).all())
            self.assertTrue((feat.loc[te, "ts"] < b["end_ts"]).all())
            self.assertEqual(int(te.sum()), (b["last_day"] - b["first_day"]).days * 24 + 24)

    def test_genisleyen_pencere(self):
        feat = self._feat(synth())
        ts0 = pd.Timestamp("2024-03-08 00:00", tz=TZ)
        b1, b2 = lw.make_blocks("2024-05-20", "2024-06-02", block_days=7)
        n1 = int(lw.split_for_block(feat, b1, ts0)[0].sum())
        n2 = int(lw.split_for_block(feat, b2, ts0)[0].sum())
        self.assertEqual(n2 - n1, 7 * 24)  # ikinci blokta eğitime birinci bloğun 7 günü eklenir

    def test_hedefi_bos_satir_egitimden_atilir_testte_kalir_ozellik_NaN_atilmaz(self):
        df = synth()
        df.loc[df["ts"] == pd.Timestamp("2024-05-10 12:00", tz=TZ), "price"] = float("nan")  # eğitim dönemi
        df.loc[df["ts"] == pd.Timestamp("2024-05-21 12:00", tz=TZ), "price"] = float("nan")  # test dönemi
        feat = self._feat(df)
        ts0 = pd.Timestamp("2024-03-02 00:00", tz=TZ)  # erken başlat: eğitimde lag_d7 NaN satırlar olsun
        b = lw.make_blocks("2024-05-20", "2024-05-26", block_days=7)[0]
        tr, te = lw.split_for_block(feat, b, ts0)
        self.assertEqual(int(feat.loc[tr, "price"].isna().sum()), 0)  # hedefi boş olan eğitimde yok
        self.assertNotIn(pd.Timestamp("2024-05-10 12:00", tz=TZ), set(feat.loc[tr, "ts"]))
        self.assertIn(pd.Timestamp("2024-05-21 12:00", tz=TZ), set(feat.loc[te, "ts"]))  # testte satır korunur
        self.assertGreater(int(feat.loc[tr, "price_lag_d7"].isna().sum()), 0)  # özellik NaN'ları atılmadı


class TestOlcum(unittest.TestCase):
    def _bt_ve_tahmin(self, fark_havasiz=10.0, fark_havali=5.0):
        df = synth(days=40)[["ts", "price"]]
        bt = bl.run_backtest(bl.price_table(df), "2024-03-20", "2024-03-25")
        ts = (bt["gun"] + pd.to_timedelta(bt["saat"], unit="h")).dt.tz_localize(TZ)
        pred = pd.DataFrame({"ts": ts, "price": bt["gercek"].to_numpy(), "blok_no": 1,
                             "tahmin_havasiz": bt["gercek"].to_numpy() + fark_havasiz,
                             "tahmin_havali": bt["gercek"].to_numpy() + fark_havali})
        return bt, pred

    def test_birlestirme_satir_kaybetmez_ve_dogru_hizalar(self):
        bt, pred = self._bt_ve_tahmin()
        pred = pred.sample(frac=1, random_state=3).reset_index(drop=True)  # sırayı karıştır
        bt_all = lw.merge_predictions(bt, pred)
        self.assertEqual(len(bt_all), len(bt))
        np.testing.assert_allclose(bt_all["tahmin_havasiz"] - bt_all["gercek"], 10.0)
        np.testing.assert_allclose(bt_all["tahmin_havali"] - bt_all["gercek"], 5.0)

    def test_ozet_baselines_ile_ayni_ve_lgbm_mae_dogru(self):
        bt, pred = self._bt_ve_tahmin()
        common = lw.common_rows_all(lw.merge_predictions(bt, pred))
        s = lw.summarize_all(common)
        b = bl.summarize(common)
        for k in ("mae_d1", "mae_d7", "mae_kural", "rmae_d7", "rmae_kural", "n"):
            self.assertAlmostEqual(s[k], b[k])  # kıyaslar baselines.summarize ile birebir aynı
        self.assertAlmostEqual(s["mae_havasiz"], 10.0)
        self.assertAlmostEqual(s["mae_havali"], 5.0)
        self.assertAlmostEqual(s["rmae_havali"], 5.0 / s["mae_d1"])

    def test_ortak_satir_kumesi_NaN_tahmin_olan_satiri_disarida_birakir(self):
        bt, pred = self._bt_ve_tahmin()
        pred.loc[0, "tahmin_havali"] = float("nan")
        bt_all = lw.merge_predictions(bt, pred)
        common = lw.common_rows_all(bt_all)
        self.assertEqual(len(common), len(bl.common_rows(bt)) - 1)

    def test_grup_ozeti_ve_etiket_bl_den_gelir(self):
        bt, pred = self._bt_ve_tahmin()
        common = lw.common_rows_all(lw.merge_predictions(bt, pred))
        g = lw.grouped(common, common["gun"].dt.strftime("%Y-%m-%d"), "gun_str")
        self.assertEqual(int(g["n"].sum()), len(common))
        self.assertIn("rmae_havali", g.columns)

    def test_ozellik_onemi_tablosu(self):
        imps = pd.DataFrame([
            {"model": "havasiz", "blok_no": 1, "a": 3.0, "b": 1.0},
            {"model": "havasiz", "blok_no": 2, "a": 1.0, "b": 1.0},
            {"model": "havali", "blok_no": 1, "a": 2.0, "b": 1.0, "c": 1.0},
        ])
        t = lw.importance_table(imps, "havasiz")
        self.assertEqual(list(t["özellik"]), ["a", "b"])  # c (hep NaN) yok
        self.assertAlmostEqual(float(t["ortalama gain payı (%)"].iloc[0]), 62.5)
        self.assertAlmostEqual(float(t["ortalama gain payı (%)"].iloc[1]), 37.5)
        self.assertEqual(list(lw.importance_table(imps, "havali")["özellik"]), ["a", "b", "c"])

    def test_nan_sayimi(self):
        df = synth(days=60)
        idx = df.index[df["ts"] == pd.Timestamp("2024-04-10 12:00", tz=TZ)]
        df.loc[idx, MODEL_FEATURES_PD2[0]] = float("nan")
        s = lw.nan_summary(df, "2024-04-05", "2024-04-15")
        self.assertEqual(s["test_rows"], 11 * 24)
        self.assertEqual(s["weather_any_nan"], 1)
        self.assertEqual(s["base_any_nan"], 0)

    @unittest.skipUnless(lw.DATA_PATH.exists() and HAS_PYARROW, "gerçek veri (dataset_weather.parquet) yok")
    def test_gercek_veride_kiyas_degerleri_baselines_md_ile_ayni(self):
        df = pd.read_parquet(lw.DATA_PATH, columns=["ts", "price"])
        bt = bl.run_backtest(bl.price_table(df))
        s = bl.summarize(bl.common_rows(bt))
        self.assertEqual(round(s["mae_d1"], 1), 519.2)
        self.assertEqual(round(s["mae_kural"], 1), 497.5)
        self.assertEqual(s["n"], 8760)


class TestKaynakKurallari(unittest.TestCase):
    def test_kodda_doldurma_yontemi_yok(self):
        src = inspect.getsource(lw)
        for yasak in ("fillna", "ffill", "bfill", "interpolate"):
            self.assertNotIn(yasak, src)

    def test_lightgbm_sadece_fonksiyon_icinde_ice_aktarilir(self):
        for satir in inspect.getsource(lw).splitlines():
            self.assertFalse(satir.startswith("import lightgbm") or satir.startswith("from lightgbm"), satir)
        self.assertIn("    import lightgbm as lgb", inspect.getsource(lw.fit_predict))

    def test_parametreler_sabit_ve_istenen_degerlerde(self):
        p = lw.LGBM_PARAMS
        self.assertEqual(p["objective"], "regression_l1")
        self.assertEqual((p["learning_rate"], p["n_estimators"], p["num_leaves"], p["min_child_samples"]),
                         (0.05, 400, 31, 50))
        self.assertEqual((p["subsample"], p["subsample_freq"], p["colsample_bytree"]), (0.8, 1, 0.8))
        self.assertEqual((p["random_state"], p["deterministic"], p["force_row_wise"]), (42, True, True))
        self.assertEqual(lw.TRAIN_START, pd.Timestamp("2024-03-08 00:00", tz=TZ))
        self.assertEqual(lw.BLOCK_DAYS, 28)

    def test_model_setleri(self):
        self.assertEqual(lw.MODEL_SETS["havasiz"], pf.FEATURES_BASE)
        self.assertEqual(lw.MODEL_SETS["havali"], pf.FEATURES_WEATHER)


@unittest.skipUnless(HAS_LGBM, "lightgbm kurulu değil")
class TestLightGBMUctanUca(unittest.TestCase):
    PARAMS = dict(objective="regression_l1", n_estimators=15, num_leaves=7, min_child_samples=5,
                  learning_rate=0.1, random_state=42, deterministic=True, force_row_wise=True, verbosity=-1)
    TS0 = pd.Timestamp("2024-03-08 00:00", tz=TZ)

    def _run(self, df):
        return lw.run_walkforward(df, "2024-05-20", "2024-06-02", train_start=self.TS0, block_days=7, params=self.PARAMS)

    def test_iki_model_satir_sayisini_korur_nan_doldurulmaz(self):
        df = synth()
        t = (df["ts"] >= pd.Timestamp("2024-05-22", tz=TZ)) & (df["ts"] < pd.Timestamp("2024-05-23", tz=TZ))
        df.loc[t, MODEL_FEATURES_PD2[2]] = float("nan")  # test döneminde hava boşluğu
        nan_once = int(df[MODEL_FEATURES_PD2[2]].isna().sum())
        pred, imps, blocks = self._run(df)
        self.assertEqual(len(pred), 14 * 24)
        self.assertEqual(set(pred["blok_no"]), {1, 2})
        self.assertFalse(pred[["tahmin_havasiz", "tahmin_havali"]].isna().any().any())
        self.assertEqual(int(df[MODEL_FEATURES_PD2[2]].isna().sum()), nan_once)  # girdi değişmedi, doldurulmadı
        orijinal = df.set_index("ts")["price"].reindex(pred["ts"]).to_numpy()
        np.testing.assert_array_equal(pred["price"].to_numpy(), orijinal)
        self.assertEqual(len(imps), 2 * len(blocks))  # blok başına 2 model
        havasiz = imps[imps["model"] == "havasiz"].dropna(axis=1, how="all").columns
        self.assertFalse(set(MODEL_FEATURES_PD2) & set(havasiz))  # havasız modelde hava özelliği yok

    def test_model_duzeyinde_sizinti_blok_baslangicindan_sonraki_fiyatlar_degisse_ilk_gun_ayni(self):
        df = synth()
        D = pd.Timestamp("2024-05-20 00:00", tz=TZ)
        mod = df.copy()
        m = mod["ts"] >= D
        mod.loc[m, "price"] = np.random.default_rng(5).uniform(1e5, 2e5, int(m.sum()))
        p1, _, _ = self._run(df)
        p2, _, _ = self._run(mod)
        ilk_gun = p1["ts"] < D + pd.Timedelta(days=1)
        for c in ("tahmin_havasiz", "tahmin_havali"):
            np.testing.assert_array_equal(p1.loc[ilk_gun, c].to_numpy(), p2.loc[ilk_gun, c].to_numpy())

    def test_fit_predict_gain_butun_ozellikler(self):
        df = synth()
        feat = pf.build_features(df)
        tr = feat[(feat["ts"] >= self.TS0) & (feat["ts"] < pd.Timestamp("2024-05-20", tz=TZ))]
        te = feat[feat["ts"] >= pd.Timestamp("2024-05-20", tz=TZ)].head(24)
        pred, gain = lw.fit_predict(tr, te, pf.FEATURES_WEATHER, self.PARAMS)
        self.assertEqual(len(pred), 24)
        self.assertEqual(list(gain), pf.FEATURES_WEATHER)


if __name__ == "__main__":
    unittest.main()
