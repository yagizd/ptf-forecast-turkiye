"""build_weather_dataset için AĞSIZ testler. Küçük sahte tablolarla çalışır.

Çalıştırma (repo kökünden): python -m unittest discover -s tests   (veya pytest)
"""

import inspect
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.data import build_weather_dataset as b
from src.data.weather import BASE_VARS, CITIES

TZ = "Europe/Istanbul"
CITY_LIST = list(CITIES)


def raw_city(n=48, start="2024-01-01 00:00", base_value=0.0, nan=None):
    """CSV'den okunmuş gibi ham şehir tablosu. time: saat dilimsiz yerel saat yazısı.

    Her hücre = base_value + satır_no. nan: {sütun: [satır numaraları]} -> o hücreler NaN.
    """
    t = pd.date_range(start, periods=n, freq="h")
    df = pd.DataFrame({"time": t.strftime("%Y-%m-%dT%H:%M")})
    for var in BASE_VARS:
        for d in (1, 2):
            df[f"{var}_previous_day{d}"] = [base_value + i for i in range(n)]
    for col, rows in (nan or {}).items():
        for r in rows:
            df.loc[r, col] = float("nan")
    return df


def all_frames(n=48, nan_by_city=None, start="2024-01-01 00:00"):
    """8 şehir için hazırlanmış (ts'li) tablolar. Şehir i'nin değerleri i + satır_no."""
    nan_by_city = nan_by_city or {}
    return {c: b.prepare_city_frame(raw_city(n=n, start=start, base_value=float(i), nan=nan_by_city.get(c)))
            for i, c in enumerate(CITY_LIST)}


def counts(k=1):
    return {c: k for c in CITY_LIST}


def base_df(n=48, start="2024-01-01 00:00", lep_nan=(5, 6)):
    ts = pd.date_range(start, periods=n, freq="h", tz=TZ)
    lep = [300.0 + i for i in range(n)]
    for i in lep_nan:
        if i < n:
            lep[i] = float("nan")
    return pd.DataFrame({
        "ts": ts,
        "price": [100.0 + i for i in range(n)],
        "consumption": [200.0 + i for i in range(n)],
        "lep": lep,
    })


TEMP2 = f"{BASE_VARS[0]}_previous_day2"
WIND2 = f"{BASE_VARS[1]}_previous_day2"
SOLAR2 = f"{BASE_VARS[2]}_previous_day2"
CLOUD2 = f"{BASE_VARS[3]}_previous_day2"


class TestSabitler(unittest.TestCase):
    def test_model_ozellikleri_tam_5_ve_pd1_yok(self):
        self.assertEqual(b.MODEL_FEATURES_PD2,
                         ["temp_mean_pd2", "cloud_mean_pd2", "precip_mean_pd2",
                          "wind_canakkale_pd2", "solar_konya_pd2"])
        self.assertEqual(len(b.MODEL_FEATURES_PD2), 5)
        for c in b.MODEL_FEATURES_PD2:
            self.assertNotIn("pd1", c)
            self.assertTrue(c.endswith("_pd2"))

    def test_pd1_sutunlari_modele_girmez(self):
        self.assertEqual(len(b.PD1_COLUMNS), 5)
        self.assertFalse(set(b.PD1_COLUMNS) & set(b.MODEL_FEATURES_PD2))
        self.assertEqual(b.WEATHER_COLUMNS, b.MODEL_FEATURES_PD2 + b.PD1_COLUMNS)

    def test_degisken_adlari_ve_sehirler_weather_ile_uyumlu(self):
        self.assertEqual(BASE_VARS, ["temperature_2m", "wind_speed_100m", "shortwave_radiation",
                                     "cloud_cover", "precipitation"])
        self.assertEqual((b.TEMP, b.WIND, b.SOLAR, b.CLOUD, b.PRECIP), tuple(BASE_VARS))
        self.assertEqual(len(CITY_LIST), 8)
        self.assertIn("canakkale", CITY_LIST)
        self.assertIn("konya", CITY_LIST)

    def test_cikti_sutun_sirasi(self):
        base = base_df()
        out, _ = b.merge_weather(base, b.build_weather_features(all_frames()))
        self.assertEqual(list(out.columns), ["ts", "price", "consumption", "lep", *b.MODEL_FEATURES_PD2, *b.PD1_COLUMNS])

    def test_kodda_doldurma_yontemi_yok(self):
        src = inspect.getsource(b)
        for yasak in ("fillna", "ffill", "bfill", "interpolate"):
            self.assertNotIn(yasak, src)


class TestOrtalama(unittest.TestCase):
    def test_8_sehir_ortalamasi_dogru(self):
        f = b.build_weather_features(all_frames())
        # şehir i değeri i + satır -> ortalama 3.5 + satır
        for r in (0, 7, 47):
            self.assertAlmostEqual(f["temp_mean_pd2"].iloc[r], 3.5 + r)
            self.assertAlmostEqual(f["cloud_mean_pd1"].iloc[r], 3.5 + r)

    def test_tek_sehirde_nan_varsa_ortalama_nan(self):
        frames = all_frames(nan_by_city={"izmir": {TEMP2: [3]}})
        f = b.build_weather_features(frames)
        self.assertTrue(pd.isna(f["temp_mean_pd2"].iloc[3]))
        self.assertEqual(int(f["temp_mean_pd2"].isna().sum()), 1)
        self.assertAlmostEqual(f["temp_mean_pd2"].iloc[4], 3.5 + 4)  # diğer saatler etkilenmez
        self.assertEqual(int(f["cloud_mean_pd2"].isna().sum()), 0)  # başka sütun etkilenmez

    def test_tek_sehir_sutunlari_sadece_o_sehirden(self):
        frames = all_frames()
        f = b.build_weather_features(frames)
        i_can, i_kon = CITY_LIST.index("canakkale"), CITY_LIST.index("konya")
        self.assertAlmostEqual(f["wind_canakkale_pd2"].iloc[10], i_can + 10)
        self.assertAlmostEqual(f["solar_konya_pd2"].iloc[10], i_kon + 10)
        self.assertAlmostEqual(f["wind_canakkale_pd1"].iloc[2], i_can + 2)

    def test_nan_doldurulmaz(self):
        nan_rows = [0, 1, 2, 10]
        frames = all_frames(nan_by_city={"canakkale": {WIND2: nan_rows}})
        f = b.build_weather_features(frames)
        w = f["wind_canakkale_pd2"]
        self.assertEqual(list(w[w.isna()].index.map(lambda t: f.index.get_loc(t))), nan_rows)
        self.assertEqual(int(w.isna().sum()), 4)
        self.assertEqual(int(f["solar_konya_pd2"].isna().sum()), 0)

    def test_baslangicta_nan_kalir(self):
        frames = all_frames(nan_by_city={c: {CLOUD2: [0, 1, 2]} for c in CITY_LIST})
        f = b.build_weather_features(frames)
        self.assertTrue(f["cloud_mean_pd2"].iloc[:3].isna().all())  # ilk satırlar NaN kalmalı (geriye/ileriye taşınmaz)
        self.assertFalse(f["cloud_mean_pd2"].iloc[3:].isna().any())


class TestBirlestirme(unittest.TestCase):
    def test_left_join_satir_sayisini_korur_fazla_hava_saati_disarida_kalir(self):
        base = base_df(n=48)
        feats = b.build_weather_features(all_frames(n=72))  # hava tarafında 24 saat fazla
        out, info = b.merge_weather(base, feats)
        self.assertEqual(len(out), 48)
        self.assertEqual(info["outside"], 24)
        self.assertEqual(info["unmatched_base"], 0)

    def test_hava_karsiligi_olmayan_dataset_saati_nan_kalir(self):
        base = base_df(n=72)
        feats = b.build_weather_features(all_frames(n=48))  # son 24 saatte hava yok
        out, info = b.merge_weather(base, feats)
        self.assertEqual(len(out), 72)
        self.assertEqual(info["unmatched_base"], 24)
        self.assertEqual(info["outside"], 0)
        self.assertTrue(out[b.WEATHER_COLUMNS].iloc[48:].isna().all().all())
        self.assertFalse(out[b.WEATHER_COLUMNS].iloc[:48].isna().any().any())

    def test_price_consumption_lep_degismez_nan_dahil(self):
        base = base_df(n=48, lep_nan=(5, 6))
        out, _ = b.merge_weather(base, b.build_weather_features(all_frames(n=48)))
        b.check_unchanged(base, out)  # hata vermemeli
        self.assertEqual(int(out["lep"].isna().sum()), 2)
        self.assertEqual(list(out.index[out["lep"].isna()]), [5, 6])

    def test_degisen_deger_yakalanir(self):
        base = base_df(n=48)
        out, _ = b.merge_weather(base, b.build_weather_features(all_frames(n=48)))
        bozuk = out.copy()
        bozuk.loc[0, "price"] += 0.001
        with self.assertRaises(AssertionError):
            b.check_unchanged(base, bozuk)
        bozuk2 = out.copy()
        bozuk2.loc[5, "lep"] = 0.0  # NaN'ı 0 yapmak da yakalanmalı
        with self.assertRaises(AssertionError):
            b.check_unchanged(base, bozuk2)


class TestHatalar(unittest.TestCase):
    def test_tekrar_eden_zaman_hata(self):
        frames = all_frames()
        frames["ankara"] = pd.concat([frames["ankara"], frames["ankara"].iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            b.validate_city_frames(frames, counts(), expected_rows=None)

    def test_eksik_sehir_hata(self):
        frames = all_frames()
        del frames["konya"]
        with self.assertRaises(ValueError):
            b.validate_city_frames(frames, counts(), expected_rows=None)

    def test_farkli_zaman_dizisi_hata(self):
        frames = all_frames()
        frames["izmir"] = b.prepare_city_frame(raw_city(start="2024-01-02 00:00"))
        with self.assertRaises(ValueError):
            b.validate_city_frames(frames, counts(), expected_rows=None)

    def test_farkli_aylik_dosya_sayisi_hata(self):
        c = counts()
        c["bursa"] = 2
        with self.assertRaises(ValueError):
            b.validate_city_frames(all_frames(), c, expected_rows=None)

    def test_beklenen_satir_sayisi_tutmazsa_hata(self):
        with self.assertRaises(ValueError):
            b.validate_city_frames(all_frames(n=48), counts(), expected_rows=47)
        b.validate_city_frames(all_frames(n=48), counts(), expected_rows=48)  # tutunca hata yok

    def test_dataset_ts_tekrari_hata(self):
        base = base_df()
        base = pd.concat([base, base.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            b.check_base(base, expected_rows=None)

    def test_dataset_satir_sayisi_tutmazsa_hata(self):
        with self.assertRaises(ValueError):
            b.check_base(base_df(n=48), expected_rows=999)
        b.check_base(base_df(n=48), expected_rows=48)

    def test_dataset_yanlis_saat_dilimi_hata(self):
        base = base_df()
        base["ts"] = base["ts"].dt.tz_convert("UTC")
        with self.assertRaises(ValueError):
            b.check_base(base, expected_rows=None)

    def test_eksik_sutun_hata(self):
        raw = raw_city().drop(columns=[TEMP2])
        with self.assertRaises(ValueError):
            b.prepare_city_frame(raw)

    def test_saat_dilimli_time_hata(self):
        raw = raw_city()
        raw["time"] = raw["time"] + "+03:00"
        with self.assertRaises(ValueError):
            b.prepare_city_frame(raw)


class TestSaatDilimi(unittest.TestCase):
    def test_naive_saat_istanbula_cevrilir(self):
        p = b.prepare_city_frame(raw_city(start="2024-01-01 00:00"))
        self.assertEqual(str(p["ts"].dt.tz), TZ)
        self.assertEqual(p["ts"].iloc[0], pd.Timestamp("2024-01-01 00:00:00+03:00"))
        self.assertEqual(p["ts"].iloc[0].utcoffset(), timedelta(hours=3))
        self.assertEqual(p["ts"].iloc[0].hour, 0)  # yerel saat korunur, kaydırılmaz
        self.assertEqual(p["ts"].iloc[23].hour, 23)

    def test_hava_ts_dataset_ts_ile_ayni_ani_gosterir(self):
        base = base_df(n=48, start="2024-01-01 00:00")
        p = b.prepare_city_frame(raw_city(n=48, start="2024-01-01 00:00"))
        self.assertTrue((base["ts"].values == p["ts"].values).all())
        out, info = b.merge_weather(base, b.build_weather_features(all_frames(n=48)))
        self.assertEqual(info["unmatched_base"], 0)  # saat kayması olsaydı eşleşmezdi


class TestYazma(unittest.TestCase):
    def test_ustune_yazmaz_overwrite_ile_yazar(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.parquet"
            df1 = pd.DataFrame({"a": [1.0, float("nan")]})
            b.write_parquet_atomic(df1, path)
            self.assertFalse((Path(d) / "x.parquet.tmp").exists())
            with self.assertRaises(FileExistsError):
                b.write_parquet_atomic(pd.DataFrame({"a": [9.0]}), path)
            self.assertEqual(len(pd.read_parquet(path)), 2)  # değişmedi
            b.write_parquet_atomic(pd.DataFrame({"a": [9.0]}), path, overwrite=True)
            self.assertEqual(len(pd.read_parquet(path)), 1)
            self.assertFalse((Path(d) / "x.parquet.tmp").exists())


class TestBeklentiOlcumu(unittest.TestCase):
    def _beklenen_desenli_tablo(self, delik=True):
        ts = pd.date_range("2024-01-01 00:00", "2026-10-20 23:00", freq="h", tz=TZ)
        out = pd.DataFrame({"ts": ts})
        for col in b.WEATHER_COLUMNS:
            out[col] = 1.0
            baslangic = (pd.Timestamp("2024-03-06 20:00", tz=TZ) if col.startswith(("wind_", "solar_"))
                         else pd.Timestamp("2024-02-04 03:00", tz=TZ))
            out.loc[out["ts"] < baslangic, col] = float("nan")
        if delik:
            m = (out["ts"] >= pd.Timestamp("2026-04-17 18:00", tz=TZ)) & (out["ts"] <= pd.Timestamp("2026-04-23 02:00", tz=TZ))
            out.loc[m, "cloud_mean_pd2"] = float("nan")
            self.assertEqual(int(m.sum()), 129)
        return out

    def test_beklenen_desen_uyusuyor(self):
        res = b.check_expectations(self._beklenen_desenli_tablo())
        self.assertTrue(all(ok for _, _, ok in res), [r for r in res if not r[2]])

    def test_delik_yoksa_uyusmuyor_diye_raporlanir(self):
        res = b.check_expectations(self._beklenen_desenli_tablo(delik=False))
        delik = [r for r in res if r[0].startswith("cloud_mean_pd2: ortada")][0]
        self.assertFalse(delik[2])  # beklenti tutmadı, güzelleştirilmedi

    def test_kucuk_veride_uyusmuyor(self):
        out, _ = b.merge_weather(base_df(), b.build_weather_features(all_frames()))
        res = b.check_expectations(out)
        self.assertFalse(all(ok for _, _, ok in res))


class TestUctanUca(unittest.TestCase):
    def _hazirla(self, d):
        d = Path(d)
        base = base_df(n=72, lep_nan=(5, 6))
        base_path = d / "dataset.parquet"
        base.to_parquet(base_path, index=False)
        wdir = d / "weather"
        for i, c in enumerate(CITY_LIST):
            (wdir / c).mkdir(parents=True)
            raw_city(n=48, base_value=float(i)).to_csv(wdir / c / f"weather_{c}_2024-01.csv", index=False)
        return base, base_path, wdir

    def test_calisir_ustune_yazmaz_hash_raporda(self):
        with tempfile.TemporaryDirectory() as d:
            base, base_path, wdir = self._hazirla(d)
            kw = dict(base_path=base_path, weather_dir=wdir, out_path=Path(d) / "dw.parquet",
                      report_path=Path(d) / "dw.md", expected_dataset_rows=72, expected_city_rows=48)
            r = b.run(**kw)
            out = pd.read_parquet(kw["out_path"])
            self.assertEqual(len(out), 72)
            self.assertEqual(r["outside"], 0)
            self.assertEqual(r["unmatched_base"], 24)
            self.assertEqual(int(out["lep"].isna().sum()), 2)  # lep NaN'ları kaldı
            self.assertEqual(int(out["temp_mean_pd2"].isna().sum()), 24)
            rapor = kw["report_path"].read_text(encoding="utf-8")
            self.assertIn(r["sha256"], rapor)
            self.assertIn("SHA-256", rapor)
            self.assertIn("BİREBİR AYNI", rapor)
            self.assertIn("UYUŞMUYOR", rapor)  # küçük veri beklentiyi tutturamaz; güzelleştirilmez
            h = b._sha256(kw["out_path"])
            with self.assertRaises(FileExistsError):
                b.run(**kw)
            self.assertEqual(b._sha256(kw["out_path"]), h)  # dosya değişmedi
            b.run(overwrite=True, **kw)  # --overwrite ile yazar
            self.assertFalse(list(Path(d).glob("*.tmp")))

    def test_eslesmeyen_hava_dosyasi_yazmadan_durur(self):
        with tempfile.TemporaryDirectory() as d:
            base, base_path, wdir = self._hazirla(d)
            # bir şehirde farklı zaman -> hata, hiçbir çıktı yazılmamalı
            raw_city(n=48, start="2024-01-02 00:00").to_csv(wdir / "izmir" / "weather_izmir_2024-01.csv", index=False)
            out_path, report_path = Path(d) / "dw.parquet", Path(d) / "dw.md"
            with self.assertRaises(ValueError):
                b.run(base_path=base_path, weather_dir=wdir, out_path=out_path, report_path=report_path,
                      expected_dataset_rows=72, expected_city_rows=48)
            self.assertFalse(out_path.exists())
            self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()
