"""weather ve weather_quality için AĞSIZ testler. Sahte JSON / sahte HTTP kullanır.

Çalıştırma (repo kökünden): python -m unittest discover -s tests   (veya pytest)
"""

import io
import json
import ssl
import tempfile
import unittest
import urllib.error
import urllib.parse
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

from src.data import weather as w
from src.data import weather_quality as q

CITY = {"test": (1.0, 2.0)}


def _fake_json(url, null_first_hours=0, rows=None):
    """URL'deki start/end'e göre sahte Open-Meteo yanıtı üretir."""
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    start, end = qs["start_date"][0], qs["end_date"][0]
    times = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    if rows is not None:
        times = times[:rows]
    hourly = {"time": times}
    for c in w.HOURLY_VARS:
        hourly[c] = [None if i < null_first_hours else float(i) for i in range(len(times))]
    return json.dumps({"hourly": hourly})


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sleeps = []
        p = mock.patch.object(w.time, "sleep", side_effect=self.sleeps.append)  # beklemeyi taklit et
        p.start()
        self.addCleanup(p.stop)
        self.calls = []

    def patch_http(self, fn):
        def wrapped(url):
            self.calls.append(url)
            return fn(url)
        p = mock.patch.object(w, "_http_get", side_effect=wrapped)
        p.start()
        self.addCleanup(p.stop)

    def run_fetch(self, start="2024-02-01", end="2024-02-03", **kw):
        return w.fetch_weather_history(
            start=start, end=end, cities=CITY, out_root=self.root / "weather",
            log_path=self.root / "run.log", **kw)

    def log_text(self):
        return (self.root / "run.log").read_text(encoding="utf-8")


class TestParcalama(unittest.TestCase):
    def test_aylik_parcalar(self):
        got = list(w._month_chunks(date(2024, 1, 15), date(2024, 3, 10)))
        self.assertEqual(got, [
            (date(2024, 1, 15), date(2024, 1, 31)),
            (date(2024, 2, 1), date(2024, 2, 29)),  # artık yıl
            (date(2024, 3, 1), date(2024, 3, 10)),
        ])

    def test_yil_sinirini_asar(self):
        got = list(w._month_chunks(date(2025, 12, 1), date(2026, 1, 5)))
        self.assertEqual(got, [(date(2025, 12, 1), date(2025, 12, 31)), (date(2026, 1, 1), date(2026, 1, 5))])

    def test_url_icerigi(self):
        url = w.build_url(41.01, 28.98, date(2024, 2, 1), date(2024, 2, 29))
        self.assertIn("previous-runs-api.open-meteo.com", url)
        self.assertNotIn("archive", url)
        self.assertIn("models=ecmwf_ifs025", url)
        self.assertIn("timezone=Europe%2FIstanbul", url)
        for c in w.HOURLY_VARS:
            self.assertIn(c, url)
        self.assertEqual(len(w.HOURLY_VARS), 10)


class TestNullDoldurulmaz(unittest.TestCase):
    def test_parse_null_nan_olur_ve_doldurulmaz(self):
        text = _fake_json("https://x/?start_date=2024-02-01&end_date=2024-02-01", null_first_hours=5)
        df = w.parse_response(text)
        self.assertEqual(len(df), 24)
        for c in w.HOURLY_VARS:
            self.assertEqual(int(df[c].isna().sum()), 5)
            self.assertEqual(df[c].iloc[5], 5.0)  # null sonrası değer olduğu gibi

    def test_eksik_sutun_hata_verir(self):
        text = json.dumps({"hourly": {"time": ["2024-02-01T00:00"]}})
        with self.assertRaises(ValueError):
            w.parse_response(text)


class TestIndirme(_Base):
    def test_dosya_yazilir_ve_null_korunur(self):
        self.patch_http(lambda url: _fake_json(url, null_first_hours=5))
        s = self.run_fetch()
        self.assertEqual(s["written"], 1)
        df = pd.read_csv(self.root / "weather" / "test" / "weather_test_2024-02.csv")
        self.assertEqual(len(df), 72)
        self.assertEqual(list(df.columns), ["time", *w.HOURLY_VARS])
        self.assertTrue(df[w.HOURLY_VARS[0]].iloc[:5].isna().all())  # doldurulmadı
        self.assertEqual(df[w.HOURLY_VARS[0]].iloc[5], 5.0)
        log = self.log_text()
        self.assertIn("satır 72 (beklenen 72)", log)
        self.assertIn("temperature_2m_d1=5", log)  # null sayısı loglandı
        self.assertFalse(list((self.root / "weather" / "test").glob("*.tmp")))

    def test_bos_yanit_loglanir_dosya_yazilmaz(self):
        bos = json.dumps({"hourly": {"time": [], **{c: [] for c in w.HOURLY_VARS}}})
        self.patch_http(lambda url: bos)
        s = self.run_fetch()
        self.assertEqual(s["empty"], [("test", "2024-02")])
        self.assertIn("BOŞ", self.log_text())
        self.assertFalse((self.root / "weather" / "test" / "weather_test_2024-02.csv").exists())

    def test_kisa_yanit_uyari_loglanir_ama_yazilir(self):
        self.patch_http(lambda url: _fake_json(url, rows=20))
        s = self.run_fetch()
        self.assertEqual(s["warnings"], [("test", "2024-02", 20, 72)])
        self.assertIn("UYARI", self.log_text())
        df = pd.read_csv(self.root / "weather" / "test" / "weather_test_2024-02.csv")
        self.assertEqual(len(df), 20)  # eksik saatler uydurulmadı

    def test_devam_edebilir_tam_ay_atlanir_son_ay_yeniden_iner(self):
        self.patch_http(lambda url: _fake_json(url))
        self.run_fetch(start="2024-01-01", end="2024-02-10")
        self.assertEqual(len(self.calls), 2)  # Ocak + Şubat
        self.calls.clear()
        s = self.run_fetch(start="2024-01-01", end="2024-02-10")
        self.assertEqual(len(self.calls), 1)  # sadece son ay (Şubat) yeniden indi
        self.assertEqual(s["skipped"], 1)

    def test_kopma_uc_denemeden_sonra_parca_basarisiz_diger_devam(self):
        def fn(url):
            if "start_date=2024-01-01" in url:
                raise ConnectionResetError("koptu")
            return _fake_json(url)
        self.patch_http(fn)
        s = self.run_fetch(start="2024-01-01", end="2024-02-29")
        self.assertEqual(s["failed"], [("test", "2024-01")])
        self.assertEqual(self.sleeps[:2], [5, 15])  # bekleme süreleri
        self.assertEqual(sum(1 for u in self.calls if "start_date=2024-01-01" in u), 3)  # tam 3 deneme
        self.assertTrue((self.root / "weather" / "test" / "weather_test_2024-02.csv").exists())
        self.assertIn("BAŞARISIZ", self.log_text())
        self.assertIsNone(s["stopped"])

    def test_429_iki_kez_gelirse_durur(self):
        def fn(url):
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, io.BytesIO(b"limit asildi"))
        self.patch_http(fn)
        s = self.run_fetch()
        self.assertEqual(len(self.calls), 2)  # ilk + 60 sn sonra 1 deneme
        self.assertIn(60, self.sleeps)
        self.assertIn("429", s["stopped"])

    def test_429_bir_kez_sonra_basari(self):
        durum = {"n": 0}

        def fn(url):
            durum["n"] += 1
            if durum["n"] == 1:
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, io.BytesIO(b"x"))
            return _fake_json(url)
        self.patch_http(fn)
        s = self.run_fetch()
        self.assertIsNone(s["stopped"])
        self.assertEqual(s["written"], 1)

    def test_baska_http_hatasi_metinle_durur(self):
        def fn(url):
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b'{"reason":"gecersiz degisken"}'))
        self.patch_http(fn)
        s = self.run_fetch()
        self.assertEqual(len(self.calls), 1)  # tekrar denenmedi
        self.assertIn("400", s["stopped"])
        self.assertIn("gecersiz degisken", s["stopped"])

    def test_kesik_json_kopma_gibi_uc_deneme_sonra_parca_atlanir(self):
        self.patch_http(lambda url: '{"hourly": {"time": ["2024-02-01T00:00"')  # kesik yanıt
        s = self.run_fetch()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(s["failed"], [("test", "2024-02")])
        self.assertEqual(self.sleeps[:2], [5, 15])
        self.assertIsNone(s["stopped"])  # tüm iş durmadı

    def test_kesik_json_iki_kez_sonra_gecerli_yanit(self):
        durum = {"n": 0}

        def fn(url):
            durum["n"] += 1
            return '{"hourly":' if durum["n"] <= 2 else _fake_json(url)
        self.patch_http(fn)
        s = self.run_fetch()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(s["written"], 1)
        self.assertEqual(s["failed"], [])

    def test_sutun_eksikse_hala_durur(self):
        eksik = json.dumps({"hourly": {"time": ["2024-02-01T00:00"]}})  # geçerli JSON ama sütunlar yok
        self.patch_http(lambda url: eksik)
        s = self.run_fetch()
        self.assertEqual(len(self.calls), 1)  # tekrar denenmedi
        self.assertIsNotNone(s["stopped"])
        self.assertEqual(s["failed"], [])

    def test_ssl_hatasi_uc_deneme_yapilir(self):
        def fn(url):
            raise ssl.SSLError("sahte ssl hatasi")
        self.patch_http(fn)
        s = self.run_fetch()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(s["failed"], [("test", "2024-02")])
        self.assertEqual(self.sleeps[:2], [5, 15])
        self.assertIsNone(s["stopped"])

    def test_http_hatasi_oserror_olsa_da_http_olarak_islenir(self):
        # HTTPError, OSError'ın alt sınıfı; yine de 3 deneme DEĞİL, hemen durmalı (400).
        def fn(url):
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b"x"))
        self.patch_http(fn)
        s = self.run_fetch()
        self.assertEqual(len(self.calls), 1)
        self.assertIn("400", s["stopped"])

    def test_sure_siniri_durdurur(self):
        self.patch_http(lambda url: _fake_json(url))
        s = self.run_fetch(max_seconds=-1)
        self.assertEqual(len(self.calls), 0)
        self.assertIn("süre sınırı", s["stopped"])


def _frame(times, **overrides):
    """10 sütunlu sahte tablo. overrides: sütun -> değer listesi."""
    t = pd.to_datetime(pd.Series(times))
    df = pd.DataFrame({"time": t})
    for c in w.HOURLY_VARS:
        df[c] = [1.0] * len(t)
    for c, vals in overrides.items():
        df[c] = vals
    return df


class TestKalite(unittest.TestCase):
    def test_ilk_dolu_ve_tam_dolu(self):
        times = pd.date_range("2024-01-01", periods=240, freq="h")  # 10 gün
        vals = [float("nan")] * 30 + [1.0] * 210
        vals[100] = float("nan")  # 2024-01-05 04:00'te ara bir null
        s = q.column_stats(pd.Series(times), pd.Series(vals))
        self.assertEqual(s["null"], 31)
        self.assertEqual(s["ilk_dolu"], pd.Timestamp("2024-01-02 06:00"))
        self.assertEqual(s["tam_dolu"], pd.Timestamp("2024-01-06"))  # son null'ın ertesi günü
        self.assertEqual(s["ilk_dolu_sonrasi_null"], 1)  # baştaki 30 null sayılmaz, sadece ortadaki delik

    def test_ilk_dolu_sonrasi_null_rapora_girer(self):
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        c = w.HOURLY_VARS[0]
        vals = [float("nan")] * 10 + [1.0] * 38
        vals[20] = float("nan")  # ortada 1 delik
        text = q.build_report({"ankara": _frame(times, **{c: vals})})
        self.assertIn("ilk dolu sonrası null", text)
        satir = [l for l in text.splitlines() if l.startswith("| ankara | " + c)][0]
        self.assertIn("| 11 | 2024-01-01 10:00 | 1 |", satir)  # null=11 (10 baş + 1 delik), ilk dolu 10:00, sonrası=1

    def test_hic_dolu_yoksa_ilk_dolu_sonrasi_none(self):
        times = pd.Series(pd.date_range("2024-03-01", periods=24, freq="h"))
        s = q.column_stats(times, pd.Series([float("nan")] * 24))
        self.assertIsNone(s["ilk_dolu_sonrasi_null"])

    def test_hic_null_yoksa_tam_dolu_ilk_gun(self):
        times = pd.Series(pd.date_range("2024-03-01 05:00", periods=48, freq="h"))
        s = q.column_stats(times, pd.Series([2.0] * 48))
        self.assertEqual(s["tam_dolu"], pd.Timestamp("2024-03-01"))
        self.assertEqual(s["ilk_dolu"], pd.Timestamp("2024-03-01 05:00"))

    def test_tamami_null(self):
        times = pd.Series(pd.date_range("2024-03-01", periods=24, freq="h"))
        s = q.column_stats(times, pd.Series([float("nan")] * 24))
        self.assertIsNone(s["ilk_dolu"])
        self.assertIsNone(s["tam_dolu"])

    def test_tekrar_ve_eksik_saat(self):
        df = _frame(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00"])
        s = q.city_stats(df)
        self.assertEqual(s["tekrar"], 1)
        self.assertEqual(s["eksik_saat"], 1)  # 02:00 yok

    def test_sehir_farki_vurgulanir(self):
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        c = w.HOURLY_VARS[0]
        a = _frame(times)
        b = _frame(times, **{c: [float("nan")] * 10 + [1.0] * 38})
        text = q.build_report({"ankara": a, "izmir": b})
        self.assertIn("ŞEHİRLER ARASI FARK VAR", text)
        self.assertIn("ankara=2024-01-01 00:00", text)
        self.assertIn("izmir=2024-01-01 10:00", text)

    def test_fark_yoksa_belirtilir(self):
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        text = q.build_report({"ankara": _frame(times), "izmir": _frame(times)})
        self.assertIn("farkı yok", text)
        self.assertNotIn("FARK VAR", text)

    def test_rapor_yok_sehir_listelenir(self):
        times = pd.date_range("2024-01-01", periods=24, freq="h")
        text = q.build_report({"ankara": _frame(times)}, absent=["konya"])
        self.assertIn("konya", text)


if __name__ == "__main__":
    unittest.main()
