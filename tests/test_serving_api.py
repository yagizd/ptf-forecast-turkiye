"""Servis API testleri (FastAPI TestClient). fastapi/httpx kurulu değilse ya da gerçek paket/tablo yoksa ATLANIR.

Çalıştırma (yalnız bu dosya): python -m unittest tests.test_serving_api -v   (unittest discover YOK)
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    from src.serving.api import app

    HAS_API, API_ATLA = True, ""
except Exception as err:  # fastapi/httpx yok: CI'ya henüz geçilmedi, patlamasın
    HAS_API, API_ATLA = False, f"fastapi/httpx import edilemedi: {type(err).__name__}"

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data" / "processed" / "model_paketi_v1"
TABLE_PATH = ROOT / "data" / "processed" / "dataset_weather_guncel.parquet"
READY = (BUNDLE_DIR / "meta.json").exists() and TABLE_PATH.exists()
ENV = {"MODEL_DIR": str(BUNDLE_DIR), "DATA_PATH": str(TABLE_PATH)}


@unittest.skipUnless(HAS_API, API_ATLA)
@unittest.skipUnless(READY, "paket/tablo yok (data/processed/model_paketi_v1 ve dataset_weather_guncel.parquet)")
class TestApiGercek(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.serving import predictor as pr

        cls.pr = pr
        cls.patcher = patch.dict(os.environ, ENV)
        cls.patcher.start()
        cls.client = TestClient(app)
        cls.client.__enter__()  # lifespan çalışsın
        cls.bundle = pr.load_bundle(BUNDLE_DIR)
        cls.table = pr.load_table(TABLE_PATH, cls.bundle["c_day"])

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.patcher.stop()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(set(j), {"durum", "model_surumu", "egitim_sonu", "tablo_son_gun", "mod"})
        self.assertEqual(j["durum"], "ok")
        self.assertEqual(j["mod"], "replay")
        self.assertEqual(j["egitim_sonu"], self.bundle["c_day"].date().isoformat())
        self.assertEqual(j["tablo_son_gun"], self.table["last_day"].date().isoformat())
        self.assertTrue(j["model_surumu"].startswith("model_paketi_v1@"))

    def test_predict_predictor_ile_birebir_ayni(self):
        for d in ("2026-09-22", "2026-10-01", "2026-10-29"):
            with self.subTest(tarih=d):
                beklenen = json.loads(json.dumps(self.pr.predict_day(self.bundle, self.table, d)))  # JSON'a gidip gelme
                r = self.client.get("/predict", params={"date": d})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json(), beklenen)

    def test_durum_kodlari(self):
        r = self.client.get("/predict", params={"date": "2026-09-21"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("egitimde goruldu", r.json()["detail"])
        r = self.client.get("/predict", params={"date": "2026-10-30"})
        self.assertEqual(r.status_code, 404)
        self.assertIn("tabloda yok", r.json()["detail"])
        self.assertEqual(self.client.get("/predict", params={"date": "abc"}).status_code, 422)
        self.assertEqual(self.client.get("/predict").status_code, 422)  # date parametresi yok


@unittest.skipUnless(HAS_API, API_ATLA)
class TestApiAcilis(unittest.TestCase):
    """Ortam değişkeni eksik ya da yol yanlışsa uygulama AÇILMAZ (gerçek paket gerekmez)."""

    def test_env_eksik(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MODEL_DIR ve DATA_PATH"):
                with TestClient(app):
                    pass

    def test_sadece_biri_var(self):
        with patch.dict(os.environ, {"MODEL_DIR": str(BUNDLE_DIR)}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MODEL_DIR ve DATA_PATH"):
                with TestClient(app):
                    pass

    def test_yol_yok(self):
        with patch.dict(os.environ, {"MODEL_DIR": str(ROOT / "yok_klasor_xyz"), "DATA_PATH": str(ROOT / "yok_xyz.parquet")}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MODEL_DIR klasoru yok"):
                with TestClient(app):
                    pass


if __name__ == "__main__":
    unittest.main()
