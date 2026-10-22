"""Open-Meteo Previous Runs ile geçmiş hava TAHMİNİ indirme (devam edebilir).

Hava verisi Open-Meteo.com (CC BY 4.0), ticari olmayan kullanım.

Önemli: bu gerçekleşen hava (ERA5) DEĞİL, "o gün elde olan tahmin"dir (_previous_day1, _previous_day2).
Böylece model, tahmin anında gerçekten bilebileceği havayı görür (sızıntı olmaz).

Çıktı: data/raw/weather/<sehir>/weather_<sehir>_YYYY-MM.csv  ('time' + 10 sütun)
Log  : data/raw/weather_run.log
Kural: null (boş) değerler ASLA doldurulmaz; CSV'de boş hücre olarak kalır.
Çalıştırma (repo kökünden): python -m src.data.weather
"""

import calendar
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "weather"
LOG_PATH = ROOT / "data" / "raw" / "weather_run.log"

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs025"
TIMEZONE = "Europe/Istanbul"
START = "2024-01-01"  # ölçümde veri ~2024 Ocak sonu / Şubat'ta başlıyor; öncesi null gelir

BASE_VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover", "precipitation"]
# 10 sütun: her değişken için 1 gün ve 2 gün önceki tahmin. Adlar API'deki gibi kalır.
HOURLY_VARS = [f"{v}_previous_day{n}" for v in BASE_VARS for n in (1, 2)]

# şehir (ASCII dosya adı) -> (enlem, boylam)
CITIES = {
    "istanbul": (41.01, 28.98),
    "ankara": (39.93, 32.86),
    "izmir": (38.42, 27.14),
    "bursa": (40.19, 29.06),
    "antalya": (36.90, 30.69),
    "adana": (37.00, 35.32),
    "canakkale": (40.15, 26.41),
    "konya": (37.87, 32.48),
}

WAITS = (5, 15)  # bağlantı kopmasında 1. ve 2. denemeden sonra bekleme (en fazla 3 deneme)
RATE_LIMIT_WAIT = 60  # HTTP 429'da bekleme (sadece 1 kez daha denenir)
PAUSE_SECONDS = 1.0  # istekler arası nezaket beklemesi
MAX_SECONDS = 1800  # toplam süre sınırı: 30 dakika

# HTTPError, URLError'ın (ve OSError'ın) alt sınıfıdır; bu yüzden except'te HTTPError ÖNCE yazılmalı.
# OSError: ssl.SSLError gibi diğer ağ hatalarını da yakalar.
CONNECTION_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException, OSError)
# Yarım kalan / bozuk yanıt gövdesi de tipik bir ağ sorunudur: kopma gibi sayılır.
BROKEN_BODY_ERRORS = (json.JSONDecodeError, UnicodeDecodeError)


class WeatherStop(Exception):
    """Tüm işi durduran hata (429 ikinci kez, başka HTTP hatası, beklenmeyen yanıt, süre sınırı)."""


class ChunkFailed(Exception):
    """Bir parça 3 denemede de koptu. Sadece o parça atlanır, iş devam eder."""


# ------------------------------------------------------------------ yardımcılar
def make_logger(path: Path):
    """Hem ekrana hem dosyaya (UTF-8) yazan log fonksiyonu. Türkçe harf konsolu çökertmez."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def _month_chunks(start: date, end: date):
    """start..end aralığını aylık (ilk gün, son gün) parçalara böler."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        yield max(cur, start), min(date(cur.year, cur.month, last_day), end)
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)


def _rows_in_csv(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)  # başlık satırı hariç


def _short(col: str) -> str:
    """Log'u kısaltmak için: temperature_2m_previous_day1 -> temperature_2m_d1"""
    return col.replace("_previous_day", "_d")


def build_url(lat: float, lon: float, start: date, end: date) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": TIMEZONE,
        "models": MODEL,
    }
    return API_URL + "?" + urllib.parse.urlencode(params, safe=",")


def _http_get(url: str) -> str:
    """Tek GET isteği. (Testlerde bu fonksiyon sahtesiyle değiştirilir.)"""
    req = urllib.request.Request(url, headers={"User-Agent": "ptf-forecast/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def parse_response(text: str) -> pd.DataFrame:
    """JSON yanıtı tabloya çevirir: 'time' + 10 sütun. null -> NaN (DOLDURULMAZ)."""
    data = json.loads(text)
    hourly = data["hourly"]
    missing = [c for c in ["time", *HOURLY_VARS] if c not in hourly]
    if missing:
        raise ValueError(f"yanıtta beklenen sütunlar yok: {missing}")
    df = pd.DataFrame({c: hourly[c] for c in ["time", *HOURLY_VARS]})
    for c in HOURLY_VARS:
        df[c] = pd.to_numeric(df[c])  # None -> NaN; başka hiçbir değişiklik yok
    return df


def _get_text_with_retry(url: str, label: str, log) -> str:
    """Bağlantı kopmasında 3 deneme (5, 15 sn). 429'da 60 sn bekleyip 1 kez daha dener.

    - Kopma ve yarım/bozuk yanıt (kesik JSON) aynı sayılır: 3 denemede de sürerse ChunkFailed
      (sadece bu parça atlanır). Sütun eksikliği ise burada değil parse_response'ta yakalanır (WeatherStop).
    - 429 ikinci kez gelirse veya başka HTTP hatası olursa WeatherStop (her şey durur).
    """
    conn_tries = 0
    rate_retried = False
    while True:
        try:
            text = _http_get(url)
            json.loads(text)  # sadece geçerli JSON mu diye bakar; kesik/bozuksa BROKEN_BODY_ERRORS fırlar
            return text
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", "replace")[:300]
            if err.code == 429:
                if rate_retried:
                    log(f"{label}: HTTP 429 ikinci kez, DURULUYOR")
                    raise WeatherStop("HTTP 429 (ikinci kez): " + body)
                rate_retried = True
                log(f"{label}: HTTP 429, {RATE_LIMIT_WAIT} sn bekleniyor")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            log(f"{label}: HTTP {err.code} hatası, DURULUYOR: {body}")
            raise WeatherStop(f"HTTP {err.code}: {body}")
        except (*CONNECTION_ERRORS, *BROKEN_BODY_ERRORS) as err:
            conn_tries += 1
            if conn_tries > len(WAITS):
                log(f"{label}: deneme {conn_tries}/{len(WAITS) + 1} koptu ({type(err).__name__}), vazgeçildi")
                raise ChunkFailed(type(err).__name__)
            wait = WAITS[conn_tries - 1]
            log(f"{label}: deneme {conn_tries}/{len(WAITS) + 1} koptu ({type(err).__name__}), {wait} sn bekleniyor")
            time.sleep(wait)


# ------------------------------------------------------------------ ana fonksiyon
def fetch_weather_history(
    start: str = START,
    end: str | None = None,
    cities: dict | None = None,
    out_root: Path | None = None,
    log_path: Path | None = None,
    max_seconds: float = MAX_SECONDS,
    pause: float = PAUSE_SECONDS,
    log=None,
) -> dict:
    """Her şehir için start..end (varsayılan: dün) hava tahminini aylık parçalarla indirir.

    - Tam inmiş ay dosyası atlanır; end'in ayı her seferinde yeniden iner.
    - Bir parça 3 denemede de koparsa 'failed' listesine yazılır, diğerlerine devam edilir.
    - Boş (0 satırlık) yanıt dosyaya yazılmaz, 'empty' listesine yazılır.
    - 429 ikinci kez, başka HTTP hatası veya süre sınırı: tüm iş DURUR (summary['stopped']).
    Dönen: {'written', 'skipped', 'failed', 'empty', 'warnings', 'stopped'}
    """
    cities = CITIES if cities is None else cities
    out_root = RAW_DIR if out_root is None else out_root
    log = log or make_logger(log_path or (out_root.parent / "weather_run.log"))

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end) if end else date.today() - timedelta(days=1)
    if end_d < start_d:
        raise ValueError(f"end ({end_d}) start'tan ({start_d}) önce olamaz.")
    chunks = list(_month_chunks(start_d, end_d))

    summary = {"written": 0, "skipped": 0, "failed": [], "empty": [], "warnings": [], "stopped": None}
    t0 = time.monotonic()
    log(f"BAŞLADI {start_d}..{end_d}, {len(cities)} şehir x {len(chunks)} ay, model={MODEL}")

    try:
        for city, (lat, lon) in cities.items():
            for c_start, c_end in chunks:
                ym = f"{c_start.year}-{c_start.month:02d}"
                label = f"{city} {ym}"
                if time.monotonic() - t0 > max_seconds:
                    raise WeatherStop(f"süre sınırı aşıldı ({max_seconds:.0f} sn), kaldığı yer: {label}")

                path = out_root / city / f"weather_{city}_{ym}.csv"
                expected = ((c_end - c_start).days + 1) * 24
                is_last_month = c_end == end_d
                if path.exists() and not is_last_month and _rows_in_csv(path) == expected:
                    summary["skipped"] += 1
                    log(f"{label}: zaten var ({expected} satır), atlandı")
                    continue

                url = build_url(lat, lon, c_start, c_end)
                try:
                    text = _get_text_with_retry(url, label, log)
                except ChunkFailed:
                    summary["failed"].append((city, ym))
                    log(f"{label}: BAŞARISIZ (diğer parçalara devam)")
                    time.sleep(pause)
                    continue
                try:
                    df = parse_response(text)
                except (ValueError, KeyError, json.JSONDecodeError) as err:
                    log(f"{label}: beklenmeyen yanıt, DURULUYOR: {type(err).__name__}: {str(err)[:200]}")
                    raise WeatherStop(f"{label}: beklenmeyen yanıt ({type(err).__name__})")

                nulls = {c: int(df[c].isna().sum()) for c in HOURLY_VARS}
                log(f"{label}: satır {len(df)} (beklenen {expected}); null: "
                    + ", ".join(f"{_short(c)}={n}" for c, n in nulls.items()))

                if len(df) == 0:
                    summary["empty"].append((city, ym))
                    log(f"{label}: BOŞ yanıt (dosya yazılmadı)")
                    time.sleep(pause)
                    continue
                if len(df) != expected:
                    summary["warnings"].append((city, ym, len(df), expected))
                    log(f"{label}: UYARI satır sayısı beklenenden farklı ({len(df)} != {expected})")

                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".csv.tmp")  # yarım dosya kalmasın diye önce geçici dosyaya yaz
                df.to_csv(tmp, index=False, encoding="utf-8")
                tmp.replace(path)
                summary["written"] += 1
                time.sleep(pause)
    except WeatherStop as stop:
        summary["stopped"] = str(stop)
        log(f"DURDU: {stop}")

    log(f"BİTTİ {time.monotonic() - t0:.1f} sn: yazılan {summary['written']}, atlanan {summary['skipped']}, "
        f"başarısız {len(summary['failed'])}, boş {len(summary['empty'])}, uyarı {len(summary['warnings'])}")
    if summary["failed"]:
        log("BAŞARISIZ PARÇALAR: " + ", ".join(f"{c} {m}" for c, m in summary["failed"]))
    if summary["empty"]:
        log("BOŞ YANIT PARÇALARI: " + ", ".join(f"{c} {m}" for c, m in summary["empty"]))
    return summary


def main() -> int:
    summary = fetch_weather_history()
    return 1 if (summary["stopped"] or summary["failed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
