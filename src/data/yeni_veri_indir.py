"""2026-10-21'den itibaren YENİ günlerin ham verisini ayrı dosyalara indirir (eski ham dosyalara dokunmaz).

Çıktılar:
  data/raw/ptf/ptf_yeni_<ilk>_<son>.csv, data/raw/rt_cons/rt_cons_yeni_..., data/raw/load_plan/load_plan_yeni_...
  data/raw/weather/<sehir>/weather_<sehir>_yeni_<ilk>_<son>.csv
Mevcut dosya varsa üzerine yazılmaz (hata verir). Kullanıcı adı/şifre okunmaz; eptr2 .env'i kendisi okur.
Kural: eksik değer doldurulmaz.
Çalıştırma: python -m src.data.yeni_veri_indir 2026-10-21 2026-10-30
"""

import sys
import time
from datetime import date
from pathlib import Path

from eptr2 import EPTR2
from eptr2.calls.consumption import get_load_plan, get_rt_cons
from eptr2.calls.dam import get_mcp

from src.data import weather as wx
from src.data.epias import ROOT, _refuse_login, _safe

RAW = ROOT / "data" / "raw"
LOG = ROOT / "data" / "processed" / "egitim_oturumu_30Ekim" / "ilerleme.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _save(df, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"var olan dosyanın üzerine yazılmaz: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(a: str, b: str) -> int:
    start, end = date.fromisoformat(a), date.fromisoformat(b)
    tag = f"yeni_{a}_{b}"
    failed = []
    eptr = EPTR2(use_dotenv=True, recycle_tgt=True, dotenv_path=str(ROOT / ".env"), tgt_path=str(ROOT))
    eptr.get_tgt = _refuse_login
    for kind, call in (("ptf", get_mcp), ("rt_cons", get_rt_cons), ("load_plan", get_load_plan)):
        path = RAW / kind / f"{kind}_{tag}.csv"
        for attempt in (1, 2, 3):
            try:
                df = call(start_date=a, end_date=b, eptr=eptr)
                _save(df, path)
                log(f"{kind}: {len(df)} satır -> {path.name}")
                break
            except FileExistsError:
                raise
            except Exception as err:  # bağlantı vb.; 3 deneme
                log(f"{kind}: deneme {attempt}/3 hata {type(err).__name__}: {_safe(err)}")
                time.sleep((5, 15, 0)[attempt - 1])
        else:
            failed.append(kind)
        time.sleep(1)
    for city, (lat, lon) in wx.CITIES.items():
        path = RAW / "weather" / city / f"weather_{city}_{tag}.csv"
        label = f"hava {city}"
        try:
            text = wx._get_text_with_retry(wx.build_url(lat, lon, start, end), label, log)
            df = wx.parse_response(text)
            _save(df, path)
            log(f"{label}: {len(df)} satır; null: " + ", ".join(f"{wx._short(c)}={int(df[c].isna().sum())}" for c in wx.HOURLY_VARS))
        except Exception as err:
            log(f"{label}: BAŞARISIZ {type(err).__name__}: {str(err)[:150]}")
            failed.append(label)
        time.sleep(1)
    log("indirme bitti; başarısız: " + (", ".join(failed) or "yok"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
