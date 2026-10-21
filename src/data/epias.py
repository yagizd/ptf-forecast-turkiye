"""EPİAŞ'tan saatlik PTF (Piyasa Takas Fiyatı) çekme.

Kullanıcı adı ve şifre buraya yazılmaz; eptr2 paketi bunları .env dosyasından okur.
"""

import os
import time
from pathlib import Path

import pandas as pd
from eptr2 import EPTR2
from eptr2.calls.dam import get_mcp
from urllib3.exceptions import MaxRetryError, ProtocolError

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# Sadece bağlantı kopmalarında tekrar denenir; başka hatalarda hemen durulur.
# eptr2 istekleri urllib3 ile atar; kopma ProtocolError, bağlantı reddi MaxRetryError olarak çıkar.
# Bunlar yerleşik ConnectionError'ın alt sınıfı DEĞİLDİR, o yüzden ayrıca yazıldı.
RETRY_ERRORS = (ProtocolError, MaxRetryError, ConnectionError)
WAITS = (5, 15)  # 1. ve 2. denemeden sonra beklenecek saniye (en fazla 3 deneme)


def _safe(err: Exception) -> str:
    """Hata mesajını kısalt; kullanıcı adı/şifre geçiyorsa sansürle."""
    msg = str(err)[:200]
    for key in ("EPTR_USERNAME", "EPTR_PASSWORD"):
        secret = os.environ.get(key)
        if secret:
            msg = msg.replace(secret, "***")
    return msg


def _refuse_login(**kwargs):
    raise RuntimeError("Bilet doldu; veri isteği sırasında giriş denemesi yapılmaz.")


def fetch_ptf_day(day: str) -> pd.DataFrame:
    """Verilen günün (YYYY-MM-DD) saatlik PTF'sini çek, data/raw/ içine CSV olarak kaydet."""
    # Giriş (şifre isteği) burada, tek sefer ve tekrar denemesiz yapılır.
    # Hata olursa istisna dışarı çıkar ve durur.
    eptr = EPTR2(
        use_dotenv=True,
        recycle_tgt=True,
        dotenv_path=str(ROOT / ".env"),
        tgt_path=str(ROOT),
    )
    # Veri istekleri tekrar denenirken şifre bir daha gönderilmesin.
    eptr.get_tgt = _refuse_login

    max_tries = len(WAITS) + 1
    for attempt in range(1, max_tries + 1):
        try:
            df = get_mcp(start_date=day, end_date=day, eptr=eptr)
            print(f"deneme {attempt}/{max_tries}: başarı")
            break
        except RETRY_ERRORS as err:
            if attempt == max_tries:
                print(f"deneme {attempt}/{max_tries}: koptu ({type(err).__name__}), vazgeçildi")
                raise
            wait = WAITS[attempt - 1]
            print(
                f"deneme {attempt}/{max_tries}: koptu ({type(err).__name__}: {_safe(err)}), "
                f"{wait} sn bekleniyor"
            )
            time.sleep(wait)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"ptf_{day}.csv", index=False)
    return df
