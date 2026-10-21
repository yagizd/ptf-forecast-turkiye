"""EPİAŞ'tan saatlik PTF (Piyasa Takas Fiyatı) çekme.

Kullanıcı adı ve şifre buraya yazılmaz; eptr2 paketi bunları .env dosyasından okur.
"""

from pathlib import Path

import pandas as pd
from eptr2 import EPTR2
from eptr2.calls.dam import get_mcp

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def fetch_ptf_day(day: str) -> pd.DataFrame:
    """Verilen günün (YYYY-MM-DD) saatlik PTF'sini çek, data/raw/ içine CSV olarak kaydet."""
    eptr = EPTR2(
        use_dotenv=True,
        recycle_tgt=True,
        dotenv_path=str(ROOT / ".env"),
        tgt_path=str(ROOT),
    )
    df = get_mcp(start_date=day, end_date=day, eptr=eptr)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"ptf_{day}.csv", index=False)
    return df
