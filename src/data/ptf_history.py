"""EPİAŞ'tan aylık parçalarla PTF geçmişi çekme (devam edebilir).

Her ay data/raw/ptf/ptf_YYYY-MM.csv olarak kaydedilir. Tam inmiş aylar tekrar indirilmez.
Kullanıcı adı ve şifre buraya yazılmaz; eptr2 paketi bunları .env dosyasından okur.
"""

import calendar
import time
from datetime import date, timedelta
from pathlib import Path

from eptr2 import EPTR2
from eptr2.calls.dam import get_mcp

from src.data.epias import RETRY_ERRORS, ROOT, WAITS, _refuse_login, _safe

PTF_DIR = ROOT / "data" / "raw" / "ptf"
PAUSE_SECONDS = 1.0  # parçalar arası nezaket beklemesi


def _month_chunks(start: date, end: date):
    """start..end aralığını aylık (ilk, son gün) parçalara böl."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        chunk_start = max(cur, start)
        chunk_end = min(date(cur.year, cur.month, last_day), end)
        yield chunk_start, chunk_end
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)


def _rows_in_csv(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)  # başlık satırı hariç


def _get_with_retry(eptr, start: date, end: date, label: str):
    """Tek bir parça için get_mcp; sadece bağlantı kopmalarında en fazla 3 deneme."""
    max_tries = len(WAITS) + 1
    for attempt in range(1, max_tries + 1):
        try:
            df = get_mcp(start_date=start.isoformat(), end_date=end.isoformat(), eptr=eptr)
            print(f"{label} deneme {attempt}/{max_tries}: başarı")
            return df
        except RETRY_ERRORS as err:
            if attempt == max_tries:
                print(f"{label} deneme {attempt}/{max_tries}: koptu ({type(err).__name__}), vazgeçildi")
                raise
            wait = WAITS[attempt - 1]
            print(
                f"{label} deneme {attempt}/{max_tries}: koptu "
                f"({type(err).__name__}: {_safe(err)}), {wait} sn bekleniyor"
            )
            time.sleep(wait)


def fetch_ptf_history(
    start: str = "2021-01-01",
    end: str | None = None,
    out_dir: Path = PTF_DIR,
) -> list[Path]:
    """start..end (YYYY-MM-DD) PTF'sini aylık parçalarla çek. end verilmezse dün.

    - Tam inmiş ay dosyası varsa atlanır (devam edilebilir).
    - end'in düştüğü ay (bugünkü kısmi ay) her çalıştırmada yeniden indirilir.
    - Bir parça tekrar denemelerden sonra da başarısız olursa DURUR; önceki aylar diskte kalır.
    Dönen değer: diskte hazır olan ay dosyalarının listesi.
    """
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end) if end else date.today() - timedelta(days=1)
    if end_d < start_d:
        raise ValueError(f"end ({end_d}) start'tan ({start_d}) önce olamaz.")

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(_month_chunks(start_d, end_d))
    eptr = None  # giriş sadece indirilecek parça varsa, tek sefer yapılır
    done: list[Path] = []

    for i, (c_start, c_end) in enumerate(chunks, 1):
        ym = f"{c_start.year}-{c_start.month:02d}"
        path = out_dir / f"ptf_{ym}.csv"
        expected = ((c_end - c_start).days + 1) * 24
        is_last_month = (c_end == end_d)
        label = f"[{i}/{len(chunks)}] {ym}"

        if path.exists() and not is_last_month and _rows_in_csv(path) == expected:
            print(f"{label}: zaten var ({expected} satır), atlandı")
            done.append(path)
            continue

        if eptr is None:
            # Giriş (şifre isteği): tek sefer, tekrar denemesiz. Hata olursa istisna çıkar ve durur.
            eptr = EPTR2(
                use_dotenv=True,
                recycle_tgt=True,
                dotenv_path=str(ROOT / ".env"),
                tgt_path=str(ROOT),
            )
            # Veri istekleri tekrar denenirken şifre bir daha gönderilmesin.
            eptr.get_tgt = _refuse_login

        try:
            df = _get_with_retry(eptr, c_start, c_end, label)
        except Exception as err:
            print(f"DURDU: {ym} ayında ({type(err).__name__}). Önceki aylar diskte duruyor.")
            raise

        if len(df) == 0:
            raise ValueError(f"{ym}: EPİAŞ boş tablo döndürdü.")
        if len(df) != expected:
            print(f"{label}: UYARI beklenen {expected} satır, gelen {len(df)}")

        tmp = path.with_suffix(".csv.tmp")  # yarım dosya kalmasın diye önce geçici dosyaya yaz
        df.to_csv(tmp, index=False)
        tmp.replace(path)
        done.append(path)
        time.sleep(PAUSE_SECONDS)

    return done
