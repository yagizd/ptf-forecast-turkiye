"""EPİAŞ'tan aylık parçalarla tüketim geçmişi çekme (devam edebilir).

İki çağrı: gerçek zamanlı tüketim (rt-cons, sütun: consumption) ve
yük tahmin planı (load-plan, sütun: lep).
Çıktı: data/raw/rt_cons/rt_cons_YYYY-MM.csv ve data/raw/load_plan/load_plan_YYYY-MM.csv.
Kullanıcı adı ve şifre buraya yazılmaz; eptr2 paketi bunları .env dosyasından okur.
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path

from eptr2 import EPTR2
from eptr2.calls.consumption import get_load_plan, get_rt_cons

from src.data.epias import RETRY_ERRORS, ROOT, WAITS, _refuse_login, _safe
from src.data.ptf_history import PAUSE_SECONDS, _month_chunks, _rows_in_csv

RAW_ROOT = ROOT / "data" / "raw"
KINDS = {
    "rt_cons": (get_rt_cons, "consumption"),
    "load_plan": (get_load_plan, "lep"),
}
MAX_EMPTY_IN_A_ROW = 3  # bundan fazla üst üste boş ay olursa o çağrı bırakılır
MAX_SECONDS = 3600  # toplam süre sınırı


class StopRun(Exception):
    """Süre sınırı aşıldı; çalışma bilerek durduruldu."""


def _get_with_retry(call, start: date, end: date, label: str, eptr, log):
    """Tek parça için çağrı; sadece bağlantı kopmalarında en fazla 3 deneme."""
    max_tries = len(WAITS) + 1
    for attempt in range(1, max_tries + 1):
        try:
            df = call(start_date=start.isoformat(), end_date=end.isoformat(), eptr=eptr)
            log(f"{label} deneme {attempt}/{max_tries}: başarı")
            return df
        except RETRY_ERRORS as err:
            if attempt == max_tries:
                log(f"{label} deneme {attempt}/{max_tries}: koptu ({type(err).__name__}), vazgeçildi")
                raise
            wait = WAITS[attempt - 1]
            log(
                f"{label} deneme {attempt}/{max_tries}: koptu "
                f"({type(err).__name__}: {_safe(err)}), {wait} sn bekleniyor"
            )
            time.sleep(wait)


def fetch_consumption_history(
    start: str = "2021-01-01",
    end: str | None = None,
    kinds: tuple[str, ...] = ("rt_cons", "load_plan"),
    out_root: Path = RAW_ROOT,
    log_path: Path | None = None,
    max_seconds: float = MAX_SECONDS,
) -> dict:
    """start..end tüketim verisini aylık parçalarla çek. end verilmezse dün.

    - Tam inmiş ay dosyası atlanır; end'in ayı her seferinde yeniden indirilir.
    - Bağlantı kopması: 3 deneme, sonra DUR. Başka hata: hemen DUR.
    - Boş tablo: loglanır, devam edilir; MAX_EMPTY_IN_A_ROW'dan fazla üst üste olursa o çağrı bırakılır.
    - Toplam süre max_seconds'u aşarsa StopRun ile DURUR.
    Dönen değer: {"files": {tür: [yollar]}, "empty": {tür: [aylar]}, "abandoned": [türler]}.
    """
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end) if end else date.today() - timedelta(days=1)
    if end_d < start_d:
        raise ValueError(f"end ({end_d}) start'tan ({start_d}) önce olamaz.")

    out_root.mkdir(parents=True, exist_ok=True)
    log_file = log_path or (out_root / "consumption_run.log")

    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    t0 = time.monotonic()
    chunks = list(_month_chunks(start_d, end_d))
    eptr = None  # giriş sadece indirilecek parça varsa, tek sefer yapılır
    active = list(kinds)
    empty_run = {k: 0 for k in kinds}
    summary = {"files": {k: [] for k in kinds}, "empty": {k: [] for k in kinds}, "abandoned": []}
    log(f"BAŞLADI {start_d}..{end_d}, {len(chunks)} ay, çağrılar: {', '.join(kinds)}")

    for i, (c_start, c_end) in enumerate(chunks, 1):
        ym = f"{c_start.year}-{c_start.month:02d}"
        expected = ((c_end - c_start).days + 1) * 24
        is_last_month = c_end == end_d

        for kind in list(active):
            if time.monotonic() - t0 > max_seconds:
                log(f"DURDU: süre sınırı aşıldı ({max_seconds:.0f} sn), kaldığı yer: {ym} {kind}")
                raise StopRun(f"süre sınırı, {ym} {kind}")

            call, value_col = KINDS[kind]
            path = out_root / kind / f"{kind}_{ym}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            label = f"[{i}/{len(chunks)}] {kind} {ym}"

            if path.exists() and not is_last_month and _rows_in_csv(path) == expected:
                log(f"{label}: zaten var ({expected} satır), atlandı")
                summary["files"][kind].append(path)
                empty_run[kind] = 0
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
                df = _get_with_retry(call, c_start, c_end, label, eptr, log)
                if len(df) > 0 and not {"date", value_col}.issubset(df.columns):
                    raise ValueError(f"beklenen sütunlar yok: {list(df.columns)}")
            except Exception as err:
                log(f"DURDU: {ym} {kind} ({type(err).__name__}: {_safe(err)}). Önceki aylar diskte duruyor.")
                raise

            if len(df) == 0:
                empty_run[kind] += 1
                summary["empty"][kind].append(ym)
                log(f"{label}: BOŞ tablo (üst üste {empty_run[kind]})")
                if empty_run[kind] > MAX_EMPTY_IN_A_ROW:
                    active.remove(kind)
                    summary["abandoned"].append(kind)
                    log(f"{kind}: {MAX_EMPTY_IN_A_ROW} aydan fazla üst üste boş, bu çağrı BIRAKILDI")
                time.sleep(PAUSE_SECONDS)
                continue

            empty_run[kind] = 0
            if len(df) != expected:
                log(f"{label}: UYARI beklenen {expected} satır, gelen {len(df)}")

            tmp = path.with_suffix(".csv.tmp")  # yarım dosya kalmasın diye önce geçici dosyaya yaz
            df.to_csv(tmp, index=False)
            tmp.replace(path)
            summary["files"][kind].append(path)
            log(f"{label}: kaydedildi ({len(df)} satır)")
            time.sleep(PAUSE_SECONDS)

    log(f"BİTTİ, toplam {time.monotonic() - t0:.1f} sn")
    return summary
