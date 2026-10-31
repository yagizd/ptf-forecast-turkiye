"""Optuna en iyi 5 deneme tablosu (SALT-OKUNUR). Model eğitimi yok.

Orijinal veritabanı (data/processed/egitim_oturumu/optuna_v.db) AÇILMAZ: önce data/processed/aralik_oturumu_31Ekim/ altına
KOPYALANIR (kopya zaten varsa o kullanılır), yalnızca kopya açılır. Çalışma (study) adları get_all_study_summaries ile bulunur.
Referans değerler (varsayılan V-MAE, seed 43, karar) reports/optuna_dogrulama.md dosyasından OKUNUR (elle yazılmaz).
Çıktı: reports/optuna_en_iyi5.md
Çalıştırma (ana .venv, optuna 5.0.0): PYTHONUTF8=1 python -m src.models.optuna_tablo
"""

import re
import shutil
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]
ORIG_DB = ROOT / "data" / "processed" / "egitim_oturumu" / "optuna_v.db"
COPY_DB = ROOT / "data" / "processed" / "aralik_oturumu_31Ekim" / "optuna_v_kopya.db"
REF_REPORT = ROOT / "reports" / "optuna_dogrulama.md"
OUT_PATH = ROOT / "reports" / "optuna_en_iyi5.md"
SENTENCE = "Optuna benimsenmedi (en iyi iyileşme eşiğin altında); tablo bilgilendirme amaçlı."


def md(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def read_reference(text: str) -> dict:
    """optuna_dogrulama.md'den okunur; bulunamayan alan None kalır (uydurma yok)."""
    ref = {"varsayilan": None, "en_iyi": None, "seed43_satiri": None, "toplam_satiri": None, "karar_benimsenmedi": False}
    m = re.search(r"Varsayılan V-MAE:\s*([\d.]+);\s*Optuna en iyi V-MAE:\s*([\d.]+)", text)
    if m:
        ref["varsayilan"], ref["en_iyi"] = float(m.group(1)), float(m.group(2))
    for line in text.splitlines():
        if line.startswith("| random_state=43"):
            ref["seed43_satiri"] = line
        if line.startswith("| toplam V-MAE"):
            ref["toplam_satiri"] = line
    ref["karar_benimsenmedi"] = "BENİMSENMEDİ" in text
    return ref


def main() -> int:
    COPY_DB.parent.mkdir(parents=True, exist_ok=True)
    if not COPY_DB.exists():
        shutil.copy2(ORIG_DB, COPY_DB)
    storage = "sqlite:///" + COPY_DB.as_posix()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    summaries = optuna.get_all_study_summaries(storage)
    ref = read_reference(REF_REPORT.read_text(encoding="utf-8"))
    # cümle yalnızca raporun kendisi benimsenmediğini söylüyorsa VE toplam kazanç satırı eşiği tutmadıysa yazılır
    esik_altinda = bool(ref["toplam_satiri"] and ref["toplam_satiri"].rstrip().endswith("TUTMADI |"))
    L = []
    if ref["karar_benimsenmedi"] and esik_altinda:
        L += [SENTENCE, ""]
    L += ["# Optuna: en iyi 5 deneme (V üzerinde)", "",
          "**Kodu ve ilk 5 denemeyi Claude (sohbet) 31 Ekim'de inceledi: veritabanı kopyasından ilk 5 V-MAE değeri bağımsız okundu, aynı.** Salt-okunur; yalnızca veritabanı KOPYASI açıldı.", "",
          f"- Kaynak: {COPY_DB.relative_to(ROOT).as_posix()} (kopya; orijinal {ORIG_DB.relative_to(ROOT).as_posix()} açılmadı).",
          f"- Çalışmalar (study): " + ", ".join(f"{s.study_name} ({s.n_trials} deneme)" for s in summaries), "",
          "## Referans (reports/optuna_dogrulama.md'den okundu)", ""]
    if ref["varsayilan"] is not None:
        L.append(f"- Varsayılan parametrelerle V-MAE: {ref['varsayilan']:.3f}; Optuna en iyi: {ref['en_iyi']:.3f}.")
    else:
        L.append("- Varsayılan V-MAE: BİLMİYORUM (rapordan okunamadı).")
    L.append(f"- Toplam V-MAE kuralı satırı: {ref['toplam_satiri'] or 'BİLMİYORUM'}")
    L.append(f"- Seed 43 satırı: {ref['seed43_satiri'] or 'BİLMİYORUM'}")
    L.append(f"- Rapordaki karar: {'BENİMSENMEDİ' if ref['karar_benimsenmedi'] else 'BENİMSENMEDİ ifadesi raporda YOK'}")
    L.append("")
    for s in summaries:
        study = optuna.load_study(study_name=s.study_name, storage=storage)
        done = [t for t in study.trials if t.state.name == "COMPLETE"]
        best = sorted(done, key=lambda t: t.value)[:5]
        params = sorted({k for t in best for k in t.params})
        rows = []
        for t in best:
            sure = (t.datetime_complete - t.datetime_start).total_seconds()
            rows.append((t.number, f"{t.value:.3f}", f"{sure:.1f}", *[(f"{t.params[k]:.4f}" if isinstance(t.params[k], float) else t.params[k]) for k in params]))
        L += [f"## {s.study_name}: tamamlanan {len(done)} / toplam {len(study.trials)} deneme", "",
              md(["deneme", "V-MAE (TL/MWh)", "süre (sn)", *params], rows), ""]
    OUT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("yazıldı", OUT_PATH.name, "| çalışmalar:", ", ".join(s.study_name for s in summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
