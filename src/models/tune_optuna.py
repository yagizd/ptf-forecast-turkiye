"""Optuna ile LightGBM ayarı: SADECE doğrulama dönemi V üzerinde (test dönemi T kilitli).

Hedef: seçilen yapılandırmanın (secilen_ozellikler.json) V yürüyen pencere MAE'si (TL/MWh).
Örnekleyici TPE seed=42; depolama sqlite (devam edebilir). Önce 5 deneme koşulur ve deneme başına süre ölçülür;
n_trials = min(400, floor(45 dk / ölçülen süre)).
Benimseme kuralı (öncesinden sabit): en iyi V-MAE varsayılandan en az %1,5 (göreli) iyi, iyileşme V'nin iki yarısında
(blok 1-7, blok 8-14) ayrı ayrı da var, VE random_state=43 ile tekrarda iyileşme en az %1,0. Üçü de tutmazsa varsayılan KALIR.

Çalıştırma: python -m src.models.tune_optuna
Kural: eksik değer doldurulmaz.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import ptf_features as pf
from src.features import ptf_features_v2 as v2
from src.models import baselines as bl
from src.models import lgbm_walkforward as lw
from src.models import validation_experiments as ve

ROOT = ve.ROOT
OUT_DIR = ve.OUT_DIR
DB_URL = "sqlite:///" + (OUT_DIR / "optuna_v.db").as_posix()
PARAMS_PATH = OUT_DIR / "secilen_parametreler.json"
REPORT_PATH = ROOT / "reports" / "optuna_dogrulama.md"
LOG_PATH = OUT_DIR / "optuna_ilerleme.log"

STUDY = "ptf_v_tpe42"
BUDGET_S = 45 * 60
MEASURE_TRIALS = 5
MAX_TRIALS = 400
MIN_REL_GAIN = 0.015
MIN_REL_GAIN_SEED43 = 0.010


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fixed_params(random_state: int = 42) -> dict:
    return dict(objective="regression_l1", subsample_freq=1, deterministic=True, force_row_wise=True,
                random_state=random_state, verbosity=-1)


def suggest(trial) -> dict:
    return {
        **fixed_params(),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
    }


class Evaluator:
    def __init__(self):
        self.df = ve.load_locked()  # T kilidi
        sec = json.loads(ve.SEC_PATH.read_text(encoding="utf-8"))
        self.groups = tuple(sec["gruplar"])
        self.norm = bool(sec["kapak_normalizasyonu_K"])
        self.sec = sec
        self.feat = v2.build_features_v2(self.df, groups=self.groups, normalize=self.norm)

    def run(self, params: dict) -> pd.DataFrame:
        pred, _, _ = ve.walkforward_config(self.df, ve.V_START, ve.V_END, groups=self.groups,
                                           normalize=self.norm, params=params, feat=self.feat)
        return pred[pred["price"].notna()]

    @staticmethod
    def mae(p: pd.DataFrame, mask=None) -> float:
        q = p if mask is None else p[mask]
        return float((q["tahmin"] - q["price"]).abs().mean())

    def halves(self, p: pd.DataFrame):
        return self.mae(p, p["blok_no"] <= 7), self.mae(p, p["blok_no"] >= 8)


def default_params(random_state: int = 42) -> dict:
    d = dict(lw.LGBM_PARAMS)
    d["random_state"] = random_state
    return d


def main() -> int:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev = Evaluator()
    t0 = time.time()
    log(f"başladı; yapılandırma {ev.sec['secilen_yapilandirma']} gruplar={list(ev.groups)} K={ev.norm}")

    p_def = ev.run(default_params(42))
    mae_def = ev.mae(p_def)
    log(f"varsayılan V-MAE {mae_def:.3f} (n={len(p_def)}) süre {time.time() - t0:.1f}s")

    def objective(trial):
        return ev.mae(ev.run(suggest(trial)))

    study = optuna.create_study(study_name=STUDY, storage=DB_URL, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    t1 = time.time()
    if done < MEASURE_TRIALS:
        study.optimize(objective, n_trials=MEASURE_TRIALS - done)
    done_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    secs = [(t.datetime_complete - t.datetime_start).total_seconds() for t in done_trials[:MEASURE_TRIALS]]
    per_trial = float(np.mean(secs))
    n_trials = min(MAX_TRIALS, math.floor(BUDGET_S / per_trial))
    log(f"deneme başına ölçülen süre {per_trial:.1f}s (ilk {MEASURE_TRIALS}); n_trials = {n_trials}")
    remaining = n_trials - len(done_trials)
    if remaining > 0:
        deadline = BUDGET_S - (time.time() - t1) + 0  # ölçüm süresi de bütçeden sayılır
        study.optimize(objective, n_trials=remaining, timeout=max(60.0, deadline),
                       callbacks=[lambda s, t: log(f"deneme {t.number} MAE {t.value:.3f} en iyi {s.best_value:.3f}")])
    trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    best = study.best_trial
    best_params = {**fixed_params(), **best.params}
    log(f"bitti: {len(trials)} deneme, en iyi {best.value:.3f} (varsayılan {mae_def:.3f})")

    # benimseme kuralı
    p_best = ev.run(best_params)
    mae_best = ev.mae(p_best)
    rel = (mae_def - mae_best) / mae_def
    h1d, h2d = ev.halves(p_def)
    h1b, h2b = ev.halves(p_best)
    rel1, rel2 = (h1d - h1b) / h1d, (h2d - h2b) / h2d
    p_def43 = ev.run(default_params(43))
    best43 = dict(best_params)
    best43["random_state"] = 43
    p_best43 = ev.run(best43)
    mae_def43, mae_best43 = ev.mae(p_def43), ev.mae(p_best43)
    rel43 = (mae_def43 - mae_best43) / mae_def43
    c1, c2, c3 = rel >= MIN_REL_GAIN, (rel1 > 0 and rel2 > 0), rel43 >= MIN_REL_GAIN_SEED43
    adopt = bool(c1 and c2 and c3)
    chosen = best_params if adopt else default_params(42)
    gerekce = (f"göreli kazanç {rel * 100:+.2f}% (eşik %1,5: {'tuttu' if c1 else 'TUTMADI'}); "
               f"yarılar {rel1 * 100:+.2f}% / {rel2 * 100:+.2f}% (ikisi de >0: {'tuttu' if c2 else 'TUTMADI'}); "
               f"seed43 kazanç {rel43 * 100:+.2f}% (eşik %1,0: {'tuttu' if c3 else 'TUTMADI'})")
    PARAMS_PATH.write_text(json.dumps({
        "benimsendi": adopt, "gerekce": gerekce, "parametreler": chosen, "en_iyi_deneme_parametreleri": best_params,
        "varsayilan_V_MAE": mae_def, "en_iyi_V_MAE": mae_best, "claude_incelemedi": True}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    top = sorted(trials, key=lambda t: t.value)[:10]
    L = ["# Optuna doğrulama ayarı (V üzerinde)", "",
         "**Claude (sohbet) bu kodu ve sonucu İNCELEMEDİ.** Test dönemi T'ye dokunulmadı.", "",
         f"- Yapılandırma: {ev.sec['secilen_yapilandirma']} (gruplar {list(ev.groups)}, K={ev.norm}); TPE seed=42; sqlite: optuna_v.db",
         f"- Ölçülen deneme başına süre (ilk 5): {per_trial:.1f} sn; n_trials = min(400, floor(2700/süre)) = {n_trials}; "
         f"tamamlanan: {len(trials)}.",
         f"- Varsayılan V-MAE: {mae_def:.3f}; Optuna en iyi V-MAE: {mae_best:.3f} (kazanç {rel * 100:+.2f}%).", "",
         "## Benimseme kuralı", "",
         bl._md_table(["ölçüt", "varsayılan", "en iyi", "kazanç", "eşik", "sonuç"], [
             ("toplam V-MAE", f"{mae_def:.3f}", f"{mae_best:.3f}", f"{rel * 100:+.2f}%", "%1,5", "tuttu" if c1 else "TUTMADI"),
             ("blok 1-7", f"{h1d:.3f}", f"{h1b:.3f}", f"{rel1 * 100:+.2f}%", ">0", "tuttu" if rel1 > 0 else "TUTMADI"),
             ("blok 8-14", f"{h2d:.3f}", f"{h2b:.3f}", f"{rel2 * 100:+.2f}%", ">0", "tuttu" if rel2 > 0 else "TUTMADI"),
             ("random_state=43", f"{mae_def43:.3f}", f"{mae_best43:.3f}", f"{rel43 * 100:+.2f}%", "%1,0", "tuttu" if c3 else "TUTMADI")]), "",
         f"**Karar: {'BENİMSENDİ' if adopt else 'BENİMSENMEDİ, varsayılan parametreler kalır'}.**", "",
         "## En iyi 10 deneme", "",
         bl._md_table(["deneme", "V-MAE", "süre (sn)", "parametreler"], [
             (t.number, f"{t.value:.3f}", f"{(t.datetime_complete - t.datetime_start).total_seconds():.1f}",
              json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.params.items()}))
             for t in top]), "",
         "## İlk 5 deneme süreleri (sn)", "", ", ".join(f"{s:.1f}" for s in secs), ""]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    log("rapor yazıldı: " + gerekce)
    return 0


if __name__ == "__main__":
    sys.exit(main())
