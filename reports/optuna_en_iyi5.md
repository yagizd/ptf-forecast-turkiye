Optuna benimsenmedi (en iyi iyileşme eşiğin altında); tablo bilgilendirme amaçlı.

# Optuna: en iyi 5 deneme (V üzerinde)

**Kodu ve ilk 5 denemeyi Claude (sohbet) 31 Ekim'de inceledi: veritabanı kopyasından ilk 5 V-MAE değeri bağımsız okundu, aynı.** Salt-okunur; yalnızca veritabanı KOPYASI açıldı.

- Kaynak: data/processed/aralik_oturumu_31Ekim/optuna_v_kopya.db (kopya; orijinal data/processed/egitim_oturumu/optuna_v.db açılmadı).
- Çalışmalar (study): ptf_v_tpe42 (141 deneme)

## Referans (reports/optuna_dogrulama.md'den okundu)

- Varsayılan parametrelerle V-MAE: 271.200; Optuna en iyi: 267.561.
- Toplam V-MAE kuralı satırı: | toplam V-MAE | 271.200 | 267.561 | +1.34% | %1,5 | TUTMADI |
- Seed 43 satırı: | random_state=43 | 270.380 | 268.610 | +0.65% | %1,0 | TUTMADI |
- Rapordaki karar: BENİMSENMEDİ

## ptf_v_tpe42: tamamlanan 141 / toplam 141 deneme

| deneme | V-MAE (TL/MWh) | süre (sn) | colsample_bytree | learning_rate | min_child_samples | n_estimators | num_leaves | reg_alpha | reg_lambda | subsample |
|---|---|---|---|---|---|---|---|---|---|---|
| 111 | 267.561 | 22.4 | 0.6450 | 0.0354 | 64 | 1327 | 62 | 1.1135 | 6.1753 | 0.8863 |
| 47 | 267.638 | 22.1 | 0.6562 | 0.0234 | 67 | 1293 | 62 | 0.4332 | 5.9121 | 0.9325 |
| 126 | 267.724 | 24.6 | 0.6039 | 0.0372 | 54 | 1420 | 62 | 1.5627 | 6.1635 | 0.9219 |
| 48 | 267.743 | 21.2 | 0.6660 | 0.0348 | 56 | 1181 | 67 | 0.7117 | 5.4412 | 0.9227 |
| 56 | 267.765 | 21.2 | 0.6112 | 0.0233 | 73 | 1281 | 58 | 2.0025 | 3.5158 | 0.9304 |

