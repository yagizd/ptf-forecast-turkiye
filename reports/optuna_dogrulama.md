# Optuna doğrulama ayarı (V üzerinde)

**Claude (sohbet) bu kodu ve sonucu İNCELEMEDİ.** Test dönemi T'ye dokunulmadı.

- Yapılandırma: R1 (gruplar ['H'], K=False); TPE seed=42; sqlite: optuna_v.db
- Ölçülen deneme başına süre (ilk 5): 8.4 sn; n_trials = min(400, floor(2700/süre)) = 321; tamamlanan: 141.
- Varsayılan V-MAE: 271.200; Optuna en iyi V-MAE: 267.561 (kazanç +1.34%).

## Benimseme kuralı

| ölçüt | varsayılan | en iyi | kazanç | eşik | sonuç |
|---|---|---|---|---|---|
| toplam V-MAE | 271.200 | 267.561 | +1.34% | %1,5 | TUTMADI |
| blok 1-7 | 261.840 | 257.975 | +1.48% | >0 | tuttu |
| blok 8-14 | 282.055 | 278.679 | +1.20% | >0 | tuttu |
| random_state=43 | 270.380 | 268.610 | +0.65% | %1,0 | TUTMADI |

**Karar: BENİMSENMEDİ, varsayılan parametreler kalır.**

## En iyi 10 deneme

| deneme | V-MAE | süre (sn) | parametreler |
|---|---|---|---|
| 111 | 267.561 | 22.4 | {"num_leaves": 62, "learning_rate": 0.0354, "n_estimators": 1327, "min_child_samples": 64, "subsample": 0.8863, "colsample_bytree": 0.645, "reg_lambda": 6.1753, "reg_alpha": 1.1135} |
| 47 | 267.638 | 22.1 | {"num_leaves": 62, "learning_rate": 0.0234, "n_estimators": 1293, "min_child_samples": 67, "subsample": 0.9325, "colsample_bytree": 0.6562, "reg_lambda": 5.9121, "reg_alpha": 0.4332} |
| 126 | 267.724 | 24.6 | {"num_leaves": 62, "learning_rate": 0.0372, "n_estimators": 1420, "min_child_samples": 54, "subsample": 0.9219, "colsample_bytree": 0.6039, "reg_lambda": 6.1635, "reg_alpha": 1.5627} |
| 48 | 267.743 | 21.2 | {"num_leaves": 67, "learning_rate": 0.0348, "n_estimators": 1181, "min_child_samples": 56, "subsample": 0.9227, "colsample_bytree": 0.666, "reg_lambda": 5.4412, "reg_alpha": 0.7117} |
| 56 | 267.765 | 21.2 | {"num_leaves": 58, "learning_rate": 0.0233, "n_estimators": 1281, "min_child_samples": 73, "subsample": 0.9304, "colsample_bytree": 0.6112, "reg_lambda": 3.5158, "reg_alpha": 2.0025} |
| 42 | 267.795 | 23.0 | {"num_leaves": 53, "learning_rate": 0.0305, "n_estimators": 1474, "min_child_samples": 57, "subsample": 0.8986, "colsample_bytree": 0.6147, "reg_lambda": 5.8535, "reg_alpha": 1.6859} |
| 52 | 267.857 | 20.2 | {"num_leaves": 56, "learning_rate": 0.0335, "n_estimators": 1261, "min_child_samples": 58, "subsample": 0.8978, "colsample_bytree": 0.6522, "reg_lambda": 5.5743, "reg_alpha": 1.1411} |
| 139 | 267.868 | 21.1 | {"num_leaves": 56, "learning_rate": 0.0316, "n_estimators": 1298, "min_child_samples": 50, "subsample": 0.9037, "colsample_bytree": 0.6264, "reg_lambda": 8.0183, "reg_alpha": 0.6741} |
| 100 | 267.975 | 21.1 | {"num_leaves": 56, "learning_rate": 0.0319, "n_estimators": 1290, "min_child_samples": 55, "subsample": 0.9006, "colsample_bytree": 0.6326, "reg_lambda": 6.9597, "reg_alpha": 0.8822} |
| 130 | 267.993 | 20.6 | {"num_leaves": 57, "learning_rate": 0.0275, "n_estimators": 1269, "min_child_samples": 69, "subsample": 0.9224, "colsample_bytree": 0.5899, "reg_lambda": 2.3167, "reg_alpha": 1.7995} |

## İlk 5 deneme süreleri (sn)

11.9, 1.7, 10.1, 5.9, 12.4

