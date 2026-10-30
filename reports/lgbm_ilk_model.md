# LightGBM ilk deneme (yürüyen pencere)

**Bu bir İLK DENEME'dir. Parametreler sabit bir tahmindir, AYAR DEĞİLDİR; bu sonuçlara bakıp parametre değiştirilmemelidir.** Ayar (Optuna) ayrı bir doğrulama setiyle yapılacak. Erken durdurma ve test verisiyle ayar yok.

## Protokol

- Eğitim başlangıcı: 2024-03-08 00:00:00+03:00 (hava verisi bundan sonra tam dolu).
- Test dönemi: 2025-10-21 .. 2026-10-20; 14 blok x 28 gün (son blok kısa olabilir).
- Her blokta eğitim: hedef zamanı blok başlangıcından ÖNCE olan tüm satırlar (genişleyen pencere); hedefi boş olanlar atılır, özellik NaN'ları kalır.
- Özellikler: price_lag_d1/d2/d7, price_d1_mean, hour, dayofweek, month, is_weekend (+ havalı modelde 5 hava sütunu, previous_day2). consumption, lep ve *_pd1 sütunları KULLANILMADI.
- Parametreler: {'objective': 'regression_l1', 'learning_rate': 0.05, 'n_estimators': 400, 'num_leaves': 31, 'min_child_samples': 50, 'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'random_state': 42, 'deterministic': True, 'force_row_wise': True, 'verbosity': -1}

- Ölçüm satır sayısı: **8760** (kıyaslar, havasız ve havalı tahminlerin hepsinin ve gerçeğin dolu olduğu satırlar). baselines.common_rows ile aynı küme mi: **evet** (8760 / 8760).
- Test döneminde en az bir özelliği NaN olan satır: temel özellikler 0, hava özellikleri 129 (toplam 8760). NaN'lar doldurulmadı; LightGBM kendisi işledi.

## Tutarlılık kontrolü (reports/baselines.md ile)

| model | baselines.md'deki MAE | bu çalışmadaki MAE | sonuç |
|---|---|---|---|
| D-1 | 519.2 | 519.2 | UYUŞUYOR |
| Kural | 497.5 | 497.5 | UYUŞUYOR |

## Toplam

| model | MAE (TL/MWh) | rMAE (D-1'e göre) |
|---|---|---|
| D-1 | 519.2 | 1.000 |
| D-7 | 598.6 | 1.153 |
| Kural | 497.5 | 0.958 |
| LightGBM havasız | 433.5 | 0.835 |
| LightGBM havalı | 419.3 | 0.807 |

## Hafta içi / hafta sonu

| grup | n | MAE D-1 | MAE D-7 | MAE Kural | MAE havasız | MAE havalı | rMAE D-7 | rMAE Kural | rMAE havasız | rMAE havalı |
|---|---|---|---|---|---|---|---|---|---|---|
| Hafta içi | 6264 | 513.4 | 606.0 | 464.7 | 435.5 | 421.0 | 1.180 | 0.905 | 0.848 | 0.820 |
| Hafta sonu | 2496 | 533.9 | 579.9 | 579.9 | 428.7 | 415.0 | 1.086 | 1.086 | 0.803 | 0.777 |

## Özel saatler (fiyat 0 / ayın tavanında / diğer)

'Tavanda' = fiyat, o ayın en yüksek fiyatına eşit ve o ayda bu fiyatta en az 5 saat var (baselines.label_special). Resmî tavan değil; sadece değerlendirme etiketi.

| saat türü | n | MAE D-1 | MAE D-7 | MAE Kural | MAE havasız | MAE havalı | rMAE D-7 | rMAE Kural | rMAE havasız | rMAE havalı |
|---|---|---|---|---|---|---|---|---|---|---|
| Diğer | 7503 | 570.2 | 663.7 | 554.4 | 471.6 | 457.4 | 1.164 | 0.972 | 0.827 | 0.802 |
| Fiyat 0 | 409 | 191.6 | 238.1 | 142.0 | 226.5 | 239.7 | 1.243 | 0.741 | 1.182 | 1.251 |
| Tavanda | 848 | 226.8 | 196.6 | 165.6 | 196.3 | 168.9 | 0.867 | 0.731 | 0.866 | 0.745 |

## Blok bazında

| blok | tarih aralığı | n | MAE D-1 | MAE D-7 | MAE Kural | MAE havasız | MAE havalı | rMAE D-7 | rMAE Kural | rMAE havasız | rMAE havalı |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2025-10-21 .. 2025-11-17 | 672 | 408.7 | 390.5 | 356.3 | 295.9 | 274.9 | 0.955 | 0.872 | 0.724 | 0.673 |
| 2 | 2025-11-18 .. 2025-12-15 | 672 | 340.4 | 453.4 | 387.1 | 292.1 | 259.1 | 1.332 | 1.137 | 0.858 | 0.761 |
| 3 | 2025-12-16 .. 2026-01-12 | 672 | 384.2 | 352.1 | 339.3 | 293.8 | 273.5 | 0.916 | 0.883 | 0.765 | 0.712 |
| 4 | 2026-01-13 .. 2026-02-09 | 672 | 451.3 | 393.3 | 388.3 | 328.7 | 292.1 | 0.872 | 0.860 | 0.728 | 0.647 |
| 5 | 2026-02-10 .. 2026-03-09 | 672 | 756.4 | 807.2 | 755.3 | 606.9 | 587.1 | 1.067 | 0.999 | 0.802 | 0.776 |
| 6 | 2026-03-10 .. 2026-04-06 | 672 | 640.1 | 886.5 | 757.0 | 612.2 | 600.0 | 1.385 | 1.183 | 0.956 | 0.937 |
| 7 | 2026-04-07 .. 2026-05-04 | 672 | 504.8 | 664.6 | 540.5 | 546.4 | 525.3 | 1.317 | 1.071 | 1.082 | 1.040 |
| 8 | 2026-05-05 .. 2026-06-01 | 672 | 339.8 | 489.4 | 348.4 | 366.8 | 436.7 | 1.440 | 1.025 | 1.080 | 1.285 |
| 9 | 2026-06-02 .. 2026-06-29 | 672 | 539.0 | 734.4 | 514.6 | 479.0 | 487.1 | 1.363 | 0.955 | 0.889 | 0.904 |
| 10 | 2026-06-30 .. 2026-07-27 | 672 | 620.1 | 658.5 | 511.6 | 491.7 | 435.0 | 1.062 | 0.825 | 0.793 | 0.701 |
| 11 | 2026-07-28 .. 2026-08-24 | 672 | 551.3 | 717.2 | 483.5 | 429.4 | 438.8 | 1.301 | 0.877 | 0.779 | 0.796 |
| 12 | 2026-08-25 .. 2026-09-21 | 672 | 581.1 | 523.1 | 487.3 | 423.1 | 405.8 | 0.900 | 0.839 | 0.728 | 0.698 |
| 13 | 2026-09-22 .. 2026-10-19 | 672 | 641.8 | 714.9 | 606.2 | 477.7 | 443.6 | 1.114 | 0.945 | 0.744 | 0.691 |
| 14 | 2026-10-20 .. 2026-10-20 | 24 | 274.6 | 501.7 | 274.6 | 214.7 | 191.0 | 1.827 | 1.000 | 0.782 | 0.696 |

## Aylık

| ay | n | MAE D-1 | MAE D-7 | MAE Kural | MAE havasız | MAE havalı | rMAE D-7 | rMAE Kural | rMAE havasız | rMAE havalı |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10 | 264 | 507.8 | 458.2 | 435.1 | 391.1 | 354.0 | 0.902 | 0.857 | 0.770 | 0.697 |
| 2025-11 | 720 | 362.4 | 459.7 | 396.5 | 291.4 | 263.5 | 1.269 | 1.094 | 0.804 | 0.727 |
| 2025-12 | 744 | 298.6 | 317.1 | 278.8 | 231.2 | 213.4 | 1.062 | 0.934 | 0.774 | 0.715 |
| 2026-01 | 744 | 403.1 | 358.2 | 352.9 | 302.8 | 276.0 | 0.889 | 0.876 | 0.751 | 0.685 |
| 2026-02 | 672 | 749.3 | 808.0 | 773.1 | 602.3 | 568.2 | 1.078 | 1.032 | 0.804 | 0.758 |
| 2026-03 | 744 | 667.6 | 844.3 | 731.9 | 578.2 | 546.1 | 1.265 | 1.096 | 0.866 | 0.818 |
| 2026-04 | 720 | 536.4 | 638.2 | 518.8 | 564.0 | 573.4 | 1.190 | 0.967 | 1.052 | 1.069 |
| 2026-05 | 744 | 341.4 | 497.6 | 360.0 | 373.4 | 423.3 | 1.458 | 1.055 | 1.094 | 1.240 |
| 2026-06 | 720 | 539.2 | 742.8 | 534.4 | 486.0 | 496.7 | 1.378 | 0.991 | 0.901 | 0.921 |
| 2026-07 | 744 | 609.5 | 728.8 | 511.5 | 511.4 | 452.1 | 1.196 | 0.839 | 0.839 | 0.742 |
| 2026-08 | 744 | 566.6 | 617.7 | 499.9 | 408.5 | 420.5 | 1.090 | 0.882 | 0.721 | 0.742 |
| 2026-09 | 720 | 609.6 | 526.7 | 498.6 | 427.3 | 401.7 | 0.864 | 0.818 | 0.701 | 0.659 |
| 2026-10 | 480 | 602.1 | 776.1 | 596.1 | 472.0 | 445.1 | 1.289 | 0.990 | 0.784 | 0.739 |

## Özellik önemi (gain, bloklar üzerinde ortalama pay)

### LightGBM havasiz

| özellik | ortalama gain payı (%) |
|---|---|
| price_lag_d1 | 40.6 |
| price_lag_d7 | 21.5 |
| price_lag_d2 | 12.2 |
| price_d1_mean | 9.4 |
| hour | 7.4 |
| dayofweek | 5.0 |
| month | 3.4 |
| is_weekend | 0.5 |

### LightGBM havali

| özellik | ortalama gain payı (%) |
|---|---|
| price_lag_d1 | 37.6 |
| price_lag_d7 | 20.3 |
| price_lag_d2 | 7.3 |
| price_d1_mean | 6.4 |
| wind_canakkale_pd2 | 5.3 |
| hour | 4.8 |
| dayofweek | 4.1 |
| temp_mean_pd2 | 3.5 |
| cloud_mean_pd2 | 3.5 |
| solar_konya_pd2 | 3.4 |
| month | 1.7 |
| precip_mean_pd2 | 1.5 |
| is_weekend | 0.5 |

## Yorum (sayılardan otomatik üretildi)

- Toplam MAE: D-1 519.2, Kural 497.5, LightGBM havasız 433.5, havalı 419.3 TL/MWh.
- LightGBM havasız: rMAE 0.835 -> D-1'i yendi; Kural'a göre daha iyi.
- LightGBM havalı: rMAE 0.807 -> D-1'i yendi; Kural'a göre daha iyi.
- Hava eklenince MAE farkı: -14.2 TL/MWh (negatif = hava yardımcı oldu). Tek bir test döneminde tek çalıştırma; fark istatistiksel olarak sınanmadı.
- Bayram günleri ayrıca ele alınmadı.
- Bu sonuçlara bakarak parametre veya özellik seti değiştirmek test dönemini 'ayar' için kullanmak olur; yapılmamalı.
