Bu sonuç bir KARAR DEĞİL; yaklaşık 9 gün var, R1/R3 seçimi için yetersiz. Önceden yazılmış kural (8/13 blok) değişmedi.

# Yeni gün değerlendirmesi (2026-10-21 sonrası)

**Claude (sohbet) bu kodu ve sonucu İNCELEMEDİ.**

- Veri: dataset_weather_guncel.parquet; yeni günler 2026-10-21..2026-10-29; 9 gün, 216 saat (tüm modellerin ve gerçeğin dolu olduğu aynı satırlar). Bu günler test dönemi T'nin DIŞINDADIR.
- R1: dondurulmuş final (gruplar ['H'], ham fiyat hedefi, varsayılan parametreler, Optuna YOK). R3: aynı özellikler, hedef = fiyat/tavan, ağırlık = tavan, tahmin x tavan(D) (tavan 4500 TL/MWh, 2026-04-04 kararı).
- Eğitim: 2024-03-08 00:00:00+03:00 .. 2026-10-20 23:00 (yeni günün öncesindeki tüm veri). Yeni günler 9 gün (<28) olduğundan tek blok kullanıldı: model bir kez eğitildi, 9 günün hepsi bu modelle tahmin edildi (eğitimin bitişinden 1..9 gün sonrası).
- Özellikler (18): price_lag_d1, price_lag_d2, price_lag_d7, price_d1_mean, hour, dayofweek, month, is_weekend, temp_mean_pd2, cloud_mean_pd2, precip_mean_pd2, wind_canakkale_pd2, solar_konya_pd2, is_holiday, is_arife, is_bridge, days_to_holiday, days_since_holiday
- Fiyatı 4500 (tavan) olan saat sayısı: 18

## Toplam

| model | MAE (TL/MWh) | rMAE (D-1'e göre) |
|---|---|---|
| D-1 | 495.2 | 1.000 |
| D-7 | 720.3 | 1.454 |
| Kural | 448.6 | 0.906 |
| R1 | 433.9 | 0.876 |
| R3 | 422.6 | 0.853 |

## Gün gün MAE

| gün | saat | MAE D-1 | MAE D-7 | MAE Kural | MAE R1 | MAE R3 | R1/R3 hangisi düşük |
|---|---|---|---|---|---|---|---|
| 2026-10-21 | 24 | 185.0 | 811.5 | 185.0 | 259.7 | 320.4 | R1 |
| 2026-10-22 | 24 | 267.4 | 795.6 | 267.4 | 256.3 | 320.5 | R1 |
| 2026-10-23 | 24 | 181.5 | 645.4 | 181.5 | 306.9 | 281.5 | R3 |
| 2026-10-24 | 24 | 492.1 | 1114.8 | 1114.8 | 794.0 | 680.4 | R3 |
| 2026-10-25 | 24 | 1082.1 | 588.2 | 588.2 | 585.5 | 567.7 | R3 |
| 2026-10-26 | 24 | 1276.8 | 727.8 | 727.8 | 491.4 | 365.5 | R3 |
| 2026-10-27 | 24 | 178.7 | 508.0 | 178.7 | 380.4 | 290.3 | R3 |
| 2026-10-28 | 24 | 263.8 | 553.1 | 263.8 | 553.8 | 685.7 | R1 |
| 2026-10-29 | 24 | 529.8 | 738.3 | 529.8 | 276.7 | 291.0 | R1 |

R3, 9 günün 5'inde R1'den düşük MAE verdi (bilgi amaçlı; 9 gün, 28 günlük tek bloktan bile kısa).

## Notlar

- 9 gün çok az; R3'ün ya da R1'in 'daha iyi' olduğu SONUCU ÇIKARILMAZ.
- Seçim kuralı (V üzerinde, ≥%1,0 göreli kazanç ve 13 tam bloktan ≥8'inde daha iyi) değişmedi; bu tablo o kuralın yerini almaz.
- Sızıntı testi: tests/test_yeni_gun_degerlendirme.py (D gününden sonraki fiyatlar değişince D'nin özellikleri değişmez).

