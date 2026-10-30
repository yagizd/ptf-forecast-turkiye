Toplam kapsama %75-%85 arasındaysa iyi; dışındaysa sebebi araştırılır, ayar yapılmaz.

# Doğrulama dönemi V için tahmin aralığı (CQR)

**Kodu ve sonucu Claude (sohbet) 30 Ekim'de inceledi; kapsama ve genişlik parquet'ten bağımsız yeniden hesaplandı, aynı çıktı.** Test dönemi T'ye dokunulmadı (veri load_locked ile yüklendi).

- Dönem V = 2024-10-21..2025-10-20; make_blocks 14 blok üretti (13 tam + 1 kısa son blok); aralık üretilen blok sayısı: 14.
- Özellikler (18): price_lag_d1, price_lag_d2, price_lag_d7, price_d1_mean, hour, dayofweek, month, is_weekend, temp_mean_pd2, cloud_mean_pd2, precip_mean_pd2, wind_canakkale_pd2, solar_konya_pd2, is_holiday, is_arife, is_bridge, days_to_holiday, days_since_holiday; gruplar ['H']; K yok.
- Yöntem: LightGBM quantile (alpha 0.1 ve 0.9) + CQR; kalibrasyon = bloktan önceki son 56 gün; hedef kapsama %80; varsayılan parametreler, ayar yok.
- Satır: 8760 (hedefi dolu 8760).

## Toplam

| sürüm | n | kapsama | ortalama genişlik (TL/MWh) |
|---|---|---|---|
| kırpılmamış | 8760 | 75.43% | 854.33 |
| [0, tavan] kırpılmış | 8760 | 75.43% | 829.33 |

**Kabul kuralı sonucu (kırpılmamış toplam kapsama 75.43%): İÇİNDE (kabul kuralına göre iyi).**

## Kontroller

- Alt > üst olan satır (kırpılmamış): 0; (kırpılmış): 0.
- Q sonsuz olan blok: yok.
- Q min/medyan/maks (sonlu olanlar): 53.3 / 97.4 / 403.0
- Kırpma etkisi: alt sınır < 0 olan satır 136; üst sınır > tavan olan satır 2128.

## Blok başına kapsama ve genişlik

| blok | n | kapsama | genişlik | kapsama (kırpılmış) | genişlik (kırpılmış) | Q |
|---|---|---|---|---|---|---|
| 1 | 672 | 76.6% | 701.3 | 76.6% | 690.7 | 61.3 |
| 2 | 672 | 80.4% | 712.9 | 80.4% | 685.6 | 98.1 |
| 3 | 672 | 83.2% | 717.1 | 83.2% | 688.8 | 120.2 |
| 4 | 672 | 77.5% | 592.2 | 77.5% | 581.1 | 64.5 |
| 5 | 672 | 72.8% | 600.0 | 72.8% | 595.8 | 80.4 |
| 6 | 672 | 63.4% | 789.0 | 63.4% | 786.7 | 65.0 |
| 7 | 672 | 44.5% | 830.2 | 44.5% | 829.2 | 123.6 |
| 8 | 672 | 67.0% | 1361.6 | 67.0% | 1348.4 | 395.5 |
| 9 | 672 | 85.0% | 1491.9 | 85.0% | 1463.2 | 403.0 |
| 10 | 672 | 89.6% | 1047.3 | 89.6% | 940.2 | 242.8 |
| 11 | 672 | 90.0% | 785.0 | 90.0% | 726.1 | 127.0 |
| 12 | 672 | 77.5% | 774.0 | 77.5% | 754.3 | 87.8 |
| 13 | 672 | 73.2% | 708.0 | 73.2% | 695.6 | 53.3 |
| 14 | 24 | 75.0% | 738.4 | 75.0% | 703.5 | 96.7 |

Not: kısa son blok (blok 14) küçük örnekle kapsama verir; yorumda dikkat. Betik süresi (ölçüm): 11 sn.

