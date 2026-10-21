# Kıyas modelleri (baseline) sonuçları

Test dönemi: 2025-10-21 .. 2026-10-20 (365 gün, 8760 saat). Üç modelin de tahmini ve gerçek fiyatı dolu olan 8760 saat değerlendirildi (0 saat boş olduğu için dışarıda kaldı).

MAE = ortalama kaç TL/MWh yanıldık. rMAE = modelin MAE'si ÷ D-1'in MAE'si (1'in altı = D-1'i yendi).

## Toplam

| model | MAE (TL/MWh) | rMAE |
|---|---|---|
| 1. Dünün aynı saati (D-1) | 519.2 | 1.000 |
| 2. Geçen haftanın aynı günü (D-7) | 598.6 | 1.153 |
| 3. Kural (Sal-Cum: D-1, Pzt/Cmt/Paz: D-7) | 497.5 | 0.958 |

## Hafta içi / hafta sonu

| grup | saat sayısı | MAE D-1 | MAE D-7 | MAE Kural | rMAE D-7 | rMAE Kural |
|---|---|---|---|---|---|---|
| Hafta içi | 6264 | 513.4 | 606.0 | 464.7 | 1.180 | 0.905 |
| Hafta sonu | 2496 | 533.9 | 579.9 | 579.9 | 1.086 | 1.086 |

## Aylık

| ay | saat sayısı | MAE D-1 | MAE D-7 | MAE Kural | rMAE D-7 | rMAE Kural |
|---|---|---|---|---|---|---|
| 2025-10 | 264 | 507.8 | 458.2 | 435.1 | 0.902 | 0.857 |
| 2025-11 | 720 | 362.4 | 459.7 | 396.5 | 1.269 | 1.094 |
| 2025-12 | 744 | 298.6 | 317.1 | 278.8 | 1.062 | 0.934 |
| 2026-01 | 744 | 403.1 | 358.2 | 352.9 | 0.889 | 0.876 |
| 2026-02 | 672 | 749.3 | 808.0 | 773.1 | 1.078 | 1.032 |
| 2026-03 | 744 | 667.6 | 844.3 | 731.9 | 1.265 | 1.096 |
| 2026-04 | 720 | 536.4 | 638.2 | 518.8 | 1.190 | 0.967 |
| 2026-05 | 744 | 341.4 | 497.6 | 360.0 | 1.458 | 1.055 |
| 2026-06 | 720 | 539.2 | 742.8 | 534.4 | 1.378 | 0.991 |
| 2026-07 | 744 | 609.5 | 728.8 | 511.5 | 1.196 | 0.839 |
| 2026-08 | 744 | 566.6 | 617.7 | 499.9 | 1.090 | 0.882 |
| 2026-09 | 720 | 609.6 | 526.7 | 498.6 | 0.864 | 0.818 |
| 2026-10 | 480 | 602.1 | 776.1 | 596.1 | 1.289 | 0.990 |

## Saate göre (0-23)

| saat | saat sayısı | MAE D-1 | MAE D-7 | MAE Kural | rMAE D-7 | rMAE Kural |
|---|---|---|---|---|---|---|
| 0 | 365 | 474.6 | 630.2 | 562.1 | 1.328 | 1.184 |
| 1 | 365 | 458.4 | 596.1 | 528.9 | 1.300 | 1.154 |
| 2 | 365 | 448.7 | 598.7 | 533.3 | 1.334 | 1.189 |
| 3 | 365 | 430.0 | 597.1 | 519.1 | 1.389 | 1.207 |
| 4 | 365 | 477.7 | 632.5 | 553.7 | 1.324 | 1.159 |
| 5 | 365 | 472.2 | 626.9 | 551.8 | 1.328 | 1.169 |
| 6 | 365 | 521.0 | 604.6 | 554.1 | 1.160 | 1.064 |
| 7 | 365 | 442.2 | 556.9 | 486.3 | 1.259 | 1.100 |
| 8 | 365 | 600.7 | 513.0 | 399.9 | 0.854 | 0.666 |
| 9 | 365 | 646.7 | 545.1 | 435.8 | 0.843 | 0.674 |
| 10 | 365 | 632.7 | 582.7 | 496.5 | 0.921 | 0.785 |
| 11 | 365 | 630.0 | 568.1 | 470.3 | 0.902 | 0.747 |
| 12 | 365 | 488.3 | 580.2 | 463.5 | 1.188 | 0.949 |
| 13 | 365 | 581.0 | 613.5 | 524.6 | 1.056 | 0.903 |
| 14 | 365 | 661.0 | 646.3 | 542.5 | 0.978 | 0.821 |
| 15 | 365 | 620.7 | 618.9 | 544.8 | 0.997 | 0.878 |
| 16 | 365 | 485.1 | 479.0 | 381.7 | 0.987 | 0.787 |
| 17 | 365 | 470.9 | 505.3 | 389.1 | 1.073 | 0.826 |
| 18 | 365 | 496.8 | 539.4 | 445.8 | 1.086 | 0.897 |
| 19 | 365 | 487.4 | 638.5 | 483.2 | 1.310 | 0.991 |
| 20 | 365 | 446.2 | 652.1 | 462.9 | 1.461 | 1.037 |
| 21 | 365 | 443.5 | 621.6 | 469.2 | 1.401 | 1.058 |
| 22 | 365 | 512.5 | 695.3 | 549.2 | 1.357 | 1.072 |
| 23 | 365 | 533.8 | 724.0 | 591.7 | 1.356 | 1.108 |

## Özel saatler (fiyat 0 veya ayın tavanında)

'Tavanda' = fiyat, o ayın en yüksek fiyatına eşit VE o ayda tam o fiyatta en az 5 saat var. Bu resmî tavan değil, ayın veriden çıkan en yüksek fiyatıdır (tüm veri setinden hesaplandı, sadece değerlendirme içindir). 'Diğer' = ikisi de değil.

| saat türü | saat sayısı | MAE D-1 | MAE D-7 | MAE Kural | rMAE D-7 | rMAE Kural |
|---|---|---|---|---|---|---|
| Diğer | 7503 | 570.2 | 663.7 | 554.4 | 1.164 | 0.972 |
| Fiyat 0 | 409 | 191.6 | 238.1 | 142.0 | 1.243 | 0.741 |
| Tavanda | 848 | 226.8 | 196.6 | 165.6 | 0.867 | 0.731 |

## Yorum

- Toplam MAE: D-1 519.2, D-7 598.6, Kural 497.5 TL/MWh. En düşük hata: 3. Kural (Sal-Cum: D-1, Pzt/Cmt/Paz: D-7).
- D-7 için rMAE 1.153: D-1'i yenemedi.
- Kural için rMAE 0.958: D-1'i yendi.
- Hafta sonu en iyi model 1. Dünün aynı saati (D-1), hafta içi 3. Kural (Sal-Cum: D-1, Pzt/Cmt/Paz: D-7).
- D-1'in MAE'si diğer saatlerde 570.2; 'Fiyat 0' saatlerinde 191.6 (409 saat); 'Tavanda' saatlerinde 226.8 (848 saat). Özel saat toplamı 1257.
- Bayram günleri ayrıca ele alınmadı; tatil etkisi hatalara karışmış olabilir.
- consumption ve lep (yük tahmini) bu modellerde kullanılmadı. Sebep: yayın saatlerini bilmiyoruz, tahmin anında elimizde olmayabilirler (sızıntı riski).
- Zaman kuralı: D günü tahmin edilirken sadece D-1 ve öncesinin fiyatı görüldü (kodda assert ve testle denetleniyor).
