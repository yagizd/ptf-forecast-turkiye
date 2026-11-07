T daha önce bir kez görüldü (ilk model); bu temiz bir sınav DEĞİL.

Claude (sohbet) bu raporun toplam sonuçlarını 6 Kasım'da inceledi: B0 ve B3, %80 ve %90 için kapsama ve genişlik, T ve V'de parquet'ten bağımsız yeniden hesaplandı, aynı çıktı. Blok, ay ve saat türü tabloları satır satır doğrulanmadı.

T KOŞUSU KABUL KURALLARI (sonuçtan ÖNCE yazıldı, DEĞİŞMEZ): TK1. Koşu tek kez; tekrar yok. TK2. Hiçbir ayar/parametre/özellik T sonucuna bakarak değiştirilmez. TK3. Bilgi eşiği (ret kuralı DEĞİL): iki düzeyde de |kapsama - hedef| <= 2,0 puan ise "hedef tuttu", değilse "hedef tutmadı"; sonuç olduğu gibi yazılır, kötü çıkan aylar güzelleştirilmez. TK4. V sonuçları (79,86 / 89,92) ile T sonuçları yan yana verilir; fark yorumlanırken "tek T dönemi, tek koşu, anlamlılık sınanmadı" notu eklenir.

# Final aralık koşusu: B3 (günlük kalibrasyon + ACI, gamma 0.05) T = 2025-10-21..2026-10-20 (8760 saat)

- Ayar yok: V'de seçilen yöntem aynen uygulandı. Tablo 21-29 Ekim 2026 günlerini İÇERMEZ.
- Sebep: BİLMİYORUZ; tavan değişimi hipotezi sınanmadı (zayıf dönemlerin nedeni araştırılmadı).
- Ham quantile sırası bozuk satır (T test satırlarında): q10 > q90 36; q05 > q95 10; 4 quantile'ın sırası bozuk (herhangi biri) 2027.

## Bilgi eşiği (TK3, ret kuralı DEĞİL)

B3 için: %80 |kapsama - hedef| = 0.01 puan; %90 |kapsama - hedef| = 0.02 puan. **Sonuç: hedef tuttu** (eşik 2.0 puan).

## Toplam (T, n=8760 / varyant)

| varyant | düzey | n | kapsama | kırpılmış genişlik | alt ıskalama | üst ıskalama | boş aralık saat | sınırsız-Q gün |
|---|---|---|---|---|---|---|---|---|
| B0 | %80 | 8760 | 71.50% | 1297.9 | 16.92% | 11.59% | 0 | 0 |
| B0 | %90 | 8760 | 84.45% | 1719.8 | 8.86% | 6.69% | 0 | 0 |
| B3 | %80 | 8760 | 79.99% | 1548.8 | 11.04% | 8.97% | 0 | 0 |
| B3 | %90 | 8760 | 90.02% | 1942.4 | 5.57% | 4.41% | 0 | 0 |

B0 = günlük güncellemesiz 56 günlük kalibrasyon (blok başında bir kez); B3 = günlük güncelleme + ACI.

## V ile T yan yana (TK4)

| varyant | düzey | V kapsama | V genişlik | T kapsama | T genişlik |
|---|---|---|---|---|---|
| B0 | %80 | 75.43% | 829.3 | 71.50% | 1297.9 |
| B0 | %90 | 84.53% | 1108.8 | 84.45% | 1719.8 |
| B3 | %80 | 79.86% | 886.9 | 79.99% | 1548.8 |
| B3 | %90 | 89.92% | 1190.9 | 90.02% | 1942.4 |

Tek T dönemi, tek koşu, anlamlılık sınanmadı.

## Blok başına kapsama

| blok | n | B0 %80 | B3 %80 | B0 %90 | B3 %90 |
|---|---|---|---|---|---|
| 1 | 672 | 79.32% | 80.65% | 89.88% | 90.48% |
| 2 | 672 | 80.36% | 80.95% | 89.14% | 89.73% |
| 3 | 672 | 75.89% | 75.45% | 91.07% | 88.99% |
| 4 | 672 | 74.11% | 77.23% | 85.27% | 87.35% |
| 5 | 672 | 43.90% | 80.36% | 62.80% | 90.77% |
| 6 | 672 | 63.54% | 79.02% | 82.14% | 90.03% |
| 7 | 672 | 75.74% | 84.82% | 87.65% | 91.67% |
| 8 | 672 | 65.03% | 80.65% | 83.33% | 90.48% |
| 9 | 672 | 73.96% | 80.65% | 86.16% | 89.58% |
| 10 | 672 | 62.80% | 75.00% | 71.13% | 88.54% |
| 11 | 672 | 70.98% | 82.14% | 85.42% | 90.92% |
| 12 | 672 | 81.70% | 83.63% | 91.82% | 91.67% |
| 13 | 672 | 81.40% | 78.72% | 91.52% | 89.73% |
| 14 (n küçük, yorumlanmaz) | 24 | 91.67% | 95.83% | 100.00% | 100.00% |

## Ay başına kapsama

| ay | n | B0 %80 | B3 %80 | B0 %90 | B3 %90 |
|---|---|---|---|---|---|
| 2025-10 | 264 | 73.11% | 74.62% | 86.74% | 88.64% |
| 2025-11 | 720 | 79.03% | 80.56% | 89.72% | 90.00% |
| 2025-12 | 744 | 82.80% | 80.38% | 92.07% | 90.46% |
| 2026-01 | 744 | 77.28% | 78.76% | 89.92% | 88.98% |
| 2026-02 | 672 | 47.32% | 73.81% | 64.58% | 86.76% |
| 2026-03 | 744 | 63.71% | 83.33% | 81.32% | 92.74% |
| 2026-04 | 720 | 67.78% | 82.50% | 82.50% | 90.00% |
| 2026-05 | 744 | 68.82% | 82.53% | 85.08% | 91.40% |
| 2026-06 | 720 | 71.81% | 78.89% | 85.42% | 88.75% |
| 2026-07 | 744 | 62.37% | 75.00% | 71.24% | 87.90% |
| 2026-08 | 744 | 73.79% | 83.33% | 87.77% | 91.94% |
| 2026-09 | 720 | 83.47% | 84.03% | 92.64% | 91.67% |
| 2026-10 | 480 | 80.62% | 77.29% | 91.25% | 89.79% |

## Ayrı satır: 2026-04 ve 2026-05

| ay | düzey | B0 (kapsama / genişlik) | B3 (kapsama / genişlik) |
|---|---|---|---|
| **2026-04** | %80 | 67.78% / 2068.2 | 82.50% / 2515.4 |
| **2026-04** | %90 | 82.50% / 2638.3 | 90.00% / 2907.8 |
| **2026-05** | %80 | 68.82% / 2007.0 | 82.53% / 2278.9 |
| **2026-05** | %90 | 85.08% / 2597.7 | 91.40% / 2816.5 |

## Fiyat 0 saatleri ve tavan saatleri için kapsama

| saat türü | düzey | B0 (n; kapsama) | B3 (n; kapsama) |
|---|---|---|---|
| fiyat 0 | %80 | 409 saat; 81.91% | 409 saat; 90.71% |
| fiyat 0 | %90 | 409 saat; 90.71% | 409 saat; 97.07% |
| fiyat = günün tavanı | %80 | 853 saat; 81.01% | 853 saat; 84.17% |
| fiyat = günün tavanı | %90 | 853 saat; 89.33% | 853 saat; 91.56% |

## B3 alpha izi (gün başına kullanılan alpha)

| düzey | min | ortalama | maks | alpha<0 gün | alpha>1 gün | gün |
|---|---|---|---|---|---|---|
| %80 | 0.0258 | 0.1628 | 0.2346 | 0 | 0 | 365 |
| %90 | 0.0033 | 0.0827 | 0.1292 | 0 | 0 | 365 |

Betik süresi (ölçüm): 22 sn.

