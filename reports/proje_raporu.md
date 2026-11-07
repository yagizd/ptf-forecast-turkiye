Claude (sohbet) bu raporu 7 Kasım 2026'da inceledi. Parquet dosyalarından sıfırdan hesaplanıp raporla aynı çıkanlar: satır sayıları; toplam MAE ve rMAE (D-1, D-7, Kural, ilk havasız, ilk havalı, final); hafta içi ve hafta sonu MAE; fiyat 0 saatleri; 2026-04 ve 2026-05 MAE; V ve T için tüm aralık tablosu (B0 ve B3, %80 ve %90); blok 7; 2026-02; bulut deliği. Satır satır doğrulanmayanlar: aylık tablonun diğer ayları, saat türü tabloları, ham quantile sayıları (36, 10, 2027; kaynak: reports/aralik_final_t.md).

# Türkiye Gün Öncesi Elektrik Fiyatı (PTF) Tahmini: Proje Raporu (taslak)

Sayıların kaynakları `data/processed/oturum_hafta8/sayilar.md`, `rapor_sayi_kaynak.md` ve ilgili `reports/` dosyalarıdır (CI ve Docker sayıları `data/processed/oturum_docker/ci_push_rapor.md` dosyasından). Hata ya da kötü çıkan yerler olduğu gibi yazılmıştır.

## 1. Problem

PTF (piyasa takas fiyatı, TL/MWh), Türkiye gün öncesi elektrik piyasasında her saat için belirlenen fiyattır. Üretici ve tüccar, piyasa kapanmadan önce ertesi günün 24 fiyatını tahmin etmek ister. Yanlış tahmin para kaybettirir.

Bu projede tahmin anı **D-1 saat 10:00** kabul edilir: yarının fiyatını, bugün 10:00'da elde olan bilgiyle tahmin ederiz. Amaç iki çıktıdır: (1) her saat için bir sayı (nokta tahmini), (2) her saat için bir aralık (%80 ve %90 hedef kapsama). Proje bir öğrenme ve portfolyo çalışmasıdır; canlı çalışan bir işlem aracı değildir.

## 2. Veri ve temizlik

Saatlik tablo 2021-01-01'den başlar. Ana tabloda 50.856 satır vardır (2021-01-01..2026-10-20); T sonrası 9 gün eklenince 51.072 satır olur (2026-10-29'a kadar). Tabloda fiyat, tüketim ve yük planı vardır; EPİAŞ verisi `eptr2` paketiyle çekildi.

Hava verisi Open-Meteo'dan gelir (CC BY 4.0, kaynak belirtilir). 8 şehir için 272 aylık dosya indirildi (8 şehir x 34 ay); son 10 gün (2026-10-21..30) için şehir başına bir dosya daha eklendi (toplam 280). Önemli seçim: gerçekleşen hava değil, **tahmin edilen hava** (`previous_day2`, yani iki gün önceki hava tahmini) kullanıldı; çünkü gerçek hava kullanmak sonuçları olduğundan iyi gösterirdi. Open-Meteo boş veriyi hata vermeden `null` olarak döndürür; bu yüzden her indirmede boş sayısı kaydedildi. Bulut verisinde tek bir delik vardır (2026-04-17..23 arası 129 saat). **Eksik değerler hiç doldurulmadı**; LightGBM boşları kendisi işler.

Tüketim ve yük planı toplandı ama özellik olarak KULLANILMADI, çünkü yayın saatleri bilinmiyor ve gelecek bilgisi sızabilir.

Fiyat tavanı zamanla değişir (3.000 TL/MWh 2024-07-01'den, 3.400 TL/MWh 2025-04-05'ten, 4.500 TL/MWh 2026-04-04'ten). Tavan tablosu EPİAŞ kurul kararlarından derlendi. 2 Nisan 2026'dan sonra yeni karar çıkıp çıkmadığı BİLMİYORUM.

## 3. Dürüst test protokolü

Veri üç dönemde kullanıldı:

- **Eğitim:** 2024-03-08'den başlar (hava verisi bu tarihte tam dolu olur). Her blokta, blok başlangıcından önceki tüm satırlar kullanılır (genişleyen pencere).
- **V (doğrulama):** 2024-10-21..2025-10-20, 8760 saat. Özellik kümesi ve yöntem seçimleri burada yapıldı.
- **T (test):** 2025-10-21..2026-10-20, 8760 saat.

Değerlendirme ileri-yürüyen (walk-forward) yapıldı: 28 günlük bloklar (13 tam blok ve kısa bir son blok); model her blokta yeniden eğitilir ve yalnız o bloktan önceki veriyi görür. Tahmin özellikleri yalnız bir gün önce bilinen bilgilerden oluşur.

**T'nin bir kez görülmüş olması.** İlk LightGBM modeli T üzerinde değerlendirildi. Bu yüzden T, tam temiz bir son sınav DEĞİLDİR. Sonradan yapılan tüm seçimler (özellik kümesi, Optuna kararı, aralık yöntemi) yalnız V'ye bakılarak yapıldı ve kabul kuralları sonuçtan ÖNCE yazıldı; ama T'nin ilk modelle bir kez görülmüş olması bir sınırlama olarak kalır. Final aralık koşusu T üzerinde tek sefer yapıldı ve sonuç olduğu gibi yazıldı.

## 4. Kıyas modelleri

Üç basit kıyas modeli kuruldu. MAE ortalama mutlak hatadır (TL/MWh); rMAE, D-1'in MAE'sine oran:

| Model | T MAE | rMAE |
|---|---|---|
| D-1 (dünün aynı saati) | 519,2 | 1,000 |
| D-7 (geçen haftanın aynı günü) | 598,6 | 1,153 |
| Kural (Sal-Cum D-1; Pzt/Cmt/Paz D-7) | 497,5 | 0,958 |

D-7 tek başına D-1'den kötüdür. Hafta içinde Kural 464,7, D-1 513,4 (Kural belirgin daha iyi). Hafta sonunda Kural 579,9 (D-7'yi kullandığı için), D-1 533,9 (D-1 daha iyi).

## 5. LightGBM ve özellikler

LightGBM bir ağaç topluluğu modelidir (gradient boosting). Ham fiyat hedef alınır, hata ölçüsü L1 (mutlak hata) kullanılır; 400 ağaç, öğrenme oranı 0,05, 31 yaprak, tohum 42. Bu değerler **varsayılan**; ayar sonucu değildir.

İlk modelde özellikler: fiyatın 1, 2, 7 gün öncesi, önceki günün ortalama fiyatı ve takvim (saat, gün, ay, hafta sonu). Havasız sürüm T'de MAE 433,5 (rMAE 0,835); havalı sürüm (8 şehirden sıcaklık, bulut, yağış ortalaması; rüzgâr Çanakkale; güneş Konya) MAE 419,3 (rMAE 0,807).

Sonra V üzerinde şu karar kuralıyla özellik kümeleri denendi: "göreli iyileşme en az %1 VE 13 tam bloğun en az 8'inde daha iyi". Sonuç (V-MAE): R0 277,34; **R1 271,20** (tatil özellikleri: bayram, arife, köprü günü ve en yakın tatile uzaklık); R2 272,30; R3 257,55 (tavana normalize edilmiş hedef). R1 seçildi (+%2,22, 9/13 blok). R2 seçilmedi. R3 +%5,03 kazandırdı ama 7/13 blokta daha iyi olduğu için kural gereği SEÇİLMEDİ (kural 8 blok ister). Final model R1'dir: 18 özellik, hedef ham fiyat.

Final model T'de **MAE 420,5 (rMAE 0,810)**. İlk havalı modelden (419,3) iyi değildir.

**Zayıf yerler (güzelleştirilmedi).** Fiyatın 0 olduğu 409 saatte final model D-1'den kötüdür (MAE 244,0; D-1 191,6): model sıfıra yapışmayı yakalayamıyor. 2026-04 (587,6; D-1 536,4) ve 2026-05 (399,8; D-1 341,4) aylarında final model D-1'den de kötüdür. Sebebi BİLMİYORUM. Tavanın 4 Nisan 2026'da yükselmesi bir hipotezdir ve SINANMADI.

## 6. Doğrulama deneyleri ve Optuna

Optuna (otomatik parametre arama) V üzerinde çalıştırıldı (141 deneme tamamlandı). Varsayılan V-MAE 271,200; en iyi 267,561; kazanç %1,34. Benimseme kuralı sonuçtan önce yazılmıştı: toplam kazanç en az %1,5 olmalı; ayrıca farklı tohumla (43) kazanç en az %1,0 olmalı. Toplam kazanç eşiği (%1,5) TUTMADI; tohum 43'te kazanç %0,65 ve o eşik de tutmadı. Blok 1-7 (+%1,48) ve blok 8-14 (+%1,20) ölçütleri tuttu. **Karar: Optuna benimsenmedi, varsayılan parametreler kaldı.** Nedeni: kazanç küçük ve farklı tohumda kararsız; aşırı uyum (overfit) riski.

## 7. Belirsizlik: CQR ve ACI

Aralıklar LightGBM quantile (yüzdelik) modelleri ve CQR (conformalized quantile regression: modelin hata dağılımına göre aralığı düzeltme) ile üretildi. Kalibrasyon bloktan önceki son 56 gündür. İki yöntem karşılaştırıldı:

- **B0:** blok başında bir kez kalibrasyon.
- **B3:** her gün kalibrasyonu güncelle ve ACI (adaptive conformal inference: bir önceki günün aralığı kaçırıp kaçırmadığına göre hata düzeyini günü gününe ayarlar), gamma 0,05.

Genişlik [0, günün tavanı] aralığına kırpılmıştır.

| Yöntem | Düzey | V kapsama | V genişlik | T kapsama | T genişlik |
|---|---|---|---|---|---|
| B0 | %80 | 75,43% | 829,3 | 71,50% | 1297,9 |
| B0 | %90 | 84,53% | 1108,8 | 84,45% | 1719,8 |
| B3 | %80 | 79,86% | 886,9 | 79,99% | 1548,8 |
| B3 | %90 | 89,92% | 1190,9 | 90,02% | 1942,4 |

Bu sayılar saklanan tahmin tablolarından sıfırdan hesaplanmış ve raporlarla aynı çıkmıştır. (Claude sohbet, 7 Kasım 2026; V'de B0, B1, B2, B3 dahil.)

B0, hem V'de hem T'de hedefin altında kalır. V'deki en kötü örnek blok 7 (2025-04-07..2025-05-04): %80 aralığı yalnız %44,49 kapsar; B3 aynı blokta %80,80 verir (genişlik 829,2 -> 1436,6). Sebebi BİLMİYORUM; tavan değişimi hipotezi sınanmadı. T'de B3 her iki düzeyde hedefe 0,01 ve 0,02 puan yaklaştı (önceden yazılan bilgi eşiği 2,0 puan: "hedef tuttu").

**Dürüst notlar:** (1) T'de B3 aralığı V'ye göre çok daha geniştir (yaklaşık 1,75 kat ve 1,63 kat); sebebi BİLMİYORUM. (2) Aylara göre kapsama değişir: örneğin 2026-02'de %90 düzeyinde B3 kapsaması 86,76%'dır. (3) Ham quantile çıktıları bazı satırlarda çaprazlanır. Kod, sıralama düzeltmesi YAPMAZ; ham (sıralanmamış) quantile'larla çalışır. T'de q10 > q90: 36 satır, q05 > q95: 10 satır, dört quantile'ın herhangi bir sıra bozukluğu: 2027 satır (8760'ın %23,1'i). Saklanan nihai aralıklarda alt sınırı üst sınırdan büyük olan satır yok (0), ama %90 aralığı %80 aralığını tam içermeyen satır var: T'de B3 için 1399/8760 saat (%16,0), V'de B3 için 1055/8760 saat (%12,0) (Claude'un hesabı, saklanan tahmin tablolarından). (4) Tek T dönemi, tek koşu; anlamlılık sınanmadı; T daha önce görüldü.

28 günlük kalibrasyon da denendi, önceden yazılan kuralı sağlamadığı için 56 gün kaldı.

## 8. Servis ve paketleme

Model, donmuş bir paket (model dosyaları ve meta bilgi) olarak kaydedilir; FastAPI ile sunulur. Servis **replay modundadır**: yalnız 2026-09-22..2026-10-29 arası 38 gün (912 saat) için cevap verir. `/predict?date=...` 24 saatlik nokta tahmini ile %80 ve %90 aralıklarını döndürür. Canlı veri hattı YOKTUR; servis "yarını" tahmin edemez.

Testler: yerel makinede `unittest discover` ile 189 test koştu ve 189'u geçti (T betiğinin 11 testi dahil). GitHub Actions iki işten oluşur: `test` (189 test koştu, 10 tanesi atlandı; atlananlar gerçek veri veya paket ister) ve `docker` (imajı derler). Dolayısıyla CI kodun sağlığını gösterir, sonuç kalitesini göstermez.

Docker: `docker/Dockerfile` ve `.dockerignore` yazıldı ve commit'lendi (derleme bağlamı 52 dosya, 265.255 bayt, ölçüldü). İmaj CI'da derlendi (çalışma 42368669858; imaj 690 MB; içinde lightgbm 4.7.0 ve pandas 3.0.6 çalışıyor; ortam değişkenleri yokken kapsayıcı açılmıyor, çıkış kodu 3). `/predict` imajın içinde model ve veriyle DENENMEDİ. İmaj bu bilgisayarda hiç çalıştırılmadı. Bulut yayını (AWS) yapılmadı; Streamlit uygulaması yalnız boş bir yer tutucudur.

## 9. Sınırlamalar ve BİLMİYORUZ listesi

- T daha önce görüldü; final sonuç temiz bir son sınav değildir.
- Tek dönem, tek koşu, anlamlılık sınanmadı.
- T'de aralık genişliğinin V'ye göre büyük olmasının sebebi bilinmiyor.
- Blok 7 / 2026-04 / 2026-05 zayıflığının sebebi bilinmiyor; tavan hipotezi sınanmadı.
- `_previous_day2` hava tahminlerinin D-1 10:00'da hazır olup olmadığı bilinmiyor; yük planı yayın saati bilinmiyor; köprü günü bilgisinin tahmin anında bilinip bilinmediği doğrulanmadı.
- 2 Nisan 2026'dan sonra yeni tavan kararı var mı bilinmiyor.
- Docker imajının derlendiği CI'da doğrulandı; `libgomp1` paketi `python:3.13-slim` + lightgbm 4.7.0 ile gerekli çıktı (onsuz lightgbm içe aktarılamadı). İmajın model ve veriyle çalıştırılması DENENMEDİ.
- Servis canlı değil, replay.
- Ham EPİAŞ verisi ve eğitilmiş modeller depoda yok; EPİAŞ yeniden dağıtım koşulları bulunamadı.
- %90 aralığı %80 aralığını her saatte içermez (T'de B3: %16,0 saat); ham quantile'lar sıralanmaz.

## 10. Öğrenilenler

1. **Önce kural, sonra sonuç.** Kabul kuralları sonuçtan önce yazıldığında kötü çıkan sonuçları (R3, Optuna, 28 gün) kabul etmek kolaylaştı.
2. **Bir test dönemi sadece bir kez temiz kalır.** T'yi ilk modelle görmek, sonraki her sonucun değerini düşürdü. Bir sonraki projede son testi kilitli tutmak gerekir.
3. **Basit kıyaslar güçlüdür.** Kural modeli hafta içinde D-1'i yendi; LightGBM'in kazancı (rMAE 0,81) bu kıyasa göre ölçülmeli.
4. **Aralıkta kapsama tek başına yetmez.** B3 hedefi tutturdu ama genişlik çok büyüdü; kapsama ve genişlik birlikte raporlanmalı.
5. **Sessiz hatalara dikkat.** API'ler boş veriyi hata vermeden döndürebilir; her indirmede boş sayısı yazılmalı.
6. **Bağımsız doğrulama işe yarar.** Sonuç sayıları saklanan tahmin tablolarından sıfırdan yeniden hesaplanınca raporlarla toplam sayılar tuttu.
7. **Dürüst eksikleri yazmak:** İmajın CI'da derlendiği ama model ile çalıştırılmadığının açıkça yazılması, sonradan oluşabilecek yanlış izlenimi önler.
