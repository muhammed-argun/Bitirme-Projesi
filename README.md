## Temel Veri Hazırlığı ve Ön İşleme Kodları

### 1. pre_processing.py (Ana Veri Ön İşleme Köprüsü)
* **Görevi:** Bilgisayardaki ham aylık Excel satış dosyalarını (`.xlsx`) tarayarak okur.
* **İşlevselliği:** İçerisinde barındırdığı `urun_siniflandir` fonksiyonu ile ürün adlarındaki anahtar kelimeleri (Örn: "1. sınıf", "lgs", "tyt", "kpss") regex tabanlı tarayarak verileri akademik seviyelere göre 7 ana sınıfa ayırır (*İlkokul, Ortaokul, LGS, Lise, YKS/Üniversite Hazırlık, Sınav Hazırlık, Diğer*). 
* **Çıktı:** Tüm aylık verileri kronolojik olarak birleştirip matris formuna getirir ve modellere doğrudan girdi sağlayan `hazir_veri_seti_gruplanmis_tumu.csv` dosyasını üretir.

### 2. comparison_file_pre_processing.py (Gerçek Zamanlı Kıyaslama Hazırlığı)
* **Görevi:** Modelin ürettiği gelecek dönem tahminlerini (Örn: Şubat 2026 tahmini), o ay gerçekleştikten sonra elde edilen gerçek dünya satış verileriyle eşleştirmek ve doğruluk analizi yapmak üzere ham yeni veriyi kategorize eder.

### 3. CV1train_and_predict.py (Çapraz Doğrulama ve Performans Analiz Kodu)
* **Görevi:** Modelin genelleme yeteneğini ölçmek amacıyla veriyi kronolojik olarak Eğitim (2024.12 öncesi), Doğrulama (2025.01 - 2025.05) ve Test olarak ayırır.
* **İşlevselliği:** Sektörel döngüselliği modele öğretebilmek adına ayları sinüs ve kosinüs bileşenlerine ayırarak (`month_sin`, `month_cos`) derin öğrenme ağına zamansal özellik (feature) olarak besler. Test kümesindeki performans analizi için gerçek değerler ile tahmin edilen değerleri MAE metriği üzerinden jüriye sunulabilir akademik doğrulukta hesaplar.

---

## Model Geliştirme Süreci (Kod Versiyonları)

### V1train_and_predict.py (Baseline - İlk Temel Model)
* **Mimari:** Temel düzeyde bir LSTM katmanı ve ardışık Dense katmanından oluşur.
* **Ölçeklendirme:** Verileri maksimum değerlerine bölerek basit bir normalizasyon uygular. 

### V2train_and_predict.py (Dinamik Sınırlı Model)
* **Gelişme:** Sektörün sezonluk dalgalanmaları (Örn: Okul açılış dönemleri olan Eylül, Ekim, Şubat ve Mart ayları) tespit edilerek modele dinamik üst limitler (`ust_limit = 4.0` veya `1.3`) eklenmiştir. Böylece patlama aylarındaki ani sıçramalar yapay olarak korunmuştur.
* **Ölçeklendirme:** Kategori bazlı Manuel Min-Max Ölçeklendirmeye geçilerek küçük hacimli kitap gruplarının büyük gruplar arasında ezilmesi engellenmiştir.

### V3train_and_predict.py (Rastgelelik Serbest / Ortalama Tabanlı Model)
* **Gelişme:** Kodun her çalışmada aynı sonucu vermesini sağlayan yapay sabit tohumlar (seed) kaldırılmıştır. Model `n_trials` (Çoklu eğitim turu) parametresiyle defalarca eğitilip tahminlerin ortalaması alınarak derin öğrenmenin varyans hatası (Overfitting riski) minimize edilmiştir.

### V4train_and_predict.py (Rafine Dinamik Sınırlar)
* **Gelişme:** Sürüm 2 ve Sürüm 3'teki aşırı tahmin sapmalarını dengelemek adına okul dönemlerine ait dinamik sınırlar daha dar ve hassas bantlara (`3.2`, `2.0` ve `1.25`) optimize edilmiştir.

### V5train_and_predict.py (Altın Parametreli Kararlı Model - Proje Temeli)
* **Gelişme:** Projenin ana omurgasını oluşturan **"Baseline Model"** sürümüdür. `LookBack=12` (Son 12 ayı inceleme), `Trials=15` (15 farklı tur eğitip ortalama alma) ve `BatchSize=4` parametreleri ile en kararlı ve tutarlı tahmin çıktılarını üretmiştir. Yapay veri sınırlamaları kaldırılarak modelin tamamen veriden öğrenmesi sağlanmıştır.

### V5train_and_predict_balanced.py (Önerilen Model - Ağırlıklandırılmış Eğitim)
* **Gelişme:** Satış hacmi yüksek olan kritik aylara ve sınıflara modelin daha fazla odaklanabilmesi için eğitim aşamasına rasyonel örneklem ağırlıkları (`sample_weight`) entegre edilmiştir. Bu sayede ciro ve hacim bazlı tahmin doğruluğu ciddi oranda artırılmıştır.

### V6train_and_predict.py (Nihai Gelişmiş Model - LeakyReLU & MAE Kayıp Fonksiyonu)
* **Mimari:** Modelin öğrenme hassasiyetini artırmak için klasik ReLU aktivasyon fonksiyonu yerine, ölü nöron problemini engelleyen **LeakyReLU** (`negative_slope=0.1`) katmanı eklenmiştir.
* **Kayıp Fonksiyonu:** Modelin eğitim optimizasyonunda uç değerlere (anlık talep patlamalarına) karşı daha dirençli ve kararlı olan **MAE (Mean Absolute Error)** kayıp fonksiyonuna geçilmiştir. Dinamik öğrenme hızı düşürücü (`ReduceLROnPlateau`) eklenerek en optimum global minimuma ulaşılması sağlanmıştır.

## Not
Kodİncelemesi.xlsx doyasında kodlardan alınan sonuçların gerçek sonuçlarla karşılaştırılması yapılmıştır.
