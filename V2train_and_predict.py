import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. KLASÖR VE AYARLARI YAPILANDIR
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# Rastgeleliği Sabitleme (Sonuçların her seferinde aynı çıkması için)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# 2. VERİ YÜKLEME VE MANUEL ÖLÇEKLENDİRME
# 'hazir_veri_seti_gruplanmis.csv' dosyasını pre_process.py oluşturmuştu.
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

# Kategori bazlı (satır bazlı) Min-Max ölçeklendirme
# Bu yöntem küçük satışlı grupların büyükler içinde kaybolmasını engeller.
mins = data.min(axis=1, keepdims=True)
maxs = data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 # Sıfıra bölme hatasını önle
scaled_data = (data - mins) / ranges

# PARAMETRELER
look_back = 12 
n_future = int(input("Kaç ay sonrasını tahmin etmek istersiniz? (Örn: 12): "))
n_trials = int(input("Kaç farklı eğitim turu yapılsın? (Örn: 5): "))

# 3. VERİ HAZIRLAMA (LSTM Formatına Uygun Hale Getirme)
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. EĞİTİM VE TAHMİN DÖNGÜSÜ
all_trial_results = []
for trial in range(n_trials):
    print(f"\nTur {trial+1}/{n_trials} eğitiliyor...")
    
    # Dengeli Model Mimarisi
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(64, return_sequences=False), 
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Kayıp (loss) azalmadığında eğitimi durdurur
    callback = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=2, verbose=0, callbacks=[callback])

    trial_preds = {}
    for i, group_name in enumerate(df.index):
        # Mevcut son 12 ayı alarak tahmin zincirini başlat
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        
        # Veri Mayıs'ta bittiği için ilk tahmin ayı Haziran (6)
        baslangic_ayi = 6 

        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            
            # --- DİNAMİK SINIR (PATLAMA AYI KONTROLÜ) ---
            su_anki_ay = (baslangic_ayi + j - 1) % 12 + 1
            
            # Eylül, Ekim, Şubat, Mart aylarında modelin daha yüksek değer üretmesine izin ver
            if su_anki_ay in [9, 10, 2, 3]:
                ust_limit = 4.0 # Patlama aylarında sınır geniş
            else:
                ust_limit = 1.3 # Diğer aylarda aşırı sıçramayı kısıtla
            
            pred = max(0, min(pred, ust_limit))
            group_preds.append(pred)
            
            # Tahmin edilen değeri pencereye ekle ve bir ay kaydır
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Ölçeklendirmeyi Tersine Çevir (Gerçek rakamlara dön)
        trial_preds[group_name] = [max(0, round(p * ranges[i][0] + mins[i][0])) for p in group_preds]

    # Her turdan çıkan dataframe'i listeye ekle
    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SONUÇLARI BİRLEŞTİR VE ORTALAMA AL
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. EXCEL'E YAZDIRMA
dosya_adi = "version2_tahmin_sonuclari_ortalama_batch2_lookback12_trials5.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print("\n" + "="*50)
print("İşlem Başarıyla Tamamlandı!")
print(f"Sonuçlar şu klasöre kaydedildi:\n{tam_yol}")
print("="*50)