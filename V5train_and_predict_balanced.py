import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. AYARLAR VE KLASÖR YAPISI
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# 2. VERİ YÜKLEME VE MANUEL ÖLÇEKLENDİRME
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

mins = data.min(axis=1, keepdims=True)
maxs = data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 
scaled_data = (data - mins) / ranges

# PARAMETRELER
look_back = 12 
n_future = 12 
n_trials = 15 
batch_size = 4

# 3. VERİ HAZIRLAMA (LSTM FORMATI)
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
y = y.reshape(-1, 1) # y'yi (N, 1) formatına getiriyoruz
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- YENİ RASYONEL DOKUNUŞ: DEĞER BAZLI AĞIRLIKLANDIRMA ---
# Satış miktarı (y) ne kadar yüksekse, o verinin hata payı o kadar ağır cezalandırılır.
# Bu sayede Ortaokul kategorisindeki 6000'lik zirveler model için 'en kritik' veri olur.
sample_weights = 1.0 + (y.flatten() * 5.0) 

# 4. EĞİTİM VE TAHMİN DÖNGÜSÜ
all_trial_results = []
for trial in range(n_trials):
    tf.keras.backend.clear_session()
    print(f"Tur {trial+1}/{n_trials} eğitiliyor... (Yüksek Değer Odaklı Ağırlıklandırma Aktif)")
    
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(64, return_sequences=False), 
        Dropout(0.2), 
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Patience değerini 20 yaparak modelin o yüksek rakamları öğrenmesi için zaman tanıdık
    callback = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    
    # Eğitimde sample_weight kullanarak yüksek satışlara odaklanıyoruz
    model.fit(X, y, 
              sample_weight=sample_weights, 
              epochs=120, 
              batch_size=batch_size, 
              verbose=0, 
              callbacks=[callback])

    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        baslangic_ayi = 6 # Haziran tahmini ile başla

        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            # Negatif değerleri engelle
            pred = max(0, pred) 
            group_preds.append(pred)
            # Pencereyi bir adım kaydır
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Ölçeklendirmeyi tersine çevirerek gerçek rakamlara dön
        trial_preds[group_name] = [max(0, round(p * ranges[i][0] + mins[i][0])) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SONUÇLARI BİRLEŞTİR VE ORTALAMA AL
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. KAYDET
dosya_adi = "V5_Value_Balanced_Final.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print("\n" + "="*50)
print("İşlem Başarıyla Tamamlandı!")
print(f"Yeni ağırlıklı tahminler kaydedildi: {dosya_adi}")
print("="*50)