import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. AYARLAR VE KLASÖR
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# Rastgelelik serbest (Version 3 mantığı) - n_trials ile ortalama alınacak
tf.keras.backend.clear_session()

# 2. VERİ YÜKLEME VE MANUEL ÖLÇEKLENDİRME
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values

mins = data.min(axis=1, keepdims=True)
maxs = data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 
scaled_data = (data - mins) / ranges

# OPTİMUM PARAMETRELER (Analiz sonuçlarına dayanarak)
look_back = 12 
n_future = 12 # 2025 Haziran - 2026 Mayıs arası
n_trials = 20 # 15'ten 20'ye çıkararak kararlılığı artırdık
batch_size = 4

# 3. VERİ HAZIRLAMA
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. EĞİTİM DÖNGÜSÜ
all_trial_results = []
for trial in range(n_trials):
    tf.keras.backend.clear_session()
    print(f"Tur {trial+1}/{n_trials} eğitiliyor...")
    
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(64, return_sequences=False), 
        Dropout(0.25), # Dropout biraz artırıldı (Overfitting kontrolü için)
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    callback = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    model.fit(X, y, epochs=120, batch_size=batch_size, verbose=0, callbacks=[callback])

    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        baslangic_ayi = 6 # Haziran

        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            su_anki_ay = (baslangic_ayi + j - 1) % 12 + 1
            
            # --- RAFİNE DİNAMİK SINIRLAR ---
            # Eylül(9) ve Şubat(2) en kritik aylar. 
            # 3. aydaki (Eylül) devasa sapmayı önlemek için limit 3.0 ile dengelendi.
            if su_anki_ay in [9, 2]:
                ust_limit = 3.2 
            elif su_anki_ay in [10, 3]:
                ust_limit = 2.0
            else:
                ust_limit = 1.25 # Stabil aylar için Version 2'den gelen optimum değer
            
            pred = max(0, min(pred, ust_limit))
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        trial_preds[group_name] = [max(0, round(p * ranges[i][0] + mins[i][0])) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. BİRLEŞTİRME
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. KAYDET
dosya_adi = "version4_tahmin_sonuclari_ortalama_batch4_lookback12_trials20.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print(f"\n{dosya_adi} dosyasına kaydedildi.")