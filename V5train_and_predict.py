import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. AYARLAR
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# 2. VERİ VE ÖLÇEKLENDİRME (V3 Mantığı)
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values
mins, maxs = data.min(axis=1, keepdims=True), data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 
scaled_data = (data - mins) / ranges

# 9. DENEYİN ALTIN PARAMETRELERİ
look_back = 12 
n_future = 12 
n_trials = 15 
batch_size = 4

# 3. VERİ HAZIRLAMA
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. EĞİTİM
all_trial_results = []
for trial in range(n_trials):
    tf.keras.backend.clear_session()
    print(f"Tur {trial+1}/{n_trials} eğitiliyor...")
    
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(64, return_sequences=False), 
        Dropout(0.2), # V3 değeri
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Öğrenme hızını (0.001) sabit tutarak Adam optimizer kullanıyoruz
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    callback = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=batch_size, verbose=0, callbacks=[callback])

    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            # Sınırları V3'teki gibi serbest bıraktık (limit yok)
            pred = max(0, pred) 
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        trial_preds[group_name] = [round(p * ranges[i][0] + mins[i][0]) for p in group_preds]
    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SONUÇ
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]
final_res.to_excel(os.path.join(test_cikti_klasor_yolu, "version5_tahmin_sonuclari_ortalama_batch4_lookback12_trials15.xlsx"))
