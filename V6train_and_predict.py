import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. HAZIRLIK
test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
if not os.path.exists(test_cikti_klasor_yolu):
    os.makedirs(test_cikti_klasor_yolu)

# 2. VERİ YÜKLEME (V3 Ölçeklendirme Mantığı - En Kararlısı)
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values
mins, maxs = data.min(axis=1, keepdims=True), data.max(axis=1, keepdims=True)
ranges = maxs - mins
ranges[ranges == 0] = 1 
scaled_data = (data - mins) / ranges

# PARAMETRELER (Version 3/5 Baz Alındı)
look_back = 12 
n_future = 12 
n_trials = 15 
batch_size = 4

# 3. VERİ SETİ OLUŞTURMA
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. EĞİTİM VE TAHMİN
all_trial_results = []
for trial in range(n_trials):
    tf.keras.backend.clear_session()
    print(f"Tur {trial+1}/{n_trials} eğitiliyor...")
    
    # RASYONEL MİMARİ GÜNCELLEMESİ
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(100, return_sequences=True), # Kapasite artırıldı
        LSTM(50),
        Dropout(0.25),
        Dense(64),
        LeakyReLU(negative_slope=0.1), # ReLU yerine LeakyReLU (Daha hassas öğrenme)
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mae') # MSE yerine MAE (Uç değerlere karşı daha dirençli)
    
    # Dinamik Eğitim Düzenleyicileri
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=0.0001)
    
    model.fit(X, y, epochs=150, batch_size=batch_size, verbose=0, callbacks=[early_stop, reduce_lr])

    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            pred = max(0, pred) 
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        trial_preds[group_name] = [round(p * ranges[i][0] + mins[i][0]) for p in group_preds]
    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. FİNALİZASYON
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]
final_res.to_excel(os.path.join(test_cikti_klasor_yolu, "version6_tahmin_sonuclari_ortalama_batch4_lookback12_trials15.xlsx"))
print("\nAnalitik olarak güçlendirilmiş V6 sonuçları hazır.")