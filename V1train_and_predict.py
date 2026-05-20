import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


test_cikti_klasor_yolu = r"E:\Muhammed Özel\Bitirme Projesi\test_sonuclari"
# 1. RASTGELELİĞİ SABİTLEME (Teknik Sorun 1 Çözümü)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# 2. VERİ YÜKLEME
df = pd.read_csv("hazir_veri_seti_gruplanmis.csv", index_col='Sinif_Grubu')
data = df.values
max_values = data.max(axis=1, keepdims=True)
max_values[max_values == 0] = 1 
scaled_data = data / max_values

# PARAMETRELER
look_back = 12  # İstediğin gibi 12 yaptım
n_future = int(input("Kaç ay sonrasını tahmin etmek istersiniz? (Örn: 11): "))
n_trials = int(input("Kaç farklı eğitim turu yapılsın? (Örn: 5): "))

# 3. VERİ HAZIRLAMA (Hatanın Çözümü Burası)
X, y = [], []
for group in scaled_data:
    for i in range(len(group) - look_back):
        X.append(group[i:(i + look_back)])
        y.append(group[i + look_back])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. EĞİTİM VE ORTALAMA ALMA DÖNGÜSÜ
all_trial_results = []
for trial in range(n_trials):
    print(f"\nTur {trial+1}/{n_trials} eğitiliyor...")
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.3), # Overfitting engelleme
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Overfitting durdurma mekanizması
    callback = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    model.fit(X, y, epochs=100, batch_size=4, verbose=0, callbacks=[callback])

    # Tahminleme (Recursive)
    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        for _ in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]

            # Mantıksal Sınırlama (Eksiye düşmeyi ve devasa artışı engeller)
            pred = max(0, min(pred, 1.5)) 
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Gerçek rakama dönüştür
        trial_preds[group_name] = [round(p * max_values[i][0]) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SONUÇLARI BİRLEŞTİR VE ORTALAMA AL
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. EXCEL'E YAZDIR
dosya_adi = "version1_tahmin_sonuclari_ortalama_batch4_lookback12_trials10.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)
print("\n" + "="*50)
print("İşlem Başarıyla Tamamlandı!")
print(f"{dosya_adi}' olarak kaydedildi.")
print("="*50)