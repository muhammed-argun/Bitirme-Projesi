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

# --- RASTGELELİK SABİTLEME KALDIRILDI ---
# Artık her trial (tur) ve her çalıştırma farklı bir başlangıç yapacak.
# Bu sayede modelin "genelleme" yeteneğini ortalama alarak artıracağız.

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
n_future = int(input("Kaç ay sonrasını tahmin etmek istersiniz? (Örn: 12): "))
n_trials = int(input("Kaç farklı eğitim turu yapılsın? (Örn: 5): "))

# 3. VERİ HAZIRLAMA
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
    # Her turda farklı bir başlangıç olması için Keras arkasındaki oturumu temizliyoruz
    tf.keras.backend.clear_session()
    
    print(f"\nTur {trial+1}/{n_trials} eğitiliyor... (Farklı ağırlıklarla başlanıyor)")
    
    model = Sequential([
        LSTM(100, return_sequences=True), # Kapasite artırıldı
        LSTM(50),
        Dropout(0.25),
        Dense(64),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Patience değerini biraz artırdık (20), modelin öğrenmesi için daha çok şans tanıdık
    callback = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=4, verbose=0, callbacks=[callback])

    trial_preds = {}
    for i, group_name in enumerate(df.index):
        current_batch = scaled_data[i, -look_back:].reshape(1, look_back, 1)
        group_preds = []
        baslangic_ayi = 6 

        for j in range(n_future):
            pred = model.predict(current_batch, verbose=0)[0][0]
            
            su_anki_ay = (baslangic_ayi + j - 1) % 12 + 1
            
            # Dinamik Sınır
            if su_anki_ay in [9, 10, 2, 3]:
                ust_limit = 4.0 
            else:
                ust_limit = 1.3 
            
            pred = max(0, min(pred, ust_limit))
            group_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        trial_preds[group_name] = [max(0, round(p * ranges[i][0] + mins[i][0])) for p in group_preds]

    all_trial_results.append(pd.DataFrame(trial_preds).T)

# 5. SONUÇLARI BİRLEŞTİR VE ORTALAMA AL
# Artık her tur farklı olduğu için bu ortalama gerçek bir değer ifade edecek.
final_res = pd.concat(all_trial_results).groupby(level=0).mean().round(0)
final_res.columns = [f"Ay_{i+1}" for i in range(n_future)]

# 6. EXCEL'E YAZDIRMA
dosya_adi = "version3_tahmin_sonuclari_ortalama_batch4_lookback12_trials25.LTSM_update.xlsx"
tam_yol = os.path.join(test_cikti_klasor_yolu, dosya_adi)
final_res.to_excel(tam_yol)

print("\n" + "="*50)
print("İşlem Başarıyla Tamamlandı!")
print(f"Rastgelelik serbest bırakıldı ve {n_trials} turun ortalaması alındı.")
print(f"Yeni dosya: {dosya_adi}")
print("="*50)