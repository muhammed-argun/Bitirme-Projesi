import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# 1. VERİYİ YÜKLE
# ======================
df = pd.read_csv("hazir_veri_seti_gruplanmis_tumu.csv", index_col='Sinif_Grubu')
data = df.values
dates = df.columns
n_groups = data.shape[0] # Grup sayısı (Hatanın çözümü için kritik)

# ======================
# 2. AY FEATURE
# ======================
months = np.array([int(col.split('.')[1]) for col in dates])
month_sin = np.sin(2 * np.pi * months / 12)
month_cos = np.cos(2 * np.pi * months / 12)

# ======================
# 3. SPLIT
# ======================
train_end = "2024.12"
val_end = "2025.05"

train_idx = np.where(dates <= train_end)[0]
val_idx = np.where((dates > train_end) & (dates <= val_end))[0]
test_idx = np.where(dates > val_end)[0]

# ======================
# 4. SCALER (SADECE TRAIN)
# ======================
scaler = MinMaxScaler()
# Veriyi transpoze edip fit ediyoruz (Sütun bazlı ölçekleme için)
scaler.fit(data[:, train_idx].T)
data_scaled = scaler.transform(data.T).T

# ======================
# 5. PARAMETRE
# ======================
look_back = 12
n_future = len(test_idx)

# ======================
# 6. DATASET OLUŞTURMA
# ======================
def create_dataset(dataset, target_indices):
    X, y = [], []
    for i in range(dataset.shape[0]):
        for idx in target_indices:
            if idx < look_back:
                continue
            
            seq = []
            for k in range(idx - look_back, idx):
                seq.append([
                    dataset[i, k],
                    month_sin[k],
                    month_cos[k]
                ])
            X.append(seq)
            y.append(dataset[i, idx])
            
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(data_scaled, train_idx)
X_val, y_val = create_dataset(data_scaled, val_idx)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# ======================
# 7. MODEL
# ======================
model = Sequential([
    Input(shape=(look_back, 3)),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ======================
# 8. TRAIN
# ======================
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[callback],
    verbose=1
)

# ======================
# 9. TEST TAHMİN (Hata Düzeltildi)
# ======================
results = {}

for i, group_name in enumerate(df.index):
    start_idx = test_idx[0]
    
    current_batch = []
    for k in range(start_idx - look_back, start_idx):
        current_batch.append([
            data_scaled[i, k],
            month_sin[k],
            month_cos[k]
        ])
    current_batch = np.array(current_batch).reshape(1, look_back, 3)

    preds = []
    for step in range(n_future):
        pred = model.predict(current_batch, verbose=0)[0][0]
        preds.append(pred)

        target_month_idx = test_idx[step]
        s_val = month_sin[target_month_idx]
        c_val = month_cos[target_month_idx]

        new_input = np.array([[[pred, s_val, c_val]]])
        current_batch = np.append(current_batch[:, 1:, :], new_input, axis=1)

    # --- HATA ÇÖZÜMÜ: INVERSE TRANSFORM ---
    # Scaler 7 grup beklediği için 7 sütunlu boş bir matris oluşturuyoruz
    dummy_mat = np.zeros((len(preds), n_groups))
    # Sadece ilgili grubun (i) sütununa tahminleri koyuyoruz
    dummy_mat[:, i] = preds
    # Tersine çevirip sadece kendi sütunumuzu geri alıyoruz
    preds_rescaled = scaler.inverse_transform(dummy_mat)[:, i]
    
    results[group_name] = np.maximum(0, np.round(preds_rescaled))

# ======================
# 10. TEST KARŞILAŞTIRMA
# ======================
test_real = data[:, test_idx]

print("\n--- TEST SONUÇLARI ---")
for i, group_name in enumerate(df.index):
    mae = mean_absolute_error(test_real[i], results[group_name])
    print(f"{group_name} TEST MAE: {mae:.2f}")

# ======================
# 11. EXCEL KAYDET
# ======================
final_df = pd.DataFrame(results).T
final_df.columns = df.columns[test_idx]
final_df.to_excel("tahmin_test_sonuclari.xlsx")

print("\nExcel kaydedildi: tahmin_test_sonuclari.xlsx")
print("Bitti.")