

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# import pickle
# import os

# # 1. Load Dataset
# data = pd.read_csv('dataset/diabetes.csv')

# # 2. Pisahkan fitur (X) dan label (y)
# X = data.drop('Outcome', axis=1)  # Semua kolom kecuali 'Outcome'
# y = data['Outcome']               # Kolom 'Outcome'

# # 3. Split data menjadi training dan testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Buat dan latih model KNN
# knn = KNeighborsClassifier(n_neighbors=5)  # Sesuaikan nilai K sesuai kebutuhan
# knn.fit(X_train, y_train)

# # 5. Simpan model ke file 'models/knn_pickle.pkl'
# output_path = 'models/knn_pickle.pkl'
# os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Pastikan folder ada
# with open(output_path, 'wb') as file:
#     pickle.dump(knn, file)

# print(f"Model KNN berhasil disimpan sebagai '{output_path}'")

# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd

# # Misalnya data pelatihan Anda berada di file CSV atau dataset lainnya
# # Anda bisa mengganti ini dengan cara Anda memuat data
# data = pd.read_csv('dataset/diabetes.csv') # Ganti dengan path dataset Anda
# X_train = data.drop('target', axis=1)  # Fitur-fitur (X) tanpa kolom target
# y_train = data['target']  # Target atau label (y)

# # Melatih scaler dan model
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)  # Menormalisasi fitur

# # Membuat dan melatih model KNN
# model = KNeighborsClassifier()
# model.fit(X_train_scaled, y_train)

# # Menyimpan model dan scaler ke file
# with open('models/knn_pickle.pkl', 'wb') as f:
#     pickle.dump(model, f)

# with open('models/scaler_pickle.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# print("Model dan scaler telah disimpan.")

import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Membaca dataset
data = pd.read_csv('dataset/diabetes.csv')  # Ganti dengan path dataset Anda

# Cek kolom yang ada untuk memastikan kolom 'Outcome' ada
print(data.columns)

# Pastikan tidak ada missing values dalam dataset
if data.isnull().sum().any():
    print("Terdapat missing values, menghapus baris dengan missing values.")
    data = data.dropna()  # Menghapus baris dengan missing values, Anda bisa menyesuaikan sesuai kebutuhan

# Memisahkan fitur dan target
X_train = data.drop('Outcome', axis=1)  # Ganti 'target' dengan 'Outcome' untuk kolom target
y_train = data['Outcome']  # Kolom target adalah 'Outcome'

# Melatih scaler dan model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Menormalisasi fitur

# Membuat dan melatih model KNN
model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)

# Menyimpan model dan scaler ke file
model_path = 'models/knn_pickle.pkl'
scaler_path = 'models/scaler_pickle.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print("Model dan scaler telah disimpan di:", model_path, "dan", scaler_path)
