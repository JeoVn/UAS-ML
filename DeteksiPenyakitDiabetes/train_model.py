

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

# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd

# # Membaca dataset
# data = pd.read_csv('dataset/diabetes.csv')  # Ganti dengan path dataset Anda

# # Cek kolom yang ada untuk memastikan kolom 'Outcome' ada
# print(data.columns)

# # Pastikan tidak ada missing values dalam dataset
# if data.isnull().sum().any():
#     print("Terdapat missing values, menghapus baris dengan missing values.")
#     data = data.dropna()  # Menghapus baris dengan missing values, Anda bisa menyesuaikan sesuai kebutuhan

# # Memisahkan fitur dan target
# X_train = data.drop('Outcome', axis=1)  # Ganti 'target' dengan 'Outcome' untuk kolom target
# y_train = data['Outcome']  # Kolom target adalah 'Outcome'

# # Melatih scaler dan model
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)  # Menormalisasi fitur

# # Membuat dan melatih model KNN
# model = KNeighborsClassifier()
# model.fit(X_train_scaled, y_train)

# # Menyimpan model dan scaler ke file
# model_path = 'models/knn_pickle.pkl'
# scaler_path = 'models/scaler_pickle.pkl'

# with open(model_path, 'wb') as f:
#     pickle.dump(model, f)

# with open(scaler_path, 'wb') as f:
#     pickle.dump(scaler, f)

# print("Model dan scaler telah disimpan di:", model_path, "dan", scaler_path)

import pickle
import pandas as pd
import numpy as np

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Implementasi KNN manual
class KNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        # Hitung jarak ke semua titik dalam data training
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Ambil indeks k tetangga terdekat
        k_indices = np.argsort(distances)[:self.n_neighbors]
        
        # Ambil label dari k tetangga terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return label yang paling sering muncul
        return np.bincount(k_nearest_labels).argmax()

# Membaca dataset
data = pd.read_csv('dataset/diabetes.csv')  # Ganti dengan path dataset Anda

# Cek kolom yang ada untuk memastikan kolom 'Outcome' ada
print(data.columns)

# Pastikan tidak ada missing values dalam dataset
if data.isnull().sum().any():
    print("Terdapat missing values, menghapus baris dengan missing values.")
    data = data.dropna()  # Menghapus baris dengan missing values, Anda bisa menyesuaikan sesuai kebutuhan

# Memisahkan fitur dan target
X_train = data.drop('Outcome', axis=1).values  # Ganti 'target' dengan 'Outcome' untuk kolom target
y_train = data['Outcome'].values  # Kolom target adalah 'Outcome'

# Melatih scaler
class StandardScalerManual:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Normalisasi fitur
scaler = StandardScalerManual()
X_train_scaled = scaler.fit_transform(X_train)  # Menormalisasi fitur

# Membuat dan melatih model KNN manual
model = KNN(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Menyimpan model dan scaler ke file
model_path = 'models/knn_manual.pkl'
scaler_path = 'models/scaler_manual.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print("Model dan scaler telah disimpan di:", model_path, "dan", scaler_path)
