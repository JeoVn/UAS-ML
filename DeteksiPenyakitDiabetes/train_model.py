

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

# import pickle
# import pandas as pd
# import numpy as np

# # Fungsi untuk menghitung jarak Euclidean
# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))

# # Implementasi KNN manual
# class KNN:
#     def __init__(self, n_neighbors=3):
#         self.n_neighbors = n_neighbors
    
#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y
    
#     def predict(self, X):
#         predictions = [self._predict_single(x) for x in X]
#         return np.array(predictions)
    
#     def _predict_single(self, x):
#         # Hitung jarak ke semua titik dalam data training
#         distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
#         # Ambil indeks k tetangga terdekat
#         k_indices = np.argsort(distances)[:self.n_neighbors]
        
#         # Ambil label dari k tetangga terdekat
#         k_nearest_labels = [self.y_train[i] for i in k_indices]
        
#         # Return label yang paling sering muncul
#         return np.bincount(k_nearest_labels).argmax()

# # Membaca dataset
# data = pd.read_csv('dataset/diabetes.csv')  # Ganti dengan path dataset Anda

# # Cek kolom yang ada untuk memastikan kolom 'Outcome' ada
# print(data.columns)

# # Pastikan tidak ada missing values dalam dataset
# if data.isnull().sum().any():
#     print("Terdapat missing values, menghapus baris dengan missing values.")
#     data = data.dropna()  # Menghapus baris dengan missing values, Anda bisa menyesuaikan sesuai kebutuhan

# # Memisahkan fitur dan target
# X_train = data.drop('Outcome', axis=1).values  # Ganti 'target' dengan 'Outcome' untuk kolom target
# y_train = data['Outcome'].values  # Kolom target adalah 'Outcome'

# # Melatih scaler
# class StandardScalerManual:
#     def __init__(self):
#         self.mean = None
#         self.std = None
    
#     def fit(self, X):
#         self.mean = X.mean(axis=0)
#         self.std = X.std(axis=0)
    
#     def transform(self, X):
#         return (X - self.mean) / self.std
    
#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)

# # Normalisasi fitur
# scaler = StandardScalerManual()
# X_train_scaled = scaler.fit_transform(X_train)  # Menormalisasi fitur

# # Membuat dan melatih model KNN manual
# model = KNN(n_neighbors=3)
# model.fit(X_train_scaled, y_train)

# # Menyimpan model dan scaler ke file
# model_path = 'models/knn_manual.pkl'
# scaler_path = 'models/scaler_manual.pkl'

# with open(model_path, 'wb') as f:
#     pickle.dump(model, f)

# with open(scaler_path, 'wb') as f:
#     pickle.dump(scaler, f)

# print("Model dan scaler telah disimpan di:", model_path, "dan", scaler_path)

# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Fungsi untuk menghitung jarak Euclidean
# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))

# # Implementasi KNN manual
# class KNN:
#     def __init__(self, n_neighbors=3):
#         self.n_neighbors = n_neighbors
    
#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y
    
#     def predict(self, X):
#         predictions = [self._predict_single(x) for x in X]
#         return np.array(predictions)
    
#     def _predict_single(self, x):
#         # Hitung jarak ke semua titik dalam data training
#         distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
#         # Ambil indeks k tetangga terdekat
#         k_indices = np.argsort(distances)[:self.n_neighbors]
        
#         # Ambil label dari k tetangga terdekat
#         k_nearest_labels = [self.y_train[i] for i in k_indices]
        
#         # Return label yang paling sering muncul
#         return np.bincount(k_nearest_labels).argmax()

# # Membaca dataset
# data = pd.read_csv('dataset/diabetes.csv')  # Ganti dengan path dataset Anda

# # Pastikan tidak ada missing values dalam dataset
# if data.isnull().sum().any():
#     print("Terdapat missing values, menghapus baris dengan missing values.")
#     data = data.dropna()  # Menghapus baris dengan missing values, Anda bisa menyesuaikan sesuai kebutuhan

# # Memisahkan fitur dan target
# X = data.drop('Outcome', axis=1).values  # Fitur
# y = data['Outcome'].values  # Target

# # Split dataset menjadi data train dan test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalisasi fitur
# class StandardScalerManual:
#     def __init__(self):
#         self.mean = None
#         self.std = None
    
#     def fit(self, X):
#         self.mean = X.mean(axis=0)
#         self.std = X.std(axis=0)
    
#     def transform(self, X):
#         return (X - self.mean) / self.std
    
#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)

# scaler = StandardScalerManual()
# X_train_scaled = scaler.fit_transform(X_train)  # Menormalisasi fitur training

# # Membuat dan melatih model KNN manual
# model = KNN(n_neighbors=3)
# model.fit(X_train_scaled, y_train)

# # Menormalisasi data uji
# X_test_scaled = scaler.transform(X_test)  # Menormalisasi fitur testing

# # Melakukan prediksi pada data uji
# y_pred = model.predict(X_test_scaled)

# # Menghitung akurasi
# accuracy = accuracy_score(y_test, y_pred)

# # Menampilkan akurasi dalam bentuk persentase
# print(f"Akurasi Model: {accuracy * 100:.2f}%")

# # Menyimpan model dan scaler ke file
# model_path = 'models/knn_manual.pkl'
# scaler_path = 'models/scaler_manual.pkl'

# with open(model_path, 'wb') as f:
#     pickle.dump(model, f)

# with open(scaler_path, 'wb') as f:
#     pickle.dump(scaler, f)

# print("Model dan scaler telah disimpan di:", model_path, "dan", scaler_path)

# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# # Membaca dataset
# data = pd.read_csv('dataset/diabetes.csv')

# # Pastikan tidak ada missing values dalam dataset
# if data.isnull().sum().any():
#     print("Terdapat missing values, menghapus baris dengan missing values.")
#     data = data.dropna()

# # Memisahkan fitur dan target
# X = data.drop('Outcome', axis=1).values  # Fitur
# y = data['Outcome'].values  # Target

# # Split dataset menjadi data train dan test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalisasi fitur
# class StandardScalerManual:
#     def __init__(self):
#         self.mean = None
#         self.std = None
    
#     def fit(self, X):
#         self.mean = X.mean(axis=0)
#         self.std = X.std(axis=0)
    
#     def transform(self, X):
#         return (X - self.mean) / self.std
    
#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)

# scaler = StandardScalerManual()
# X_train_scaled = scaler.fit_transform(X_train)  # Menormalisasi fitur training

# # Hyperparameter Tuning menggunakan GridSearchCV
# param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'metric': ['euclidean', 'manhattan', 'minkowski']}
# knn = KNeighborsClassifier()

# # Melakukan GridSearchCV untuk tuning hyperparameter
# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_scaled, y_train)

# # Menampilkan parameter terbaik
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score (Accuracy):", grid_search.best_score_)

# # Menggunakan parameter terbaik untuk model final
# best_knn = grid_search.best_estimator_

# # Melakukan normalisasi pada data uji
# X_test_scaled = scaler.transform(X_test)

# # Melakukan prediksi pada data uji
# y_pred = best_knn.predict(X_test_scaled)

# # Menghitung akurasi
# accuracy = accuracy_score(y_test, y_pred)

# # Menampilkan akurasi dalam bentuk persentase
# print(f"Akurasi Model: {accuracy * 100:.2f}%")

# # Menyimpan model dan scaler ke file
# model_path = 'models/knn_manual_best.pkl'
# scaler_path = 'models/scaler_manual.pkl'

# with open(model_path, 'wb') as f:
#     pickle.dump(best_knn, f)

# with open(scaler_path, 'wb') as f:
#     pickle.dump(scaler, f)

# print("Model dan scaler telah disimpan di:", model_path, "dan", scaler_path)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin

# Custom KNN Implementation
class CustomKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return max(set(k_nearest_labels), key=k_nearest_labels.count)

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported metric")

# Load and preprocess the dataset
data = pd.read_csv('dataset/diabetes.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Balance dataset with oversampling
X_resampled, y_resampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=y[y == 0].shape[0], random_state=42)
X_balanced = np.vstack((X[y == 0], X_resampled))
y_balanced = np.hstack((y[y == 0], y_resampled))

# Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# 3. Split data into training and test sets
train_ratio = 0.8  # 80% data untuk training
# untuk balance akurasi dan cross vali nya
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Menampilkan jumlah data training dan data testing
print(f"Jumlah data training: {len(X_train)} ({train_ratio * 100:.0f}%)")
print(f"Jumlah data testing: {len(X_test)} ({(1 - train_ratio) * 100:.0f}%)")

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(CustomKNN(), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate best model
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Cross-validation score
cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from imblearn.over_sampling import SMOTE

# # 1. Load dataset
# data = pd.read_csv('dataset/diabetes.csv')
# X = data.iloc[:, :-1]  # Semua kolom kecuali target
# y = data.iloc[:, -1]  # Kolom target

# # 2. Handle missing values
# X.fillna(X.median(), inplace=True)  # Ganti nilai kosong dengan median

# # 3. Balance dataset (SMOTE)
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # 4. Split data into training and test sets
# train_ratio = 0.8  # 80% data untuk training
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=(1 - train_ratio), random_state=42)

# # Print dataset split info
# print(f"Jumlah data training: {len(X_train)} ({train_ratio * 100:.0f}%)")
# print(f"Jumlah data testing: {len(X_test)} ({(1 - train_ratio) * 100:.0f}%)")

# # 5. Normalize data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # 6. Hyperparameter tuning using GridSearchCV
# param_grid = {
#     'n_neighbors': range(1, 31),  # Uji nilai k dari 1 hingga 30
#     'weights': ['uniform', 'distance'],  # Coba bobot uniform dan distance
#     'metric': ['euclidean', 'manhattan', 'minkowski']  # Jarak yang berbeda
# }

# knn = KNeighborsClassifier()
# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # 7. Best parameters and evaluation
# best_knn = grid_search.best_estimator_
# print("Best Parameters:", grid_search.best_params_)

# # Cross-validation score
# cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5, scoring='accuracy')
# print(f"Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")

# # Evaluate on test set
# y_pred = best_knn.predict(X_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Calculate and display test accuracy
# test_accuracy = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# # 8. Save the model (optional)
# import joblib
# joblib.dump(best_knn, "optimized_knn_model.pkl")
