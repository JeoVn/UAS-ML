

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

# 1. Load Dataset
data = pd.read_csv('dataset/diabetes.csv')

# 2. Pisahkan fitur (X) dan label (y)
X = data.drop('Outcome', axis=1)  # Semua kolom kecuali 'Outcome'
y = data['Outcome']               # Kolom 'Outcome'

# 3. Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Buat dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Sesuaikan nilai K sesuai kebutuhan
knn.fit(X_train, y_train)

# 5. Simpan model ke file 'models/knn_pickle.pkl'
output_path = 'models/knn_pickle.pkl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Pastikan folder ada
with open(output_path, 'wb') as file:
    pickle.dump(knn, file)

print(f"Model KNN berhasil disimpan sebagai '{output_path}'")



