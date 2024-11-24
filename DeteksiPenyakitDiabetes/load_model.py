# import pickle
# import os

# # Tentukan path ke model
# output_path = 'models/knn_pickle.pkl'

# # Pastikan direktori dan file model ada
# if os.path.exists(output_path):
#     # Muat model yang disimpan
#     with open(output_path, 'rb') as file:
#         model = pickle.load(file)

#     # Tampilkan detail model
#     print("Model berhasil dimuat!")
#     print(model)
# else:
#     print(f"File model tidak ditemukan di path: {output_path}")

import pickle
import os
import numpy as np

# Tentukan path ke model
output_path = 'models/knn_pickle.pkl'

# Pastikan direktori dan file model ada
if os.path.exists(output_path):
    # Muat model yang disimpan
    with open(output_path, 'rb') as file:
        model = pickle.load(file)

    # Tampilkan detail model
    print("Model berhasil dimuat!")
    print(model)
    
    # Contoh prediksi (Anda bisa mengganti ini dengan data input nyata)
    data_input = np.array([6, 148, 72, 35, 0, 33.6, 0.627, 50])  # Misalnya, data seorang pasien
    data_input = data_input.reshape(1, -1)  # Model membutuhkan input dalam bentuk 2D array
    
    # Lakukan prediksi
    prediction = model.predict(data_input)
    
    # Tampilkan hasil prediksi
    print(f"Hasil Prediksi: {'Diabetes' if prediction[0] == 1 else 'Tidak Diabetes'}")

else:
    print(f"File model tidak ditemukan di path: {output_path}")
