import pickle
import os
import numpy as np

output_path = 'models/knn_pickle.pkl'

if os.path.exists(output_path):
    with open(output_path, 'rb') as file:
        model = pickle.load(file)
    print("Model berhasil dimuat!")
    print(model)
    
else:
    print(f"File model tidak ditemukan di path: {output_path}")