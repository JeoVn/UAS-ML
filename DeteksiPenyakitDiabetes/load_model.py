import pickle
import os
model_paths = {
    'knn_pickle': 'models/knn_pickle.pkl',
    'knn_best': 'models/knn_best.pkl'
}

loaded_models = {}

for model_name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, 'rb') as file:
            loaded_models[model_name] = pickle.load(file)
        print(f"Model {model_name} berhasil dimuat!")
    else:
        print(f"File model {model_name}.pkl tidak ditemukan di path: {path}")

if 'knn_best' in loaded_models:
    print("Model terbaik :", loaded_models['knn_best'])

if 'knn_pickle' in loaded_models:
    print("Model biasa :", loaded_models['knn_pickle'])
