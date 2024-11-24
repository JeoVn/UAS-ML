import sklearn
print(sklearn.__version__)

from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import os

# app = Flask(__name__)

# # Tentukan path absolut ke file model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models','knn_pickle.pkl')

# @app.route('/', methods=['POST', 'GET'])
# def index():
#     if request.method == 'POST':
#         try:
#             # Cek apakah file model ada
#             if not os.path.exists(MODEL_PATH):
#                 return render_template('index.html', error="File model 'knn_pickle.pkl' tidak ditemukan. Pastikan file berada di lokasi yang benar.")

#             # Load model
#             with open(MODEL_PATH, 'rb') as r:
#                 model = pickle.load(r)
#         except Exception as e:
#             return render_template('index.html', error=f"Terjadi kesalahan saat memuat model: {e}")

#         try:
#             # Ambil data dari form dan pastikan validitas input
#             melahirkan = float(request.form['melahirkan'])
#             glukosa = float(request.form['glukosa'])
#             darah = float(request.form['darah'])
#             kulit = float(request.form['kulit'])
#             insulin = float(request.form['insulin'])
#             bmi = float(request.form['bmi'])
#             riwayat = float(request.form['riwayat_diabetes'])  # Adjusted to match the form field name
#             umur = float(request.form['umur'])

#             # Buat array untuk prediksi
#             datas = np.array([melahirkan, glukosa, darah, kulit, insulin, bmi, riwayat, umur])
#             datas = datas.reshape(1, -1)

#             # Prediksi menggunakan model
#             isDiabetes = model.predict(datas)

#             # Format hasil prediksi untuk ditampilkan
#             result = "Diabetes" if isDiabetes[0] == 1 else "Tidak Diabetes"
#             return render_template('index.html', result=result)  # Pass the result to the template
#         except ValueError:
#             return render_template('index.html', error="Input tidak valid. Pastikan semua data yang dimasukkan adalah angka.")
#         except Exception as e:
#             return render_template('index.html', error=f"Terjadi kesalahan saat melakukan prediksi: {e}")
#     else:
#         # Tampilkan form input
#         return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=True)


import pickle
import numpy as np
import os
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Tentukan path absolut ke file model dan scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'knn_pickle.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler_pickle.pkl')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Cek apakah file model dan scaler ada
            if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
                return render_template('index.html', error="File model atau scaler tidak ditemukan. Pastikan file berada di lokasi yang benar.")

            # Load model
            with open(MODEL_PATH, 'rb') as r:
                model = pickle.load(r)

            # Load scaler
            with open(SCALER_PATH, 'rb') as s:
                scaler = pickle.load(s)
                
        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan saat memuat model atau scaler: {e}")

        try:
            # Ambil data dari form dan pastikan validitas input
            melahirkan = float(request.form['melahirkan'])
            glukosa = float(request.form['glukosa'])
            darah = float(request.form['darah'])
            kulit = float(request.form['kulit'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            riwayat = float(request.form['riwayat_diabetes'])  # Adjusted to match the form field name
            umur = float(request.form['umur'])

            # Buat array untuk prediksi
            datas = np.array([melahirkan, glukosa, darah, kulit, insulin, bmi, riwayat, umur])
            datas = datas.reshape(1, -1)

            # Transformasi data menggunakan scaler
            datas_scaled = scaler.transform(datas)

            # Prediksi menggunakan model
            isDiabetes = model.predict(datas_scaled)

            # Format hasil prediksi untuk ditampilkan
            result = "Diabetes" if isDiabetes[0] == 1 else "Tidak Diabetes"
            return render_template('index.html', result=result)  # Pass the result to the template
        except ValueError:
            return render_template('index.html', error="Input tidak valid. Pastikan semua data yang dimasukkan adalah angka.")
        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan saat melakukan prediksi: {e}")
    else:
        # Tampilkan form input
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)



# import sklearn
# print(sklearn.__version__)

# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import os

# app = Flask(__name__)

# # Tentukan path absolut ke file model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models','knn_pickle.pkl')

# # Tentukan path absolut ke file model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'knn_pickle.pkl')

# @app.route('/', methods=['POST', 'GET'])
# def index():
#     result = None
#     error = None

#     if request.method == 'POST':
#         try:
#             # Cek apakah file model ada
#             if not os.path.exists(MODEL_PATH):
#                 error = "File model 'knn_pickle.pkl' tidak ditemukan. Pastikan file berada di lokasi yang benar."
#                 return render_template('index.html', result=result, error=error)

#             # Load model
#             with open(MODEL_PATH, 'rb') as r:
#                 model = pickle.load(r)
#         except Exception as e:
#             error = f"Terjadi kesalahan saat memuat model: {e}"
#             return render_template('index.html', result=result, error=error)

#         try:
#             # Ambil data dari form
#             melahirkan = float(request.form['melahirkan'])
#             glukosa = float(request.form['glukosa'])
#             darah = float(request.form['darah'])
#             kulit = float(request.form['kulit'])
#             insulin = float(request.form['insulin'])
#             bmi = float(request.form['bmi'])
#             riwayat = float(request.form['riwayat_diabetes'])  # Perbaiki nama field jika perlu
#             umur = float(request.form['umur'])

#             # Buat array untuk prediksi
#             datas = np.array([melahirkan, glukosa, darah, kulit, insulin, bmi, riwayat, umur])
#             datas = datas.reshape(1, -1)

#             # Prediksi menggunakan model
#             isDiabetes = model.predict(datas)

#             # Tentukan hasil prediksi
#             result = "Diabetes" if isDiabetes[0] == 1 else "Tidak Diabetes"
            
#         except ValueError:
#             error = "Input tidak valid. Pastikan semua data yang dimasukkan adalah angka."
        
#         return render_template('index.html', result=result, error=error)
#     else:
#         return render_template('index.html', result=result, error=error)

# if __name__ == "__main__":
#     app.run(debug=True)
