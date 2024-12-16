import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin

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
            raise ValueError("metrik salah")

data = pd.read_csv('dataset/diabetes.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_resampled, y_resampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=y[y == 0].shape[0], random_state=42)
X_balanced = np.vstack((X[y == 0], X_resampled))
y_balanced = np.hstack((y[y == 0], y_resampled))

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

print(f"Jumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(CustomKNN(), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

report = classification_report(y_test, y_pred, output_dict=True)
f1_score_class_0 = report['0']['f1-score']
f1_score_class_1 = report['1']['f1-score']
precision_class_0 = report['0']['precision']
precision_class_1 = report['1']['precision']
recall_class_0 = report['0']['recall']
recall_class_1 = report['1']['recall']

print(f"F1-score untuk kelas 0: {f1_score_class_0:.2f}")
print(f"F1-score untuk kelas 1: {f1_score_class_1:.2f}")
print(f"Precision untuk kelas 0: {precision_class_0:.2f}")
print(f"Precision untuk kelas 1: {precision_class_1:.2f}")
print(f"Recall untuk kelas 0: {recall_class_0:.2f}")
print(f"Recall untuk kelas 1: {recall_class_1:.2f}")

print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")

