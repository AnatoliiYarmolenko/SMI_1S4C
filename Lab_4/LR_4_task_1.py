# Імпорт необхідних бібліотек
import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# -----------------------------
# 1. Завантаження вхідних даних
# -----------------------------
input_file = 'Lab_4/data_singlevar_regr.txt'

# Файл має формат CSV з комами
data = np.loadtxt(input_file, delimiter=',')

# Поділ на X (вхідна змінна) і y (вихідна змінна)
X, y = data[:, :-1], data[:, -1]

# -----------------------------
# 2. Розбиття даних на навчальні та тестові
# -----------------------------
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# -----------------------------
# 3. Створення та навчання моделі
# -----------------------------
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# -----------------------------
# 4. Прогнозування результатів
# -----------------------------
y_test_pred = regressor.predict(X_test)

# -----------------------------
# 5. Побудова графіка
# -----------------------------
plt.scatter(X_test, y_test, color='green', label='Справжні дані')
plt.plot(X_test, y_test_pred, color='black', linewidth=3, label='Прогноз моделі')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Лінійна регресія (одна змінна)')
plt.show()

# -----------------------------
# 6. Оцінка якості моделі
# -----------------------------
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# -----------------------------
# 7. Збереження моделі у файл
# -----------------------------
output_model_file = 'model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

print("\nМодель збережено у файл:", output_model_file)

# -----------------------------
# 8. Завантаження моделі з файлу
# -----------------------------
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Перевірка завантаженої моделі
y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
