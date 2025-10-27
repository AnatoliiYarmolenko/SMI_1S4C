import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# 1. Завантаження вхідних даних
# -----------------------------
input_file = 'Lab_4/data_regr_2.txt'

# Файл має формат CSV з комами
data = np.loadtxt(input_file, delimiter=',')

X, y = data[:, :-1], data[:, -1]

# Побудова моделі лінійної регресії
model = LinearRegression()
model.fit(X, y)

# Отримання параметрів моделі
a = model.coef_[0]
b = model.intercept_

# Прогноз
y_pred = model.predict(X)

# Оцінка якості
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# Вивід результатів
print(f"Рівняння регресії: y = {a:.3f}x + {b:.3f}")
print(f"Коефіцієнт детермінації R²: {r2:.3f}")
print(f"Середньоквадратична помилка (MSE): {mse:.3f}")

# Побудова графіка
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_pred, color='red', linewidth=2, label='Регресійна пряма')
plt.title('Лінійна регресія (одна змінна)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
