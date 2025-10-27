# --------------------------------------------------------
# Лінійна регресія на наборі даних про діабет
# --------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# --------------------------------------------------------
# 1. Завантаження даних
# --------------------------------------------------------
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# --------------------------------------------------------
# 2. Розбиття на навчальну та тестову вибірки
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# --------------------------------------------------------
# 3. Створення та навчання моделі лінійної регресії
# --------------------------------------------------------
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# --------------------------------------------------------
# 4. Прогнозування результатів
# --------------------------------------------------------
y_pred = regr.predict(X_test)

# --------------------------------------------------------
# 5. Оцінка якості моделі
# --------------------------------------------------------
print("Коефіцієнти регресії:", regr.coef_)
print("Вільний член (intercept):", regr.intercept_)
print("R2 score:", round(r2_score(y_test, y_pred), 3))
print("Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, y_pred), 3))
print("Mean Squared Error (MSE):", round(mean_squared_error(y_test, y_pred), 3))

# --------------------------------------------------------
# 6. Побудова графіка
# --------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0), label='Передбачені значення')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3, label='Ідеальна відповідність')
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
ax.set_title('Порівняння справжніх і передбачених значень (Діабет)')
ax.legend()
plt.show()
