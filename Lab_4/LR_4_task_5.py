# --------------------------------------------------------
# Побудова лінійної та поліноміальної регресії (варіант 2)
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --------------------------------------------------------
# 1. Генерація випадкових даних
# --------------------------------------------------------
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X**2 + X + 2 + np.random.randn(m, 1)  # модель з шумом

# --------------------------------------------------------
# 2. Лінійна регресія
# --------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

print("\nЛінійна регресія:")
print("Коефіцієнт:", lin_reg.coef_)
print("Вільний член:", lin_reg.intercept_)
print("R2 =", round(r2_score(y, y_lin_pred), 3))
print("MAE =", round(mean_absolute_error(y, y_lin_pred), 3))
print("MSE =", round(mean_squared_error(y, y_lin_pred), 3))

# --------------------------------------------------------
# 3. Поліноміальна регресія (ступінь 2)
# --------------------------------------------------------
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("\nЗначення X[0]:", X[0])
print("Відповідне X_poly[0]:", X_poly[0])

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_poly_pred = lin_reg_poly.predict(X_poly)

print("\nПоліноміальна регресія (ступінь 2):")
print("Вільний член:", lin_reg_poly.intercept_)
print("Коефіцієнти:", lin_reg_poly.coef_)
print("R2 =", round(r2_score(y, y_poly_pred), 3))
print("MAE =", round(mean_absolute_error(y, y_poly_pred), 3))
print("MSE =", round(mean_squared_error(y, y_poly_pred), 3))

# --------------------------------------------------------
# 4. Побудова графіків
# --------------------------------------------------------
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
y_lin_new = lin_reg.predict(X_new)
X_new_poly = poly_features.transform(X_new)
y_poly_new = lin_reg_poly.predict(X_new_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Вихідні дані')
plt.plot(X_new, y_lin_new, color='red', linewidth=2, label='Лінійна регресія')
plt.plot(X_new, y_poly_new, color='green', linewidth=2, label='Поліноміальна регресія (2 ступінь)')
plt.title('Порівняння лінійної та поліноміальної регресії')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------------
# 5. Математичні рівняння моделей
# --------------------------------------------------------
print("\nМодельні дані (справжнє рівняння): y = 0.6 * X^2 + X + 2 + шум")
print("Отримана модель лінійної регресії: y = {:.3f} * X + {:.3f}".format(float(lin_reg.coef_), float(lin_reg.intercept_)))
print("Отримана модель поліноміальної регресії: y = {:.4f} * X^2 + {:.4f} * X + {:.4f}".format(
    lin_reg_poly.coef_[0,1], lin_reg_poly.coef_[0,0], float(lin_reg_poly.intercept_)))
