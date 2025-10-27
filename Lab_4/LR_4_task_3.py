# --------------------------------------------------------
# Багатовимірна лінійна та поліноміальна регресія
# --------------------------------------------------------

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# --------------------------------------------------------
# 1. Завантаження вхідних даних
# --------------------------------------------------------
input_file = 'Lab_4/data_multivar_regr.txt'  # файл із даними
data = np.loadtxt(input_file, delimiter=',')

# Поділ на вхідні змінні (X) і вихідну (y)
X, y = data[:, :-1], data[:, -1]

# --------------------------------------------------------
# 2. Розбиття даних на навчальні та тестові
# --------------------------------------------------------
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# --------------------------------------------------------
# 3. Лінійна регресія
# --------------------------------------------------------
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

# Прогнозування
y_test_pred = linear_regressor.predict(X_test)

# --------------------------------------------------------
# 4. Оцінка якості лінійного регресора
# --------------------------------------------------------
print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# --------------------------------------------------------
# 5. Поліноміальна регресія (ступінь 10)
# --------------------------------------------------------
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# --------------------------------------------------------
# 6. Перевірка на вибірковій точці
# --------------------------------------------------------
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

print("\nLinear regression prediction:", linear_regressor.predict(datapoint))
print("Polynomial regression prediction:", poly_linear_model.predict(poly_datapoint))