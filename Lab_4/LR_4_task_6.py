import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Генерація даних (з попереднього завдання) ---
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X ** 2 + X + 4 + np.random.randn(m, 1)

# --- 2. Функція для побудови кривих навчання ---
def plot_learning_curves(model, X, y, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []

    # Навчання на все більших підмножинах даних
    for m in range(5, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    # Побудова графіка
    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="Навчальний набір")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="Перевірочний набір")
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 3. Лінійна модель ---
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання — лінійна регресія")

# --- 4. Поліноміальна модель 10-го ступеня ---
poly_reg_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(poly_reg_10, X, y, "Криві навчання — поліноміальна регресія (ступінь 10)")

# --- 5. Поліноміальна модель 2-го ступеня ---
poly_reg_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(poly_reg_2, X, y, "Криві навчання — поліноміальна регресія (ступінь 2)")
