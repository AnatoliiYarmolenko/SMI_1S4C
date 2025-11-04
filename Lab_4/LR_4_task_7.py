import numpy as np
import matplotlib.pyplot as plt

# --- 1. Введення даних (Варіант 2) ---
# Експериментально отримані значення X та Y
x_data = np.array([-1, -1, 0, 1, 2, 3])
y_data = np.array([-1, 0, 1, 1, 3, 5])

print(f"Дані X: {x_data}")
print(f"Дані Y: {y_data}\n")

# --- 2. Обчислення сум для методу найменших квадратів ---
n = len(x_data)            # Кількість точок (n)
sum_x = np.sum(x_data)     # Сума X
sum_y = np.sum(y_data)     # Сума Y
sum_x2 = np.sum(x_data**2) # Сума X^2
sum_xy = np.sum(x_data * y_data) # Сума X*Y

print("--- Проміжні розрахунки ---")
print(f"Кількість точок (n): {n}")
print(f"Σx = {sum_x}")
print(f"Σy = {sum_y}")
print(f"Σx^2 = {sum_x2}")
print(f"Σxy = {sum_xy}\n")

# --- 3. Складання та розв'язання системи "нормальних рівнянь" ---
# Ми розв'язуємо систему Ax = B, де x = [β0, β1]
# A = [[n,   sum_x],
#      [sum_x, sum_x2]]
# B = [[sum_y],
#      [sum_xy]]

A = np.array([[n, sum_x],
              [sum_x, sum_x2]])

B = np.array([sum_y, sum_xy])

# Використовуємо np.linalg.solve для розв'язання системи
# beta - це вектор [β0, β1]
try:
    beta = np.linalg.solve(A, B)
    beta_0 = beta[0]
    beta_1 = beta[1]

    print("--- Результати (коефіцієнти) ---")
    print(f"Коефіцієнт β0 (зсув): {beta_0:.3f}")
    print(f"Коефіцієнт β1 (нахил): {beta_1:.3f}")
    print(f"\nРівняння апроксимуючої функції: y = {beta_0:.3f} + {beta_1:.3f}x")

    # --- 4. Побудова графіків ---
    # Створюємо набір точок для побудови лінії регресії
    # Беремо трохи ширший діапазон, ніж min і max x_data
    x_line = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 100)
    # Розраховуємо y для цих точок за нашим рівнянням
    y_line = beta_0 + beta_1 * x_line

    # Налаштування графіка
    plt.figure(figsize=(10, 6))
    
    # 1. Графік експериментальних точок
    plt.scatter(x_data, y_data, color='red', s=100, label='Експериментальні точки (X, Y)')
    
    # 2. Графік апроксимуючої функції (лінії регресії)
    plt.plot(x_line, y_line, color='blue', label=f'Апроксимуюча функція\ny = {beta_0:.3f} + {beta_1:.3f}x')
    
    # Додавання підписів та легенди
    plt.title('Лінійна регресія за методом найменших квадратів (Варіант 2)')
    plt.xlabel('Величина X')
    plt.ylabel('Величина Y')
    plt.legend()
    plt.grid(True)
    
    # Показати графік
    plt.show()

except np.linalg.LinAlgError:
    print("Помилка: неможливо розв'язати систему рівнянь. Можливо, матриця A є сингулярною.")