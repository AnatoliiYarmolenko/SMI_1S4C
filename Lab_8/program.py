import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Створення даних ---
np.random.seed(42)
x_data = np.random.rand(1000, 1).astype(np.float32)
noise = np.random.normal(0, 2, size=x_data.shape)
y_data = 2 * x_data + 1 + noise

# --- 2. Створення простої моделі ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# --- 3. Компіляція моделі ---
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mse')

# --- 4. Навчання ---
history = model.fit(x_data, y_data, epochs=2000, batch_size=100, verbose=0)


# --- 5. Проміжні значення (кожні 100 епох) ---
for i in range(100, 2001, 100):
    loss_val = history.history['loss'][i-1]
    print(f"Епоха {i:4d}: втрата={loss_val:.4f}")
    
# --- 6. Результати ---
weights = model.get_weights()
k, b = weights[0][0][0], weights[1][0]
print(f"Фінальні параметри: k={k:.4f}, b={b:.4f}")


# --- 7. Графік втрат ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.title('Зміна втрат під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Loss')

# --- 8. Графік регресії ---
plt.subplot(1,2,2)
plt.scatter(x_data, y_data, label='Дані')
plt.plot(x_data, model.predict(x_data), color='red', label='Модель')
plt.title('Результат навчання')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
        
plt.tight_layout()
plt.show()
