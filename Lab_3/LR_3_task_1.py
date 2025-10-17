import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Визначення Всесвітів (Діапазонів) ---

# Вхідні змінні
# Припустимо, температура від 0 до 100 градусів
universe_temp = np.arange(0, 101, 1)
# Припустимо, напір - це умовна шкала від 0 до 10
universe_pressure = np.arange(0, 11, 1)

# Вихідні змінні
# Кут повороту від -90 (макс. вліво) до +90 (макс. вправо) 
universe_angle = np.arange(-90, 91, 1)

# --- 2. Створення Вхідних та Вихідних Змінних ---
temperature = ctrl.Antecedent(universe_temp, 'temperature')
pressure = ctrl.Antecedent(universe_pressure, 'pressure')

hot_tap = ctrl.Consequent(universe_angle, 'hot_tap')
cold_tap = ctrl.Consequent(universe_angle, 'cold_tap')

# --- 3. Визначення Функцій Належності (Membership Functions) ---

# Temperature Terms: 'cold', 'cool', 'warm', 'not_very_hot', 'hot'
temperature['cold'] = fuzz.trapmf(temperature.universe, [0, 0, 10, 25])
temperature['cool'] = fuzz.trimf(temperature.universe, [15, 30, 45])
temperature['warm'] = fuzz.trimf(temperature.universe, [40, 50, 60])
temperature['not_very_hot'] = fuzz.trimf(temperature.universe, [55, 70, 85])
temperature['hot'] = fuzz.trapmf(temperature.universe, [75, 90, 100, 100])

# Pressure Terms: 'weak', 'not_very_strong', 'strong'
pressure['weak'] = fuzz.trapmf(pressure.universe, [0, 0, 2, 4])
pressure['not_very_strong'] = fuzz.trimf(pressure.universe, [3, 5, 7])
pressure['strong'] = fuzz.trapmf(pressure.universe, [6, 8, 10, 10])

# Angle Terms: 'Large Left' (LL), 'Medium Left' (ML), 'Small Left' (SL),
# 'No Change' (NC), 'Small Right' (SR), 'Medium Right' (MR), 'Large Right' (LR)
# "вліво" = negative, "вправо" = positive [cite: 256, 257]
hot_tap['LL'] = fuzz.trimf(hot_tap.universe, [-90, -90, -60])
hot_tap['ML'] = fuzz.trimf(hot_tap.universe, [-75, -45, -15])
hot_tap['SL'] = fuzz.trimf(hot_tap.universe, [-30, -15, 0])
hot_tap['NC'] = fuzz.trimf(hot_tap.universe, [-10, 0, 10])
hot_tap['SR'] = fuzz.trimf(hot_tap.universe, [0, 15, 30])
hot_tap['MR'] = fuzz.trimf(hot_tap.universe, [15, 45, 75])
hot_tap['LR'] = fuzz.trimf(hot_tap.universe, [60, 90, 90])

# Використовуємо ті ж терміни для крана холодної води
cold_tap['LL'] = fuzz.trimf(cold_tap.universe, [-90, -90, -60])
cold_tap['ML'] = fuzz.trimf(cold_tap.universe, [-75, -45, -15])
cold_tap['SL'] = fuzz.trimf(cold_tap.universe, [-30, -15, 0])
cold_tap['NC'] = fuzz.trimf(cold_tap.universe, [-10, 0, 10])
cold_tap['SR'] = fuzz.trimf(cold_tap.universe, [0, 15, 30])
cold_tap['MR'] = fuzz.trimf(cold_tap.universe, [15, 45, 75])
cold_tap['LR'] = fuzz.trimf(cold_tap.universe, [60, 90, 90])

# (Необов'язково) Перегляд функцій належності
# temperature.view()
# pressure.view()
# hot_tap.view()

# --- 4. Визначення Експертних Правил  ---
# (ML: середній вліво, MR: середній вправо, SL: невеликий вліво, SR: невеликий вправо, LR: великий вправо, NC: не змінювати)

rule1 = ctrl.Rule(temperature['hot'] & pressure['strong'],
                  (hot_tap['ML'], cold_tap['MR'])) # 
rule2 = ctrl.Rule(temperature['hot'] & pressure['not_very_strong'],
                  (hot_tap['NC'], cold_tap['MR'])) # [cite: 259]
rule3 = ctrl.Rule(temperature['not_very_hot'] & pressure['strong'],
                  (hot_tap['SL'], cold_tap['NC'])) # [cite: 260]
rule4 = ctrl.Rule(temperature['not_very_hot'] & pressure['weak'],
                  (hot_tap['SR'], cold_tap['SR'])) # [cite: 261]
rule5 = ctrl.Rule(temperature['warm'] & pressure['not_very_strong'],
                  (hot_tap['NC'], cold_tap['NC'])) # 
rule6 = ctrl.Rule(temperature['cool'] & pressure['strong'],
                  (hot_tap['MR'], cold_tap['ML'])) # [cite: 263]
rule7 = ctrl.Rule(temperature['cool'] & pressure['not_very_strong'],
                  (hot_tap['MR'], cold_tap['SL'])) # [cite: 264]
rule8 = ctrl.Rule(temperature['cold'] & pressure['weak'],
                  (hot_tap['LR'], cold_tap['NC'])) # 
rule9 = ctrl.Rule(temperature['cold'] & pressure['strong'],
                  (hot_tap['ML'], cold_tap['MR'])) # [cite: 266]
rule10 = ctrl.Rule(temperature['warm'] & pressure['strong'],
                   (hot_tap['SL'], cold_tap['SL'])) # [cite: 267]
rule11 = ctrl.Rule(temperature['warm'] & pressure['weak'],
                   (hot_tap['SR'], cold_tap['SR'])) # [cite: 268]

# --- 5. Створення Системи Керування ---
tap_control_system = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11]
)
tap_simulation = ctrl.ControlSystemSimulation(tap_control_system)

# --- 6. Запуск Симуляції (Приклад) ---

# Приклад 1: Вода дуже гаряча (95) і напір сильний (9)
tap_simulation.input['temperature'] = 95
tap_simulation.input['pressure'] = 9
tap_simulation.compute()

print("--- Приклад 1: Гаряча вода (95), Сильний напір (9) ---")
print(f"Поворот гарячого крана: {tap_simulation.output['hot_tap']:.2f} градусів")
print(f"Поворот холодного крана: {tap_simulation.output['cold_tap']:.2f} градусів")
# Очікуваний результат: гарячий кран - вліво (негативне), холодний - вправо (позитивне) 

# Приклад 2: Вода холодна (10) і напір слабкий (2)
tap_simulation.input['temperature'] = 10
tap_simulation.input['pressure'] = 2
tap_simulation.compute()

print("\n--- Приклад 2: Холодна вода (10), Слабкий напір (2) ---")
print(f"Поворот гарячого крана: {tap_simulation.output['hot_tap']:.2f} градусів")
print(f"Поворот холодного крана: {tap_simulation.output['cold_tap']:.2f} градусів")
# Очікуваний результат: гарячий кран - сильно вправо (позитивне) 

# --- 7. Побудова 3D Поверхонь Керування (аналог Рис. 7 ) ---

# Готуємо 2D сітку для вхідних даних
temp_range = np.linspace(universe_temp.min(), universe_temp.max(), 30)
pres_range = np.linspace(universe_pressure.min(), universe_pressure.max(), 30)
temp_grid, pres_grid = np.meshgrid(temp_range, pres_range)

# Готуємо 2D сітку для вихідних даних
hot_tap_output = np.zeros_like(temp_grid)
cold_tap_output = np.zeros_like(temp_grid)

# Розрахунок виходів для кожної точки сітки
for i in range(temp_grid.shape[0]):
    for j in range(temp_grid.shape[1]):
        tap_simulation.input['temperature'] = temp_grid[i, j]
        tap_simulation.input['pressure'] = pres_grid[i, j]
        
        try:
            tap_simulation.compute()
            hot_tap_output[i, j] = tap_simulation.output['hot_tap']
            cold_tap_output[i, j] = tap_simulation.output['cold_tap']
        except:
            # Обробка випадків, де правила не спрацьовують
            hot_tap_output[i, j] = 0
            cold_tap_output[i, j] = 0

# Побудова 3D-графіка для гарячого крана
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(temp_grid, pres_grid, hot_tap_output, cmap='viridis')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Pressure (0-10)')
ax1.set_zlabel('Hot Tap Angle (degrees)')
ax1.set_title('Hot Tap Control Surface')
fig1.colorbar(surf1, shrink=0.5, aspect=5)

# Побудова 3D-графіка для холодного крана
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(temp_grid, pres_grid, cold_tap_output, cmap='plasma')
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Pressure (0-10)')
ax2.set_zlabel('Cold Tap Angle (degrees)')
ax2.set_title('Cold Tap Control Surface')
fig2.colorbar(surf2, shrink=0.5, aspect=5)

plt.show()