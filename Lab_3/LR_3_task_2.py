import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Визначення Всесвітів (Діапазонів) ---

# Вхідні змінні
# Припустимо, температура - це відхилення від "комфортної" (0)
# від -10 (дуже холодно) до +10 (дуже тепло)
universe_temp = np.arange(-10, 11, 1)
# Швидкість зміни: від -5 (швидко холоднішає) до +5 (швидко теплішає)
universe_speed = np.arange(-5, 6, 0.5)

# Вихідна змінна
# Дія регулятора: від -100 (макс. холод) до +100 (макс. тепло)
universe_action = np.arange(-100, 101, 10)

# --- 2. Створення Вхідних та Вихідних Змінних ---
temperature = ctrl.Antecedent(universe_temp, 'temperature')
speed = ctrl.Antecedent(universe_speed, 'speed')
action = ctrl.Consequent(universe_action, 'action')

# --- 3. Визначення Функцій Належності (Terms) ---

# Temperature Terms: 5 термінів
temperature['very_cold'] = fuzz.trapmf(temperature.universe, [-10, -10, -8, -6])
temperature['cold'] = fuzz.trimf(temperature.universe, [-7, -5, -3])
temperature['normal'] = fuzz.trimf(temperature.universe, [-4, 0, 4])
temperature['hot'] = fuzz.trimf(temperature.universe, [3, 5, 7])
temperature['very_hot'] = fuzz.trapmf(temperature.universe, [6, 8, 10, 10])

# Speed Terms: 3 терміни
speed['negative'] = fuzz.trapmf(speed.universe, [-5, -5, -2, 0])
speed['zero'] = fuzz.trimf(speed.universe, [-1, 0, 1])
speed['positive'] = fuzz.trapmf(speed.universe, [0, 2, 5, 5])

# Action Terms: 5 термінів
# Вліво = 'холод' (негативні значення), Вправо = 'тепло' (позитивні)
action['LL'] = fuzz.trapmf(action.universe, [-100, -100, -80, -60]) # Великий вліво
action['SL'] = fuzz.trimf(action.universe, [-70, -50, -30])      # Невеликий вліво
action['Off'] = fuzz.trimf(action.universe, [-20, 0, 20])       # Вимкнути
action['SR'] = fuzz.trimf(action.universe, [30, 50, 70])       # Невеликий вправо
action['LR'] = fuzz.trapmf(action.universe, [60, 80, 100, 100]) # Великий вправо

# Перегляд функцій належності
# temperature.view()
# speed.view()
# action.view()

# --- 4. Визначення Експертних Правил  ---
# (LL: великий вліво, SL: невеликий вліво, Off: вимкнути, SR: невеликий вправо, LR: великий вправо)

rule1 = ctrl.Rule(temperature['very_hot'] & speed['positive'], action['LL'])
rule2 = ctrl.Rule(temperature['very_hot'] & speed['negative'], action['SL'])
rule3 = ctrl.Rule(temperature['hot'] & speed['positive'], action['LL'])
rule4 = ctrl.Rule(temperature['hot'] & speed['negative'], action['Off'])
rule5 = ctrl.Rule(temperature['very_cold'] & speed['negative'], action['LR'])
rule6 = ctrl.Rule(temperature['very_cold'] & speed['positive'], action['SR'])
#rule7 = ctrl.Rule(temperature['cold'] & speed['negative'], action['LL']) #див. аналіз
rule7 = ctrl.Rule(temperature['cold'] & speed['negative'], action['LR']) #див. аналіз

rule8 = ctrl.Rule(temperature['cold'] & speed['positive'], action['Off'])
rule9 = ctrl.Rule(temperature['very_hot'] & speed['zero'], action['LL'])
rule10 = ctrl.Rule(temperature['hot'] & speed['zero'], action['SL'])
rule11 = ctrl.Rule(temperature['very_cold'] & speed['zero'], action['LR'])
rule12 = ctrl.Rule(temperature['cold'] & speed['zero'], action['SR'])
rule13 = ctrl.Rule(temperature['normal'] & speed['positive'], action['SL'])
rule14 = ctrl.Rule(temperature['normal'] & speed['negative'], action['SR'])
rule15 = ctrl.Rule(temperature['normal'] & speed['zero'], action['Off'])

# --- 5. Створення Системи Керування ---
ac_control_system = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
     rule10, rule11, rule12, rule13, rule14, rule15]
)
ac_simulation = ctrl.ControlSystemSimulation(ac_control_system)

# --- 6. Запуск Симуляції (Приклади) ---

# Приклад 1: "Дуже тепло" (+8) і "стає тепліше" (+3)
ac_simulation.input['temperature'] = 8
ac_simulation.input['speed'] = 3
ac_simulation.compute()
print("--- Приклад 1: Дуже тепло (+8), стає тепліше (+3) ---")
print(f"Дія регулятора: {ac_simulation.output['action']:.2f} (Очікується: 'Великий вліво')")

# Приклад 2: "Нормальна" (0) і "стає холодніше" (-2)
ac_simulation.input['temperature'] = 0
ac_simulation.input['speed'] = -2
ac_simulation.compute()
print("\n--- Приклад 2: Нормальна (0), стає холодніше (-2) ---")
print(f"Дія регулятора: {ac_simulation.output['action']:.2f} (Очікується: 'Невеликий вправо')")

# Приклад 3: "Дуже холодно" (-9) і "стає холодніше" (-4)
ac_simulation.input['temperature'] = -9
ac_simulation.input['speed'] = -4
ac_simulation.compute()
print("\n--- Приклад 3: Дуже холодно (-9), стає холодніше (-4) ---")
print(f"Дія регулятора: {ac_simulation.output['action']:.2f} (Очікується: 'Великий вправо')")

# --- 7. Побудова 3D Поверхні Керування ---
temp_range = np.linspace(universe_temp.min(), universe_temp.max(), 30)
speed_range = np.linspace(universe_speed.min(), universe_speed.max(), 30)
temp_grid, speed_grid = np.meshgrid(temp_range, speed_range)

action_output = np.zeros_like(temp_grid)

for i in range(temp_grid.shape[0]):
    for j in range(temp_grid.shape[1]):
        ac_simulation.input['temperature'] = temp_grid[i, j]
        ac_simulation.input['speed'] = speed_grid[i, j]
        
        try:
            ac_simulation.compute()
            action_output[i, j] = ac_simulation.output['action']
        except:
            action_output[i, j] = 0

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(temp_grid, speed_grid, action_output, cmap='coolwarm')

ax.set_xlabel('Temperature (Відхилення від норми)')
ax.set_ylabel('Speed (Швидкість зміни)')
ax.set_zlabel('Action (Дія регулятора)')
ax.set_title('Air Conditioner Control Surface')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()