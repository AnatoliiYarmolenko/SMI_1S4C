from sklearn.datasets import load_iris

# Завантаження набору даних
iris_dataset = load_iris()

# Вивід ключів словника
print("Ключі iris_dataset:\n", iris_dataset.keys(), "\n")

# Вивід частини опису набору даних
print("Опис набору даних (початок):\n")
print(iris_dataset['DESCR'][:193] + "\n...")

# Вивід назв відповідей (сортів ірисів)
print("\nНазви відповідей (target names):")
print(iris_dataset['target_names'])

# Вивід назв ознак (feature names)
print("\nНазви ознак (feature names):")
print(iris_dataset['feature_names'])

# Вивід типу даних і форми масиву data
print("\nТип масиву data:", type(iris_dataset['data']))
print("Форма масиву data:", iris_dataset['data'].shape)

# Вивід перших 5 рядків даних (ознаки перших 5 ірисів)
print("\nПерші 5 прикладів (ознаки):")
print(iris_dataset['data'][:5])

# Вивід типу масиву target
print("\nТип масиву target:", type(iris_dataset['target']))

# Вивід відповідей (міток класів)
print("\nВідповіді (мітки класів):")
print(iris_dataset['target'])

# Розшифровка класів
print("\nРозшифровка міток класів:")
for i, name in enumerate(iris_dataset['target_names']):
    print(f"{i} - {name}")
