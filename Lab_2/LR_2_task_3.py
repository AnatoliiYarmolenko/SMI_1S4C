import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# === КРОК 1. ЗАВАНТАЖЕННЯ ТА ВИВЧЕННЯ ДАНИХ ===

# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# --- нижче закоментовано все, що не потрібно для КРОКУ 2 ---
# # Перевірка розміру датасету
# print("Розмір датасету (рядки, стовпці):")
# print(dataset.shape)
# print()
# # Перегляд перших 20 рядків
# print("Перші 20 рядків датасету:")
# print(dataset.head(20))
# print()
# # Статистичне зведення
# print("Статистичне зведення по атрибутах:")
# print(dataset.describe())
# print()
# # Кількість екземплярів у кожному класі
# print("Розподіл за класами:")
# print(dataset.groupby('class').size())

# === КРОК 2: ВІЗУАЛІЗАЦІЯ ДАНИХ ===

# Діаграма розмаху (Boxplot)
#dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
#pyplot.suptitle("Діаграма розмаху атрибутів набору даних Iris")
#pyplot.show()

# Гістограма розподілу
#dataset.hist()
#pyplot.suptitle("Гістограми розподілу атрибутів Iris")
#pyplot.show()

# Матриця діаграм розсіювання (Scatter Matrix)
#scatter_matrix(dataset)
#pyplot.suptitle("Матриця діаграм розсіювання для Iris")
#pyplot.show()

# === КРОК 3. СТВОРЕННЯ НАВЧАЛЬНОГО ТА ТЕСТОВОГО НАБОРІВ ===

# Розділення датасету на ознаки (X) і цільові мітки (y)
array = dataset.values
X = array[:, 0:4]  # перші 4 стовпці — ознаки
y = array[:, 4]    # п’ятий стовпець — клас

# Розділення X і y на навчальну та тестову вибірки
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# Виведемо розміри вибірок для перевірки
#print("Розмір навчальної вибірки:", X_train.shape)
#print("Розмір тестової вибірки:", X_validation.shape)

# === КРОК 4. КЛАСИФІКАЦІЯ (ПОБУДОВА МОДЕЛІ) ===

# Створюємо список моделей для тестування
#models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))

# Оцінка моделей через 10-кратну стратифіковану крос-валідацію
#results = []
#names = []

#print("Оцінка моделей (accuracy, mean ± std):")
#for name, model in models:
 #   kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
  #  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
   # results.append(cv_results)
    #names.append(name)
    #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Порівняння алгоритмів через діаграму розмаху
#pyplot.boxplot(results, labels=names)
#pyplot.title('Порівняння точності алгоритмів класифікації')
#pyplot.ylabel('Accuracy')
#pyplot.show()

# === КРОК 5. ОПТИМІЗАЦІЯ ПАРАМЕТРІВ МОДЕЛІ ===

# === КРОК 6. ОТРИМАННЯ ПРОГНОЗУ (ПЕРЕДБАЧЕННЯ НА ТРЕНУВАЛЬНОМУ НАБОРІ) ===
# Вибираємо модель (тут SVM) і навчаємо на навчальній вибірці
model = SVC(gamma='auto')
model.fit(X_train, Y_train)

# Робимо прогноз на тестовій (контрольній) вибірці
#predictions = model.predict(X_validation)

# === КРОК 7. ОЦІНКА ЯКОСТІ МОДЕЛІ ===

#accuracy = accuracy_score(Y_validation, predictions)
#print("Точність моделі на контрольному наборі: %.2f%%" % (accuracy*100))

# Матриця плутанини
#print("\nМатриця плутанини:")
#print(confusion_matrix(Y_validation, predictions))

# Детальний звіт по метриках
#print("\nЗвіт по метриках класифікації:")
#print(classification_report(Y_validation, predictions))

# === КРОК 8. ОТРИМАННЯ ПРОГНОЗУ (ЗАСТОСУВАННЯ МОДЕЛІ ДЛЯ ПЕРЕДБАЧЕННЯ) ===

# Нові дані – вимірювання знайденої квітки
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new:", X_new.shape)

# Використовуємо вже навчену модель для прогнозу
prediction = model.predict(X_new)

# Виводимо результат
print("Прогнозований клас (індекс):", prediction)
print(f"\nПрогнозований сорт ірису:: {prediction[0]}")
